"""
Google Gemini API Client for TrumorGPT.
Supports dual-key rotation with rate limiting (15 RPM, 500 RPD per key).
Replaces Ollama client for verification tasks.
"""

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import httpx
from loguru import logger
from pydantic import BaseModel, ValidationError

# Load .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from trumourgpt.config.settings import get_settings
from trumourgpt.llm.schemas import (
    ClaimDecomposition,
    EvidenceAnalysis,
    ExplanationGeneration,
    FactVerification,
    LLMResponse,
    TASK_MODELS,
    TASK_SCHEMAS,
    TripleExtraction,
)

T = TypeVar("T", bound=BaseModel)


@dataclass
class GeminiClientConfig:
    """Configuration for Gemini client."""
    # Dual API keys for rotation
    api_key_1: str = ""
    api_key_2: str = ""

    # Model configuration
    model: str = "gemini-1.5-flash"  # Default to 1.5 Flash
    temperature: float = 0.0
    top_p: float = 0.9
    top_k: int = 40
    max_output_tokens: int = 2048

    # Rate limiting per key
    rpm_limit: int = 15          # Requests per minute
    rpd_limit: int = 500         # Requests per day

    # Timeouts and retries
    timeout_seconds: int = 120
    max_retries: int = 5
    retry_backoff_factor: float = 2.0
    # Special rate limit handling
    rate_limit_wait_seconds: int = 65  # Wait for minute window to reset

    # JSON format
    use_json_format: bool = True
    json_schema_validation: bool = True

    # Caching
    cache_dir: str = "data/cache/gemini"
    enable_cache: bool = True


@dataclass
class RateLimiterState:
    """State for rate limiting per API key."""
    key_index: int = 0
    minute_requests: int = 0
    day_requests: int = 0
    minute_window_start: float = field(default_factory=time.time)
    day_window_start: float = field(default_factory=time.time)
    last_request_time: float = 0


@dataclass
class TokenUsage:
    """Token usage tracking."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class CacheEntry:
    """Cache entry for LLM responses."""
    prompt_hash: str
    response: LLMResponse
    timestamp: float = field(default_factory=time.time)


class GeminiClient:
    """
    Async Gemini client with dual-key rotation, rate limiting,
    structured output, retries, caching, and token tracking.
    """

    def __init__(self, config: Optional[GeminiClientConfig] = None):
        self.config = config or self._load_config()
        self._client: Optional[httpx.AsyncClient] = None
        self._cache: Dict[str, CacheEntry] = {}
        self._cache_path = Path(self.config.cache_dir)
        self._cache_path.mkdir(parents=True, exist_ok=True)
        self._token_usage = TokenUsage()
        self._rate_limiter = RateLimiterState()
        self._load_cache()

        # Validate API keys
        if not self.config.api_key_1 and not self.config.api_key_2:
            # Try to load from environment
            self.config.api_key_1 = os.environ.get("GEMINI_API_KEY_1", "")
            self.config.api_key_2 = os.environ.get("GEMINI_API_KEY_2", "")

        if not self.config.api_key_1 and not self.config.api_key_2:
            logger.warning("No Gemini API keys configured. Set GEMINI_API_KEY_1 and GEMINI_API_KEY_2 in .env or pass to config.")

    def _load_config(self) -> GeminiClientConfig:
        """Load config from settings/environment."""
        settings = get_settings()

        # Explicitly read from environment (in case .env wasn't loaded by settings)
        api_key_1 = os.environ.get("GEMINI_API_KEY_1", "")
        api_key_2 = os.environ.get("GEMINI_API_KEY_2", "")

        # Also try to read from .env file directly if env vars not set
        if not api_key_1 or not api_key_2:
            env_path = Path(".env")
            if env_path.exists():
                try:
                    with open(env_path) as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith("GEMINI_API_KEY_1="):
                                api_key_1 = line.split("=", 1)[1].strip()
                            elif line.startswith("GEMINI_API_KEY_2="):
                                api_key_2 = line.split("=", 1)[1].strip()
                except Exception as e:
                    logger.warning(f"Failed to read .env file: {e}")

        return GeminiClientConfig(
            api_key_1=api_key_1,
            api_key_2=api_key_2,
            model=getattr(settings.llm, "gemini_model", "gemini-1.5-flash"),
            temperature=settings.llm.temperature,
            top_p=settings.llm.top_p,
            top_k=settings.llm.top_k,
            max_output_tokens=settings.llm.num_predict,
            timeout_seconds=settings.llm.timeout_seconds,
            max_retries=settings.llm.max_retries,
            retry_backoff_factor=settings.llm.retry_backoff_factor,
            use_json_format=settings.llm.use_json_format,
            json_schema_validation=settings.llm.json_schema_validation,
        )

    def _get_current_key(self) -> str:
        """Get the current API key based on rotation."""
        keys = [k for k in [self.config.api_key_1, self.config.api_key_2] if k]
        if not keys:
            raise ValueError("No Gemini API keys available")
        return keys[self._rate_limiter.key_index % len(keys)]

    def _rotate_key(self) -> None:
        """Rotate to the next API key."""
        keys = [k for k in [self.config.api_key_1, self.config.api_key_2] if k]
        if len(keys) > 1:
            self._rate_limiter.key_index = (self._rate_limiter.key_index + 1) % len(keys)
            # Reset counters for new key
            self._rate_limiter.minute_requests = 0
            self._rate_limiter.minute_window_start = time.time()
            logger.info(f"Rotated to API key index {self._rate_limiter.key_index}")

    async def _check_rate_limit(self) -> None:
        """Check and enforce rate limits, rotating keys if needed."""
        now = time.time()

        # Reset minute window if expired
        if now - self._rate_limiter.minute_window_start >= 60:
            self._rate_limiter.minute_requests = 0
            self._rate_limiter.minute_window_start = now

        # Reset day window if expired
        if now - self._rate_limiter.day_window_start >= 86400:
            self._rate_limiter.day_requests = 0
            self._rate_limiter.day_window_start = now

        # Check if current key is rate limited
        keys = [k for k in [self.config.api_key_1, self.config.api_key_2] if k]
        if len(keys) > 1:
            # Try current key first
            if (self._rate_limiter.minute_requests >= self.config.rpm_limit or
                self._rate_limiter.day_requests >= self.config.rpd_limit):
                self._rotate_key()
                # After rotation, re-check
                await self._check_rate_limit()
        else:
            # Single key - wait if rate limited
            if self._rate_limiter.minute_requests >= self.config.rpm_limit:
                wait_time = 60 - (now - self._rate_limiter.minute_window_start)
                if wait_time > 0:
                    logger.warning(f"Rate limit reached, waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                    self._rate_limiter.minute_requests = 0
                    self._rate_limiter.minute_window_start = time.time()

            if self._rate_limiter.day_requests >= self.config.rpd_limit:
                wait_time = 86400 - (now - self._rate_limiter.day_window_start)
                if wait_time > 0:
                    logger.warning(f"Daily rate limit reached, waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                    self._rate_limiter.day_requests = 0
                    self._rate_limiter.day_window_start = time.time()

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout_seconds)
            )
        return self._client

    def _hash_prompt(self, prompt: str, schema: Optional[Dict] = None) -> str:
        """Create deterministic hash for prompt + schema."""
        content = prompt + (json.dumps(schema, sort_keys=True) if schema else "")
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def _load_cache(self) -> None:
        """Load response cache from disk."""
        cache_file = self._cache_path / "response_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                for entry_data in data.get("entries", []):
                    entry = CacheEntry(
                        prompt_hash=entry_data["prompt_hash"],
                        response=LLMResponse(**entry_data["response"]),
                        timestamp=entry_data["timestamp"]
                    )
                    self._cache[entry.prompt_hash] = entry
                logger.info(f"Loaded {len(self._cache)} cached Gemini responses")
            except Exception as e:
                logger.warning(f"Failed to load Gemini cache: {e}")

    def _save_cache(self) -> None:
        """Save response cache to disk."""
        if not self.config.enable_cache:
            return
        cache_file = self._cache_path / "response_cache.json"
        try:
            data = {
                "entries": [
                    {
                        "prompt_hash": entry.prompt_hash,
                        "response": entry.response.model_dump(),
                        "timestamp": entry.timestamp
                    }
                    for entry in self._cache.values()
                ]
            }
            with open(cache_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save Gemini cache: {e}")

    def _get_cached(self, prompt_hash: str) -> Optional[LLMResponse]:
        """Get cached response if exists."""
        if not self.config.enable_cache:
            return None
        entry = self._cache.get(prompt_hash)
        if entry:
            logger.debug(f"Cache hit for prompt hash: {prompt_hash[:8]}...")
            return entry.response
        return None

    def _set_cached(self, prompt_hash: str, response: LLMResponse) -> None:
        """Cache response."""
        if not self.config.enable_cache:
            return
        self._cache[prompt_hash] = CacheEntry(prompt_hash=prompt_hash, response=response)

    def _convert_schema_for_gemini(self, schema: Dict) -> Dict:
        """Convert JSON schema to Gemini-compatible format.

        Gemini API has specific requirements:
        - No 'additionalProperties' at root level
        - All properties must be defined explicitly
        - 'type' field format differences
        """
        import copy
        converted = copy.deepcopy(schema)

        # Remove additionalProperties from root level
        if "additionalProperties" in converted:
            del converted["additionalProperties"]

        # Recursively clean the schema
        def clean_schema(obj):
            if isinstance(obj, dict):
                # Remove additionalProperties at any level
                if "additionalProperties" in obj:
                    del obj["additionalProperties"]

                # Process properties
                if "properties" in obj:
                    for prop_name, prop_schema in obj["properties"].items():
                        clean_schema(prop_schema)

                # Process items in arrays
                if "items" in obj:
                    clean_schema(obj["items"])

                # Process anyOf, allOf, oneOf
                for key in ["anyOf", "allOf", "oneOf"]:
                    if key in obj:
                        for item in obj[key]:
                            clean_schema(item)

                # Fix type field - ensure it's a string, not a list
                if "type" in obj and isinstance(obj["type"], list):
                    # Use the first non-null type
                    types = [t for t in obj["type"] if t != "null"]
                    if types:
                        obj["type"] = types[0]
                    else:
                        obj["type"] = "string"

        clean_schema(converted)
        return converted

    def _build_payload(self, prompt: str, schema: Optional[Dict] = None) -> Dict[str, Any]:
        """Build the request payload for Gemini API."""
        # Build generation config
        generation_config = {
            "temperature": self.config.temperature,
            "topP": self.config.top_p,
            "topK": self.config.top_k,
            "maxOutputTokens": self.config.max_output_tokens,
        }

        # Add response schema if using JSON format
        if self.config.use_json_format and schema:
            generation_config["responseMimeType"] = "application/json"
            # Convert schema to Gemini-compatible format
            generation_config["responseSchema"] = self._convert_schema_for_gemini(schema)

        return {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": generation_config,
        }

    async def _generate_with_retry(
        self,
        prompt: str,
        schema: Optional[Dict] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate with exponential backoff retry and rate limiting."""
        await self._check_rate_limit()

        client = self._get_client()
        model = model or self.config.model
        temperature = temperature if temperature is not None else self.config.temperature

        payload = self._build_payload(prompt, schema)
        api_key = self._get_current_key()

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

        last_error = None
        for attempt in range(self.config.max_retries + 1):
            try:
                start_time = time.time()
                response = await client.post(url, json=payload)
                latency_ms = int((time.time() - start_time) * 1000)

                # Update rate limiter
                self._rate_limiter.minute_requests += 1
                self._rate_limiter.day_requests += 1
                self._rate_limiter.last_request_time = time.time()

                if response.status_code == 429:
                    # Rate limited - try rotating key first
                    logger.warning(f"Rate limited (429), rotating key...")
                    self._rotate_key()
                    api_key = self._get_current_key()
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
                    last_error = "Rate limited, rotated key"

                    # Check if both keys are exhausted - if so, wait for minute window
                    if self._rate_limiter.minute_requests >= self.config.rpm_limit:
                        wait_time = self.config.rate_limit_wait_seconds
                        logger.warning(f"Both keys rate limited. Waiting {wait_time}s for minute window reset...")
                        await asyncio.sleep(wait_time)
                        # Reset minute counter after wait
                        self._rate_limiter.minute_requests = 0
                        self._rate_limiter.minute_window_start = time.time()
                    else:
                        await asyncio.sleep(self.config.retry_backoff_factor ** attempt)
                    continue

                response.raise_for_status()
                data = response.json()

                # Extract response text and token usage
                candidates = data.get("candidates", [])
                if not candidates:
                    raise RuntimeError("No candidates in response")

                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if not parts:
                    raise RuntimeError("No parts in response")

                response_text = parts[0].get("text", "")

                # Token usage from usageMetadata
                usage = data.get("usageMetadata", {})
                prompt_tokens = usage.get("promptTokenCount", 0)
                completion_tokens = usage.get("candidatesTokenCount", 0)

                self._token_usage.prompt_tokens += prompt_tokens
                self._token_usage.completion_tokens += completion_tokens
                self._token_usage.total_tokens = (
                    self._token_usage.prompt_tokens + self._token_usage.completion_tokens
                )

                return {
                    "response": response_text,
                    "latency_ms": latency_ms,
                    "token_usage": {
                        "prompt_tokens": self._token_usage.prompt_tokens,
                        "completion_tokens": self._token_usage.completion_tokens,
                        "total_tokens": self._token_usage.total_tokens,
                    }
                }

            except httpx.TimeoutException as e:
                last_error = f"Timeout after {self.config.timeout_seconds}s: {e}"
            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}: {e.response.text}"
                if e.response.status_code == 400:
                    # Bad request - don't retry
                    break
            except Exception as e:
                last_error = f"Unexpected error: {e}"

            if attempt < self.config.max_retries:
                wait_time = self.config.retry_backoff_factor ** attempt
                logger.warning(f"Attempt {attempt + 1} failed: {last_error}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

        raise RuntimeError(f"All retries exhausted. Last error: {last_error}")

    async def generate_structured(
        self,
        task: str,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """
        Generate structured output for a specific task.

        Args:
            task: Task identifier (claim_decomposition, triple_extraction, etc.)
            prompt: Input prompt
            model: Override model
            temperature: Override temperature

        Returns:
            LLMResponse with validated data
        """
        if task not in TASK_SCHEMAS:
            raise ValueError(f"Unknown task: {task}. Available: {list(TASK_SCHEMAS.keys())}")

        schema = TASK_SCHEMAS[task] if self.config.use_json_format else None
        model_class = TASK_MODELS[task]

        # Check cache
        prompt_hash = self._hash_prompt(prompt, schema)
        cached = self._get_cached(prompt_hash)
        if cached:
            return cached

        # Generate
        logger.debug(f"Generating {task} with model {model or self.config.model}")
        result = await self._generate_with_retry(prompt, schema, model, temperature)

        # Parse and validate
        raw_response = result["response"]
        success = True
        data = None
        error = None

        try:
            if self.config.use_json_format:
                parsed = json.loads(raw_response)
                validated = model_class(**parsed)
                data = validated.model_dump()
            else:
                # Try to parse as JSON anyway
                parsed = json.loads(raw_response)
                validated = model_class(**parsed)
                data = validated.model_dump()
        except (json.JSONDecodeError, ValidationError) as e:
            success = False
            error = f"Validation failed: {e}"
            logger.error(f"Structured output validation failed for {task}: {e}")
            logger.debug(f"Raw response: {raw_response[:500]}...")

        response = LLMResponse(
            task=task,
            model=model or self.config.model,
            success=success,
            data=data,
            raw_response=raw_response,
            error=error,
            latency_ms=result["latency_ms"],
            token_usage=result["token_usage"]
        )

        # Cache successful responses
        if success:
            self._set_cached(prompt_hash, response)

        return response

    async def generate_embeddings(
        self,
        texts: Union[str, List[str]],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """Generate embeddings using Gemini embedding model."""
        # Note: We'll use Ollama for embeddings since we don't have Gemini embeddings set up
        # This is a placeholder that returns dummy embeddings
        if isinstance(texts, str):
            texts = [texts]
        logger.warning("Gemini embeddings not implemented, returning dummy embeddings")
        return [[0.0] * 768 for _ in texts]

    def get_token_usage(self) -> TokenUsage:
        """Get current token usage statistics."""
        return self._token_usage

    def reset_token_usage(self) -> None:
        """Reset token usage counters."""
        self._token_usage = TokenUsage()

    async def health_check(self) -> bool:
        """Check if Gemini API is healthy."""
        try:
            client = self._get_client()
            api_key = self._get_current_key()
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.config.model}?key={api_key}"
            response = await client.get(url)
            return response.status_code == 200
        except Exception:
            return False

    def clear_cache(self) -> None:
        """Clear response cache."""
        self._cache.clear()
        cache_file = self._cache_path / "response_cache.json"
        if cache_file.exists():
            cache_file.unlink()
        logger.info("Gemini response cache cleared")


# Synchronous wrapper for simpler usage
class SyncGeminiClient:
    """Synchronous wrapper around GeminiClient."""

    def __init__(self, config: Optional[GeminiClientConfig] = None):
        self._async_client = GeminiClient(config)

    def __enter__(self) -> "SyncGeminiClient":
        return self

    def __exit__(self, *args) -> None:
        # Run cleanup in event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        loop.run_until_complete(self._async_client.__aexit__(None, None, None))

    def _run_async(self, coro):
        """Run async coroutine, handling both running and non-running event loops."""
        # First check if there's a running loop
        try:
            loop = asyncio.get_running_loop()
            # We're in a running loop - run in a separate thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        except RuntimeError:
            # No running loop
            pass

        # Try to get existing loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Loop exists but is running - use thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()

        # No loop or loop not running - create new one
        if loop is None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            return loop.run_until_complete(coro)
        finally:
            # Don't close the loop if it was pre-existing
            pass

    def generate_structured(
        self,
        task: str,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """Synchronous structured generation."""
        return self._run_async(
            self._async_client.generate_structured(task, prompt, model, temperature)
        )

    def generate_embeddings(
        self,
        texts: Union[str, List[str]],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """Synchronous embedding generation."""
        return self._run_async(
            self._async_client.generate_embeddings(texts, model)
        )

    def health_check(self) -> bool:
        """Synchronous health check."""
        return self._run_async(self._async_client.health_check())

    def get_token_usage(self) -> TokenUsage:
        """Get token usage."""
        return self._async_client.get_token_usage()

    def reset_token_usage(self) -> None:
        """Reset token usage."""
        self._async_client.reset_token_usage()

    def clear_cache(self) -> None:
        """Clear cache."""
        self._async_client.clear_cache()


def get_gemini_client(config: Optional[GeminiClientConfig] = None) -> GeminiClient:
    """Get async Gemini client instance."""
    return GeminiClient(config)


def get_sync_gemini_client(config: Optional[GeminiClientConfig] = None) -> SyncGeminiClient:
    """Get synchronous Gemini client instance."""
    return SyncGeminiClient(config)