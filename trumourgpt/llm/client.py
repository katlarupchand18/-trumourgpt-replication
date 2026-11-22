"""
Ollama HTTP Client for TrumorGPT.
Supports structured JSON output, retries, caching, and token tracking.
"""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import httpx
from loguru import logger
from pydantic import BaseModel, ValidationError

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
class OllamaClientConfig:
    """Configuration for Ollama client."""
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2:3b"
    embedding_model: str = "nomic-embed-text:latest"
    temperature: float = 0.0
    top_p: float = 0.9
    top_k: int = 40
    num_predict: int = 2048
    num_ctx: int = 8192
    timeout_seconds: int = 120
    max_retries: int = 3
    retry_backoff_factor: float = 2.0
    use_json_format: bool = True
    json_schema_validation: bool = True
    cache_dir: str = "data/cache/llm"
    enable_cache: bool = True


@dataclass
class TokenUsage:
    """Token usage tracking."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_eval_count: int = 0
    eval_count: int = 0


@dataclass
class CacheEntry:
    """Cache entry for LLM responses."""
    prompt_hash: str
    response: LLMResponse
    timestamp: float = field(default_factory=time.time)


class OllamaClient:
    """
    Async Ollama client with structured output, retries, caching, and validation.
    """

    def __init__(self, config: Optional[OllamaClientConfig] = None):
        self.config = config or self._load_config()
        self._client: Optional[httpx.AsyncClient] = None
        self._cache: Dict[str, CacheEntry] = {}
        self._cache_path = Path(self.config.cache_dir)
        self._cache_path.mkdir(parents=True, exist_ok=True)
        self._token_usage = TokenUsage()
        self._load_cache()

    def _load_config(self) -> OllamaClientConfig:
        """Load config from settings."""
        settings = get_settings()
        return OllamaClientConfig(
            base_url=settings.llm.base_url,
            model=settings.llm.model,
            embedding_model=settings.llm.embedding_model,
            temperature=settings.llm.temperature,
            top_p=settings.llm.top_p,
            top_k=settings.llm.top_k,
            num_predict=settings.llm.num_predict,
            num_ctx=settings.llm.num_ctx,
            timeout_seconds=settings.llm.timeout_seconds,
            max_retries=settings.llm.max_retries,
            retry_backoff_factor=settings.llm.retry_backoff_factor,
            use_json_format=settings.llm.use_json_format,
            json_schema_validation=settings.llm.json_schema_validation,
        )

    async def __aenter__(self) -> "OllamaClient":
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=httpx.Timeout(self.config.timeout_seconds)
        )
        return self

    async def __aexit__(self, *args) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        self._save_cache()

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
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
                logger.info(f"Loaded {len(self._cache)} cached LLM responses")
            except Exception as e:
                logger.warning(f"Failed to load LLM cache: {e}")

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
            logger.warning(f"Failed to save LLM cache: {e}")

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

    async def _generate_with_retry(
        self,
        prompt: str,
        schema: Optional[Dict] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate with exponential backoff retry."""
        client = self._get_client()
        model = model or self.config.model
        temperature = temperature if temperature is not None else self.config.temperature

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "num_predict": self.config.num_predict,
                "num_ctx": self.config.num_ctx,
            }
        }

        if self.config.use_json_format and schema:
            payload["format"] = schema

        last_error = None
        for attempt in range(self.config.max_retries + 1):
            try:
                start_time = time.time()
                response = await client.post("/api/generate", json=payload)
                latency_ms = int((time.time() - start_time) * 1000)

                response.raise_for_status()
                data = response.json()

                # Track token usage
                self._token_usage.prompt_tokens += data.get("prompt_eval_count", 0)
                self._token_usage.completion_tokens += data.get("eval_count", 0)
                self._token_usage.total_tokens = (
                    self._token_usage.prompt_tokens + self._token_usage.completion_tokens
                )
                self._token_usage.prompt_eval_count = data.get("prompt_eval_count", 0)
                self._token_usage.eval_count = data.get("eval_count", 0)

                return {
                    "response": data.get("response", ""),
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
        """Generate embeddings using Ollama embedding model."""
        client = self._get_client()
        model = model or self.config.embedding_model

        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            payload = {"model": model, "prompt": text}
            response = await client.post("/api/embeddings", json=payload)
            response.raise_for_status()
            data = response.json()
            embeddings.append(data["embedding"])

        return embeddings

    async def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        model: Optional[str] = None
    ) -> List[List[float]]:
        """Generate embeddings in batches."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = await self.generate_embeddings(batch, model)
            all_embeddings.extend(embeddings)
        return all_embeddings

    def get_token_usage(self) -> TokenUsage:
        """Get current token usage statistics."""
        return self._token_usage

    def reset_token_usage(self) -> None:
        """Reset token usage counters."""
        self._token_usage = TokenUsage()

    async def health_check(self) -> bool:
        """Check if Ollama server is healthy."""
        try:
            client = self._get_client()
            response = await client.get("/api/tags")
            return response.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        client = self._get_client()
        response = await client.get("/api/tags")
        response.raise_for_status()
        return response.json().get("models", [])

    def clear_cache(self) -> None:
        """Clear response cache."""
        self._cache.clear()
        cache_file = self._cache_path / "response_cache.json"
        if cache_file.exists():
            cache_file.unlink()
        logger.info("LLM response cache cleared")


# Synchronous wrapper for simpler usage
class SyncOllamaClient:
    """Synchronous wrapper around OllamaClient."""

    def __init__(self, config: Optional[OllamaClientConfig] = None):
        self._async_client = OllamaClient(config)

    def __enter__(self) -> "SyncOllamaClient":
        return self

    def __exit__(self, *args) -> None:
        # Run cleanup in event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        loop.run_until_complete(self._async_client.__aexit__(None, None, None))

    def generate_structured(
        self,
        task: str,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """Synchronous structured generation."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            self._async_client.generate_structured(task, prompt, model, temperature)
        )

    def generate_embeddings(
        self,
        texts: Union[str, List[str]],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """Synchronous embedding generation."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            self._async_client.generate_embeddings(texts, model)
        )

    def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        model: Optional[str] = None
    ) -> List[List[float]]:
        """Synchronous batch embedding generation."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            self._async_client.generate_embeddings_batch(texts, batch_size, model)
        )

    def health_check(self) -> bool:
        """Synchronous health check."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self._async_client.health_check())

    def list_models(self) -> List[Dict[str, Any]]:
        """Synchronous model listing."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self._async_client.list_models())

    def get_token_usage(self) -> TokenUsage:
        """Get token usage."""
        return self._async_client.get_token_usage()

    def reset_token_usage(self) -> None:
        """Reset token usage."""
        self._async_client.reset_token_usage()

    def clear_cache(self) -> None:
        """Clear cache."""
        self._async_client.clear_cache()


# Convenience function for quick access
def get_ollama_client(config: Optional[OllamaClientConfig] = None) -> OllamaClient:
    """Get async Ollama client instance."""
    return OllamaClient(config)


def get_sync_ollama_client(config: Optional[OllamaClientConfig] = None) -> SyncOllamaClient:
    """Get synchronous Ollama client instance."""
    return SyncOllamaClient(config)