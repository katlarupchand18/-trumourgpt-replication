"""
Ollama Embeddings Wrapper with Caching.
Supports batch encoding, normalization, and disk caching for reproducibility.
"""

import hashlib
import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import numpy as np

from loguru import logger
from trumourgpt.config.settings import get_settings
from trumourgpt.llm.client import SyncOllamaClient, OllamaClientConfig


@dataclass
class EmbeddingConfig:
    """Configuration for embeddings."""
    model: str = "nomic-embed-text:latest"
    batch_size: int = 32
    normalize: bool = True
    cache_dir: str = "data/cache/embeddings"
    dimension: int = 768
    ollama_base_url: str = "http://localhost:11434"
    timeout_seconds: int = 60


@dataclass
class CacheEntry:
    """Cache entry for embeddings."""
    text_hash: str
    embedding: np.ndarray
    model: str
    timestamp: float = field(default_factory=time.time)


class EmbeddingCache:
    """Disk-based embedding cache for reproducibility."""

    def __init__(self, cache_dir: str, model: str):
        self.cache_dir = Path(cache_dir) / model.replace(":", "_").replace("/", "_")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self._memory_cache: Dict[str, np.ndarray] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load cache index."""
        index_file = self.cache_dir / "index.json"
        if index_file.exists():
            try:
                with open(index_file) as f:
                    self._index = json.load(f)
                logger.debug(f"Loaded embedding cache index with {len(self._index)} entries")
            except Exception as e:
                logger.warning(f"Failed to load embedding cache index: {e}")
                self._index = {}
        else:
            self._index = {}

    def _save_index(self) -> None:
        """Save cache index."""
        index_file = self.cache_dir / "index.json"
        try:
            with open(index_file, "w") as f:
                json.dump(self._index, f)
        except Exception as e:
            logger.warning(f"Failed to save embedding cache index: {e}")

    def _get_hash(self, text: str) -> str:
        """Compute hash for text."""
        return hashlib.sha256(text.encode()).hexdigest()[:32]

    def get(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache."""
        text_hash = self._get_hash(text)

        # Check memory cache first
        if text_hash in self._memory_cache:
            return self._memory_cache[text_hash]

        # Check disk cache
        if text_hash in self._index:
            cache_file = self.cache_dir / f"{text_hash}.npy"
            if cache_file.exists():
                try:
                    embedding = np.load(cache_file)
                    self._memory_cache[text_hash] = embedding
                    return embedding
                except Exception as e:
                    logger.warning(f"Failed to load cached embedding: {e}")

        return None

    def set(self, text: str, embedding: np.ndarray) -> None:
        """Store embedding in cache."""
        text_hash = self._get_hash(text)

        # Save to memory
        self._memory_cache[text_hash] = embedding

        # Save to disk
        cache_file = self.cache_dir / f"{text_hash}.npy"
        try:
            np.save(cache_file, embedding)
            self._index[text_hash] = {
                "model": self.model,
                "timestamp": time.time(),
                "dimension": embedding.shape[0]
            }
            self._save_index()
        except Exception as e:
            logger.warning(f"Failed to save embedding to cache: {e}")

    def clear(self) -> None:
        """Clear cache."""
        self._memory_cache.clear()
        self._index.clear()
        for f in self.cache_dir.glob("*.npy"):
            f.unlink()
        index_file = self.cache_dir / "index.json"
        if index_file.exists():
            index_file.unlink()
        logger.info("Embedding cache cleared")

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "model": self.model,
            "entries": len(self._index),
            "memory_entries": len(self._memory_cache),
            "cache_dir": str(self.cache_dir),
        }


class OllamaEmbeddings:
    """
    Ollama embeddings wrapper with batching, normalization, and caching.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or self._load_config()
        self._client: Optional[SyncOllamaClient] = None
        self._cache = EmbeddingCache(self.config.cache_dir, self.config.model)
        self._embedding_dim: Optional[int] = None

    def _load_config(self) -> EmbeddingConfig:
        """Load config from settings."""
        settings = get_settings()
        return EmbeddingConfig(
            model=settings.embeddings.model,
            batch_size=settings.embeddings.batch_size,
            normalize=settings.embeddings.normalize,
            cache_dir=settings.embeddings.cache_dir,
            dimension=settings.embeddings.dimension,
            ollama_base_url=settings.llm.base_url,
        )

    def _get_client(self) -> SyncOllamaClient:
        """Get or create Ollama client."""
        if self._client is None:
            client_config = OllamaClientConfig(
                base_url=self.config.ollama_base_url,
                embedding_model=self.config.model,
            )
            self._client = SyncOllamaClient(client_config)
        return self._client

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for processing
            show_progress: Show progress bar
            use_cache: Use disk cache

        Returns:
            Embeddings array of shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.zeros((0, self.get_dimension()))

        batch_size = batch_size or self.config.batch_size
        embeddings = []

        # Process in batches
        from tqdm import tqdm
        iterator = range(0, len(texts), batch_size)
        if show_progress and len(texts) > batch_size:
            iterator = tqdm(iterator, desc="Encoding embeddings", total=(len(texts) + batch_size - 1) // batch_size)

        for i in iterator:
            batch = texts[i:i + batch_size]
            batch_embeddings = self._encode_batch(batch, use_cache)
            embeddings.extend(batch_embeddings)

        result = np.array(embeddings, dtype=np.float32)

        # Normalize if requested
        if self.config.normalize:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            result = result / norms

        return result

    def _encode_batch(self, texts: List[str], use_cache: bool) -> List[np.ndarray]:
        """Encode a batch of texts."""
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        # Check cache
        if use_cache:
            for idx, text in enumerate(texts):
                cached = self._cache.get(text)
                if cached is not None:
                    embeddings.append(cached)
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(idx)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))

        # Encode uncached texts
        if uncached_texts:
            client = self._get_client()
            new_embeddings = client.generate_embeddings(uncached_texts, self.config.model)

            # Convert to numpy and cache
            for text, emb_list in zip(uncached_texts, new_embeddings):
                emb = np.array(emb_list, dtype=np.float32)
                if use_cache:
                    self._cache.set(text, emb)
                embeddings.append(emb)

        # Reorder to match input order
        if use_cache and uncached_indices:
            result = [None] * len(texts)
            cache_idx = 0
            uncached_idx = 0
            for i in range(len(texts)):
                if i in uncached_indices:
                    result[i] = embeddings[uncached_idx]
                    uncached_idx += 1
                else:
                    result[i] = embeddings[cache_idx]
                    cache_idx += 1
            return result

        return embeddings

    def encode_single(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Encode a single text."""
        return self.encode(text, use_cache=use_cache)[0]

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        if self._embedding_dim is None:
            # Get dimension from model info or test encode
            test_emb = self.encode_single("test", use_cache=False)
            self._embedding_dim = test_emb.shape[0]
        return self._embedding_dim

    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        emb1 = self.encode_single(text1)
        emb2 = self.encode_single(text2)
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    def similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """Compute pairwise cosine similarity matrix."""
        embeddings = self.encode(texts)
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = embeddings / norms
        # Cosine similarity
        return normalized @ normalized.T

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self._cache.stats()

    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self._cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self._client:
            self._client.__exit__(*args)


# Convenience functions
def create_embeddings(config: Optional[EmbeddingConfig] = None) -> OllamaEmbeddings:
    """Create OllamaEmbeddings instance."""
    return OllamaEmbeddings(config)


def encode_texts(
    texts: List[str],
    model: str = "nomic-embed-text:latest",
    batch_size: int = 32,
    normalize: bool = True,
    cache_dir: str = "data/cache/embeddings",
) -> np.ndarray:
    """Quick encode function."""
    config = EmbeddingConfig(
        model=model,
        batch_size=batch_size,
        normalize=normalize,
        cache_dir=cache_dir,
    )
    with OllamaEmbeddings(config) as embedder:
        return embedder.encode(texts)