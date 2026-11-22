"""LLM module for TrumorGPT."""

from trumourgpt.llm.client import OllamaClient, OllamaClientConfig
from trumourgpt.llm.gemini_client import GeminiClient, SyncGeminiClient, GeminiClientConfig
from trumourgpt.llm.qwen_client import QwenClient, QwenClientConfig, get_qwen_client
from trumourgpt.llm.schemas import (
    ClaimDecomposition,
    TripleExtraction,
    EvidenceAnalysis,
    FactVerification,
    ExplanationGeneration,
    LLMResponse,
)

__all__ = [
    "OllamaClient",
    "OllamaClientConfig",
    "GeminiClient",
    "SyncGeminiClient",
    "GeminiClientConfig",
    "QwenClient",
    "QwenClientConfig",
    "get_qwen_client",
    "ClaimDecomposition",
    "TripleExtraction",
    "EvidenceAnalysis",
    "FactVerification",
    "ExplanationGeneration",
    "LLMResponse",
]