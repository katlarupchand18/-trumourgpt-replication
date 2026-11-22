"""
TrumorGPT Configuration Management
Uses Pydantic Settings for type-safe configuration with YAML + environment variable support.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml


class LLMConfig(BaseSettings):
    """LLM/Ollama configuration."""
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2:3b"
    embedding_model: str = "nomic-embed-text:latest"

    # Gemini configuration
    gemini_model: str = "gemini-1.5-flash"

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


class EmbeddingsConfig(BaseSettings):
    """Embedding model configuration."""
    provider: str = "ollama"
    model: str = "nomic-embed-text:latest"
    batch_size: int = 32
    normalize: bool = True
    cache_dir: str = "data/cache/embeddings"
    dimension: int = 768


class TopicModelingConfig(BaseSettings):
    """LDA topic modeling configuration."""
    num_topics: int = 20
    alpha: str = "auto"
    eta: str = "auto"
    iterations: int = 1000
    passes: int = 10
    random_state: int = 42
    min_df: int = 5
    max_df: float = 0.95

    corpus_source: str = "pubmed_abstracts"
    corpus_path: str = "data/corpus/health_corpus.txt"
    model_path: str = "data/cache/lda_model"

    health_topic_keywords: List[str] = Field(default_factory=lambda: [
        "health", "medical", "medicine", "disease", "pandemic", "epidemic",
        "virus", "vaccine", "treatment", "therapy", "diagnosis", "symptom",
        "patient", "hospital", "doctor", "nurse", "clinical", "trial",
        "drug", "pharmaceutical", "fda", "cdc", "who", "public health",
        "covid", "coronavirus", "sars", "infection", "immunity", "antibody"
    ])
    health_topic_boost: float = 1.5


class SentenceCentralityConfig(BaseSettings):
    """Topic-enhanced sentence centrality configuration (Paper §III-B)."""
    eta: float = 0.7  # Paper: η = 0.7

    similarity_threshold: float = 0.1
    max_edges_per_node: int = 50

    pagerank_alpha: float = 0.85
    pagerank_max_iter: int = 100
    pagerank_tol: float = 1e-6

    top_k_sentences: int = 10


class TopicTextRankConfig(BaseSettings):
    """Topic-Specific TextRank configuration (Paper §III-C)."""
    damping_factor: float = 0.85
    max_iterations: int = 500
    convergence_tol: float = 1e-6

    health_topic_boost: float = 1.5  # Paper: α = 1.5
    similarity_threshold: float = 0.1

    top_k_keywords: int = 20


class KnowledgeGraphConfig(BaseSettings):
    """Semantic health knowledge graph configuration."""
    backend: str = "networkx"
    storage_path: str = "data/cache/knowledge_graph"

    entity_types: List[str] = Field(default_factory=lambda: [
        "DISEASE", "MEDICATION", "VACCINE", "HEALTH_ORGANIZATION",
        "PUBLIC_POLICY", "MEDICAL_CONDITION", "QUANTITY", "DATE", "LOCATION"
    ])

    relation_types: List[str] = Field(default_factory=lambda: [
        "causes", "prevents", "treats", "increases", "decreases",
        "associated_with", "approved_by", "recommended_by", "contradicts"
    ])

    few_shot_examples: int = 5
    max_triples_per_claim: int = 50
    min_confidence: float = 0.5
    deduplicate: bool = True


class GraphRAGConfig(BaseSettings):
    """GraphRAG retrieval configuration (Paper §III-E)."""
    use_weighted_jaccard: bool = False
    weight_by_centrality: bool = False

    top_k_graphs: int = 5
    similarity_threshold: float = 0.005

    dbpedia_path: str = "data/dbpedia/health_triples.nt"
    dbpedia_filter_keywords: List[str] = Field(default_factory=lambda: [
        "health", "medical", "medicine", "disease", "pandemic", "epidemic",
        "virus", "vaccine", "treatment", "therapy", "hospital", "clinical"
    ])


class VerificationConfig(BaseSettings):
    """Fact verification configuration."""
    labels: List[str] = Field(default_factory=lambda: ["TRUE", "FALSE", "UNDETERMINED"])
    few_shot_examples: int = 3
    max_evidence_passages: int = 10
    max_evidence_tokens: int = 4000
    check_negation: bool = True
    check_temporal: bool = True
    check_numerical: bool = True


class DatasetConfig(BaseSettings):
    """Dataset configuration (Paper §IV-B)."""
    label_mapping: Dict[str, List[str]] = Field(default_factory=lambda: {
        "TRUE": ["True", "Mostly True", "Half True"],
        "FALSE": ["Mostly False", "False", "Pants on Fire"]
    })

    target_size: int = 600
    true_count: int = 300
    false_count: int = 300

    categories: List[str] = Field(default_factory=lambda: ["Health Care", "Coronavirus"])

    splits: Dict[str, int] = Field(default_factory=lambda: {
        "dev": 100, "val": 100, "test": 400
    })

    raw_path: str = "data/raw/politifact_claims.json"
    processed_path: str = "data/processed/politifact_processed.json"
    splits_path: str = "data/processed/splits.json"


class EvaluationConfig(BaseSettings):
    """Evaluation configuration."""
    metrics: List[str] = Field(default_factory=lambda: [
        "accuracy", "precision", "recall", "f1", "confusion_matrix"
    ])
    per_class_metrics: bool = True
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000
    results_dir: str = "experiments/results"
    save_predictions: bool = True
    save_error_analysis: bool = True


class ExperimentConfig(BaseSettings):
    """Experiment tracking configuration."""
    name: str = "trumourgpt_replication"
    track_api_usage: bool = True
    track_tokens: bool = True
    track_latency: bool = True
    checkpoint_interval: int = 50
    seed: int = 42
    deterministic: bool = True


class LoggingConfig(BaseSettings):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    file: str = "logs/trumourgpt.log"
    rotation: str = "10 MB"
    retention: str = "7 days"


class Settings(BaseSettings):
    """Main settings container."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore"
    )

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    topic_modeling: TopicModelingConfig = Field(default_factory=TopicModelingConfig)
    sentence_centrality: SentenceCentralityConfig = Field(default_factory=SentenceCentralityConfig)
    topic_textrank: TopicTextRankConfig = Field(default_factory=TopicTextRankConfig)
    knowledge_graph: KnowledgeGraphConfig = Field(default_factory=KnowledgeGraphConfig)
    graphrag: GraphRAGConfig = Field(default_factory=GraphRAGConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Settings":
        """Load settings from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


# Global settings instance
_settings: Optional[Settings] = None


def get_settings(config_path: Optional[str | Path] = None) -> Settings:
    """Get or create global settings instance."""
    global _settings
    if _settings is None:
        if config_path and Path(config_path).exists():
            _settings = Settings.from_yaml(config_path)
        else:
            _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset global settings (useful for testing)."""
    global _settings
    _settings = None