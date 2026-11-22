"""
TrumorGPT Main Pipeline with Qwen2.5 backend.
Integrates all components: preprocessing, embeddings, topic modeling, sentence centrality,
TextRank, entity/relation extraction, knowledge graph, GraphRAG, verification.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

from trumourgpt.config.settings import get_settings, reset_settings
from trumourgpt.preprocessing.text import TextPreprocessor, PreprocessingConfig
from trumourgpt.embeddings.ollama_embeddings import OllamaEmbeddings, EmbeddingConfig
from trumourgpt.topic_modeling.lda import LDATopicModel, LDAConfig
from trumourgpt.sentence_centrality.centrality import SentenceCentrality, CentralityConfig
from trumourgpt.topic_textrank.textrank import TopicSpecificTextRank, TextRankConfig
from trumourgpt.entity_extraction.extractor import EntityExtractor, ExtractionConfig as EntityConfig
from trumourgpt.relation_extraction.extractor import RelationExtractor, ExtractionConfig as RelationConfig
from trumourgpt.knowledge_graph.graph import SemanticHealthKnowledgeGraph, KnowledgeGraphConfig
from trumourgpt.retrieval.graphrag import GraphRAGRetriever, RetrievalConfig
from trumourgpt.verification.fact_checker_qwen import FactCheckerQwen, VerificationConfig, VerificationResult
from trumourgpt.evaluation.dataset import PolitiFactDataset, DatasetConfig
from trumourgpt.evaluation.metrics import evaluate_binary, EvaluationMetrics
from trumourgpt.llm import QwenClientConfig


@dataclass
class PipelineConfig:
    """Configuration for the full pipeline."""
    # Component configs (use defaults from settings)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    topic_modeling: LDAConfig = field(default_factory=LDAConfig)
    sentence_centrality: CentralityConfig = field(default_factory=CentralityConfig)
    topic_textrank: TextRankConfig = field(default_factory=TextRankConfig)
    entity_extraction: EntityConfig = field(default_factory=EntityConfig)
    relation_extraction: RelationConfig = field(default_factory=RelationConfig)
    knowledge_graph: KnowledgeGraphConfig = field(default_factory=KnowledgeGraphConfig)
    graphrag: RetrievalConfig = field(default_factory=RetrievalConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # Qwen configuration
    qwen: QwenClientConfig = field(default_factory=QwenClientConfig)

    # Pipeline options
    use_sentence_centrality: bool = True
    use_topic_textrank: bool = True
    use_graphrag: bool = True
    save_intermediate: bool = True
    output_dir: str = "experiments/results"


@dataclass
class PipelineResult:
    """Result of pipeline execution on a claim."""
    claim_id: str
    claim_text: str
    verification_result: VerificationResult
    preprocessing_time: float
    embedding_time: float
    centrality_time: float
    textrank_time: float
    kg_time: float
    retrieval_time: float
    verification_time: float
    total_time: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "verification": self.verification_result.to_dict(),
            "timing": {
                "preprocessing": self.preprocessing_time,
                "embedding": self.embedding_time,
                "centrality": self.centrality_time,
                "textrank": self.textrank_time,
                "kg": self.kg_time,
                "retrieval": self.retrieval_time,
                "verification": self.verification_time,
                "total": self.total_time,
            }
        }


class TrumorGPTPipelineQwen:
    """
    Main TrumorGPT Pipeline with Qwen2.5 backend (Paper: complete workflow from query to fact verification).
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._components = {}
        self._initialized = False

    def initialize(self) -> None:
        """Initialize all pipeline components."""
        logger.info("Initializing TrumorGPT pipeline with Qwen2.5...")

        # Load settings from config.yaml (force reload to avoid cached defaults)
        reset_settings()
        settings = get_settings("config.yaml")

        # Preprocessing
        self._components["preprocessor"] = TextPreprocessor(self.config.preprocessing)

        # Embeddings
        self._components["embedder"] = OllamaEmbeddings(self.config.embeddings)

        # Topic modeling (LDA)
        self._components["lda_model"] = LDATopicModel(self.config.topic_modeling)
        # Try to load existing model
        if not self._components["lda_model"].load():
            logger.warning("LDA model not found. Will need to train.")

        # Sentence centrality
        self._components["centrality"] = SentenceCentrality(self.config.sentence_centrality)
        self._components["centrality"].set_models(
            self._components["embedder"],
            self._components["lda_model"]
        )

        # Topic-Specific TextRank
        self._components["textrank"] = TopicSpecificTextRank(self.config.topic_textrank)
        self._components["textrank"].set_lda_model(self._components["lda_model"])

        # Entity extraction
        self._components["entity_extractor"] = EntityExtractor(self.config.entity_extraction)

        # Relation extraction
        self._components["relation_extractor"] = RelationExtractor(self.config.relation_extraction)

        # Knowledge graph
        self._components["knowledge_graph"] = SemanticHealthKnowledgeGraph(self.config.knowledge_graph)

        # GraphRAG retriever - use config from settings (loaded from config.yaml)
        retriever_config = RetrievalConfig(
            use_weighted_jaccard=settings.graphrag.use_weighted_jaccard,
            weight_by_centrality=settings.graphrag.weight_by_centrality,
            top_k_graphs=settings.graphrag.top_k_graphs,
            similarity_threshold=settings.graphrag.similarity_threshold,
            dbpedia_path=settings.graphrag.dbpedia_path,
            dbpedia_filter_keywords=settings.graphrag.dbpedia_filter_keywords,
        )
        self._components["retriever"] = GraphRAGRetriever(
            self._components["knowledge_graph"],
            retriever_config
        )
        # Load DBpedia health knowledge base
        self._components["retriever"].load_dbpedia_health()

        # Fact checker with Qwen
        self._components["fact_checker"] = FactCheckerQwen(
            self._components["retriever"],
            self.config.verification,
            self.config.qwen
        )

        self._initialized = True
        logger.info("Pipeline initialized successfully")

    def train_lda(self, corpus_texts: List[str]) -> None:
        """Train LDA model on corpus."""
        logger.info("Training LDA model...")
        self._components["lda_model"].train(corpus_texts)
        # Update dependent components
        self._components["centrality"].set_models(
            self._components["embedder"],
            self._components["lda_model"]
        )
        self._components["textrank"].set_lda_model(self._components["lda_model"])

    def build_knowledge_graph(self, texts: List[str], source_ids: List[str]) -> None:
        """Build knowledge graph from texts."""
        logger.info(f"Building knowledge graph from {len(texts)} texts...")
        kg = self._components["knowledge_graph"]

        for text, source_id in zip(texts, source_ids):
            # Extract entities
            entities = self._components["entity_extractor"].extract_entities(text, source_id)

            # Extract relations
            relations = self._components["relation_extractor"].extract_relations(text, entities, source_id)

            # Add to KG
            kg.add_triples_from_extraction(relations, entities, source_id, text)

        # Index for retrieval
        self._components["retriever"].index_subgraphs({"main_kg": kg})

        logger.info(f"Knowledge graph built with {len(kg)} triples")

    def process_claim(self, claim: str, claim_id: str = "unknown") -> PipelineResult:
        """
        Process a single claim through the full pipeline.
        Paper workflow: Query → Gx → GraphRAG → Verification
        """
        if not self._initialized:
            self.initialize()

        start_total = time.time()

        # 1. Preprocessing
        start = time.time()
        processed = self._components["preprocessor"].process(claim, claim_id)
        preprocessing_time = time.time() - start
        sentences = processed.sentences

        if not sentences:
            logger.warning(f"No sentences extracted from claim {claim_id}")
            return self._empty_result(claim_id, claim, time.time() - start_total)

        # 2. Embeddings
        start = time.time()
        embeddings = self._components["embedder"].encode(sentences)
        embedding_time = time.time() - start

        # 3. Topic-enhanced sentence centrality (Paper §III-B)
        centrality_time = 0.0
        central_sentences = sentences
        if self.config.use_sentence_centrality and self._components["lda_model"].model is not None:
            start = time.time()
            centrality_result = self._components["centrality"].compute_centrality(sentences)
            centrality_time = time.time() - start
            central_sentences = centrality_result.top_sentences
            logger.debug(f"Central sentences: {central_sentences[:3]}")

        # 4. Topic-Specific TextRank (Paper §III-C)
        textrank_time = 0.0
        key_sentences = central_sentences
        if self.config.use_topic_textrank and self._components["lda_model"].model is not None:
            start = time.time()
            textrank_result = self._components["textrank"].compute_textrank(
                central_sentences, embeddings[:len(central_sentences)]
            )
            textrank_time = time.time() - start
            key_sentences = textrank_result.top_sentences
            logger.debug(f"Key sentences: {key_sentences[:3]}")

        # 5. Knowledge graph construction for query (Gx)
        kg_time = 0.0
        query_kg = None
        if self.config.use_graphrag:
            start = time.time()
            # Use retriever's build_query_graph which includes co-occurrence fallback
            query_text = " ".join(key_sentences)
            query_kg = self._components["retriever"].build_query_graph(query_text, claim_id)
            kg_time = time.time() - start

        # 6. GraphRAG Retrieval (Paper §III-E)
        retrieval_time = 0.0
        evidence = []
        if self.config.use_graphrag:
            start = time.time()
            evidence = self._components["retriever"].retrieve(claim, top_k=self.config.verification.max_evidence_passages)
            retrieval_time = time.time() - start
            logger.debug(f"Retrieved {len(evidence)} evidence graphs")

        # 7. Fact Verification (Paper §III-F)
        start = time.time()
        verification_result = self._components["fact_checker"].verify_claim(
            claim, claim_id, evidence=evidence, query_text=" ".join(key_sentences)
        )
        verification_time = time.time() - start

        total_time = time.time() - start_total

        return PipelineResult(
            claim_id=claim_id,
            claim_text=claim,
            verification_result=verification_result,
            preprocessing_time=preprocessing_time,
            embedding_time=embedding_time,
            centrality_time=centrality_time,
            textrank_time=textrank_time,
            kg_time=kg_time,
            retrieval_time=retrieval_time,
            verification_time=verification_time,
            total_time=total_time,
        )

    def _empty_result(self, claim_id: str, claim: str, total_time: float) -> PipelineResult:
        """Create empty result for claims with no sentences."""
        from trumourgpt.verification.fact_checker_qwen import VerificationResult, VerdictLabel
        empty_verification = VerificationResult(
            claim_id=claim_id,
            claim_text=claim,
            predicted_label=VerdictLabel.UNDETERMINED,
            confidence=0.0,
            evidence_ids=[],
            supporting_evidence=[],
            contradicting_evidence=[],
            reasoning="No sentences extracted from claim",
            uncertainty_reason="No sentences extracted",
            explanation="No sentences extracted from claim",
            key_evidence=[],
            model_identifier=self.config.qwen.model_name,
            retrieved_evidence=[],
            evidence_relationship="NEITHER",
        )
        return PipelineResult(
            claim_id=claim_id,
            claim_text=claim,
            verification_result=empty_verification,
            preprocessing_time=0.0,
            embedding_time=0.0,
            centrality_time=0.0,
            textrank_time=0.0,
            kg_time=0.0,
            retrieval_time=0.0,
            verification_time=0.0,
            total_time=total_time,
        )

    def process_batch(self, claims: List[str], claim_ids: Optional[List[str]] = None) -> List[PipelineResult]:
        """Process multiple claims."""
        if claim_ids is None:
            claim_ids = [f"claim_{i}" for i in range(len(claims))]

        return [self.process_claim(claim, cid) for claim, cid in zip(claims, claim_ids)]

    def __enter__(self) -> "TrumorGPTPipelineQwen":
        return self

    def __exit__(self, *args) -> None:
        # Clean up Qwen client
        if "fact_checker" in self._components and hasattr(self._components["fact_checker"], "_client"):
            client = self._components["fact_checker"]._client
            if client is not None:
                client.__exit__()