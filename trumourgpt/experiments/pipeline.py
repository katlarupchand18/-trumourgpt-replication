"""
TrumorGPT Main Pipeline.
Integrates all components: preprocessing, embeddings, topic modeling, sentence centrality,
TextRank, entity/relation extraction, knowledge graph, GraphRAG, verification.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

from trumourgpt.config.settings import get_settings
from trumourgpt.preprocessing.text import TextPreprocessor, PreprocessingConfig
from trumourgpt.embeddings.ollama_embeddings import OllamaEmbeddings, EmbeddingConfig
from trumourgpt.topic_modeling.lda import LDATopicModel, LDAConfig
from trumourgpt.sentence_centrality.centrality import SentenceCentrality, CentralityConfig
from trumourgpt.topic_textrank.textrank import TopicSpecificTextRank, TextRankConfig
from trumourgpt.entity_extraction.extractor import EntityExtractor, ExtractionConfig as EntityConfig
from trumourgpt.relation_extraction.extractor import RelationExtractor, ExtractionConfig as RelationConfig
from trumourgpt.knowledge_graph.graph import SemanticHealthKnowledgeGraph, KnowledgeGraphConfig
from trumourgpt.retrieval.graphrag import GraphRAGRetriever, RetrievalConfig
from trumourgpt.verification.fact_checker import FactChecker, VerificationConfig, VerificationResult
from trumourgpt.evaluation.dataset import PolitiFactDataset, DatasetConfig
from trumourgpt.evaluation.metrics import evaluate_binary, EvaluationMetrics


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


class TrumorGPTPipeline:
    """
    Main TrumorGPT Pipeline (Paper: complete workflow from query to fact verification).
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._components = {}
        self._initialized = False

    def initialize(self) -> None:
        """Initialize all pipeline components."""
        logger.info("Initializing TrumorGPT pipeline...")

        # Load settings from config.yaml (force reload to avoid cached defaults)
        from trumourgpt.config.settings import get_settings, reset_settings
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

        # Fact checker
        self._components["fact_checker"] = FactChecker(
            self._components["retriever"],
            self.config.verification
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
        if self.config.use_graphrag:
            start = time.time()
            # Use the query KG built earlier (with raw predicates for negation preservation)
            query_text = " ".join(key_sentences)
            evidence = self._components["retriever"].retrieve(claim, query_kg=query_kg)
            retrieval_time = time.time() - start
        else:
            evidence = []

        # 7. Fact Verification (Paper §III-D, III-E)
        start = time.time()
        query_text = " ".join(key_sentences)
        verification_result = self._components["fact_checker"].verify_claim(
            claim, claim_id, evidence=evidence, query_text=query_text
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
        """Create empty result for failed processing."""
        from trumourgpt.verification.fact_checker import VerificationResult
        from trumourgpt.llm.schemas import VerdictLabel

        empty_verification = VerificationResult(
            claim_id=claim_id,
            claim_text=claim,
            predicted_label=VerdictLabel.UNDETERMINED,
            confidence=0.0,
            evidence_ids=[],
            supporting_evidence=[],
            contradicting_evidence=[],
            reasoning="No sentences extracted from claim",
            uncertainty_reason="Empty claim",
            explanation="Unable to process claim.",
            key_evidence=[],
            model_identifier="trumourgpt-pipeline",
            retrieved_evidence=[],
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

    def process_batch(
        self,
        claims: List[str],
        claim_ids: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> List[PipelineResult]:
        """Process multiple claims."""
        if claim_ids is None:
            claim_ids = [f"claim_{i}" for i in range(len(claims))]

        results = []
        for claim, cid in zip(claims, claim_ids):
            logger.info(f"Processing {cid}: {claim[:80]}...")
            result = self.process_claim(claim, cid)
            results.append(result)

            # Save checkpoint
            if save_path and len(results) % 10 == 0:
                self.save_results(results, save_path)

        if save_path:
            self.save_results(results, save_path)

        return results

    def save_results(self, results: List[PipelineResult], path: str) -> None:
        """Save results to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [r.to_dict() for r in results]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(results)} results to {path}")

    def load_results(self, path: str) -> List[PipelineResult]:
        """Load results from file."""
        with open(path) as f:
            data = json.load(f)

        # Reconstruct PipelineResult objects (simplified)
        results = []
        for item in data:
            vr = VerificationResult(**item["verification"])
            result = PipelineResult(
                claim_id=item["claim_id"],
                claim_text=item["claim_text"],
                verification_result=vr,
                **item["timing"]
            )
            results.append(result)

        return results


class ExperimentRunner:
    """Run controlled experiments comparing different pipeline configurations."""

    def __init__(self, base_config: PipelineConfig):
        self.base_config = base_config
        self.results: Dict[str, List[PipelineResult]] = {}

    def run_experiment(
        self,
        name: str,
        claims: List[str],
        claim_ids: List[str],
        config_overrides: Dict[str, Any] = None,
    ) -> List[PipelineResult]:
        """Run experiment with given configuration."""
        logger.info(f"Running experiment: {name}")

        # Create modified config
        config = PipelineConfig()
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Run pipeline
        pipeline = TrumorGPTPipeline(config)
        pipeline.initialize()
        results = pipeline.process_batch(claims, claim_ids)

        self.results[name] = results
        return results

    def run_ablation_study(
        self,
        claims: List[str],
        claim_ids: List[str],
    ) -> Dict[str, List[PipelineResult]]:
        """Run ablation study: full pipeline vs. components removed."""
        ablations = {
            "full": {},
            "no_centrality": {"use_sentence_centrality": False},
            "no_textrank": {"use_topic_textrank": False},
            "no_graphrag": {"use_graphrag": False},
            "no_centrality_textrank": {"use_sentence_centrality": False, "use_topic_textrank": False},
            "llm_only": {"use_sentence_centrality": False, "use_topic_textrank": False, "use_graphrag": False},
        }

        results = {}
        for name, overrides in ablations.items():
            logger.info(f"Ablation: {name}")
            results[name] = self.run_experiment(name, claims, claim_ids, overrides)

        return results

    def evaluate_results(
        self,
        results: List[PipelineResult],
        ground_truth: Dict[str, str],
    ) -> EvaluationMetrics:
        """Evaluate pipeline results against ground truth."""
        y_true = []
        y_pred = []
        claim_ids = []

        for result in results:
            cid = result.claim_id
            if cid in ground_truth:
                y_true.append(ground_truth[cid])
                y_pred.append(result.verification_result.predicted_label.value)
                claim_ids.append(cid)

        return evaluate_binary(y_true, y_pred, "TRUE", "FALSE")

    def compare_experiments(self, ground_truth: Dict[str, str]) -> Dict[str, EvaluationMetrics]:
        """Compare all experiment results."""
        comparison = {}
        for name, results in self.results.items():
            comparison[name] = self.evaluate_results(results, ground_truth)
        return comparison


def create_sample_corpus() -> List[str]:
    """Create sample health corpus for LDA training."""
    return [
        "COVID-19 vaccines prevent severe disease and hospitalization.",
        "SARS-CoV-2 virus causes COVID-19 disease spreading through respiratory droplets.",
        "mRNA vaccines teach cells to make protein triggering immune response.",
        "Clinical trials show COVID-19 vaccines high efficacy against symptomatic infection.",
        "CDC recommends COVID-19 vaccination for everyone aged 6 months and older.",
        "Vaccine side effects typically mild including sore arm fatigue fever.",
        "Long COVID causes persistent symptoms fatigue brain fog shortness of breath.",
        "New variants SARS-CoV-2 emerge requiring updated vaccines.",
        "Public health measures masking distancing reduce virus transmission.",
        "WHO declared COVID-19 pandemic March 2020.",
        "Influenza vaccines updated annually match circulating strains.",
        "Flu vaccination reduces risk flu illness 40-60 percent.",
        "Pneumococcal vaccines prevent pneumonia meningitis bloodstream infections.",
        "HPV vaccine prevents cancers caused by human papillomavirus infection.",
        "Measles vaccine highly effective two doses providing 97% protection.",
        "Polio eliminated most countries through vaccination programs.",
        "Hepatitis B vaccine prevents liver disease liver cancer.",
        "Rotavirus vaccine protects children severe diarrheal disease.",
        "Shingles vaccine reduces risk shingles post-herpetic neuralgia.",
        "Tetanus vaccine prevents lockjaw disease bacterial infection.",
        "Antibiotics treat bacterial infections ineffective against viruses.",
        "Antiviral medications reduce severity viral infections like influenza.",
        "Monoclonal antibodies provide targeted treatment for COVID-19.",
        "Corticosteroids reduce inflammation severe COVID-19 cases.",
        "Remdesivir antiviral drug approved COVID-19 treatment.",
        "Paxlovid oral antiviral treating mild moderate COVID-19.",
        "Ventilators support breathing critically ill COVID-19 patients.",
        "Clinical trials test safety efficacy new treatments.",
        "Randomized controlled trials provide highest quality evidence.",
        "Peer review ensures scientific rigor medical research.",
        "Meta-analyses combine results multiple studies.",
        "Systematic reviews synthesize all available evidence topic.",
        "FDA approves drugs based on clinical trial data.",
        "Drug development takes 10-15 years discovery approval.",
        "Phase 1 trials test safety small groups healthy volunteers.",
        "Phase 2 trials test efficacy side effects patients.",
        "Phase 3 trials confirm efficacy large patient populations.",
        "Phase 4 surveillance monitors drugs after approval.",
        "Vaccine development follows similar phased approach.",
        "Emergency use authorization allows faster access during crises.",
        "Herd immunity occurs enough people immune stop spread.",
        "Vaccine hesitancy major challenge public health.",
        "Misinformation vaccines spreads rapidly social media.",
        "Fact-checking helps combat health misinformation.",
        "Health literacy enables people make informed decisions.",
        "Public trust health institutions crucial compliance.",
        "Epidemiologists study disease patterns populations.",
        "Contact tracing identifies people exposed infectious diseases.",
        "Quarantine separates exposed people prevent spread.",
        "Isolation separates sick people healthy people.",
        "Testing identifies infected individuals isolation.",
        "PCR tests detect viral genetic material high accuracy.",
        "Rapid antigen tests provide quick results lower sensitivity.",
        "Wastewater surveillance tracks virus levels communities.",
        "Genomic sequencing identifies new virus variants.",
        "Spike protein mutations affect virus transmissibility immune escape.",
        "Variant classification includes variants concern interest.",
        "Delta variant more transmissible original strain.",
        "Omicron variant many spike protein mutations.",
        "Bivalent vaccines target original strain Omicron variants.",
        "Vaccine effectiveness varies variant time since vaccination.",
        "Booster doses restore protection new variants.",
        "Immunocompromised people may need additional vaccine doses.",
    ]