"""
Topic-Enhanced Sentence Centrality (Paper Section III-B).
Combines BERT embeddings with LDA topic distributions for sentence representation.
Builds weighted sentence graph and computes PageRank centrality.
"""

import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from loguru import logger

from trumourgpt.config.settings import get_settings
from trumourgpt.embeddings.ollama_embeddings import OllamaEmbeddings, EmbeddingConfig
from trumourgpt.topic_modeling.lda import LDATopicModel, LDAConfig, TopicDistribution


@dataclass
class CentralityConfig:
    """Configuration for topic-enhanced sentence centrality."""
    # Paper: η = 0.7
    eta: float = 0.7

    # Graph construction
    similarity_threshold: float = 0.1
    max_edges_per_node: int = 50

    # PageRank
    pagerank_alpha: float = 0.85
    pagerank_max_iter: int = 100
    pagerank_tol: float = 1e-6

    # Output
    top_k_sentences: int = 10

    # Models
    embedding_model: str = "nomic-embed-text:latest"
    lda_num_topics: int = 20


@dataclass
class SentenceCentralityResult:
    """Result of sentence centrality computation."""
    sentences: List[str]
    embeddings: np.ndarray  # Topic-enhanced embeddings
    centrality_scores: np.ndarray  # PageRank scores
    ranked_indices: np.ndarray  # Indices sorted by centrality (descending)
    top_sentences: List[str]
    similarity_matrix: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentences": self.sentences,
            "centrality_scores": self.centrality_scores.tolist(),
            "ranked_indices": self.ranked_indices.tolist(),
            "top_sentences": self.top_sentences,
        }


def compute_topic_enhanced_embeddings(
    sentences: List[str],
    bert_embeddings: np.ndarray,
    lda_topic_dists: List[TopicDistribution],
    eta: float = 0.7,
) -> np.ndarray:
    """
    Compute topic-enhanced sentence embeddings as per paper Equation (vs = [η × norm(BERT_s); (1-η) × norm(LDA_s)]).

    Args:
        sentences: List of sentences
        bert_embeddings: BERT embeddings shape (n_sentences, bert_dim)
        lda_topic_dists: List of TopicDistribution objects
        eta: Weight for BERT embeddings (paper uses 0.7)

    Returns:
        Topic-enhanced embeddings shape (n_sentences, bert_dim + num_topics)
    """
    n = len(sentences)
    if n == 0:
        bert_dim = bert_embeddings.shape[1] if bert_embeddings.size > 0 else 0
        lda_dim = lda_topic_dists[0].topic_probs.shape[0] if lda_topic_dists else 0
        # If both are 0 but we have bert_embeddings with known dim, use that
        if bert_dim == 0 and lda_dim == 0 and bert_embeddings.ndim == 2:
            bert_dim = bert_embeddings.shape[1]
        return np.zeros((0, bert_dim + lda_dim))

    # Normalize BERT embeddings
    bert_norm = np.linalg.norm(bert_embeddings, axis=1, keepdims=True)
    bert_norm[bert_norm == 0] = 1
    bert_normalized = bert_embeddings / bert_norm

    # Get LDA topic distributions
    lda_probs = np.array([td.topic_probs for td in lda_topic_dists])

    # Normalize LDA distributions (already normalized but ensure)
    lda_norm = np.linalg.norm(lda_probs, axis=1, keepdims=True)
    lda_norm[lda_norm == 0] = 1
    lda_normalized = lda_probs / lda_norm

    # Concatenate with weights
    # v_s = [η × normalized(BERT_s); (1-η) × normalized(LDA_s)]
    enhanced = np.hstack([
        eta * bert_normalized,
        (1 - eta) * lda_normalized
    ])

    logger.debug(f"Computed topic-enhanced embeddings: {enhanced.shape}")
    return enhanced


def build_sentence_graph(
    embeddings: np.ndarray,
    similarity_threshold: float = 0.1,
    max_edges_per_node: int = 50,
) -> nx.Graph:
    """
    Build weighted sentence graph from embeddings.
    Edge weight = cosine similarity between topic-enhanced embeddings.

    Args:
        embeddings: Topic-enhanced embeddings (n_sentences, dim)
        similarity_threshold: Minimum cosine similarity for edge
        max_edges_per_node: Maximum edges per node (sparsity)

    Returns:
        NetworkX graph with weighted edges
    """
    n = embeddings.shape[0]
    if n == 0:
        return nx.Graph()

    # Compute cosine similarity matrix
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = embeddings / norms
    sim_matrix = normalized @ normalized.T

    # Build graph
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        # Get similarities for node i
        similarities = sim_matrix[i]

        # Find neighbors above threshold
        neighbor_indices = np.where(similarities >= similarity_threshold)[0]
        neighbor_indices = neighbor_indices[neighbor_indices != i]  # Exclude self

        # Sort by similarity and take top k
        if len(neighbor_indices) > max_edges_per_node:
            top_k = np.argsort(similarities[neighbor_indices])[-max_edges_per_node:]
            neighbor_indices = neighbor_indices[top_k]

        # Add edges
        for j in neighbor_indices:
            weight = float(similarities[j])
            G.add_edge(i, j, weight=weight)

    logger.debug(f"Built sentence graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def compute_pagerank_centrality(
    graph: nx.Graph,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Compute PageRank centrality on sentence graph.

    Args:
        graph: NetworkX graph with weighted edges
        alpha: Damping factor (paper uses 0.85)
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        PageRank scores for each node
    """
    if graph.number_of_nodes() == 0:
        return np.array([])

    # Use NetworkX PageRank with weight attribute
    pagerank = nx.pagerank(
        graph,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        weight='weight'
    )

    # Convert to array in node order
    scores = np.array([pagerank.get(i, 0.0) for i in range(graph.number_of_nodes())])

    logger.debug(f"Computed PageRank centrality: {scores.shape}")
    return scores


class SentenceCentrality:
    """
    Topic-Enhanced Sentence Centrality pipeline (Paper Section III-B).
    """

    def __init__(self, config: Optional[CentralityConfig] = None):
        self.config = config or self._load_config()
        self._embedder: Optional[OllamaEmbeddings] = None
        self._lda_model: Optional[LDATopicModel] = None

    def _load_config(self) -> CentralityConfig:
        settings = get_settings()
        return CentralityConfig(
            eta=settings.sentence_centrality.eta,
            similarity_threshold=settings.sentence_centrality.similarity_threshold,
            max_edges_per_node=settings.sentence_centrality.max_edges_per_node,
            pagerank_alpha=settings.sentence_centrality.pagerank_alpha,
            pagerank_max_iter=settings.sentence_centrality.pagerank_max_iter,
            pagerank_tol=settings.sentence_centrality.pagerank_tol,
            top_k_sentences=settings.sentence_centrality.top_k_sentences,
            embedding_model=settings.embeddings.model,
            lda_num_topics=settings.topic_modeling.num_topics,
        )

    def _get_embedder(self) -> OllamaEmbeddings:
        if self._embedder is None:
            self._embedder = OllamaEmbeddings(EmbeddingConfig(
                model=self.config.embedding_model,
                normalize=True,
            ))
        return self._embedder

    def _get_lda_model(self) -> LDATopicModel:
        if self._lda_model is None:
            self._lda_model = LDATopicModel(LDAConfig(
                num_topics=self.config.lda_num_topics,
            ))
            # Try to load existing model
            if not self._lda_model.load():
                logger.warning("LDA model not found. Will need to train first.")
        return self._lda_model

    def set_models(self, embedder: OllamaEmbeddings, lda_model: LDATopicModel) -> None:
        """Set pre-initialized models."""
        self._embedder = embedder
        self._lda_model = lda_model

    def compute_centrality(self, sentences: List[str]) -> SentenceCentralityResult:
        """
        Compute topic-enhanced sentence centrality for a list of sentences.

        Args:
            sentences: List of sentences

        Returns:
            SentenceCentralityResult with rankings and scores
        """
        if not sentences:
            return SentenceCentralityResult(
                sentences=[],
                embeddings=np.zeros((0, 0)),
                centrality_scores=np.array([]),
                ranked_indices=np.array([]),
                top_sentences=[],
                similarity_matrix=np.zeros((0, 0)),
            )

        # Get BERT embeddings
        embedder = self._get_embedder()
        bert_embeddings = embedder.encode(sentences)

        # Get LDA topic distributions
        lda_model = self._get_lda_model()
        if lda_model.model is None:
            raise RuntimeError("LDA model not trained. Train LDA first.")

        lda_dists = lda_model.get_topic_distributions_batch(sentences)

        # Compute topic-enhanced embeddings
        enhanced_embeddings = compute_topic_enhanced_embeddings(
            sentences,
            bert_embeddings,
            lda_dists,
            self.config.eta
        )

        # Build sentence graph
        graph = build_sentence_graph(
            enhanced_embeddings,
            self.config.similarity_threshold,
            self.config.max_edges_per_node
        )

        # Compute PageRank centrality
        centrality_scores = compute_pagerank_centrality(
            graph,
            self.config.pagerank_alpha,
            self.config.pagerank_max_iter,
            self.config.pagerank_tol
        )

        # Rank sentences by centrality (descending)
        ranked_indices = np.argsort(centrality_scores)[::-1]
        top_k = min(self.config.top_k_sentences, len(sentences))
        top_sentences = [sentences[i] for i in ranked_indices[:top_k]]

        # Compute similarity matrix for reference
        norms = np.linalg.norm(enhanced_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = enhanced_embeddings / norms
        sim_matrix = normalized @ normalized.T

        return SentenceCentralityResult(
            sentences=sentences,
            embeddings=enhanced_embeddings,
            centrality_scores=centrality_scores,
            ranked_indices=ranked_indices,
            top_sentences=top_sentences,
            similarity_matrix=sim_matrix,
        )

    def extract_central_sentences(self, text: str, top_k: Optional[int] = None) -> List[str]:
        """Extract central sentences from text."""
        from trumourgpt.preprocessing.text import TextPreprocessor
        preprocessor = TextPreprocessor()
        processed = preprocessor.process(text)
        sentences = processed.sentences

        result = self.compute_centrality(sentences)
        k = top_k or self.config.top_k_sentences
        return result.top_sentences[:k]


# Convenience functions
def compute_topic_enhanced_embeddings_simple(
    sentences: List[str],
    bert_embeddings: np.ndarray,
    lda_probs: np.ndarray,
    eta: float = 0.7,
) -> np.ndarray:
    """Simple version without TopicDistribution objects."""
    bert_norm = np.linalg.norm(bert_embeddings, axis=1, keepdims=True)
    bert_norm[bert_norm == 0] = 1
    bert_normalized = bert_embeddings / bert_norm

    lda_norm = np.linalg.norm(lda_probs, axis=1, keepdims=True)
    lda_norm[lda_norm == 0] = 1
    lda_normalized = lda_probs / lda_norm

    return np.hstack([
        eta * bert_normalized,
        (1 - eta) * lda_normalized
    ])


def build_sentence_graph_simple(
    embeddings: np.ndarray,
    threshold: float = 0.1,
    max_edges: int = 50,
) -> nx.Graph:
    """Simple graph builder."""
    return build_sentence_graph(embeddings, threshold, max_edges)


def compute_pagerank_simple(
    graph: nx.Graph,
    alpha: float = 0.85,
) -> np.ndarray:
    """Simple PageRank."""
    return compute_pagerank_centrality(graph, alpha=alpha)