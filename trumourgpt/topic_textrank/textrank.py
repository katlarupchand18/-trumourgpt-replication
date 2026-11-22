"""
Topic-Specific TextRank (TST) - Paper Section III-C.
Adapts TextRank with topic-specific teleportation for health-related content.
Implements the exact algorithm from the paper with α=1.5, d=0.85, max_iter=500, tol=1e-6.
"""

import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, Set
from loguru import logger

from trumourgpt.config.settings import get_settings
from trumourgpt.topic_modeling.lda import LDATopicModel, TopicDistribution, HealthTopicIdentifier


@dataclass
class TextRankConfig:
    """Configuration for Topic-Specific TextRank."""
    # Paper parameters
    damping_factor: float = 0.85      # Paper: d = 0.85
    max_iterations: int = 500         # Paper: max 500 iterations
    convergence_tol: float = 1e-6     # Paper: L1 tolerance 1e-6

    # Health topic boost
    health_topic_boost: float = 1.5   # Paper: α = 1.5

    # Graph construction
    similarity_threshold: float = 0.1

    # Output
    top_k_keywords: int = 20


@dataclass
class TextRankResult:
    """Result of Topic-Specific TextRank."""
    sentences: List[str]
    scores: np.ndarray  # TST scores for each sentence
    ranked_indices: np.ndarray  # Indices sorted by score (descending)
    top_sentences: List[str]
    transition_matrix: np.ndarray
    topic_relevance: np.ndarray  # R(v_i) for each sentence
    converged: bool
    iterations: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentences": self.sentences,
            "scores": self.scores.tolist(),
            "ranked_indices": self.ranked_indices.tolist(),
            "top_sentences": self.top_sentences,
            "converged": self.converged,
            "iterations": self.iterations,
        }


def compute_topic_relevance(
    lda_dists: List[TopicDistribution],
    health_topic_ids: List[int],
    health_boost: float = 1.5,
) -> np.ndarray:
    """
    Compute topic relevance scores R(v_i) as per paper.

    R(v_i) = Σ(β_k * t_i,k) / Σ(β_k * t_j,k)
    where β_k = α if topic k is health-related, else 1.

    Args:
        lda_dists: List of TopicDistribution for each sentence
        health_topic_ids: List of health-related topic indices
        health_boost: α parameter (paper uses 1.5)

    Returns:
        Topic relevance scores for each sentence (normalized to sum to 1)
    """
    n = len(lda_dists)
    if n == 0:
        return np.array([])

    num_topics = lda_dists[0].topic_probs.shape[0]

    # Build β vector
    beta = np.ones(num_topics)
    beta[health_topic_ids] = health_boost

    # Compute weighted topic sum for each sentence
    weighted_sums = np.zeros(n)
    for i, dist in enumerate(lda_dists):
        weighted_sums[i] = np.sum(dist.topic_probs * beta)

    # Normalize: R(v_i) = weighted_i / Σ(weighted_j)
    total = weighted_sums.sum()
    if total > 0:
        relevance = weighted_sums / total
    else:
        relevance = np.ones(n) / n

    logger.debug(f"Computed topic relevance: {relevance.shape}, sum={relevance.sum():.6f}")
    return relevance


def build_transition_matrix(
    embeddings: np.ndarray,
    topic_relevance: np.ndarray,
    damping_factor: float = 0.85,
    similarity_threshold: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build TST transition matrix P' = d·P + (1-d)·E as per paper.

    P is column-stochastic matrix from similarity graph.
    E = u·e^T where u is topic relevance vector, e is all-ones vector.

    Handles dangling nodes (nodes with no outgoing edges) by using
    topic relevance as fallback, ensuring P' is column-stochastic.

    Args:
        embeddings: Sentence embeddings (n, dim)
        topic_relevance: R(v_i) scores (n,)
        damping_factor: d parameter
        similarity_threshold: Edge threshold

    Returns:
        (P_prime, P) transition matrices
    """
    n = embeddings.shape[0]
    if n == 0:
        return np.zeros((0, 0)), np.zeros((0, 0))

    # Compute cosine similarity matrix
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = embeddings / norms
    sim_matrix = normalized @ normalized.T

    # Build adjacency with threshold
    adj_matrix = sim_matrix.copy()
    adj_matrix[adj_matrix < similarity_threshold] = 0
    np.fill_diagonal(adj_matrix, 0)  # No self-loops

    # Build column-stochastic P
    # P_ij = w'_ji / Σ_k w'_jk (column normalization)
    col_sums = adj_matrix.sum(axis=0)

    # Identify dangling nodes (columns with zero sum)
    dangling = col_sums == 0

    # For dangling nodes, we'll use topic relevance as the distribution
    # This ensures column-stochasticity for P
    col_sums_dangling_safe = col_sums.copy()
    col_sums_dangling_safe[dangling] = 1  # Avoid division by zero for now
    P = adj_matrix / col_sums_dangling_safe  # Broadcasting: (n,n) / (n,) -> columns normalized

    # Fix dangling node columns: replace with topic relevance vector
    if np.any(dangling):
        # For each dangling node (column j), set P[:, j] = u (topic_relevance)
        P[:, dangling] = topic_relevance.reshape(-1, 1)

    # Verify column sums = 1
    col_sums_P = P.sum(axis=0)
    if not np.allclose(col_sums_P, 1.0, atol=1e-6):
        logger.warning(f"P column sums not exactly 1: min={col_sums_P.min():.6f}, max={col_sums_P.max():.6f}")

    # Build E = u·e^T (each column is u)
    u = topic_relevance.reshape(-1, 1)  # Column vector
    e = np.ones((1, n))  # Row vector
    E = u @ e  # Outer product: (n,1) @ (1,n) = (n,n)

    # Verify E column sums = 1
    # logger.debug(f"E column sums: {E.sum(axis=0)[:5]}...")

    # Build P' = d·P + (1-d)·E
    P_prime = damping_factor * P + (1 - damping_factor) * E

    # Verify P' column sums = 1
    col_sums_prime = P_prime.sum(axis=0)
    if not np.allclose(col_sums_prime, 1.0, atol=1e-6):
        logger.warning(f"P' column sums not exactly 1: min={col_sums_prime.min():.6f}, max={col_sums_prime.max():.6f}")

    logger.debug(f"Built transition matrices: P'={P_prime.shape}, P={P.shape}")
    return P_prime, P


def run_textrank_iteration(
    P_prime: np.ndarray,
    max_iterations: int = 500,
    convergence_tol: float = 1e-6,
    initial_scores: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, bool, int]:
    """
    Run TST power iteration: TST(t+1) = P' · TST(t).

    Args:
        P_prime: Transition matrix (n, n)
        max_iterations: Maximum iterations
        convergence_tol: L1 convergence tolerance
        initial_scores: Initial TST vector (default: uniform)

    Returns:
        (final_scores, converged, iterations)
    """
    n = P_prime.shape[0]
    if n == 0:
        return np.array([]), True, 0

    # Initialize uniformly as per paper
    if initial_scores is None:
        TST = np.ones(n) / n
    else:
        TST = initial_scores.copy()
        # Ensure it's a probability distribution
        TST = TST / TST.sum() if TST.sum() > 0 else np.ones(n) / n

    converged = False
    iterations = 0

    for iteration in range(max_iterations):
        TST_new = P_prime @ TST

        # L1 convergence check
        diff = np.abs(TST_new - TST).sum()

        TST = TST_new
        iterations = iteration + 1

        if diff < convergence_tol:
            converged = True
            logger.debug(f"TST converged after {iterations} iterations (diff={diff:.2e})")
            break

    if not converged:
        logger.warning(f"TST did not converge after {max_iterations} iterations (final diff={diff:.2e})")

    # Ensure non-negative and normalized
    TST = np.maximum(TST, 0)
    TST = TST / TST.sum() if TST.sum() > 0 else np.ones(n) / n

    return TST, converged, iterations


class TopicSpecificTextRank:
    """
    Topic-Specific TextRank pipeline (Paper Section III-C).
    """

    def __init__(self, config: Optional[TextRankConfig] = None):
        self.config = config or self._load_config()
        self._lda_model: Optional[LDATopicModel] = None

    def _load_config(self) -> TextRankConfig:
        settings = get_settings()
        return TextRankConfig(
            damping_factor=settings.topic_textrank.damping_factor,
            max_iterations=settings.topic_textrank.max_iterations,
            convergence_tol=settings.topic_textrank.convergence_tol,
            health_topic_boost=settings.topic_textrank.health_topic_boost,
            similarity_threshold=settings.topic_textrank.similarity_threshold,
            top_k_keywords=settings.topic_textrank.top_k_keywords,
        )

    def set_lda_model(self, lda_model: LDATopicModel) -> None:
        """Set pre-trained LDA model."""
        self._lda_model = lda_model

    def compute_textrank(
        self,
        sentences: List[str],
        embeddings: np.ndarray,
        lda_dists: Optional[List[TopicDistribution]] = None,
    ) -> TextRankResult:
        """
        Compute Topic-Specific TextRank scores.

        Args:
            sentences: List of sentences
            embeddings: Sentence embeddings (topic-enhanced or plain)
            lda_dists: LDA topic distributions (required for topic relevance)

        Returns:
            TextRankResult with scores and rankings
        """
        n = len(sentences)
        if n == 0:
            return TextRankResult(
                sentences=[],
                scores=np.array([]),
                ranked_indices=np.array([]),
                top_sentences=[],
                transition_matrix=np.zeros((0, 0)),
                topic_relevance=np.array([]),
                converged=True,
                iterations=0,
            )

        # Get topic relevance scores
        if lda_dists is None:
            if self._lda_model is None:
                raise ValueError("LDA model required for topic relevance. Provide lda_dists or set_lda_model().")
            lda_dists = self._lda_model.get_topic_distributions_batch(sentences)

        if self._lda_model is None or self._lda_model.health_identifier is None:
            raise ValueError("Health topics not identified in LDA model.")

        health_topic_ids = self._lda_model.health_identifier.health_topic_ids

        topic_relevance = compute_topic_relevance(
            lda_dists,
            health_topic_ids,
            self.config.health_topic_boost
        )

        # Build transition matrix
        P_prime, P = build_transition_matrix(
            embeddings,
            topic_relevance,
            self.config.damping_factor,
            self.config.similarity_threshold
        )

        # Run power iteration
        scores, converged, iterations = run_textrank_iteration(
            P_prime,
            self.config.max_iterations,
            self.config.convergence_tol
        )

        # Rank sentences
        ranked_indices = np.argsort(scores)[::-1]
        top_k = min(self.config.top_k_keywords, n)
        top_sentences = [sentences[i] for i in ranked_indices[:top_k]]

        return TextRankResult(
            sentences=sentences,
            scores=scores,
            ranked_indices=ranked_indices,
            top_sentences=top_sentences,
            transition_matrix=P_prime,
            topic_relevance=topic_relevance,
            converged=converged,
            iterations=iterations,
        )

    def extract_keywords(
        self,
        text: str,
        top_k: Optional[int] = None,
    ) -> List[str]:
        """Extract keywords from text using TST."""
        from trumourgpt.preprocessing.text import TextPreprocessor
        preprocessor = TextPreprocessor()
        processed = preprocessor.process(text)
        sentences = processed.sentences

        if not sentences:
            return []

        # Get embeddings
        from trumourgpt.embeddings.ollama_embeddings import OllamaEmbeddings, EmbeddingConfig
        embedder = OllamaEmbeddings(EmbeddingConfig())
        embeddings = embedder.encode(sentences)

        result = self.compute_textrank(sentences, embeddings)
        k = top_k or self.config.top_k_keywords
        return result.top_sentences[:k]


# Verification functions for testing
def verify_transition_matrix(P_prime: np.ndarray) -> Dict[str, Any]:
    """Verify transition matrix properties as per paper."""
    n = P_prime.shape[0]
    results = {
        "shape": P_prime.shape,
        "nonnegative": np.all(P_prime >= 0),
        "column_sums": P_prime.sum(axis=0).tolist(),
        "column_sums_close_to_1": np.allclose(P_prime.sum(axis=0), 1.0, atol=1e-6),
        "row_sums": P_prime.sum(axis=1).tolist(),
    }
    return results


def verify_convergence(P_prime: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    """Verify TST convergence properties."""
    # Check stationary distribution: P' * scores ≈ scores
    residual = P_prime @ scores - scores
    results = {
        "stationary_residual": np.abs(residual).max(),
        "scores_sum": scores.sum(),
        "scores_nonnegative": np.all(scores >= 0),
        "scores_normalized": np.isclose(scores.sum(), 1.0, atol=1e-6),
    }
    return results


def demo_textrank_on_example() -> Dict[str, Any]:
    """Demo TST on a simple example."""
    # Simple test with 3 sentences
    sentences = [
        "COVID-19 vaccines prevent severe disease.",
        "The vaccine is safe and effective.",
        "Weather is nice today.",
    ]

    # Mock embeddings (3 sentences, 4 dims)
    embeddings = np.array([
        [0.8, 0.1, 0.05, 0.05],  # Health-related
        [0.7, 0.2, 0.05, 0.05],  # Health-related
        [0.1, 0.1, 0.4, 0.4],    # Non-health
    ], dtype=np.float32)

    # Mock LDA distributions (4 topics, topics 0,1 are health)
    from trumourgpt.topic_modeling.lda import TopicDistribution
    lda_dists = [
        TopicDistribution(
            topic_ids=np.arange(4),
            topic_probs=np.array([0.6, 0.3, 0.05, 0.05]),
            dominant_topic=0, dominant_prob=0.6
        ),
        TopicDistribution(
            topic_ids=np.arange(4),
            topic_probs=np.array([0.5, 0.4, 0.05, 0.05]),
            dominant_topic=0, dominant_prob=0.5
        ),
        TopicDistribution(
            topic_ids=np.arange(4),
            topic_probs=np.array([0.05, 0.05, 0.45, 0.45]),
            dominant_topic=2, dominant_prob=0.45
        ),
    ]

    config = TextRankConfig(
        damping_factor=0.85,
        health_topic_boost=1.5,
    )

    # Compute topic relevance
    topic_relevance = compute_topic_relevance(lda_dists, [0, 1], 1.5)
    print(f"Topic relevance: {topic_relevance}")

    # Build transition matrix
    P_prime, P = build_transition_matrix(embeddings, topic_relevance, 0.85)
    print(f"P' column sums: {P_prime.sum(axis=0)}")

    # Run TST
    scores, converged, iterations = run_textrank_iteration(P_prime)
    print(f"TST scores: {scores}")
    print(f"Converged: {converged}, Iterations: {iterations}")

    return {
        "topic_relevance": topic_relevance.tolist(),
        "scores": scores.tolist(),
        "converged": converged,
        "iterations": iterations,
    }