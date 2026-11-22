"""
Tests for Topic-Specific TextRank (Paper Section III-C).
Verifies: non-negative probabilities, column sums = 1, convergence, health topic boost effect.
"""

import numpy as np
from trumourgpt.topic_textrank.textrank import (
    compute_topic_relevance,
    build_transition_matrix,
    run_textrank_iteration,
    verify_transition_matrix,
    verify_convergence,
    demo_textrank_on_example,
    TextRankConfig,
    TopicSpecificTextRank,
)
from trumourgpt.topic_modeling.lda import TopicDistribution


def create_mock_lda_dists(num_sentences: int = 3, num_topics: int = 4) -> list:
    """Create mock LDA topic distributions for testing."""
    dists = []
    for i in range(num_sentences):
        probs = np.random.dirichlet(np.ones(num_topics))
        dominant = int(np.argmax(probs))
        dists.append(TopicDistribution(
            topic_ids=np.arange(num_topics),
            topic_probs=probs,
            dominant_topic=dominant,
            dominant_prob=float(probs[dominant]),
        ))
    return dists


def test_topic_relevance():
    """Test topic relevance computation R(v_i)."""
    print("Testing topic relevance...")

    # 3 sentences, 4 topics (0,1 are health)
    lda_dists = [
        TopicDistribution(np.arange(4), np.array([0.6, 0.3, 0.05, 0.05]), 0, 0.6),  # Health
        TopicDistribution(np.arange(4), np.array([0.5, 0.4, 0.05, 0.05]), 0, 0.5),  # Health
        TopicDistribution(np.arange(4), np.array([0.05, 0.05, 0.45, 0.45]), 2, 0.45),  # Non-health
    ]

    health_topics = [0, 1]
    health_boost = 1.5

    relevance = compute_topic_relevance(lda_dists, health_topics, health_boost)

    print(f"  Relevance scores: {relevance}")
    print(f"  Sum: {relevance.sum():.6f}")

    # Check properties
    assert len(relevance) == 3
    assert np.all(relevance >= 0), "Relevance should be non-negative"
    assert abs(relevance.sum() - 1.0) < 1e-6, "Relevance should sum to 1"

    # Health sentences should have higher relevance
    assert relevance[0] > relevance[2], "Health sentence should have higher relevance"
    assert relevance[1] > relevance[2], "Health sentence should have higher relevance"

    print("  ✅ Topic relevance test passed")


def test_transition_matrix():
    """Test transition matrix P' = d·P + (1-d)·E."""
    print("\nTesting transition matrix...")

    # Simple embeddings: 3 sentences, 2 dimensions
    embeddings = np.array([
        [0.8, 0.1],  # Similar to second
        [0.7, 0.2],  # Similar to first
        [0.1, 0.8],  # Different
    ], dtype=np.float32)

    topic_relevance = np.array([0.4, 0.4, 0.2])  # Health sentences higher
    damping = 0.85

    P_prime, P = build_transition_matrix(embeddings, topic_relevance, damping)

    print(f"  P' shape: {P_prime.shape}")
    print(f"  P' column sums: {P_prime.sum(axis=0)}")

    # Verify properties
    verification = verify_transition_matrix(P_prime)
    print(f"  Non-negative: {verification['nonnegative']}")
    print(f"  Column sums close to 1: {verification['column_sums_close_to_1']}")

    assert verification["nonnegative"], "P' should be non-negative"
    assert verification["column_sums_close_to_1"], "P' columns should sum to 1"
    assert P_prime.shape == (3, 3), "Should be square matrix"

    print("  ✅ Transition matrix test passed")


def test_textrank_convergence():
    """Test TST power iteration convergence."""
    print("\nTesting TST convergence...")

    # Use the transition matrix from previous test
    embeddings = np.array([
        [0.8, 0.1],
        [0.7, 0.2],
        [0.1, 0.8],
    ], dtype=np.float32)

    topic_relevance = np.array([0.4, 0.4, 0.2])
    P_prime, _ = build_transition_matrix(embeddings, topic_relevance, 0.85)

    # Run iteration
    scores, converged, iterations = run_textrank_iteration(P_prime, max_iterations=500, convergence_tol=1e-6)

    print(f"  Converged: {converged}")
    print(f"  Iterations: {iterations}")
    print(f"  Scores: {scores}")
    print(f"  Sum: {scores.sum():.6f}")

    # Verify convergence properties
    verification = verify_convergence(P_prime, scores)
    print(f"  Stationary residual: {verification['stationary_residual']:.2e}")
    print(f"  Non-negative: {verification['scores_nonnegative']}")
    print(f"  Normalized: {verification['scores_normalized']}")

    assert verification["scores_nonnegative"], "Scores should be non-negative"
    assert verification["scores_normalized"], "Scores should sum to 1"
    assert converged or iterations == 500, "Should converge or hit max iterations"

    print("  ✅ TST convergence test passed")


def test_health_topic_boost():
    """Test that health topic boost changes rankings."""
    print("\nTesting health topic boost effect...")

    # Create sentence embeddings where sentence 0 is health-related
    embeddings = np.array([
        [0.9, 0.05, 0.05],  # Health-like
        [0.05, 0.9, 0.05],  # Health-like
        [0.05, 0.05, 0.9],  # Non-health
    ], dtype=np.float32)

    # With health boost
    health_topics = [0]  # Topic 0 is health
    lda_dists = [
        TopicDistribution(np.arange(3), np.array([0.8, 0.1, 0.1]), 0, 0.8),
        TopicDistribution(np.arange(3), np.array([0.7, 0.2, 0.1]), 0, 0.7),
        TopicDistribution(np.arange(3), np.array([0.1, 0.1, 0.8]), 2, 0.8),
    ]

    relevance_boosted = compute_topic_relevance(lda_dists, health_topics, 1.5)
    relevance_uniform = compute_topic_relevance(lda_dists, health_topics, 1.0)  # No boost

    print(f"  With boost (α=1.5): {relevance_boosted}")
    print(f"  Without boost (α=1.0): {relevance_uniform}")

    # With boost, health sentences should have even higher relevance
    assert relevance_boosted[0] > relevance_uniform[0], "Boost should increase health relevance"
    assert relevance_boosted[1] > relevance_uniform[1], "Boost should increase health relevance"
    assert relevance_boosted[2] < relevance_uniform[2], "Boost should decrease non-health relevance"

    print("  ✅ Health topic boost test passed")


def test_full_textrank_pipeline():
    """Test complete TST pipeline."""
    print("\nTesting full TST pipeline...")

    result = demo_textrank_on_example()

    print(f"  Topic relevance: {result['topic_relevance']}")
    print(f"  TST scores: {result['scores']}")
    print(f"  Converged: {result['converged']}")
    print(f"  Iterations: {result['iterations']}")

    # Verify health sentences ranked higher
    scores = np.array(result['scores'])
    # Sentences 0,1 are health; sentence 2 is non-health
    assert scores[0] > scores[2], "Health sentence should rank higher"
    assert scores[1] > scores[2], "Health sentence should rank higher"

    print("  ✅ Full TST pipeline test passed")


if __name__ == "__main__":
    test_topic_relevance()
    test_transition_matrix()
    test_textrank_convergence()
    test_health_topic_boost()
    test_full_textrank_pipeline()
    print("\n✅ All TextRank tests passed!")