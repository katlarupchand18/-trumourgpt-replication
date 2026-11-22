"""
Tests for Topic-Enhanced Sentence Centrality (Paper Section III-B).
Verifies: vector dimensions, normalization, concatenation, cosine similarity, PageRank.
"""

import numpy as np
from trumourgpt.sentence_centrality.centrality import (
    compute_topic_enhanced_embeddings,
    build_sentence_graph,
    compute_pagerank_centrality,
    compute_topic_enhanced_embeddings_simple,
    build_sentence_graph_simple,
    compute_pagerank_simple,
)
from trumourgpt.topic_modeling.lda import TopicDistribution


def create_test_data(num_sentences: int = 5, bert_dim: int = 10, num_topics: int = 4) -> tuple:
    """Create test data for sentence centrality."""
    np.random.seed(42)

    # BERT embeddings
    bert_embeddings = np.random.randn(num_sentences, bert_dim).astype(np.float32)

    # LDA topic distributions
    lda_dists = []
    for i in range(num_sentences):
        probs = np.random.dirichlet(np.ones(num_topics))
        dominant = int(np.argmax(probs))
        lda_dists.append(TopicDistribution(
            topic_ids=np.arange(num_topics),
            topic_probs=probs.astype(np.float32),
            dominant_topic=dominant,
            dominant_prob=float(probs[dominant]),
        ))

    return bert_embeddings, lda_dists


def test_topic_enhanced_embeddings():
    """Test topic-enhanced embedding computation: v_s = [η × norm(BERT); (1-η) × norm(LDA)]"""
    print("Testing topic-enhanced embeddings...")

    bert_embeddings, lda_dists = create_test_data(num_sentences=5, bert_dim=10, num_topics=4)
    eta = 0.7

    enhanced = compute_topic_enhanced_embeddings(
        ["sentence"] * 5,  # dummy sentences
        bert_embeddings,
        lda_dists,
        eta
    )

    print(f"  BERT shape: {bert_embeddings.shape}")
    print(f"  Enhanced shape: {enhanced.shape}")
    print(f"  Expected dim: {bert_embeddings.shape[1] + 4}")

    # Check dimensions
    assert enhanced.shape[0] == 5, "Should have 5 sentences"
    assert enhanced.shape[1] == 10 + 4, "Should be BERT dim + num topics"

    # Check normalization
    for i in range(5):
        bert_part = enhanced[i, :10]
        lda_part = enhanced[i, 10:]

        bert_norm = np.linalg.norm(bert_part)
        lda_norm = np.linalg.norm(lda_part)

        # BERT part should have norm = eta
        assert abs(bert_norm - eta) < 1e-5, f"BERT norm should be {eta}, got {bert_norm}"
        # LDA part should have norm = 1-eta
        assert abs(lda_norm - (1-eta)) < 1e-5, f"LDA norm should be {1-eta}, got {lda_norm}"

    print("  ✅ Topic-enhanced embeddings test passed")


def test_topic_enhanced_embeddings_simple():
    """Test simple version without TopicDistribution objects."""
    print("\nTesting simple topic-enhanced embeddings...")

    bert_embeddings, lda_dists = create_test_data(5, 10, 4)
    lda_probs = np.array([d.topic_probs for d in lda_dists])
    eta = 0.7

    enhanced = compute_topic_enhanced_embeddings_simple(
        ["sentence"] * 5,
        bert_embeddings,
        lda_probs,
        eta
    )

    assert enhanced.shape == (5, 14)

    # Check same normalization
    for i in range(5):
        bert_norm = np.linalg.norm(enhanced[i, :10])
        lda_norm = np.linalg.norm(enhanced[i, 10:])
        assert abs(bert_norm - eta) < 1e-5
        assert abs(lda_norm - (1-eta)) < 1e-5

    print("  ✅ Simple embeddings test passed")


def test_sentence_graph():
    """Test sentence graph construction with cosine similarity."""
    print("\nTesting sentence graph...")

    # Create embeddings with known similarity structure
    embeddings = np.array([
        [1.0, 0.0, 0.0],      # Sentence 0
        [0.9, 0.1, 0.0],      # Similar to 0
        [0.0, 1.0, 0.0],      # Sentence 2
        [0.0, 0.9, 0.1],      # Similar to 2
        [0.0, 0.0, 1.0],      # Sentence 4
    ], dtype=np.float32)

    graph = build_sentence_graph(
        embeddings,
        similarity_threshold=0.5,
        max_edges_per_node=10
    )

    print(f"  Nodes: {graph.number_of_nodes()}")
    print(f"  Edges: {graph.number_of_edges()}")

    # Check edges exist for similar pairs
    assert graph.has_edge(0, 1), "Sentences 0 and 1 should be connected"
    assert graph.has_edge(2, 3), "Sentences 2 and 3 should be connected"

    # Check edge weights are cosine similarities
    weight_01 = graph[0][1]['weight']
    expected_01 = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
    assert abs(weight_01 - expected_01) < 1e-5, f"Weight should be cosine similarity"

    print("  ✅ Sentence graph test passed")


def test_pagerank_centrality():
    """Test PageRank centrality computation."""
    print("\nTesting PageRank centrality...")

    # Create a graph with clear centrality structure
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(5))

    # Star graph: node 0 is center
    for i in range(1, 5):
        G.add_edge(0, i, weight=1.0)

    # Add some connections among leaves
    G.add_edge(1, 2, weight=0.5)
    G.add_edge(3, 4, weight=0.5)

    scores = compute_pagerank_centrality(G, alpha=0.85)

    print(f"  PageRank scores: {scores}")

    # Node 0 should have highest centrality (center of star)
    assert np.argmax(scores) == 0, "Center node should have highest PageRank"

    # Scores should sum to 1
    assert abs(scores.sum() - 1.0) < 1e-6, "PageRank scores should sum to 1"

    # All scores non-negative
    assert np.all(scores >= 0), "PageRank scores should be non-negative"

    print("  ✅ PageRank centrality test passed")


def test_centrality_with_known_vectors():
    """Test centrality with synthetic vectors where we know expected ranking."""
    print("\nTesting centrality with known vectors...")

    # Create 4 sentences where sentence 0 is most central (similar to all others)
    embeddings = np.array([
        [0.5, 0.5, 0.5, 0.5],  # Central - similar to all
        [1.0, 0.0, 0.0, 0.0],  # Peripheral
        [0.0, 1.0, 0.0, 0.0],  # Peripheral
        [0.0, 0.0, 1.0, 0.0],  # Peripheral
    ], dtype=np.float32)

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    graph = build_sentence_graph(embeddings, similarity_threshold=0.1)
    scores = compute_pagerank_centrality(graph)

    print(f"  Scores: {scores}")
    print(f"  Ranked indices: {np.argsort(scores)[::-1]}")

    # Sentence 0 should be ranked highest
    ranked = np.argsort(scores)[::-1]
    assert ranked[0] == 0, "Central sentence should be ranked first"

    print("  ✅ Known vectors centrality test passed")


def test_edge_cases():
    """Test edge cases."""
    print("\nTesting edge cases...")

    # Empty input
    enhanced = compute_topic_enhanced_embeddings([], np.zeros((0, 10)), [], 0.7)
    assert enhanced.shape == (0, 10)

    # Single sentence
    bert = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    lda_dists = [TopicDistribution(np.arange(2), np.array([0.6, 0.4]), 0, 0.6)]
    enhanced = compute_topic_enhanced_embeddings(["test"], bert, lda_dists, 0.7)
    assert enhanced.shape == (1, 5)

    # Zero vectors (should not crash)
    bert_zero = np.zeros((2, 3))
    lda_dists = [
        TopicDistribution(np.arange(2), np.array([0.5, 0.5]), 0, 0.5),
        TopicDistribution(np.arange(2), np.array([0.5, 0.5]), 0, 0.5),
    ]
    enhanced = compute_topic_enhanced_embeddings(["a", "b"], bert_zero, lda_dists, 0.7)
    assert enhanced.shape == (2, 5)
    # Should handle zero norms gracefully

    print("  ✅ Edge cases test passed")


if __name__ == "__main__":
    test_topic_enhanced_embeddings()
    test_topic_enhanced_embeddings_simple()
    test_sentence_graph()
    test_pagerank_centrality()
    test_centrality_with_known_vectors()
    test_edge_cases()
    print("\n✅ All Sentence Centrality tests passed!")