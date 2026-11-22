"""
Tests for Jaccard triple similarity (Paper Equation 5).
"""

import numpy as np
from trumourgpt.graph_similarity.jaccard import (
    compute_jaccard_triple_similarity,
    compute_weighted_jaccard_similarity,
    verify_paper_example,
    verify_identical_graphs,
    verify_disjoint_graphs,
    verify_subset,
    run_all_verifications,
)


def test_paper_example():
    """Test Jaccard with paper's example."""
    result = verify_paper_example()

    print("Paper Example Verification:")
    print(f"  Tx: {result['Tx']}")
    print(f"  Ti: {result['Ti']}")
    print(f"  Intersection: {result['intersection']}")
    print(f"  Union: {result['union']}")
    print(f"  Similarity: {result['similarity']:.4f}")
    print(f"  Expected: {result['expected']:.4f}")
    print(f"  Matches: {result['matches_expected']}")

    assert result["matches_expected"], f"Expected {result['expected']:.4f}, got {result['similarity']:.4f}"
    assert abs(result["similarity"] - 1/3) < 0.01


def test_identical_graphs():
    """Test identical graphs have similarity 1.0."""
    result = verify_identical_graphs()
    print(f"\nIdentical Graphs: similarity={result['similarity']:.4f}, expected={result['expected']:.4f}")
    assert result["correct"]


def test_disjoint_graphs():
    """Test disjoint graphs have similarity 0.0."""
    result = verify_disjoint_graphs()
    print(f"Disjoint Graphs: similarity={result['similarity']:.4f}, expected={result['expected']:.4f}")
    assert result["correct"]


def test_subset():
    """Test subset similarity."""
    result = verify_subset()
    print(f"Subset: similarity={result['similarity']:.4f}, expected={result['expected']:.4f}")
    assert result["correct"]


def test_weighted_jaccard():
    """Test weighted Jaccard similarity."""
    # Same triples, different weights
    triples_x = {
        ("A", "relates", "B"): 1.0,
        ("B", "causes", "C"): 0.5,
    }
    triples_y = {
        ("A", "relates", "B"): 0.8,
        ("B", "causes", "C"): 0.3,
        ("C", "treats", "D"): 0.7,
    }

    sim = compute_weighted_jaccard_similarity(triples_x, triples_y)

    # Intersection: min(1.0, 0.8) + min(0.5, 0.3) = 0.8 + 0.3 = 1.1
    # Union: max(1.0, 0.8) + max(0.5, 0.3) + max(0, 0.7) = 1.0 + 0.5 + 0.7 = 2.2
    # Expected: 1.1 / 2.2 = 0.5
    expected = 0.5

    print(f"Weighted Jaccard: similarity={sim:.4f}, expected={expected:.4f}")
    assert abs(sim - expected) < 0.01


def test_normalization():
    """Test triple normalization."""
    from trumourgpt.graph_similarity.jaccard import normalize_triple, normalize_triple_set

    triple = ("Moderna", "PREVENTS", "Covid")
    normalized = normalize_triple(triple)
    assert normalized == ("moderna", "prevents", "covid")

    triples = {("A", "RELATES", "B"), ("C", "causes", "D")}
    normalized_set = normalize_triple_set(triples)
    assert ("a", "relates", "b") in normalized_set
    assert ("c", "causes", "d") in normalized_set


def test_all_verifications():
    """Run all verification tests."""
    results = run_all_verifications()
    print("\nAll Verifications:")
    for name, result in results.items():
        if name != "all_passed":
            status = "PASS" if result.get("matches_expected", result.get("correct", False)) else "FAIL"
            print(f"  {name}: {status}")

    assert results["all_passed"], "Some verifications failed"


if __name__ == "__main__":
    test_paper_example()
    test_identical_graphs()
    test_disjoint_graphs()
    test_subset()
    test_weighted_jaccard()
    test_normalization()
    test_all_verifications()
    print("\n✅ All Jaccard tests passed!")