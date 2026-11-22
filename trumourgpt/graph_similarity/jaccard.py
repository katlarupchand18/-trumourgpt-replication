"""
Graph Similarity - Jaccard Triple Similarity.
Paper Section III-E, Equation 5: S(Gx, Gi) = |Tx ∩ Ti| / |Tx ∪ Ti|
Weighted variant: S(Gx, Gi) = Σ f(t) for t in Tx ∩ Ti / Σ f(t) for t in Tx ∪ Ti
"""

from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional, Any, Callable
from loguru import logger


@dataclass
class GraphSimilarityConfig:
    """Configuration for graph similarity."""
    use_weighted: bool = False
    weight_function: Optional[Callable[[Tuple[str, str, str]], float]] = None


def compute_jaccard_triple_similarity(
    triples_x: Set[Tuple[str, str, str]],
    triples_y: Set[Tuple[str, str, str]],
) -> float:
    """
    Compute standard Jaccard similarity between two sets of triples.

    S(Gx, Gi) = |Tx ∩ Ti| / |Tx ∪ Ti|

    Args:
        triples_x: Set of triples from graph Gx (query)
        triples_y: Set of triples from graph Gi (knowledge base)

    Returns:
        Jaccard similarity score in [0, 1]
    """
    if not triples_x and not triples_y:
        return 0.0

    intersection = triples_x & triples_y
    union = triples_x | triples_y

    if not union:
        return 0.0

    return len(intersection) / len(union)


def compute_weighted_jaccard_similarity(
    triples_x: Dict[Tuple[str, str, str], float],
    triples_y: Dict[Tuple[str, str, str], float],
) -> float:
    """
    Compute weighted Jaccard similarity (Paper Equation 5).

    S(Gx, Gi) = Σ_{t ∈ Tx ∩ Ti} f(t) / Σ_{t ∈ Tx ∪ Ti} f(t)

    Args:
        triples_x: Dict mapping triple -> weight for query graph
        triples_y: Dict mapping triple -> weight for knowledge graph

    Returns:
        Weighted Jaccard similarity score in [0, 1]
    """
    all_triples = set(triples_x.keys()) | set(triples_y.keys())

    if not all_triples:
        return 0.0

    numerator = 0.0
    denominator = 0.0

    for triple in all_triples:
        weight_x = triples_x.get(triple, 0.0)
        weight_y = triples_y.get(triple, 0.0)

        numerator += min(weight_x, weight_y)
        denominator += max(weight_x, weight_y)

    if denominator == 0:
        return 0.0

    return numerator / denominator


def compute_triple_similarity_matrix(
    query_triples: Set[Tuple[str, str, str]],
    kg_triples_list: List[Set[Tuple[str, str, str]]],
) -> List[float]:
    """
    Compute similarity between query and multiple knowledge graphs.

    Returns:
        List of similarity scores
    """
    return [
        compute_jaccard_triple_similarity(query_triples, kg_triples)
        for kg_triples in kg_triples_list
    ]


def normalize_triple(triple: Tuple[str, str, str]) -> Tuple[str, str, str]:
    """Normalize triple for comparison (lowercase, strip)."""
    return (
        triple[0].lower().strip(),
        triple[1].lower().strip(),
        triple[2].lower().strip(),
    )


def normalize_triple_set(triples: Set[Tuple[str, str, str]]) -> Set[Tuple[str, str, str]]:
    """Normalize a set of triples."""
    return {normalize_triple(t) for t in triples}


def normalize_triple_dict(triples: Dict[Tuple[str, str, str], float]) -> Dict[Tuple[str, str, str], float]:
    """Normalize keys in triple weight dictionary."""
    return {normalize_triple(k): v for k, v in triples.items()}


# Verification functions for testing with paper examples
def verify_paper_example() -> Dict[str, Any]:
    """
    Verify Jaccard implementation with paper example:

    Tx = {(Moderna, prevents, COVID), (Pfizer, treats, cancer)}
    Ti = {(Moderna, prevents, COVID), (A.Z., prevents, flu)}

    Tx ∩ Ti = {(Moderna, prevents, COVID)}
    Tx ∪ Ti = {(Moderna, prevents, COVID), (Pfizer, treats, cancer), (A.Z., prevents, flu)}
    S = 1/3 ≈ 0.33
    """
    # Paper example
    Tx = {
        ("Moderna", "prevents", "COVID"),
        ("Pfizer", "treats", "cancer"),
    }
    Ti = {
        ("Moderna", "prevents", "COVID"),
        ("A.Z.", "prevents", "flu"),
    }

    sim = compute_jaccard_triple_similarity(Tx, Ti)

    result = {
        "Tx": list(Tx),
        "Ti": list(Ti),
        "intersection": list(Tx & Ti),
        "union": list(Tx | Ti),
        "similarity": sim,
        "expected": 1/3,
        "matches_expected": abs(sim - 1/3) < 0.01,
    }

    logger.info(f"Paper example verification: {result}")
    return result


def verify_identical_graphs() -> Dict[str, Any]:
    """Verify similarity of identical graphs is 1.0."""
    triples = {
        ("A", "relates", "B"),
        ("B", "causes", "C"),
        ("C", "treats", "D"),
    }

    sim = compute_jaccard_triple_similarity(triples, triples)

    return {
        "similarity": sim,
        "expected": 1.0,
        "correct": sim == 1.0,
    }


def verify_disjoint_graphs() -> Dict[str, Any]:
    """Verify similarity of disjoint graphs is 0.0."""
    triples_x = {("A", "relates", "B")}
    triples_y = {("C", "causes", "D")}

    sim = compute_jaccard_triple_similarity(triples_x, triples_y)

    return {
        "similarity": sim,
        "expected": 0.0,
        "correct": sim == 0.0,
    }


def verify_subset() -> Dict[str, Any]:
    """Verify similarity when one is subset of other."""
    triples_x = {("A", "relates", "B"), ("B", "causes", "C")}
    triples_y = {("A", "relates", "B")}

    sim = compute_jaccard_triple_similarity(triples_x, triples_y)

    return {
        "similarity": sim,
        "expected": 0.5,  # 1/2
        "correct": abs(sim - 0.5) < 0.01,
    }


def run_all_verifications() -> Dict[str, Any]:
    """Run all verification tests."""
    results = {
        "paper_example": verify_paper_example(),
        "identical_graphs": verify_identical_graphs(),
        "disjoint_graphs": verify_disjoint_graphs(),
        "subset": verify_subset(),
    }

    all_passed = all(r.get("matches_expected", r.get("correct", False)) for r in results.values())
    results["all_passed"] = all_passed

    return results


# Weight function examples (for weighted Jaccard)
def centrality_weight_function(triple: Tuple[str, str, str], centrality_scores: Dict[str, float]) -> float:
    """Weight function based on entity centrality (Paper: f(t) from centrality scores)."""
    head, _, tail = triple
    return (centrality_scores.get(head, 0.0) + centrality_scores.get(tail, 0.0)) / 2


def frequency_weight_function(triple: Tuple[str, str, str], frequencies: Dict[Tuple[str, str, str], int]) -> float:
    """Weight function based on triple frequency."""
    return float(frequencies.get(triple, 1))


def uniform_weight_function(triple: Tuple[str, str, str]) -> float:
    """Uniform weight function (standard Jaccard)."""
    return 1.0


def apply_weight_function(
    triples: Set[Tuple[str, str, str]],
    weight_fn: Callable[[Tuple[str, str, str]], float],
) -> Dict[Tuple[str, str, str], float]:
    """Apply weight function to set of triples."""
    return {t: weight_fn(t) for t in triples}


if __name__ == "__main__":
    # Run verifications
    results = run_all_verifications()
    print(json.dumps(results, indent=2))