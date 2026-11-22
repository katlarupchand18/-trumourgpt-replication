"""
Tests for Evaluation Metrics (Paper Section IV).
Verifies: Accuracy, Precision, Recall, F1, Confusion Matrix, Bootstrap CI.
"""

import numpy as np
from trumourgpt.evaluation.metrics import (
    compute_accuracy,
    compute_precision_recall_f1,
    compute_confusion_matrix,
    evaluate_binary,
    compute_per_class_metrics,
    bootstrap_confidence_interval,
    verify_paper_metrics,
    EvaluationMetrics,
)


def test_basic_metrics():
    """Test basic metric computations."""
    print("Testing basic metrics...")

    y_true = ["TRUE", "TRUE", "FALSE", "FALSE", "TRUE", "FALSE"]
    y_pred = ["TRUE", "FALSE", "FALSE", "FALSE", "TRUE", "TRUE"]

    # Accuracy
    acc = compute_accuracy(y_true, y_pred)
    expected_acc = 4/6  # 4 correct out of 6
    assert abs(acc - expected_acc) < 1e-6, f"Accuracy: expected {expected_acc}, got {acc}"
    print(f"  Accuracy: {acc:.4f} (expected {expected_acc:.4f})")

    # Precision, Recall, F1
    prec, rec, f1 = compute_precision_recall_f1(y_true, y_pred, "TRUE")
    # TP=2, FP=1, FN=1
    expected_prec = 2/3
    expected_rec = 2/3
    expected_f1 = 2/3
    assert abs(prec - expected_prec) < 1e-6
    assert abs(rec - expected_rec) < 1e-6
    assert abs(f1 - expected_f1) < 1e-6
    print(f"  Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    print("  ✅ Basic metrics test passed")


def test_confusion_matrix():
    """Test confusion matrix computation."""
    print("\nTesting confusion matrix...")

    y_true = ["TRUE", "TRUE", "FALSE", "FALSE"]
    y_pred = ["TRUE", "FALSE", "FALSE", "TRUE"]

    cm, labels = compute_confusion_matrix(y_true, y_pred)

    print(f"  Labels: {labels}")
    print(f"  Confusion Matrix:\n{cm}")

    # Should be 2x2
    assert cm.shape == (2, 2)
    # Diagonal: correct predictions
    assert cm[0, 0] == 1  # TRUE->TRUE
    assert cm[1, 1] == 1  # FALSE->FALSE
    # Off-diagonal: errors
    assert cm[0, 1] == 1  # TRUE->FALSE
    assert cm[1, 0] == 1  # FALSE->TRUE

    print("  ✅ Confusion matrix test passed")


def test_evaluate_binary():
    """Test full binary evaluation."""
    print("\nTesting binary evaluation...")

    # Create test case with known values
    # 300 TRUE, 300 FALSE
    y_true = ["TRUE"] * 300 + ["FALSE"] * 300
    # Predictions: TP=255, FN=45, FP=24, TN=276 (paper numbers)
    y_pred = ["TRUE"] * 255 + ["FALSE"] * 45 + ["TRUE"] * 24 + ["FALSE"] * 276

    metrics = evaluate_binary(y_true, y_pred, "TRUE", "FALSE")

    print(f"  Accuracy: {metrics.accuracy:.4f}")
    print(f"  Precision: {metrics.precision:.4f}")
    print(f"  Recall: {metrics.recall:.4f}")
    print(f"  F1: {metrics.f1:.4f}")
    print(f"  TP: {metrics.true_positives}, TN: {metrics.true_negatives}")
    print(f"  FP: {metrics.false_positives}, FN: {metrics.false_negatives}")

    # Check against paper values
    assert abs(metrics.accuracy - 0.885) < 0.005
    assert abs(metrics.precision - 0.914) < 0.005
    assert abs(metrics.recall - 0.850) < 0.005
    assert abs(metrics.f1 - 0.881) < 0.005

    # Check confusion matrix values
    assert metrics.true_positives == 255
    assert metrics.true_negatives == 276
    assert metrics.false_positives == 24
    assert metrics.false_negatives == 45

    print("  ✅ Binary evaluation test passed")


def test_per_class_metrics():
    """Test per-class metrics."""
    print("\nTesting per-class metrics...")

    y_true = ["TRUE", "TRUE", "FALSE", "FALSE", "TRUE"]
    y_pred = ["TRUE", "FALSE", "FALSE", "TRUE", "TRUE"]

    metrics = compute_per_class_metrics(y_true, y_pred)

    print(f"  Per-class metrics: {metrics}")

    # TRUE class: TP=2, FP=1, FN=1
    assert abs(metrics["TRUE"]["precision"] - 2/3) < 1e-6
    assert abs(metrics["TRUE"]["recall"] - 2/3) < 1e-6
    assert metrics["TRUE"]["support"] == 3

    # FALSE class: TP=1, FP=1, FN=1
    assert abs(metrics["FALSE"]["precision"] - 0.5) < 1e-6
    assert abs(metrics["FALSE"]["recall"] - 0.5) < 1e-6
    assert metrics["FALSE"]["support"] == 2

    print("  ✅ Per-class metrics test passed")


def test_bootstrap_ci():
    """Test bootstrap confidence interval."""
    print("\nTesting bootstrap CI...")

    # Large enough sample for bootstrap
    np.random.seed(42)
    y_true = ["TRUE"] * 100 + ["FALSE"] * 100
    y_pred = ["TRUE"] * 90 + ["FALSE"] * 10 + ["TRUE"] * 10 + ["FALSE"] * 90

    ci = bootstrap_confidence_interval(y_true, y_pred, compute_accuracy, n_bootstrap=100, confidence=0.95)

    print(f"  95% CI for accuracy: [{ci[0]:.4f}, {ci[1]:.4f}]")

    # CI should contain true accuracy
    true_acc = compute_accuracy(y_true, y_pred)
    assert ci[0] <= true_acc <= ci[1], f"True accuracy {true_acc} not in CI {ci}"

    # CI should be reasonable width
    assert ci[1] - ci[0] < 0.2, "CI too wide"

    print("  ✅ Bootstrap CI test passed")


def test_edge_cases():
    """Test edge cases."""
    print("\nTesting edge cases...")

    # Empty lists
    acc = compute_accuracy([], [])
    assert acc == 0.0

    prec, rec, f1 = compute_precision_recall_f1([], [], "TRUE")
    assert prec == 0.0 and rec == 0.0 and f1 == 0.0

    # All correct
    y_true = ["TRUE", "FALSE", "TRUE"]
    y_pred = ["TRUE", "FALSE", "TRUE"]
    acc = compute_accuracy(y_true, y_pred)
    assert acc == 1.0

    # All wrong
    y_pred = ["FALSE", "TRUE", "FALSE"]
    acc = compute_accuracy(y_true, y_pred)
    assert acc == 0.0

    # Single class
    y_true = ["TRUE", "TRUE"]
    y_pred = ["TRUE", "TRUE"]
    metrics = evaluate_binary(y_true, y_pred, "TRUE", "FALSE")
    assert metrics.accuracy == 1.0
    assert metrics.recall == 1.0
    assert metrics.precision == 1.0

    # No positive predictions
    y_true = ["TRUE", "TRUE"]
    y_pred = ["FALSE", "FALSE"]
    prec, rec, f1 = compute_precision_recall_f1(y_true, y_pred, "TRUE")
    assert prec == 0.0
    assert rec == 0.0
    assert f1 == 0.0

    print("  ✅ Edge cases test passed")


def test_paper_metrics_verification():
    """Verify metrics match paper's reported numbers."""
    print("\nVerifying paper metrics...")

    result = verify_paper_metrics()

    print(f"  Computed: {result['computed']}")
    print(f"  Expected: {result['expected']}")
    print(f"  Matches: {result['matches']}")

    assert result["matches"]["accuracy"], f"Accuracy mismatch: {result['computed']['accuracy']:.4f} vs {result['expected']['accuracy']:.4f}"
    assert result["matches"]["precision"], f"Precision mismatch"
    assert result["matches"]["recall"], f"Recall mismatch"
    assert result["matches"]["f1"], f"F1 mismatch"

    print("  ✅ Paper metrics verification passed")


if __name__ == "__main__":
    test_basic_metrics()
    test_confusion_matrix()
    test_evaluate_binary()
    test_per_class_metrics()
    test_bootstrap_ci()
    test_edge_cases()
    test_paper_metrics_verification()
    print("\n✅ All Evaluation Metrics tests passed!")