"""
Evaluation Metrics for TrumorGPT.
Paper Section IV: Accuracy, Precision, Recall, F1-score, Confusion Matrix.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
from loguru import logger


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int

    # Per-class metrics
    precision_per_class: Dict[str, float] = None
    recall_per_class: Dict[str, float] = None
    f1_per_class: Dict[str, float] = None

    # Additional
    num_samples: int = 0
    num_abstentions: int = 0
    num_errors: int = 0
    confidence_interval: Optional[Tuple[float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "true_positives": self.true_positives,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision_per_class": self.precision_per_class,
            "recall_per_class": self.recall_per_class,
            "f1_per_class": self.f1_per_class,
            "num_samples": self.num_samples,
            "num_abstentions": self.num_abstentions,
            "num_errors": self.num_errors,
            "confidence_interval": self.confidence_interval,
        }

    def __str__(self) -> str:
        return (f"Accuracy: {self.accuracy:.4f} | "
                f"Precision: {self.precision:.4f} | "
                f"Recall: {self.recall:.4f} | "
                f"F1: {self.f1:.4f} | "
                f"TP: {self.true_positives} TN: {self.true_negatives} "
                f"FP: {self.false_positives} FN: {self.false_negatives}")


def compute_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Label order (default: sorted unique labels)

    Returns:
        (confusion_matrix, labels)
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    label_to_idx = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)

    for true, pred in zip(y_true, y_pred):
        if true in label_to_idx and pred in label_to_idx:
            cm[label_to_idx[true], label_to_idx[pred]] += 1

    return cm, labels


def compute_accuracy(y_true: List[str], y_pred: List[str]) -> float:
    """Compute accuracy."""
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def compute_precision_recall_f1(
    y_true: List[str],
    y_pred: List[str],
    positive_label: str = "TRUE",
    average: str = "binary",
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, F1-score.

    Args:
        y_true: Ground truth
        y_pred: Predictions
        positive_label: Label considered positive
        average: 'binary', 'macro', 'micro', 'weighted'

    Returns:
        (precision, recall, f1)
    """
    if not y_true:
        return 0.0, 0.0, 0.0

    labels = list(set(y_true) | set(y_pred))

    if average == "binary":
        # Binary classification metrics for positive_label
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive_label and p == positive_label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive_label and p == positive_label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive_label and p != positive_label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1

    elif average in ("macro", "micro", "weighted"):
        # Multi-class
        precisions = []
        recalls = []
        f1s = []
        supports = []

        for label in labels:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            supports.append(sum(1 for t in y_true if t == label))

        if average == "macro":
            return np.mean(precisions), np.mean(recalls), np.mean(f1s)
        elif average == "micro":
            # Global TP, FP, FN
            total_tp = sum(1 for t, p in zip(y_true, y_pred) if t == p)
            total_fp = sum(1 for t, p in zip(y_true, y_pred) if t != p)
            # For micro, precision = recall = accuracy
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            return precision, precision, precision
        elif average == "weighted":
            total_support = sum(supports)
            if total_support == 0:
                return 0.0, 0.0, 0.0
            precision = sum(p * s for p, s in zip(precisions, supports)) / total_support
            recall = sum(r * s for r, s in zip(recalls, supports)) / total_support
            f1 = sum(f * s for f, s in zip(f1s, supports)) / total_support
            return precision, recall, f1

    return 0.0, 0.0, 0.0


def compute_per_class_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute precision, recall, F1 for each class."""
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    metrics = {}
    for label in labels:
        # Treat each label as positive in one-vs-rest
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = sum(1 for t in y_true if t == label)

        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    return metrics


def evaluate_binary(
    y_true: List[str],
    y_pred: List[str],
    positive_label: str = "TRUE",
    negative_label: str = "FALSE",
    labels: List[str] = None,
) -> EvaluationMetrics:
    """
    Evaluate binary classification.
    Maps labels to positive/negative.
    """
    # Normalize labels
    if labels is None:
        labels = [negative_label, positive_label]

    # Filter valid predictions
    valid_pairs = [(t, p) for t, p in zip(y_true, y_pred) if t in labels and p in labels]
    if not valid_pairs:
        return EvaluationMetrics(
            accuracy=0.0, precision=0.0, recall=0.0, f1=0.0,
            true_positives=0, true_negatives=0, false_positives=0, false_negatives=0,
            num_samples=len(y_true),
        )

    y_true_valid, y_pred_valid = zip(*valid_pairs)

    # Compute confusion matrix values
    tp = sum(1 for t, p in valid_pairs if t == positive_label and p == positive_label)
    tn = sum(1 for t, p in valid_pairs if t == negative_label and p == negative_label)
    fp = sum(1 for t, p in valid_pairs if t == negative_label and p == positive_label)
    fn = sum(1 for t, p in valid_pairs if t == positive_label and p == negative_label)

    # Metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Per-class metrics
    per_class = compute_per_class_metrics(list(y_true_valid), list(y_pred_valid), labels)
    precision_per_class = {k: v["precision"] for k, v in per_class.items()}
    recall_per_class = {k: v["recall"] for k, v in per_class.items()}
    f1_per_class = {k: v["f1"] for k, v in per_class.items()}

    # Count abstentions and errors
    num_abstentions = sum(1 for p in y_pred if p not in labels)
    num_errors = sum(1 for t, p in zip(y_true, y_pred) if p not in labels)

    return EvaluationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        precision_per_class=precision_per_class,
        recall_per_class=recall_per_class,
        f1_per_class=f1_per_class,
        num_samples=len(valid_pairs),
        num_abstentions=num_abstentions,
        num_errors=num_errors,
    )


def bootstrap_confidence_interval(
    y_true: List[str],
    y_pred: List[str],
    metric_fn: callable = compute_accuracy,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Args:
        y_true: Ground truth
        y_pred: Predictions
        metric_fn: Function to compute metric
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        random_state: Random seed

    Returns:
        (lower_bound, upper_bound)
    """
    np.random.seed(random_state)
    n = len(y_true)
    if n == 0:
        return (0.0, 0.0)

    bootstrap_scores = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        sample_true = [y_true[i] for i in indices]
        sample_pred = [y_pred[i] for i in indices]
        score = metric_fn(sample_true, sample_pred)
        bootstrap_scores.append(score)

    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_scores, alpha * 100)
    upper = np.percentile(bootstrap_scores, (1 - alpha) * 100)

    return (float(lower), float(upper))


def print_evaluation_report(metrics: EvaluationMetrics, labels: List[str] = None) -> None:
    """Print formatted evaluation report."""
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"Samples evaluated: {metrics.num_samples}")
    print(f"Abstentions: {metrics.num_abstentions}")
    print(f"Errors: {metrics.num_errors}")
    print("-" * 60)
    print(f"Accuracy:  {metrics.accuracy:.4f}")
    print(f"Precision: {metrics.precision:.4f}")
    print(f"Recall:    {metrics.recall:.4f}")
    print(f"F1-Score:  {metrics.f1:.4f}")
    print("-" * 60)
    print("Confusion Matrix:")
    print(f"  TP: {metrics.true_positives:4d}  FP: {metrics.false_positives:4d}")
    print(f"  FN: {metrics.false_negatives:4d}  TN: {metrics.true_negatives:4d}")
    print("-" * 60)

    if metrics.precision_per_class:
        print("Per-Class Metrics:")
        for label in (labels or sorted(metrics.precision_per_class.keys())):
            if label in metrics.precision_per_class:
                print(f"  {label:15s}: P={metrics.precision_per_class[label]:.4f} "
                      f"R={metrics.recall_per_class[label]:.4f} "
                      f"F1={metrics.f1_per_class[label]:.4f}")

    if metrics.confidence_interval:
        print("-" * 60)
        print(f"95% CI for Accuracy: [{metrics.confidence_interval[0]:.4f}, {metrics.confidence_interval[1]:.4f}]")

    print("=" * 60)


# Verification with paper numbers
def verify_paper_metrics() -> Dict[str, Any]:
    """Verify metrics computation with paper's reported numbers."""
    # Paper: TrumorGPT 88.5% accuracy, 91.4% precision, 85.0% recall, 88.1% F1
    # Confusion matrix (normalized): TP=85%, TN=92%
    # 600 samples: 300 TRUE, 300 FALSE

    # Reconstruct from normalized confusion matrix
    # TP = 85% of 300 = 255
    # TN = 92% of 300 = 276
    # FP = 300 - 276 = 24
    # FN = 300 - 255 = 45

    y_true = ["TRUE"] * 300 + ["FALSE"] * 300
    y_pred = ["TRUE"] * 255 + ["FALSE"] * 45 + ["TRUE"] * 24 + ["FALSE"] * 276

    metrics = evaluate_binary(y_true, y_pred, "TRUE", "FALSE")

    expected = {
        "accuracy": 0.885,
        "precision": 0.914,
        "recall": 0.850,
        "f1": 0.881,
    }

    result = {
        "computed": {
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
        },
        "expected": expected,
        "matches": {
            "accuracy": abs(metrics.accuracy - expected["accuracy"]) < 0.005,
            "precision": abs(metrics.precision - expected["precision"]) < 0.005,
            "recall": abs(metrics.recall - expected["recall"]) < 0.005,
            "f1": abs(metrics.f1 - expected["f1"]) < 0.005,
        },
    }

    return result