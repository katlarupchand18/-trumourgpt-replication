"""Evaluation module for TrumorGPT."""

from trumourgpt.evaluation.metrics import (
    EvaluationMetrics,
    compute_accuracy,
    compute_precision_recall_f1,
    compute_confusion_matrix,
    bootstrap_confidence_interval,
)
from trumourgpt.evaluation.dataset import (
    PolitiFactDataset,
    DatasetConfig,
    Claim,
    LabelMapping,
)

__all__ = [
    "EvaluationMetrics",
    "compute_accuracy",
    "compute_precision_recall_f1",
    "compute_confusion_matrix",
    "bootstrap_confidence_interval",
    "PolitiFactDataset",
    "DatasetConfig",
    "Claim",
    "LabelMapping",
]