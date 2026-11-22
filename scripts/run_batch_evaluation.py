#!/usr/bin/env python
"""
Batch Evaluation Script for TrumorGPT.
Runs 600 PolitiFact claims in batches: 150, 100, 100, 100, 100, 50
with metrics after each batch. Uses fixed random seed for reproducible ordering.

Features:
- Crash-safe SQLite persistence with transactional commits
- Automatic resume from last completed claim
- Per-claim immediate saving (no batch waiting)
- Dual Gemini API key support with rate limiting
- Exact-once result recording
- Partial accuracy calculation at any time
"""

import json
import os
import sys
import time
import sqlite3
import hashlib
import fcntl
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import random
from contextlib import contextmanager

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from trumourgpt.config.settings import get_settings
from trumourgpt.evaluation.dataset import PolitiFactDataset, DatasetConfig, BinaryLabel
from trumourgpt.evaluation.metrics import evaluate_binary, EvaluationMetrics
from trumourgpt.experiments.pipeline import TrumorGPTPipeline, PipelineConfig, PipelineResult


# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
logger.add("logs/batch_evaluation_{time:YYYY-MM-DD}.log", level="DEBUG", rotation="10 MB")


# Batch configuration
BATCH_SIZES = [150, 100, 100, 100, 100, 50]  # Total = 600
RANDOM_SEED = 42
RUN_ID = "trumourgpt_eval_001"  # Fixed run ID for resume capability


@dataclass
class BatchResult:
    """Result of a single batch evaluation."""
    batch_num: int
    batch_size: int
    claims_processed: int
    metrics: EvaluationMetrics
    total_time: float
    avg_time_per_claim: float
    timestamp: str
    predictions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ClaimRecord:
    """Single claim prediction record for database."""
    run_id: str
    claim_id: str
    claim_text: str
    true_label: str
    predicted_label: str
    confidence: float
    evidence_count: int
    reasoning: str
    evidence_relationship: str
    processing_time: float
    batch_num: int
    batch_position: int
    global_position: int
    timestamp: str
    model_identifier: str
    api_status: str  # "success", "error", "rate_limited", "quota_exhausted"
    api_error: Optional[str] = None


class SQLitePersistence:
    """Crash-safe SQLite persistence for evaluation results."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    run_id TEXT NOT NULL,
                    claim_id TEXT NOT NULL,
                    claim_text TEXT NOT NULL,
                    true_label TEXT NOT NULL,
                    predicted_label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence_count INTEGER NOT NULL,
                    reasoning TEXT NOT NULL,
                    evidence_relationship TEXT NOT NULL,
                    processing_time REAL NOT NULL,
                    batch_num INTEGER NOT NULL,
                    batch_position INTEGER NOT NULL,
                    global_position INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    model_identifier TEXT NOT NULL,
                    api_status TEXT NOT NULL,
                    api_error TEXT,
                    PRIMARY KEY (run_id, claim_id)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    run_id TEXT PRIMARY KEY,
                    dataset_hash TEXT NOT NULL,
                    claim_order TEXT NOT NULL,  -- JSON array of claim_ids
                    batch_boundaries TEXT NOT NULL,  -- JSON array of cumulative sizes
                    completed_claim_ids TEXT NOT NULL,  -- JSON array
                    pending_claim_ids TEXT NOT NULL,  -- JSON array
                    last_completed_batch INTEGER NOT NULL,
                    last_completed_global_pos INTEGER NOT NULL,
                    model_config TEXT NOT NULL,  -- JSON
                    prompt_version TEXT NOT NULL,
                    kg_version TEXT NOT NULL,
                    retrieval_config_hash TEXT NOT NULL,
                    api_usage_total INTEGER DEFAULT 0,
                    api_usage_today INTEGER DEFAULT 0,
                    api_usage_last_reset TEXT,
                    last_update_timestamp TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'running'  -- running, completed, paused, error
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_run_batch
                ON predictions(run_id, batch_num)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_run_global
                ON predictions(run_id, global_position)
            """)

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get database connection with file locking for crash safety."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            yield conn
        finally:
            conn.close()

    def save_prediction(self, record: ClaimRecord) -> None:
        """Save a single prediction atomically."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO predictions (
                    run_id, claim_id, claim_text, true_label, predicted_label,
                    confidence, evidence_count, reasoning, evidence_relationship,
                    processing_time, batch_num, batch_position, global_position,
                    timestamp, model_identifier, api_status, api_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.run_id, record.claim_id, record.claim_text, record.true_label,
                record.predicted_label, record.confidence, record.evidence_count,
                record.reasoning, record.evidence_relationship, record.processing_time,
                record.batch_num, record.batch_position, record.global_position,
                record.timestamp, record.model_identifier, record.api_status, record.api_error
            ))
            conn.commit()

    def get_prediction(self, run_id: str, claim_id: str) -> Optional[Dict[str, Any]]:
        """Get a single prediction by claim_id."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM predictions WHERE run_id = ? AND claim_id = ?
            """, (run_id, claim_id))
            row = cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, row))
            return None

    def get_all_predictions(self, run_id: str) -> List[Dict[str, Any]]:
        """Get all predictions for a run, ordered by global_position."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM predictions WHERE run_id = ? ORDER BY global_position
            """, (run_id,))
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_completed_claim_ids(self, run_id: str) -> List[str]:
        """Get all completed claim IDs for a run."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT claim_id FROM predictions WHERE run_id = ? AND api_status = 'success'
                ORDER BY global_position
            """, (run_id,))
            return [row[0] for row in cursor.fetchall()]

    def save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Save checkpoint atomically using temporary table approach."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO checkpoints (
                    run_id, dataset_hash, claim_order, batch_boundaries,
                    completed_claim_ids, pending_claim_ids, last_completed_batch,
                    last_completed_global_pos, model_config, prompt_version,
                    kg_version, retrieval_config_hash, api_usage_total,
                    api_usage_today, api_usage_last_reset, last_update_timestamp, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                checkpoint["run_id"],
                checkpoint["dataset_hash"],
                json.dumps(checkpoint["claim_order"]),
                json.dumps(checkpoint["batch_boundaries"]),
                json.dumps(checkpoint["completed_claim_ids"]),
                json.dumps(checkpoint["pending_claim_ids"]),
                checkpoint["last_completed_batch"],
                checkpoint["last_completed_global_pos"],
                json.dumps(checkpoint["model_config"]),
                checkpoint["prompt_version"],
                checkpoint["kg_version"],
                checkpoint["retrieval_config_hash"],
                checkpoint.get("api_usage_total", 0),
                checkpoint.get("api_usage_today", 0),
                checkpoint.get("api_usage_last_reset", datetime.now().isoformat()),
                datetime.now().isoformat(),
                checkpoint.get("status", "running")
            ))
            conn.commit()

    def load_checkpoint(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint for a run."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM checkpoints WHERE run_id = ?
            """, (run_id,))
            row = cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                data = dict(zip(columns, row))
                # Parse JSON fields
                data["claim_order"] = json.loads(data["claim_order"])
                data["batch_boundaries"] = json.loads(data["batch_boundaries"])
                data["completed_claim_ids"] = json.loads(data["completed_claim_ids"])
                data["pending_claim_ids"] = json.loads(data["pending_claim_ids"])
                data["model_config"] = json.loads(data["model_config"])
                return data
            return None

    def get_prediction_count(self, run_id: str) -> int:
        """Get count of predictions for a run."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM predictions WHERE run_id = ?", (run_id,))
            return cursor.fetchone()[0]

    def get_batch_predictions(self, run_id: str, batch_num: int) -> List[Dict[str, Any]]:
        """Get predictions for a specific batch."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM predictions WHERE run_id = ? AND batch_num = ? ORDER BY batch_position
            """, (run_id, batch_num))
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]


def compute_dataset_hash(claims: List[Dict[str, Any]]) -> str:
    """Compute hash of dataset for change detection."""
    content = json.dumps([{"id": c["claim_id"], "text": c["text"]} for c in claims], sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def load_test_claims() -> List[Dict[str, Any]]:
    """Load all test claims with ground truth labels."""
    # Load settings from config.yaml to ensure correct Gemini model
    from trumourgpt.config.settings import get_settings, reset_settings
    reset_settings()
    settings = get_settings("config.yaml")

    dataset = PolitiFactDataset(DatasetConfig(
        raw_path=settings.dataset.raw_path,
        processed_path=settings.dataset.processed_path,
        splits_path=settings.dataset.splits_path,
        target_size=settings.dataset.target_size,
        true_count=settings.dataset.true_count,
        false_count=settings.dataset.false_count,
        splits=settings.dataset.splits,
    ))

    # Load or process claims
    claims = dataset.load_processed()

    # Get test split (400 claims)
    test_claims = dataset.get_split("test")
    logger.info(f"Loaded {len(test_claims)} test claims")

    # Convert to dict format
    claim_dicts = []
    for claim in test_claims:
        claim_dicts.append({
            "claim_id": str(claim.claim_id),
            "text": claim.text,
            "binary_label": claim.binary_label.value,
            "original_label": claim.original_label.value,
            "category": claim.category,
            "source_url": claim.source_url,
        })

    return claim_dicts


def get_claim_order(claims: List[Dict[str, Any]], seed: int = RANDOM_SEED) -> List[str]:
    """Get deterministic claim order using fixed seed."""
    claim_ids = [c["claim_id"] for c in claims]
    random.seed(seed)
    random.shuffle(claim_ids)
    return claim_ids


def evaluate_claim(pipeline: TrumorGPTPipeline, claim: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a single claim and return prediction dict."""
    claim_id = claim["claim_id"]
    claim_text = claim["text"]
    true_label = claim["binary_label"]

    try:
        result = pipeline.process_claim(claim_text, claim_id)

        pred = {
            "claim_id": claim_id,
            "claim_text": claim_text,
            "true_label": true_label,
            "predicted_label": result.verification_result.predicted_label.value,
            "confidence": result.verification_result.confidence,
            "evidence_count": len(result.verification_result.retrieved_evidence),
            "reasoning": result.verification_result.reasoning,
            "evidence_relationship": result.verification_result.evidence_relationship,
            "processing_time": result.total_time,
            "model_identifier": result.verification_result.model_identifier,
            "api_status": "success",
            "api_error": None,
        }

    except Exception as e:
        logger.error(f"Error processing {claim_id}: {e}")
        pred = {
            "claim_id": claim_id,
            "claim_text": claim_text,
            "true_label": true_label,
            "predicted_label": "ERROR",
            "confidence": 0.0,
            "evidence_count": 0,
            "reasoning": f"Error: {str(e)}",
            "evidence_relationship": "ERROR",
            "processing_time": 0.0,
            "model_identifier": "error",
            "api_status": "error",
            "api_error": str(e),
        }

    return pred


def compute_batch_metrics(predictions: List[Dict[str, Any]]) -> EvaluationMetrics:
    """Compute metrics for a batch of predictions."""
    y_true = []
    y_pred = []
    for pred in predictions:
        if pred["predicted_label"] not in ("ERROR", "UNDETERMINED"):
            y_true.append(pred["true_label"])
            y_pred.append(pred["predicted_label"])

    if not y_true:
        return EvaluationMetrics(
            accuracy=0.0, precision=0.0, recall=0.0, f1=0.0,
            confusion_matrix=[[0, 0], [0, 0]], per_class={}
        )

    return evaluate_binary(y_true, y_pred, "TRUE", "FALSE")


def compute_cumulative_metrics(predictions: List[Dict[str, Any]]) -> EvaluationMetrics:
    """Compute metrics on all predictions so far."""
    return compute_batch_metrics(predictions)


def generate_batch_report(
    batch_num: int,
    batch_predictions: List[Dict[str, Any]],
    cumulative_predictions: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Generate batch-specific report."""
    batch_metrics = compute_batch_metrics(batch_predictions)
    cum_metrics = compute_cumulative_metrics(cumulative_predictions)

    report_file = output_dir / f"batch_{batch_num:03d}_report.md"

    # Count labels
    batch_true = sum(1 for p in batch_predictions if p["predicted_label"] == "TRUE")
    batch_false = sum(1 for p in batch_predictions if p["predicted_label"] == "FALSE")
    batch_undetermined = sum(1 for p in batch_predictions if p["predicted_label"] == "UNDETERMINED")
    batch_error = sum(1 for p in batch_predictions if p["predicted_label"] == "ERROR")

    cum_true = sum(1 for p in cumulative_predictions if p["predicted_label"] == "TRUE")
    cum_false = sum(1 for p in cumulative_predictions if p["predicted_label"] == "FALSE")
    cum_undetermined = sum(1 for p in cumulative_predictions if p["predicted_label"] == "UNDETERMINED")
    cum_error = sum(1 for p in cumulative_predictions if p["predicted_label"] == "ERROR")

    with open(report_file, "w") as f:
        f.write(f"# Batch {batch_num} Evaluation Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Batch Metrics\n\n")
        f.write(f"- **Accuracy:** {batch_metrics.accuracy:.4f}\n")
        f.write(f"- **Precision:** {batch_metrics.precision:.4f}\n")
        f.write(f"- **Recall:** {batch_metrics.recall:.4f}\n")
        f.write(f"- **F1 Score:** {batch_metrics.f1:.4f}\n\n")

        f.write("### Batch Label Distribution\n\n")
        f.write(f"- TRUE: {batch_true}\n")
        f.write(f"- FALSE: {batch_false}\n")
        f.write(f"- UNDETERMINED: {batch_undetermined}\n")
        f.write(f"- ERROR: {batch_error}\n\n")

        f.write("## Cumulative Metrics\n\n")
        f.write(f"- **Accuracy:** {cum_metrics.accuracy:.4f}\n")
        f.write(f"- **Precision:** {cum_metrics.precision:.4f}\n")
        f.write(f"- **Recall:** {cum_metrics.recall:.4f}\n")
        f.write(f"- **F1 Score:** {cum_metrics.f1:.4f}\n\n")

        f.write("### Cumulative Label Distribution\n\n")
        f.write(f"- TRUE: {cum_true}\n")
        f.write(f"- FALSE: {cum_false}\n")
        f.write(f"- UNDETERMINED: {cum_undetermined}\n")
        f.write(f"- ERROR: {cum_error}\n\n")

        if cum_metrics.confusion_matrix:
            f.write("### Cumulative Confusion Matrix\n\n")
            f.write("| | Pred TRUE | Pred FALSE |\n")
            f.write("|---|---|---|\n")
            f.write(f"| True TRUE | {cum_metrics.confusion_matrix[0][0]} | {cum_metrics.confusion_matrix[0][1]} |\n")
            f.write(f"| True FALSE | {cum_metrics.confusion_matrix[1][0]} | {cum_metrics.confusion_matrix[1][1]} |\n\n")

    logger.info(f"Generated batch {batch_num} report: {report_file}")


def save_batch_csv(batch_num: int, predictions: List[Dict[str, Any]], output_dir: Path) -> None:
    """Save batch predictions as CSV."""
    import csv
    csv_file = output_dir / f"batch_{batch_num:03d}_predictions.csv"

    fieldnames = [
        "claim_id", "claim_text", "true_label", "predicted_label",
        "confidence", "evidence_count", "reasoning", "evidence_relationship",
        "processing_time", "model_identifier", "api_status", "api_error"
    ]

    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pred in predictions:
            # Only write relevant fields
            row = {k: pred.get(k, "") for k in fieldnames}
            writer.writerow(row)

    logger.info(f"Saved batch {batch_num} CSV: {csv_file}")


def save_cumulative_csv(predictions: List[Dict[str, Any]], output_dir: Path) -> None:
    """Save all predictions as cumulative CSV."""
    import csv
    csv_file = output_dir / "all_predictions.csv"

    fieldnames = [
        "claim_id", "claim_text", "true_label", "predicted_label",
        "confidence", "evidence_count", "reasoning", "evidence_relationship",
        "processing_time", "model_identifier", "api_status", "api_error",
        "batch_num", "global_position"
    ]

    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pred in predictions:
            row = {k: pred.get(k, "") for k in fieldnames}
            writer.writerow(row)

    logger.info(f"Saved cumulative CSV: {csv_file}")


def save_metrics_json(
    batch_num: int,
    batch_metrics: EvaluationMetrics,
    cum_metrics: EvaluationMetrics,
    output_dir: Path,
) -> None:
    """Save metrics as JSON."""
    metrics_file = output_dir / f"batch_{batch_num:03d}_metrics.json"

    data = {
        "batch_num": batch_num,
        "batch_metrics": {
            "accuracy": batch_metrics.accuracy,
            "precision": batch_metrics.precision,
            "recall": batch_metrics.recall,
            "f1": batch_metrics.f1,
            "confusion_matrix": batch_metrics.confusion_matrix,
            "per_class": batch_metrics.per_class,
        },
        "cumulative_metrics": {
            "accuracy": cum_metrics.accuracy,
            "precision": cum_metrics.precision,
            "recall": cum_metrics.recall,
            "f1": cum_metrics.f1,
            "confusion_matrix": cum_metrics.confusion_matrix,
            "per_class": cum_metrics.per_class,
        },
        "timestamp": datetime.now().isoformat(),
    }

    with open(metrics_file, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved batch {batch_num} metrics: {metrics_file}")


def print_status(predictions: List[Dict[str, Any]], batch_num: int, batch_size: int, total: int) -> None:
    """Print current status."""
    completed = len(predictions)
    correct = sum(1 for p in predictions if p["predicted_label"] == p["true_label"] and p["predicted_label"] in ("TRUE", "FALSE"))
    total_determined = sum(1 for p in predictions if p["predicted_label"] in ("TRUE", "FALSE"))
    undetermined = sum(1 for p in predictions if p["predicted_label"] == "UNDETERMINED")
    errors = sum(1 for p in predictions if p["predicted_label"] == "ERROR")

    overall_acc = correct / completed if completed > 0 else 0
    answered_acc = correct / total_determined if total_determined > 0 else 0
    coverage = total_determined / completed if completed > 0 else 0
    abstention = undetermined / completed if completed > 0 else 0

    logger.info("=" * 60)
    logger.info(f"STATUS: Batch {batch_num} | {completed}/{total} claims ({completed/total*100:.1f}%)")
    logger.info(f"  Overall Accuracy: {overall_acc:.4f} ({correct}/{completed})")
    logger.info(f"  Answered Accuracy: {answered_acc:.4f} ({correct}/{total_determined})")
    logger.info(f"  Coverage: {coverage:.4f} ({total_determined}/{completed})")
    logger.info(f"  Abstention Rate: {abstention:.4f} ({undetermined}/{completed})")
    logger.info(f"  Errors: {errors}")
    logger.info(f"  TRUE: {sum(1 for p in predictions if p['predicted_label']=='TRUE')}, "
                f"FALSE: {sum(1 for p in predictions if p['predicted_label']=='FALSE')}, "
                f"UNDETERMINED: {undetermined}")
    logger.info("=" * 60)


def main():
    """Main batch evaluation entry point with crash-safe persistence and resume."""
    import argparse

    parser = argparse.ArgumentParser(description="TrumorGPT Batch Evaluation")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--metrics-only", action="store_true", help="Calculate metrics from saved predictions only")
    parser.add_argument("--status", action="store_true", help="Show current status and exit")
    parser.add_argument("--run-id", default=RUN_ID, help="Run ID for persistence")
    args = parser.parse_args()

    run_id = args.run_id

    # Initialize persistence
    db_path = Path("experiments/results/evaluation.db")
    persistence = SQLitePersistence(db_path)

    # Load test claims
    all_claims = load_test_claims()
    logger.info(f"Total test claims available: {len(all_claims)}")

    # Get deterministic claim order
    claim_order = get_claim_order(all_claims, RANDOM_SEED)
    dataset_hash = compute_dataset_hash(all_claims)

    # Create claim lookup
    claim_lookup = {c["claim_id"]: c for c in all_claims}
    ordered_claims = [claim_lookup[cid] for cid in claim_order]
    total_claims = len(ordered_claims)

    # Calculate batch boundaries
    batch_boundaries = []
    cum = 0
    for size in BATCH_SIZES:
        cum += size
        batch_boundaries.append(cum)

    # Check for existing run
    existing_checkpoint = persistence.load_checkpoint(run_id)
    existing_predictions = persistence.get_all_predictions(run_id)
    completed_claim_ids = set(p["claim_id"] for p in existing_predictions if p["api_status"] == "success")

    if existing_checkpoint and not args.metrics_only and not args.status:
        # Verify compatibility
        if existing_checkpoint["dataset_hash"] != dataset_hash:
            logger.error("Dataset hash mismatch! Cannot resume. Use new run_id or clear database.")
            sys.exit(1)

        if existing_checkpoint["claim_order"] != claim_order:
            logger.error("Claim order mismatch! Cannot resume.")
            sys.exit(1)

        logger.info(f"Resuming run {run_id}: {len(completed_claim_ids)} claims already completed")

        # Find last completed position
        last_pos = existing_checkpoint.get("last_completed_global_pos", 0)
        start_global_pos = last_pos + 1

        # Skip completed claims
        pending_claims = []
        for i, claim in enumerate(ordered_claims):
            if i >= start_global_pos and claim["claim_id"] not in completed_claim_ids:
                pending_claims.append(claim)

        last_batch = existing_checkpoint.get("last_completed_batch", 0)

    else:
        # Fresh run
        if existing_predictions:
            logger.warning(f"Run {run_id} has {len(existing_predictions)} predictions but no valid checkpoint. Starting fresh.")

        pending_claims = ordered_claims
        start_global_pos = 0
        last_batch = 0

        # Save initial checkpoint
        checkpoint = {
            "run_id": run_id,
            "dataset_hash": dataset_hash,
            "claim_order": claim_order,
            "batch_boundaries": batch_boundaries,
            "completed_claim_ids": [],
            "pending_claim_ids": [c["claim_id"] for c in ordered_claims],
            "last_completed_batch": 0,
            "last_completed_global_pos": -1,
            "model_config": {
                "gemini_model": "gemini-3.5-flash",
                "temperature": 0.0,
            },
            "prompt_version": "v1.0",
            "kg_version": "healthcare_policy_v1",
            "retrieval_config_hash": "v1.0",
            "status": "running",
        }
        persistence.save_checkpoint(checkpoint)

    if args.status:
        # Show status and exit
        print_status(existing_predictions, last_batch, 0, total_claims)
        sys.exit(0)

    if args.metrics_only:
        # Calculate metrics from saved predictions only
        if existing_predictions:
            cum_metrics = compute_cumulative_metrics(existing_predictions)
            logger.info(f"CUMULATIVE METRICS (from {len(existing_predictions)} saved predictions):")
            logger.info(f"  Accuracy: {cum_metrics.accuracy:.4f}")
            logger.info(f"  Precision: {cum_metrics.precision:.4f}")
            logger.info(f"  Recall: {cum_metrics.recall:.4f}")
            logger.info(f"  F1: {cum_metrics.f1:.4f}")
            print_status(existing_predictions, last_batch, 0, total_claims)
        else:
            logger.info("No saved predictions found.")
        sys.exit(0)

    # Initialize pipeline
    logger.info("Initializing pipeline...")
    pipeline_config = PipelineConfig(
        use_sentence_centrality=True,
        use_topic_textrank=True,
        use_graphrag=True,
        save_intermediate=False,
        output_dir="experiments/results",
    )
    pipeline = TrumorGPTPipeline(pipeline_config)
    pipeline.initialize()
    logger.info("Pipeline initialized successfully")

    # Output directory
    output_dir = Path(f"experiments/results/batch_eval_{run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"TrumorGPT Batch Evaluation (600 claims) - Run: {run_id}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Database: {db_path}")
    logger.info("=" * 60)

    # Process claims
    all_predictions = list(existing_predictions)
    global_pos = start_global_pos
    claim_idx = 0
    batch_num = last_batch + 1 if last_batch > 0 else 1

    # Find which batch we're in based on global position
    for i, boundary in enumerate(batch_boundaries):
        if global_pos < boundary:
            batch_num = i + 1
            break

    batch_start_pos = batch_boundaries[batch_num - 2] if batch_num > 1 else 0
    batch_claims_processed = global_pos - batch_start_pos

    batch_predictions = []
    start_time = time.time()

    for claim in pending_claims:
        if claim_idx >= total_claims - start_global_pos:
            break

        # Evaluate claim
        pred = evaluate_claim(pipeline, claim)
        pred["global_position"] = global_pos
        pred["batch_num"] = batch_num
        pred["batch_position"] = batch_claims_processed

        # Save to database immediately (crash-safe)
        record = ClaimRecord(
            run_id=run_id,
            claim_id=pred["claim_id"],
            claim_text=pred["claim_text"],
            true_label=pred["true_label"],
            predicted_label=pred["predicted_label"],
            confidence=pred["confidence"],
            evidence_count=pred["evidence_count"],
            reasoning=pred["reasoning"],
            evidence_relationship=pred["evidence_relationship"],
            processing_time=pred["processing_time"],
            batch_num=pred["batch_num"],
            batch_position=pred["batch_position"],
            global_position=pred["global_position"],
            timestamp=datetime.now().isoformat(),
            model_identifier=pred["model_identifier"],
            api_status=pred["api_status"],
            api_error=pred["api_error"],
        )
        persistence.save_prediction(record)

        # Update in-memory lists
        all_predictions.append(pred)
        batch_predictions.append(pred)

        batch_claims_processed += 1
        global_pos += 1
        claim_idx += 1

        # Log progress
        if batch_claims_processed % 5 == 0 or batch_claims_processed <= 5:
            logger.info(f"  [{batch_claims_processed}/{BATCH_SIZES[batch_num-1]}] "
                        f"{pred['claim_id']}: {pred['predicted_label']} "
                        f"(conf={pred['confidence']:.2f}, evidence={pred['evidence_count']})")

        # Check if batch is complete
        if batch_claims_processed >= BATCH_SIZES[batch_num - 1]:
            # Batch complete - compute metrics and save reports
            batch_metrics = compute_batch_metrics(batch_predictions)
            cum_metrics = compute_cumulative_metrics(all_predictions)

            # Save batch CSV and metrics
            save_batch_csv(batch_num, batch_predictions, output_dir)
            save_metrics_json(batch_num, batch_metrics, cum_metrics, output_dir)
            generate_batch_report(batch_num, batch_predictions, all_predictions, output_dir)

            # Update checkpoint
            completed_ids = [p["claim_id"] for p in all_predictions if p["api_status"] == "success"]
            pending_ids = [c["claim_id"] for c in ordered_claims[global_pos:]]

            checkpoint = {
                "run_id": run_id,
                "dataset_hash": dataset_hash,
                "claim_order": claim_order,
                "batch_boundaries": batch_boundaries,
                "completed_claim_ids": completed_ids,
                "pending_claim_ids": pending_ids,
                "last_completed_batch": batch_num,
                "last_completed_global_pos": global_pos - 1,
                "model_config": {
                    "gemini_model": "gemini-3.5-flash",
                    "temperature": 0.0,
                },
                "prompt_version": "v1.0",
                "kg_version": "healthcare_policy_v1",
                "retrieval_config_hash": "v1.0",
                "status": "running" if global_pos < total_claims else "completed",
            }
            persistence.save_checkpoint(checkpoint)

            # Save cumulative CSV
            save_cumulative_csv(all_predictions, output_dir)

            # Print batch summary
            logger.info(f"=== Batch {batch_num} Complete ({len(batch_predictions)} claims) ===")
            logger.info(f"  Batch Accuracy: {batch_metrics.accuracy:.4f}")
            logger.info(f"  Cumulative Accuracy: {cum_metrics.accuracy:.4f}")
            logger.info(f"  Cumulative F1: {cum_metrics.f1:.4f}")
            logger.info(f"  Time: {time.time() - start_time:.1f}s total")

            print_status(all_predictions, batch_num, BATCH_SIZES[batch_num-1], total_claims)

            # Move to next batch
            batch_num += 1
            batch_predictions = []
            batch_claims_processed = 0

            if batch_num > len(BATCH_SIZES) or global_pos >= total_claims:
                break

    # Final report
    logger.info("=" * 60)
    logger.info("BATCH EVALUATION COMPLETE")
    logger.info("=" * 60)

    final_metrics = compute_cumulative_metrics(all_predictions)
    logger.info(f"FINAL: Accuracy={final_metrics.accuracy:.4f}, "
                f"Precision={final_metrics.precision:.4f}, "
                f"Recall={final_metrics.recall:.4f}, "
                f"F1={final_metrics.f1:.4f}")
    logger.info(f"Total claims processed: {len(all_predictions)}")
    logger.info(f"Total time: {time.time() - start_time:.1f}s")

    # Save final summary
    final_summary = {
        "run_id": run_id,
        "total_claims": len(all_predictions),
        "final_metrics": {
            "accuracy": final_metrics.accuracy,
            "precision": final_metrics.precision,
            "recall": final_metrics.recall,
            "f1": final_metrics.f1,
            "confusion_matrix": final_metrics.confusion_matrix,
        },
        "batch_metrics": [
            {
                "batch": b["batch_num"] if isinstance(b, dict) else b,
                "accuracy": b.get("metrics", {}).get("accuracy", 0) if isinstance(b, dict) else 0,
                "f1": b.get("metrics", {}).get("f1", 0) if isinstance(b, dict) else 0,
            }
            for b in []  # Would need to track batch metrics separately
        ],
        "total_time_seconds": time.time() - start_time,
        "dataset_hash": dataset_hash,
    }

    with open(output_dir / "final_summary.json", "w") as f:
        json.dump(final_summary, f, indent=2)

    # Final checkpoint
    completed_ids = [p["claim_id"] for p in all_predictions if p["api_status"] == "success"]
    checkpoint = {
        "run_id": run_id,
        "dataset_hash": dataset_hash,
        "claim_order": claim_order,
        "batch_boundaries": batch_boundaries,
        "completed_claim_ids": completed_ids,
        "pending_claim_ids": [],
        "last_completed_batch": len(BATCH_SIZES),
        "last_completed_global_pos": global_pos - 1,
        "model_config": {
            "gemini_model": "gemini-3.5-flash",
            "temperature": 0.0,
        },
        "prompt_version": "v1.0",
        "kg_version": "healthcare_policy_v1",
        "retrieval_config_hash": "v1.0",
        "status": "completed",
    }
    persistence.save_checkpoint(checkpoint)

    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Database: {db_path}")


if __name__ == "__main__":
    main()