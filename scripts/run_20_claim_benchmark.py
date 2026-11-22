#!/usr/bin/env python
"""Run 20-claim dev benchmark to measure retrieval improvement."""
import json
import time
from trumourgpt.experiments.pipeline import TrumorGPTPipeline, PipelineConfig
from trumourgpt.evaluation.dataset import PolitiFactDataset, DatasetConfig
from trumourgpt.evaluation.metrics import evaluate_binary


def main():
    print("=" * 80)
    print("20-CLAIM DEV BENCHMARK")
    print("=" * 80)

    # Load dataset
    print("\nLoading PolitiFact dataset...")
    dataset = PolitiFactDataset(DatasetConfig())
    dataset.process()
    dev_claims = dataset.get_split("dev")  # 100 claims

    # Take first 20 claims
    benchmark_claims = dev_claims[:20]
    print(f"Running benchmark on {len(benchmark_claims)} claims")

    # Initialize pipeline
    config = PipelineConfig()
    pipeline = TrumorGPTPipeline(config)
    pipeline.initialize()

    # Process claims
    print("\nProcessing claims...")
    results = []
    ground_truth = {}

    for i, claim_data in enumerate(benchmark_claims):
        claim_text = claim_data.text
        claim_id = str(claim_data.claim_id)
        true_label = claim_data.binary_label.value

        ground_truth[claim_id] = true_label

        print(f"\n[{i+1}/20] Claim {claim_id}: {claim_text[:80]}...")
        print(f"    True label: {true_label}")

        start = time.time()
        result = pipeline.process_claim(claim_text, claim_id)
        elapsed = time.time() - start

        pred_label = result.verification_result.predicted_label.value
        confidence = result.verification_result.confidence
        evidence_count = len(result.verification_result.retrieved_evidence)

        print(f"    Predicted: {pred_label} (conf={confidence:.2f})")
        print(f"    Evidence retrieved: {evidence_count}")
        print(f"    Time: {elapsed:.1f}s")

        results.append(result)

    # Evaluate
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    y_true = []
    y_pred = []
    for result in results:
        cid = result.claim_id
        if cid in ground_truth:
            y_true.append(ground_truth[cid])
            y_pred.append(result.verification_result.predicted_label.value)

    metrics = evaluate_binary(y_true, y_pred, "TRUE", "FALSE")

    print(f"\nAccuracy: {metrics.accuracy:.2%}")
    print(f"Precision: {metrics.precision:.2%}")
    print(f"Recall: {metrics.recall:.2%}")
    print(f"F1: {metrics.f1:.2%}")

    print(f"\nConfusion Matrix:")
    print(f"  TRUE->TRUE: {metrics.true_positives}, FALSE->FALSE: {metrics.true_negatives}")
    print(f"  TRUE->FALSE: {metrics.false_negatives}, FALSE->TRUE: {metrics.false_positives}")
    print(f"  Abstentions (UNDETERMINED): {metrics.num_abstentions}")

    # Per-claim breakdown
    print("\nPer-claim results:")
    correct = 0
    for result in results:
        cid = result.claim_id
        true_l = ground_truth.get(cid, "?")
        pred_l = result.verification_result.predicted_label.value
        conf = result.verification_result.confidence
        match = "✓" if true_l == pred_l else "✗"
        if true_l == pred_l:
            correct += 1
        print(f"  {match} {cid}: true={true_l}, pred={pred_l}, conf={conf:.2f}")

    print(f"\nCorrect: {correct}/{len(results)} = {correct/len(results):.2%}")

    # Save results
    output_path = f"experiments/results/benchmark_20_claims_{int(time.time())}.json"
    pipeline.save_results(results, output_path)
    print(f"\nResults saved to {output_path}")

    return metrics


if __name__ == "__main__":
    main()