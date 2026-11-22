#!/usr/bin/env python
"""Run fast dev evaluation using pattern-only extraction (no LLM fallback)."""
import json
import time
from trumourgpt.experiments.pipeline import TrumorGPTPipeline, PipelineConfig
from trumourgpt.evaluation.dataset import PolitiFactDataset, DatasetConfig
from trumourgpt.evaluation.metrics import evaluate_binary, print_evaluation_report
from trumourgpt.retrieval.graphrag import RetrievalConfig


def main():
    print("=" * 80)
    print("FAST DEV SET EVALUATION (100 claims, pattern-only)")
    print("=" * 80)

    # Load dataset
    print("\nLoading PolitiFact dataset...")
    dataset = PolitiFactDataset(DatasetConfig())
    dataset.process()
    dev_claims = dataset.get_split("dev")  # 100 claims

    print(f"Running evaluation on {len(dev_claims)} claims")

    # Initialize pipeline with custom config
    config = PipelineConfig()
    pipeline = TrumorGPTPipeline(config)

    # Override retrieval config to disable LLM fallback
    # We'll monkey-patch the retriever after initialization
    pipeline.initialize()

    # Disable LLM fallback for faster evaluation
    orig_build = pipeline._components["retriever"].build_query_graph
    pipeline._components["retriever"].build_query_graph = lambda query, query_id: orig_build(query, query_id, use_llm_fallback=False)

    # Process claims
    print("\nProcessing claims (pattern-only, no LLM)...")
    results = []
    ground_truth = {}

    for i, claim_data in enumerate(dev_claims):
        claim_text = claim_data.text
        claim_id = str(claim_data.claim_id)
        true_label = claim_data.binary_label.value

        ground_truth[claim_id] = true_label

        if i % 10 == 0:
            print(f"  [{i+1}/100] {claim_id}: {claim_text[:60]}...")

        start = time.time()
        result = pipeline.process_claim(claim_text, claim_id)
        elapsed = time.time() - start

        pred_label = result.verification_result.predicted_label.value
        confidence = result.verification_result.confidence
        evidence_count = len(result.verification_result.retrieved_evidence)

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
    print_evaluation_report(metrics, ["FALSE", "TRUE"])

    # Per-claim breakdown
    print("\nPer-claim results:")
    correct = 0
    undetermined = 0
    for result in results:
        cid = result.claim_id
        true_l = ground_truth.get(cid, "?")
        pred_l = result.verification_result.predicted_label.value
        conf = result.verification_result.confidence
        evidence = len(result.verification_result.retrieved_evidence)
        match = "✓" if true_l == pred_l else "✗"
        if true_l == pred_l:
            correct += 1
        if pred_l == "UNDETERMINED":
            undetermined += 1
        print(f"  {match} {cid}: true={true_l}, pred={pred_l}, conf={conf:.2f}, evidence={evidence}")

    print(f"\nCorrect predictions: {correct}/{len(results)} = {correct/len(results):.2%}")
    print(f"UNDETERMINED: {undetermined}/{len(results)} = {undetermined/len(results):.2%}")

    # Save results
    output_path = f"experiments/results/fast_dev_eval_{int(time.time())}.json"
    pipeline.save_results(results, output_path)
    print(f"\nResults saved to {output_path}")

    return metrics


if __name__ == "__main__":
    main()