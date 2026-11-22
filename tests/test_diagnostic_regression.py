"""
Regression tests for the three diagnostic claims.
Ensures the fact checker correctly handles:
1. Positive claim with matching evidence (SUPPORTS -> TRUE)
2. Negative claim with contradicting evidence (CONTRADICTS -> FALSE)
3. Negated claim with matching negated evidence (SUPPORTS -> TRUE)
4. Consistency between evidence_relationship and predicted_label
"""

import pytest
from trumourgpt.experiments.pipeline import TrumorGPTPipeline
from trumourgpt.llm.schemas import VerdictLabel
from trumourgpt.verification.fact_checker import validate_verification_consistency, analyze_claim_evidence_negation
from trumourgpt.retrieval.graphrag import RetrievedEvidence
from trumourgpt.knowledge_graph.graph import Triple


@pytest.fixture(scope="module")
def pipeline():
    """Initialize pipeline once for all tests."""
    p = TrumorGPTPipeline()
    p.initialize()
    return p


class TestDiagnosticClaims:
    """Test the three diagnostic claims from the paper replication."""

    def test_sars_cov2_causes_covid19(self, pipeline):
        """SARS-CoV-2 causes COVID-19 - should be TRUE with SUPPORTS."""
        claim = "SARS-CoV-2 causes COVID-19."
        result = pipeline.process_claim(claim, "diagnostic_1")
        vr = result.verification_result

        assert vr.predicted_label == VerdictLabel.TRUE, f"Expected TRUE, got {vr.predicted_label.value}"
        assert vr.evidence_relationship == "SUPPORTS", f"Expected SUPPORTS, got {vr.evidence_relationship}"
        assert vr.confidence > 0.5, f"Low confidence: {vr.confidence}"
        assert len(vr.retrieved_evidence) > 0, "No evidence retrieved"

        # Verify the matching triple is correct (check all retrieved evidence)
        found = False
        for ev in vr.retrieved_evidence:
            matching = ev.matching_triples
            for t in matching:
                if t.head == "SARS-CoV-2" and t.relation == "causes" and "COVID-19" in t.tail:
                    found = True
                    break
            if found:
                break
        assert found, f"Expected SARS-CoV-2 causes COVID-19 triple in matching evidence"

    def test_vaccines_cause_infertility(self, pipeline):
        """COVID-19 vaccines cause infertility - should be FALSE with CONTRADICTS."""
        claim = "COVID-19 vaccines cause infertility."
        result = pipeline.process_claim(claim, "diagnostic_2")
        vr = result.verification_result

        assert vr.predicted_label == VerdictLabel.FALSE, f"Expected FALSE, got {vr.predicted_label.value}"
        assert vr.evidence_relationship == "CONTRADICTS", f"Expected CONTRADICTS, got {vr.evidence_relationship}"
        assert vr.confidence > 0.5, f"Low confidence: {vr.confidence}"
        assert len(vr.retrieved_evidence) > 0, "No evidence retrieved"

        # Verify we get contradicting evidence (entity-pair match with "does not cause")
        found = False
        for ev in vr.retrieved_evidence:
            matching = ev.matching_triples
            for t in matching:
                if "does not cause" in t.relation.lower() and "infertility" in t.tail.lower():
                    found = True
                    break
            if found:
                break
        assert found, f"Expected contradicting evidence with 'does not cause infertility'"

    def test_mrna_vaccines_do_not_alter_dna(self, pipeline):
        """mRNA vaccines do not alter human DNA - should be TRUE with SUPPORTS."""
        claim = "mRNA vaccines do not alter human DNA."
        result = pipeline.process_claim(claim, "diagnostic_3")
        vr = result.verification_result

        assert vr.predicted_label == VerdictLabel.TRUE, f"Expected TRUE, got {vr.predicted_label.value}"
        assert vr.evidence_relationship == "SUPPORTS", f"Expected SUPPORTS, got {vr.evidence_relationship}"
        assert vr.confidence > 0.5, f"Low confidence: {vr.confidence}"
        assert len(vr.retrieved_evidence) > 0, "No evidence retrieved"

        # Verify the matching triple preserves negation (check all retrieved evidence)
        found = False
        for ev in vr.retrieved_evidence:
            matching = ev.matching_triples
            for t in matching:
                if "do not alter" in t.relation.lower() and "dna" in t.tail.lower():
                    found = True
                    break
            if found:
                break
        assert found, f"Expected negated relation 'do not alter DNA'"


class TestConsistencyValidation:
    """Test the consistency validation and post-hoc correction logic."""

    def test_validate_consistency_supports_true(self):
        """SUPPORTS + TRUE should be consistent."""
        from trumourgpt.llm.schemas import FactVerification
        v = FactVerification(
            claim_id="test", claim_text="test",
            evidence_relationship="SUPPORTS", predicted_label=VerdictLabel.TRUE,
            confidence=0.9, reasoning="test", model_identifier="test"
        )
        is_consistent, _ = validate_verification_consistency(v)
        assert is_consistent

    def test_validate_consistency_contradicts_false(self):
        """CONTRADICTS + FALSE should be consistent."""
        from trumourgpt.llm.schemas import FactVerification
        v = FactVerification(
            claim_id="test", claim_text="test",
            evidence_relationship="CONTRADICTS", predicted_label=VerdictLabel.FALSE,
            confidence=0.9, reasoning="test", model_identifier="test"
        )
        is_consistent, _ = validate_verification_consistency(v)
        assert is_consistent

    def test_validate_consistency_neither_undetermined(self):
        """NEITHER + UNDETERMINED should be consistent."""
        from trumourgpt.llm.schemas import FactVerification
        v = FactVerification(
            claim_id="test", claim_text="test",
            evidence_relationship="NEITHER", predicted_label=VerdictLabel.UNDETERMINED,
            confidence=0.9, reasoning="test", model_identifier="test"
        )
        is_consistent, _ = validate_verification_consistency(v)
        assert is_consistent

    def test_validate_inconsistency_supports_false(self):
        """SUPPORTS + FALSE should be inconsistent."""
        from trumourgpt.llm.schemas import FactVerification
        v = FactVerification(
            claim_id="test", claim_text="test",
            evidence_relationship="SUPPORTS", predicted_label=VerdictLabel.FALSE,
            confidence=0.9, reasoning="test", model_identifier="test"
        )
        is_consistent, error = validate_verification_consistency(v)
        assert not is_consistent
        assert "SUPPORTS" in error and "FALSE" in error

    def test_validate_inconsistency_contradicts_true(self):
        """CONTRADICTS + TRUE should be inconsistent."""
        from trumourgpt.llm.schemas import FactVerification
        v = FactVerification(
            claim_id="test", claim_text="test",
            evidence_relationship="CONTRADICTS", predicted_label=VerdictLabel.TRUE,
            confidence=0.9, reasoning="test", model_identifier="test"
        )
        is_consistent, error = validate_verification_consistency(v)
        assert not is_consistent
        assert "CONTRADICTS" in error and "TRUE" in error

    def test_analyze_negation_contradiction(self):
        """Positive claim vs negated evidence should return CONTRADICTS."""
        claim = "COVID-19 vaccines cause infertility."
        evidence = [RetrievedEvidence(
            graph_id="test",
            similarity_score=0.5,
            matching_triples=[Triple(head="COVID-19 vaccine", relation="does not cause", tail="infertility",
                                   source_text="test", source_id="test", confidence=0.9,
                                   head_type="entity", tail_type="entity")],
            all_triples=[],
            source_texts=["test"],
            metadata={}
        )]
        result = analyze_claim_evidence_negation(claim, evidence)
        assert result == "CONTRADICTS"

    def test_analyze_negation_support_both_positive(self):
        """Positive claim vs positive evidence should return SUPPORTS."""
        claim = "SARS-CoV-2 causes COVID-19."
        evidence = [RetrievedEvidence(
            graph_id="test",
            similarity_score=0.5,
            matching_triples=[Triple(head="SARS-CoV-2", relation="causes", tail="COVID-19",
                                   source_text="test", source_id="test", confidence=0.9,
                                   head_type="entity", tail_type="entity")],
            all_triples=[],
            source_texts=["test"],
            metadata={}
        )]
        result = analyze_claim_evidence_negation(claim, evidence)
        assert result == "SUPPORTS"

    def test_analyze_negation_support_both_negated(self):
        """Negated claim vs negated evidence should return SUPPORTS."""
        claim = "mRNA vaccines do not alter human DNA."
        evidence = [RetrievedEvidence(
            graph_id="test",
            similarity_score=0.5,
            matching_triples=[Triple(head="mRNA vaccines", relation="do not alter", tail="human DNA",
                                   source_text="test", source_id="test", confidence=0.9,
                                   head_type="entity", tail_type="entity")],
            all_triples=[],
            source_texts=["test"],
            metadata={}
        )]
        result = analyze_claim_evidence_negation(claim, evidence)
        assert result == "SUPPORTS"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])