"""
Fact Verification for TrumorGPT with Qwen2.5 backend.
Paper Section III-E: LLM with GraphRAG evidence for TRUE/FALSE/UNDETERMINED classification.
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from loguru import logger

from trumourgpt.config.settings import get_settings
from trumourgpt.llm import QwenClient, QwenClientConfig
from trumourgpt.llm.schemas import (
    FactVerification,
    ExplanationGeneration,
    VerdictLabel,
    FACT_VERIFICATION_SCHEMA,
    EXPLANATION_GENERATION_SCHEMA,
)
from trumourgpt.retrieval.graphrag import GraphRAGRetriever, RetrievedEvidence


@dataclass
class VerificationConfig:
    """Configuration for fact verification."""
    few_shot_examples: int = 3
    max_evidence_passages: int = 10
    max_evidence_tokens: int = 4000
    check_negation: bool = True
    check_temporal: bool = True
    check_numerical: bool = True
    temperature: float = 0.0


@dataclass
class VerificationResult:
    """Result of fact verification."""
    claim_id: str
    claim_text: str
    predicted_label: VerdictLabel
    confidence: float
    evidence_ids: List[str]
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    reasoning: str
    uncertainty_reason: Optional[str]
    explanation: str
    key_evidence: List[str]
    model_identifier: str
    retrieved_evidence: List[RetrievedEvidence]
    evidence_relationship: Optional[str] = None  # SUPPORTS, CONTRADICTS, NEITHER

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "predicted_label": self.predicted_label.value,
            "confidence": self.confidence,
            "evidence_ids": self.evidence_ids,
            "supporting_evidence": self.supporting_evidence,
            "contradicting_evidence": self.contradicting_evidence,
            "reasoning": self.reasoning,
            "uncertainty_reason": self.uncertainty_reason,
            "explanation": self.explanation,
            "key_evidence": self.key_evidence,
            "model_identifier": self.model_identifier,
            "retrieved_evidence": [e.to_dict() for e in self.retrieved_evidence],
        }


# Few-shot examples for verification
VERIFICATION_EXAMPLES = [
    # CRITICAL EXAMPLE: Positive claim contradicted by negated evidence -> CONTRADICTS -> FALSE
    {
        "claim": "COVID-19 vaccines cause infertility.",
        "evidence": [
            "Source: peer_reviewed_studies - COVID-19 vaccines do not cause infertility.",
            "Source: cdc_statements - CDC: There is no evidence that COVID-19 vaccines cause fertility problems.",
            "Source: asrm_statements - ASRM: COVID-19 vaccines do not affect fertility.",
        ],
        "verdict": "FALSE",
        "evidence_relationship": "CONTRADICTS",
        "reasoning": "The claim states vaccines CAUSE infertility. The evidence from multiple authoritative sources states vaccines do NOT cause infertility. The claim and evidence contradict each other. Therefore evidence_relationship = CONTRADICTS → predicted_label = FALSE.",
    },
    # CRITICAL EXAMPLE: Positive claim contradicted by negated evidence -> CONTRADICTS -> FALSE
    {
        "claim": "Vaccines cause autism.",
        "evidence": [
            "Source: cdc_statements - CDC: Vaccines do not cause autism.",
            "Source: peer_reviewed_studies - Multiple large-scale studies: No link between vaccines and autism.",
            "Source: who_statements - WHO: Vaccines are safe and do not cause autism.",
        ],
        "verdict": "FALSE",
        "evidence_relationship": "CONTRADICTS",
        "reasoning": "The claim states vaccines CAUSE autism. The evidence states vaccines do NOT cause autism. They contradict. Therefore evidence_relationship = CONTRADICTS → predicted_label = FALSE.",
    },
    # CRITICAL EXAMPLE: Positive claim supported by positive evidence -> SUPPORTS -> TRUE
    {
        "claim": "SARS-CoV-2 causes COVID-19.",
        "evidence": [
            "Source: cdc_statements - CDC: SARS-CoV-2 is the virus that causes COVID-19 disease.",
            "Source: who_statements - WHO: SARS-CoV-2 is the virus responsible for COVID-19.",
        ],
        "verdict": "TRUE",
        "evidence_relationship": "SUPPORTS",
        "reasoning": "The claim states SARS-CoV-2 CAUSES COVID-19. The evidence states SARS-CoV-2 CAUSES COVID-19. They agree. Therefore evidence_relationship = SUPPORTS → predicted_label = TRUE.",
    },
    # CRITICAL EXAMPLE: Negated claim supported by negated evidence -> SUPPORTS -> TRUE
    {
        "claim": "mRNA vaccines do not alter human DNA.",
        "evidence": [
            "Source: vaccine_safety - CDC: mRNA vaccines do not alter human DNA.",
            "Source: vaccine_safety - mRNA does not enter the cell nucleus where DNA is located.",
            "Source: vaccine_safety - CDC confirms mRNA vaccines cannot alter DNA.",
        ],
        "verdict": "TRUE",
        "evidence_relationship": "SUPPORTS",
        "reasoning": "The claim states mRNA vaccines do NOT alter human DNA. The evidence states mRNA vaccines do NOT alter human DNA. They agree (both negated). Therefore evidence_relationship = SUPPORTS → predicted_label = TRUE.",
    },
    # CRITICAL EXAMPLE: Negated claim contradicted by positive evidence -> CONTRADICTS -> FALSE
    {
        "claim": "mRNA vaccines do not prevent COVID-19.",
        "evidence": [
            "Source: cdc_statements - CDC: COVID-19 vaccines prevent COVID-19.",
            "Source: peer_reviewed_studies - Clinical trials show high efficacy against symptomatic infection.",
        ],
        "verdict": "FALSE",
        "evidence_relationship": "CONTRADICTS",
        "reasoning": "The claim states vaccines do NOT prevent COVID-19. The evidence states vaccines DO prevent COVID-19. They contradict. Therefore evidence_relationship = CONTRADICTS → predicted_label = FALSE.",
    },
    # Additional clear example: "X causes Y" vs "X does not cause Y"
    {
        "claim": "Flu vaccine causes the flu.",
        "evidence": [
            "Source: cdc_statements - CDC: Flu vaccines do not cause the flu.",
            "Source: peer_reviewed_studies - Inactivated flu vaccines cannot cause influenza illness.",
        ],
        "verdict": "FALSE",
        "evidence_relationship": "CONTRADICTS",
        "reasoning": "The claim states flu vaccine CAUSES the flu. The evidence states flu vaccine does NOT cause the flu. They contradict. Therefore evidence_relationship = CONTRADICTS → predicted_label = FALSE.",
    },
]


def build_verification_prompt(
    claim: str,
    evidence: List[RetrievedEvidence],
    examples: List[Dict],
) -> str:
    """Build few-shot prompt for fact verification."""

    # Format evidence - present each source text as separate evidence with its source
    evidence_text = ""
    ev_idx = 1
    for ev in evidence:
        for src in ev.source_texts[:3]:  # Limit to top 3 sources per evidence
            source_name = "Knowledge Base"
            if "CDC" in src[:50]:
                source_name = "CDC"
            elif "WHO" in src[:50]:
                source_name = "WHO"
            elif "ASRM" in src[:50]:
                source_name = "ASRM"
            elif "FDA" in src[:50]:
                source_name = "FDA"
            elif "study" in src[:50].lower() or "research" in src[:50].lower():
                source_name = "Peer-reviewed Study"

            evidence_text += f"\nEvidence {ev_idx} (Source: {source_name}):\n  {src[:400]}\n"
            ev_idx += 1

    # Use only 1-2 most relevant examples
    example_text = ""
    for i, ex in enumerate(examples[:2]):
        example_text += f"\nExample {i+1}:\nClaim: {ex['claim']}\nEvidence: {json.dumps(ex['evidence'])}\nVerdict: {ex['verdict']}\nReasoning: {ex['reasoning']}\n"

    prompt = f"""You are a fact-checker. Compare the CLAIM to the EVIDENCE and determine if the evidence SUPPORTS, CONTRADICTS, or is NEITHER.

CLAIM: {claim}

EVIDENCE:{evidence_text}

RULES:
- SUPPORTS: Evidence agrees with claim (both say X happens, or both say X does NOT happen)
- CONTRADICTS: Evidence disagrees with claim (claim says X happens, evidence says X does NOT happen, or vice versa)
- NEITHER: Evidence is irrelevant or insufficient

LOGIC - FOLLOW THESE EXACTLY:
- "X causes Y" + "X causes Y" -> SUPPORTS -> TRUE
- "X causes Y" + "X does NOT cause Y" -> CONTRADICTS -> FALSE
- "X does NOT cause Y" + "X does NOT cause Y" -> SUPPORTS -> TRUE
- "X does NOT cause Y" + "X causes Y" -> CONTRADICTS -> FALSE

STEP-BY-STEP REASONING (REQUIRED):
1. Identify the core assertion in the CLAIM (e.g., "vaccines cause infertility" = positive assertion of causation)
2. Identify the core assertion in the EVIDENCE (e.g., "vaccines do not cause infertility" = negative assertion of causation)
3. Compare: Do they agree or disagree?
   - If both assert the SAME thing (both positive or both negative) -> SUPPORTS
   - If one asserts positive and the other negative -> CONTRADICTS
   - If unrelated -> NEITHER
4. Map to label: SUPPORTS -> TRUE, CONTRADICTS -> FALSE, NEITHER -> UNDETERMINED

CONSISTENCY CHECK (MANDATORY):
- If evidence_relationship = SUPPORTS → predicted_label MUST be TRUE
- If evidence_relationship = CONTRADICTS → predicted_label MUST be FALSE
- If evidence_relationship = NEITHER → predicted_label MUST be UNDETERMINED

{example_text}

Now analyze the CLAIM above against the EVIDENCE above.
Think step by step in your reasoning field.

Return JSON with: claim_id, claim_text, evidence_relationship (SUPPORTS/CONTRADICTS/NEITHER), predicted_label (TRUE/FALSE/UNDETERMINED), confidence (0-1), evidence_ids, supporting_evidence, contradicting_evidence, reasoning, uncertainty_reason, model_identifier."""
    return prompt


def validate_verification_consistency(verification: FactVerification) -> tuple[bool, str]:
    """
    Validate that evidence_relationship and predicted_label are logically consistent.

    Returns:
        (is_consistent, error_message)
    """
    rel = verification.evidence_relationship
    label = verification.predicted_label

    if rel == "SUPPORTS" and label != VerdictLabel.TRUE:
        return False, f"Inconsistent: evidence_relationship=SUPPORTS but predicted_label={label.value} (expected TRUE)"
    if rel == "CONTRADICTS" and label != VerdictLabel.FALSE:
        return False, f"Inconsistent: evidence_relationship=CONTRADICTS but predicted_label={label.value} (expected FALSE)"
    if rel == "NEITHER" and label != VerdictLabel.UNDETERMINED:
        return False, f"Inconsistent: evidence_relationship=NEITHER but predicted_label={label.value} (expected UNDETERMINED)"
    if rel not in ("SUPPORTS", "CONTRADICTS", "NEITHER"):
        return False, f"Invalid evidence_relationship: {rel}"

    return True, ""


def fix_verification_consistency(verification: FactVerification) -> FactVerification:
    """
    Fix inconsistent verification by correcting predicted_label based on evidence_relationship.
    The evidence_relationship is the primary judgment; predicted_label follows deterministically.
    """
    rel = verification.evidence_relationship

    if rel == "SUPPORTS":
        corrected_label = VerdictLabel.TRUE
    elif rel == "CONTRADICTS":
        corrected_label = VerdictLabel.FALSE
    elif rel == "NEITHER":
        corrected_label = VerdictLabel.UNDETERMINED
    else:
        # Invalid relationship - default to UNDETERMINED
        corrected_label = VerdictLabel.UNDETERMINED

    if verification.predicted_label != corrected_label:
        logger.warning(
            f"Fixed inconsistent verification: evidence_relationship={rel}, "
            f"predicted_label {verification.predicted_label.value} -> {corrected_label.value}"
        )
        return FactVerification(
            claim_id=verification.claim_id,
            claim_text=verification.claim_text,
            evidence_relationship=verification.evidence_relationship,
            predicted_label=corrected_label,
            confidence=verification.confidence,
            evidence_ids=verification.evidence_ids,
            supporting_evidence=verification.supporting_evidence,
            contradicting_evidence=verification.contradicting_evidence,
            reasoning=verification.reasoning,
            uncertainty_reason=verification.uncertainty_reason,
            model_identifier=verification.model_identifier,
        )
    return verification


def analyze_claim_evidence_negation(
    claim: str,
    evidence: List[RetrievedEvidence],
) -> Optional[str]:
    """
    Post-hoc analysis: Check if claim and evidence have clear negation contradiction.

    Returns:
        "CONTRADICTS" if claim asserts X and evidence asserts NOT X (or vice versa)
        "SUPPORTS" if both assert same (both positive or both negative)
        None if unclear
    """
    claim_lower = claim.lower()

    # Check if claim contains negation
    claim_negated = any(neg in claim_lower for neg in ["do not", "does not", "doesn't", "don't", "not ", "no ", "never", "cannot", "can't"])

    # Analyze matching triples from evidence
    for ev in evidence:
        for triple in ev.matching_triples:
            relation_lower = triple.relation.lower()

            # Check if evidence relation contains negation
            evidence_negated = any(neg in relation_lower for neg in ["do not", "does not", "doesn't", "don't", "not ", "no ", "never", "cannot", "can't"])

            # If claim and evidence have opposite negation on same entity pair -> CONTRADICTS
            if claim_negated != evidence_negated:
                logger.info(f"Negation mismatch detected: claim_negated={claim_negated}, evidence_negated={evidence_negated}, relation={triple.relation}")
                return "CONTRADICTS"

            # If both have same negation -> SUPPORTS
            if claim_negated and evidence_negated:
                return "SUPPORTS"
            if not claim_negated and not evidence_negated:
                return "SUPPORTS"

    return None


def apply_post_hoc_correction(
    verification: FactVerification,
    claim: str,
    evidence: List[RetrievedEvidence],
) -> FactVerification:
    """
    Apply post-hoc correction based on actual claim/evidence triple analysis.
    This overrides LLM judgment when there's a clear structural contradiction.
    """
    derived_relationship = analyze_claim_evidence_negation(claim, evidence)

    if derived_relationship is None:
        return verification

    # If LLM relationship contradicts structural analysis, fix it
    if verification.evidence_relationship != derived_relationship:
        logger.warning(
            f"Post-hoc correction: LLM said {verification.evidence_relationship}, "
            f"structural analysis says {derived_relationship}. Overriding."
        )

        # Determine correct label from derived relationship
        if derived_relationship == "SUPPORTS":
            corrected_label = VerdictLabel.TRUE
        elif derived_relationship == "CONTRADICTS":
            corrected_label = VerdictLabel.FALSE
        else:
            corrected_label = VerdictLabel.UNDETERMINED

        return FactVerification(
            claim_id=verification.claim_id,
            claim_text=verification.claim_text,
            evidence_relationship=derived_relationship,
            predicted_label=corrected_label,
            confidence=verification.confidence,
            evidence_ids=verification.evidence_ids,
            supporting_evidence=verification.supporting_evidence,
            contradicting_evidence=verification.contradicting_evidence,
            reasoning=f"[Corrected] {verification.reasoning}",
            uncertainty_reason=verification.uncertainty_reason,
            model_identifier=verification.model_identifier,
        )

    return verification


def build_explanation_prompt(
    claim: str,
    verdict: VerdictLabel,
    evidence: List[RetrievedEvidence],
    reasoning: str,
) -> str:
    """Build prompt for explanation generation."""

    evidence_summary = ""
    for ev in evidence[:3]:
        for src in ev.source_texts[:2]:
            evidence_summary += f"- {src[:300]}\n"

    prompt = f"""Generate a concise, evidence-grounded explanation for the fact-check verdict.

Claim: {claim}
Verdict: {verdict.value}
Reasoning: {reasoning}
Key Evidence:
{evidence_summary}

Return JSON with: claim_id, verdict, explanation, key_evidence, model_identifier.

Requirements:
- 2-3 sentences maximum
- Cite specific evidence
- No fabricated citations
- Clear and accessible language"""
    return prompt


class FactCheckerQwen:
    """
    Fact verification using Qwen2.5 with GraphRAG evidence.
    Paper: "TrumorGPT determines the answer based on the similarity scores... integrates structured representation with generative capabilities"
    """

    def __init__(
        self,
        retriever: GraphRAGRetriever,
        config: Optional[VerificationConfig] = None,
        qwen_config: Optional[QwenClientConfig] = None,
    ):
        self.retriever = retriever
        self.config = config or VerificationConfig()
        self._client: Optional[QwenClient] = None
        self._qwen_config = qwen_config or QwenClientConfig()

    def _get_client(self) -> QwenClient:
        if self._client is None:
            self._client = QwenClient(self._qwen_config)
        return self._client

    def verify_claim(
        self,
        claim: str,
        claim_id: str = "unknown",
        evidence: Optional[List[RetrievedEvidence]] = None,
        query_text: Optional[str] = None,
    ) -> VerificationResult:
        """
        Verify a health claim using GraphRAG retrieval and LLM reasoning.
        """
        # Retrieve evidence if not provided
        if evidence is None:
            if query_text is not None:
                evidence = self.retriever.retrieve(claim, top_k=self.config.max_evidence_passages, query_text=query_text)
            else:
                evidence = self.retriever.retrieve(claim, top_k=self.config.max_evidence_passages)

        if not evidence:
            logger.warning(f"No evidence retrieved for claim: {claim_id}")
            return self._create_undetermined_result(claim, claim_id, "No evidence retrieved")

        # Run verification
        verification = self._run_verification(claim, claim_id, evidence)

        # Generate explanation
        explanation = self._generate_explanation(claim, claim_id, verification, evidence)

        return VerificationResult(
            claim_id=claim_id,
            claim_text=claim,
            predicted_label=verification.predicted_label,
            confidence=verification.confidence,
            evidence_ids=verification.evidence_ids,
            supporting_evidence=verification.supporting_evidence,
            contradicting_evidence=verification.contradicting_evidence,
            reasoning=verification.reasoning,
            uncertainty_reason=verification.uncertainty_reason,
            explanation=explanation.explanation,
            key_evidence=explanation.key_evidence,
            model_identifier=verification.model_identifier,
            retrieved_evidence=evidence,
            evidence_relationship=verification.evidence_relationship,
        )

    def _run_verification(
        self,
        claim: str,
        claim_id: str,
        evidence: List[RetrievedEvidence],
    ) -> FactVerification:
        """Run LLM verification with consistency validation and retry."""
        client = self._get_client()

        prompt = build_verification_prompt(
            claim, evidence, VERIFICATION_EXAMPLES[:self.config.few_shot_examples]
        )

        max_retries = 2
        for attempt in range(max_retries + 1):
            response = client.generate_structured("fact_verification", prompt)

            if not response.success or response.data is None:
                logger.error(f"Verification failed (attempt {attempt + 1}/{max_retries + 1}): {response.error}")
                if attempt == max_retries:
                    return self._create_undetermined_fact_verification(claim, claim_id, "LLM verification failed")
                continue

            try:
                verification = FactVerification(**response.data)
            except Exception as e:
                logger.error(f"Verification parsing failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                if attempt == max_retries:
                    return self._create_undetermined_fact_verification(claim, claim_id, "LLM verification parsing failed")
                continue

            # Validate consistency between evidence_relationship and predicted_label
            is_consistent, error_msg = validate_verification_consistency(verification)
            if is_consistent:
                # Apply post-hoc correction based on structural analysis of claim/evidence triples
                verification = apply_post_hoc_correction(verification, claim, evidence)
                return verification

            logger.warning(f"Inconsistent verification (attempt {attempt + 1}/{max_retries + 1}): {error_msg}")

            if attempt < max_retries:
                # Retry with explicit consistency reminder
                prompt += f"\n\nPREVIOUS ATTEMPT WAS INCONSISTENT: {error_msg}\nRemember: SUPPORTS->TRUE, CONTRADICTS->FALSE, NEITHER->UNDETERMINED"
                continue
            else:
                # Final attempt: fix deterministically based on evidence_relationship
                logger.warning(f"Fixing inconsistency deterministically: {error_msg}")
                return fix_verification_consistency(verification)

        return self._create_undetermined_fact_verification(claim, claim_id, "LLM verification failed after retries")

    def _generate_explanation(
        self,
        claim: str,
        claim_id: str,
        verification: FactVerification,
        evidence: List[RetrievedEvidence],
    ) -> ExplanationGeneration:
        """Generate explanation for verdict."""
        client = self._get_client()

        prompt = build_explanation_prompt(
            claim, verification.predicted_label, evidence, verification.reasoning
        )

        response = client.generate_structured("explanation_generation", prompt)

        if not response.success or response.data is None:
            logger.warning(f"Explanation generation failed: {response.error}")
            return ExplanationGeneration(
                claim_id=claim_id,
                verdict=verification.predicted_label,
                explanation=verification.reasoning,
                key_evidence=verification.supporting_evidence[:3],
                model_identifier=verification.model_identifier,
            )

        return ExplanationGeneration(**response.data)

    def _create_undetermined_result(
        self,
        claim: str,
        claim_id: str,
        reason: str,
    ) -> VerificationResult:
        """Create UNDETERMINED result."""
        model_name = self._qwen_config.model_name
        return VerificationResult(
            claim_id=claim_id,
            claim_text=claim,
            predicted_label=VerdictLabel.UNDETERMINED,
            confidence=0.0,
            evidence_ids=[],
            supporting_evidence=[],
            contradicting_evidence=[],
            reasoning=f"Unable to determine: {reason}",
            uncertainty_reason=reason,
            explanation=f"Insufficient evidence to verify this claim. {reason}",
            key_evidence=[],
            model_identifier=model_name,
            retrieved_evidence=[],
            evidence_relationship="NEITHER",
        )

    def _create_undetermined_fact_verification(
        self,
        claim: str,
        claim_id: str,
        reason: str,
    ) -> FactVerification:
        model_name = self._qwen_config.model_name
        return FactVerification(
            claim_id=claim_id,
            claim_text=claim,
            predicted_label=VerdictLabel.UNDETERMINED,
            confidence=0.0,
            evidence_ids=[],
            supporting_evidence=[],
            contradicting_evidence=[],
            reasoning=f"Unable to determine: {reason}",
            uncertainty_reason=reason,
            model_identifier=model_name,
        )

    def verify_batch(
        self,
        claims: List[str],
        claim_ids: Optional[List[str]] = None,
    ) -> List[VerificationResult]:
        """Verify multiple claims."""
        if claim_ids is None:
            claim_ids = [f"claim_{i}" for i in range(len(claims))]

        return [self.verify_claim(claim, cid) for claim, cid in zip(claims, claim_ids)]


# Convenience function
def verify_claim_simple(
    claim: str,
    retriever: GraphRAGRetriever,
    config: Optional[VerificationConfig] = None,
    qwen_config: Optional[QwenClientConfig] = None,
) -> VerificationResult:
    """Quick verification function."""
    checker = FactCheckerQwen(retriever, config, qwen_config)
    return checker.verify_claim(claim)