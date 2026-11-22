"""
Pydantic schemas for structured LLM output validation.
All LLM tasks must return validated JSON matching these schemas.
"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class VerdictLabel(str, Enum):
    """Fact verification verdict labels (Paper: TRUE, FALSE, UNDETERMINED)."""
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNDETERMINED = "UNDETERMINED"


class AtomicClaim(BaseModel):
    """Single atomic claim from decomposition."""
    text: str = Field(..., description="The atomic claim text")
    entities: List[str] = Field(default_factory=list, description="Entities mentioned")
    relations: List[str] = Field(default_factory=list, description="Relations/predicates")
    negation: bool = Field(default=False, description="Whether claim contains negation")
    temporal_info: Optional[str] = Field(default=None, description="Temporal constraints")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Decomposition confidence")


class ClaimDecomposition(BaseModel):
    """Task A: Claim decomposition output."""
    claim_id: str = Field(..., description="Unique claim identifier")
    original_claim: str = Field(..., description="Original claim text")
    atomic_claims: List[AtomicClaim] = Field(..., min_length=1, description="Decomposed atomic claims")
    key_entities: List[str] = Field(default_factory=list, description="All key entities")
    key_relations: List[str] = Field(default_factory=list, description="All key relations")
    temporal_constraints: List[str] = Field(default_factory=list, description="Temporal constraints")
    negation_present: bool = Field(default=False, description="Any negation in claim")


class Triple(BaseModel):
    """Subject-Predicate-Object triple with provenance."""
    subject: str = Field(..., description="Head entity")
    predicate: str = Field(..., description="Relation/predicate")
    object: str = Field(..., description="Tail entity")
    source_text: str = Field(..., description="Source text span")
    source_id: str = Field(..., description="Source document identifier")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence")


class TripleExtraction(BaseModel):
    """Task B: Knowledge graph triple extraction output."""
    claim_id: str = Field(..., description="Claim identifier")
    source_text: str = Field(..., description="Source text for extraction")
    source_id: str = Field(..., description="Source document ID")
    triples: List[Triple] = Field(default_factory=list, description="Extracted triples")
    entities_found: List[str] = Field(default_factory=list, description="Unique entities")
    relations_found: List[str] = Field(default_factory=list, description="Unique relations")


class EvidenceItem(BaseModel):
    """Single evidence item for analysis."""
    evidence_id: str = Field(..., description="Evidence identifier")
    text: str = Field(..., description="Evidence passage text")
    source: str = Field(..., description="Source identifier")
    source_url: Optional[str] = Field(default=None, description="Source URL")
    triple_matches: List[Triple] = Field(default_factory=list, description="Matching triples")


class EvidenceAnalysis(BaseModel):
    """Task C: Evidence analysis output."""
    claim_id: str = Field(..., description="Claim identifier")
    claim_text: str = Field(..., description="Claim text")
    supporting_evidence: List[EvidenceItem] = Field(default_factory=list, description="Supporting evidence")
    contradicting_evidence: List[EvidenceItem] = Field(default_factory=list, description="Contradicting evidence")
    irrelevant_evidence: List[EvidenceItem] = Field(default_factory=list, description="Irrelevant evidence")
    insufficient_evidence: bool = Field(default=False, description="Whether evidence is insufficient overall")
    analysis_summary: str = Field(..., description="Summary of evidence analysis")


class FactVerification(BaseModel):
    """Task D: Fact verification output."""
    claim_id: str = Field(..., description="Claim identifier")
    claim_text: str = Field(..., description="Claim text")
    # Intermediate step: explicit relationship between claim and evidence
    evidence_relationship: str = Field(
        ...,
        description="Relationship between claim and evidence: SUPPORTS, CONTRADICTS, or NEITHER"
    )
    predicted_label: VerdictLabel = Field(..., description="TRUE, FALSE, or UNDETERMINED")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    evidence_ids: List[str] = Field(default_factory=list, description="Supporting evidence IDs")
    supporting_evidence: List[str] = Field(default_factory=list, description="Supporting evidence texts")
    contradicting_evidence: List[str] = Field(default_factory=list, description="Contradicting evidence texts")
    reasoning: str = Field(..., description="Step-by-step verification reasoning")
    uncertainty_reason: Optional[str] = Field(default=None, description="Reason if UNDETERMINED")
    model_identifier: str = Field(..., description="LLM model used")


class ExplanationGeneration(BaseModel):
    """Task E: Explanation generation output."""
    claim_id: str = Field(..., description="Claim identifier")
    verdict: VerdictLabel = Field(..., description="Final verdict")
    explanation: str = Field(..., description="Concise evidence-grounded explanation")
    key_evidence: List[str] = Field(default_factory=list, description="Key evidence citations")
    model_identifier: str = Field(..., description="LLM model used")


class EntityExtractionItem(BaseModel):
    """Single extracted entity."""
    text: str = Field(..., description="Entity text")
    type: str = Field(..., description="Entity type")
    start: int = Field(..., description="Start character position")
    end: int = Field(..., description="End character position")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Extraction confidence")


class EntityExtraction(BaseModel):
    """Entity extraction output."""
    claim_id: str = Field(..., description="Claim identifier")
    original_text: str = Field(..., description="Original text")
    entities: List[EntityExtractionItem] = Field(default_factory=list, description="Extracted entities")


class LLMResponse(BaseModel):
    """Generic LLM response wrapper."""
    task: str = Field(..., description="Task identifier")
    model: str = Field(..., description="Model used")
    success: bool = Field(..., description="Whether parsing succeeded")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Parsed structured data")
    raw_response: Optional[str] = Field(default=None, description="Raw LLM response")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    latency_ms: int = Field(default=0, description="Response latency in ms")
    token_usage: Optional[Dict[str, int]] = Field(default=None, description="Token usage if available")


# JSON Schema definitions for Ollama format="json" parameter
CLAIM_DECOMPOSITION_SCHEMA = {
    "type": "object",
    "properties": {
        "claim_id": {"type": "string"},
        "original_claim": {"type": "string"},
        "atomic_claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "entities": {"type": "array", "items": {"type": "string"}},
                    "relations": {"type": "array", "items": {"type": "string"}},
                    "negation": {"type": "boolean"},
                    "temporal_info": {"type": ["string", "null"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["text"]
            }
        },
        "key_entities": {"type": "array", "items": {"type": "string"}},
        "key_relations": {"type": "array", "items": {"type": "string"}},
        "temporal_constraints": {"type": "array", "items": {"type": "string"}},
        "negation_present": {"type": "boolean"}
    },
    "required": ["claim_id", "original_claim", "atomic_claims"],
    "additionalProperties": False
}

TRIPLE_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "claim_id": {"type": "string"},
        "source_text": {"type": "string"},
        "source_id": {"type": "string"},
        "triples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "source_text": {"type": "string"},
                    "source_id": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["subject", "predicate", "object", "source_text", "source_id"]
            }
        },
        "entities_found": {"type": "array", "items": {"type": "string"}},
        "relations_found": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["claim_id", "source_text", "source_id"],
    "additionalProperties": False
}

EVIDENCE_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "claim_id": {"type": "string"},
        "claim_text": {"type": "string"},
        "supporting_evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "evidence_id": {"type": "string"},
                    "text": {"type": "string"},
                    "source": {"type": "string"},
                    "source_url": {"type": ["string", "null"]},
                    "triple_matches": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "subject": {"type": "string"},
                                "predicate": {"type": "string"},
                                "object": {"type": "string"},
                                "source_text": {"type": "string"},
                                "source_id": {"type": "string"},
                                "confidence": {"type": "number"}
                            },
                            "required": ["subject", "predicate", "object", "source_text", "source_id"]
                        }
                    }
                },
                "required": ["evidence_id", "text", "source"]
            }
        },
        "contradicting_evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "evidence_id": {"type": "string"},
                    "text": {"type": "string"},
                    "source": {"type": "string"},
                    "source_url": {"type": ["string", "null"]},
                    "triple_matches": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "subject": {"type": "string"},
                                "predicate": {"type": "string"},
                                "object": {"type": "string"},
                                "source_text": {"type": "string"},
                                "source_id": {"type": "string"},
                                "confidence": {"type": "number"}
                            },
                            "required": ["subject", "predicate", "object", "source_text", "source_id"]
                        }
                    }
                },
                "required": ["evidence_id", "text", "source"]
            }
        },
        "irrelevant_evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "evidence_id": {"type": "string"},
                    "text": {"type": "string"},
                    "source": {"type": "string"},
                    "source_url": {"type": ["string", "null"]},
                    "triple_matches": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "subject": {"type": "string"},
                                "predicate": {"type": "string"},
                                "object": {"type": "string"},
                                "source_text": {"type": "string"},
                                "source_id": {"type": "string"},
                                "confidence": {"type": "number"}
                            },
                            "required": ["subject", "predicate", "object", "source_text", "source_id"]
                        }
                    }
                },
                "required": ["evidence_id", "text", "source"]
            }
        },
        "insufficient_evidence": {"type": "boolean"},
        "analysis_summary": {"type": "string"}
    },
    "required": ["claim_id", "claim_text"],
    "additionalProperties": False
}

FACT_VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "claim_id": {"type": "string"},
        "claim_text": {"type": "string"},
        "evidence_relationship": {
            "type": "string",
            "enum": ["SUPPORTS", "CONTRADICTS", "NEITHER"],
            "description": "Relationship between claim and evidence: SUPPORTS, CONTRADICTS, or NEITHER"
        },
        "predicted_label": {"type": "string", "enum": ["TRUE", "FALSE", "UNDETERMINED"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
        "supporting_evidence": {"type": "array", "items": {"type": "string"}},
        "contradicting_evidence": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
        "uncertainty_reason": {"type": ["string", "null"]},
        "model_identifier": {"type": "string"}
    },
    "required": ["claim_id", "claim_text", "evidence_relationship", "predicted_label", "confidence", "reasoning", "model_identifier"],
    "additionalProperties": False
}

EXPLANATION_GENERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "claim_id": {"type": "string"},
        "verdict": {"type": "string", "enum": ["TRUE", "FALSE", "UNDETERMINED"]},
        "explanation": {"type": "string"},
        "key_evidence": {"type": "array", "items": {"type": "string"}},
        "model_identifier": {"type": "string"}
    },
    "required": ["claim_id", "verdict", "explanation", "model_identifier"],
    "additionalProperties": False
}

ENTITY_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "claim_id": {"type": "string"},
        "original_text": {"type": "string"},
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "type": {"type": "string"},
                    "start": {"type": "integer"},
                    "end": {"type": "integer"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["text", "type", "start", "end"]
            }
        }
    },
    "required": ["claim_id", "original_text", "entities"],
    "additionalProperties": False
}


TASK_SCHEMAS = {
    "claim_decomposition": CLAIM_DECOMPOSITION_SCHEMA,
    "triple_extraction": TRIPLE_EXTRACTION_SCHEMA,
    "evidence_analysis": EVIDENCE_ANALYSIS_SCHEMA,
    "fact_verification": FACT_VERIFICATION_SCHEMA,
    "explanation_generation": EXPLANATION_GENERATION_SCHEMA,
    "entity_extraction": ENTITY_EXTRACTION_SCHEMA,
}


TASK_MODELS = {
    "claim_decomposition": ClaimDecomposition,
    "triple_extraction": TripleExtraction,
    "evidence_analysis": EvidenceAnalysis,
    "fact_verification": FactVerification,
    "explanation_generation": ExplanationGeneration,
    "entity_extraction": EntityExtraction,
}