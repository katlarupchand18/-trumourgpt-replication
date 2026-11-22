"""
Relation Extraction for TrumorGPT.
Extracts subject-predicate-object triples from text using few-shot LLM prompting.
Paper Section III-A, III-D: Relations for semantic health knowledge graph.
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from enum import Enum
from loguru import logger

from trumourgpt.config.settings import get_settings
from trumourgpt.llm.client import SyncOllamaClient, OllamaClientConfig
from trumourgpt.entity_extraction.extractor import EntityType, ExtractedEntity


class RelationType(str, Enum):
    """Health domain relation types (Paper: causes, prevents, treats, etc.)."""
    CAUSES = "causes"
    PREVENTS = "prevents"
    TREATS = "treats"
    INCREASES = "increases"
    DECREASES = "decreases"
    ASSOCIATED_WITH = "associated_with"
    APPROVED_BY = "approved_by"
    RECOMMENDED_BY = "recommended_by"
    CONTRADICTS = "contradicts"
    TRANSMITS = "transmits"
    DIAGNOSES = "diagnoses"
    SYMPTOM_OF = "symptom_of"
    RISK_FACTOR_FOR = "risk_factor_for"
    CONTRAINDICATED_FOR = "contraindicated_for"
    INTERACTS_WITH = "interacts_with"
    REDUCES = "reduces"
    ENHANCES = "enhances"
    INHIBITS = "inhibits"
    ACTIVATES = "activates"
    BINDS_TO = "binds_to"
    EXPRESSES = "expresses"
    REGULATES = "regulates"

    # Political/healthcare-policy relations (from KB)
    BLAMES = "blames"
    ATTACKS = "attacks"
    CRITICIZES = "criticizes"
    ACCUSES = "accuses"
    CLAIMED = "claimed"
    ENDORSED = "endorsed"
    SUPPORTED = "supported"
    COMPARED = "compared"
    CITED = "cited"
    VOTED_FOR = "voted for"
    VOTED_AGAINST = "voted against"
    PREVENTED = "prevented"
    PROPOSED = "proposed"
    GUT = "gut"
    COST = "cost"
    ESTIMATED = "estimated"
    BROKE = "broke"
    LEFT = "left"
    SPREAD = "spread"
    WOULD_ALLOW = "would have allowed"
    WOULD_COST = "would cost"
    UPHELD = "upheld"
    LOWERED = "lowered"
    PROJECTS = "projects"
    ALLOWS = "allows"
    NEGOTIATE = "negotiate"
    ENROLLED = "enrolled"
    GREW = "grew"
    SHOWS = "shows"
    REPORTED = "reported"


@dataclass
class ExtractedRelation:
    """Extracted relation triple with provenance."""
    subject: str
    subject_type: EntityType
    predicate: RelationType
    object: str
    object_type: EntityType
    confidence: float
    source_text: str
    source_span: Tuple[int, int]  # Character span in source text
    raw_predicate: str = ""  # Original predicate text before normalization

    def to_triple(self) -> Tuple[str, str, str]:
        """Return as (subject, predicate, object) triple using normalized predicate."""
        return (self.subject, self.predicate.value, self.object)

    def to_triple_raw(self) -> Tuple[str, str, str]:
        """Return as (subject, predicate, object) triple using raw predicate."""
        raw = self.raw_predicate if self.raw_predicate else self.predicate.value
        return (self.subject, raw, self.object)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "subject_type": self.subject_type.value,
            "predicate": self.predicate.value,
            "raw_predicate": self.raw_predicate,
            "object": self.object,
            "object_type": self.object_type.value,
            "confidence": self.confidence,
            "source_text": self.source_text,
            "source_span": self.source_span,
        }


@dataclass
class ExtractionConfig:
    """Configuration for relation extraction."""
    few_shot_examples: int = 5
    max_relations_per_text: int = 50
    min_confidence: float = 0.5
    use_llm: bool = True
    fallback_to_patterns: bool = True

    # Relation types to extract
    relation_types: List[RelationType] = field(default_factory=lambda: [
        RelationType.CAUSES, RelationType.PREVENTS, RelationType.TREATS,
        RelationType.INCREASES, RelationType.DECREASES, RelationType.ASSOCIATED_WITH,
        RelationType.APPROVED_BY, RelationType.RECOMMENDED_BY, RelationType.CONTRADICTS,
        RelationType.TRANSMITS, RelationType.SYMPTOM_OF, RelationType.RISK_FACTOR_FOR,
    ])


# Few-shot examples for relation extraction
RELATION_EXTRACTION_EXAMPLES = [
    {
        "text": "SARS-CoV-2 causes COVID-19.",
        "entities": [
            {"text": "SARS-CoV-2", "type": "VIRUS", "start": 0, "end": 9},
            {"text": "COVID-19", "type": "DISEASE", "start": 11, "end": 18},
        ],
        "relations": [
            {"subject": "SARS-CoV-2", "predicate": "causes", "object": "COVID-19"},
        ]
    },
    {
        "text": "COVID-19 vaccines prevent severe disease caused by SARS-CoV-2.",
        "entities": [
            {"text": "COVID-19 vaccines", "type": "VACCINE", "start": 0, "end": 17},
            {"text": "severe disease", "type": "MEDICAL_CONDITION", "start": 26, "end": 40},
            {"text": "SARS-CoV-2", "type": "VIRUS", "start": 48, "end": 58},
        ],
        "relations": [
            {"subject": "COVID-19 vaccines", "predicate": "prevents", "object": "severe disease"},
            {"subject": "SARS-CoV-2", "predicate": "causes", "object": "severe disease"},
        ]
    },
    {
        "text": "The CDC recommends Pfizer vaccine for everyone aged 6 months and older.",
        "entities": [
            {"text": "CDC", "type": "HEALTH_ORGANIZATION", "start": 4, "end": 7},
            {"text": "Pfizer vaccine", "type": "VACCINE", "start": 19, "end": 31},
        ],
        "relations": [
            {"subject": "CDC", "predicate": "recommends", "object": "Pfizer vaccine"},
        ]
    },
    {
        "text": "Clinical trials show mRNA vaccines reduce hospitalization by 95%.",
        "entities": [
            {"text": "Clinical trials", "type": "TREATMENT", "start": 0, "end": 15},
            {"text": "mRNA vaccines", "type": "VACCINE", "start": 22, "end": 35},
            {"text": "hospitalization", "type": "MEDICAL_CONDITION", "start": 36, "end": 49},
            {"text": "95%", "type": "QUANTITY", "start": 53, "end": 56},
        ],
        "relations": [
            {"subject": "mRNA vaccines", "predicate": "reduces", "object": "hospitalization"},
            {"subject": "Clinical trials", "predicate": "show", "object": "mRNA vaccines reduce hospitalization"},
        ]
    },
    {
        "text": "Long COVID causes fatigue and brain fog in patients.",
        "entities": [
            {"text": "Long COVID", "type": "DISEASE", "start": 0, "end": 10},
            {"text": "fatigue", "type": "SYMPTOM", "start": 16, "end": 23},
            {"text": "brain fog", "type": "SYMPTOM", "start": 28, "end": 37},
            {"text": "patients", "type": "PERSON", "start": 41, "end": 49},
        ],
        "relations": [
            {"subject": "Long COVID", "predicate": "causes", "object": "fatigue"},
            {"subject": "Long COVID", "predicate": "causes", "object": "brain fog"},
        ]
    },
    {
        "text": "Governor DeSantis signed legislation blocking vaccine mandates.",
        "entities": [
            {"text": "Governor DeSantis", "type": "PERSON", "start": 0, "end": 17},
            {"text": "legislation", "type": "PUBLIC_POLICY", "start": 23, "end": 34},
            {"text": "vaccine mandates", "type": "PUBLIC_POLICY", "start": 40, "end": 56},
        ],
        "relations": [
            {"subject": "Governor DeSantis", "predicate": "signed", "object": "legislation"},
            {"subject": "legislation", "predicate": "blocks", "object": "vaccine mandates"},
        ]
    },
]


# Predicate normalization: map LLM natural language predicates to RelationType enum
PREDICATE_NORMALIZATION = {
    # Causes variants
    "causes": RelationType.CAUSES,
    "cause": RelationType.CAUSES,
    "caused by": RelationType.CAUSES,
    "is caused by": RelationType.CAUSES,
    "leads to": RelationType.CAUSES,
    "led to": RelationType.CAUSES,
    "results in": RelationType.CAUSES,
    "result in": RelationType.CAUSES,
    "triggers": RelationType.CAUSES,
    "induces": RelationType.CAUSES,

    # Prevents variants
    "prevents": RelationType.PREVENTS,
    "prevent": RelationType.PREVENTS,
    "protects against": RelationType.PREVENTS,
    "reduces risk of": RelationType.PREVENTS,
    "lowers risk of": RelationType.PREVENTS,
    "is effective at preventing": RelationType.PREVENTS,
    "are effective at preventing": RelationType.PREVENTS,
    "blocks": RelationType.PREVENTS,
    "inhibits": RelationType.PREVENTS,

    # Treats variants
    "treats": RelationType.TREATS,
    "treat": RelationType.TREATS,
    "cures": RelationType.TREATS,
    "therapy for": RelationType.TREATS,
    "treatment for": RelationType.TREATS,

    # Increases variants
    "increases": RelationType.INCREASES,
    "increase": RelationType.INCREASES,
    "raises": RelationType.INCREASES,
    "elevates": RelationType.INCREASES,
    "boosts": RelationType.INCREASES,
    "enhances": RelationType.INCREASES,

    # Decreases variants
    "decreases": RelationType.DECREASES,
    "decrease": RelationType.DECREASES,
    "reduces": RelationType.DECREASES,
    "reducing": RelationType.DECREASES,
    "lowers": RelationType.DECREASES,
    "lowers": RelationType.DECREASES,
    "diminishes": RelationType.DECREASES,

    # Associated with variants
    "associated with": RelationType.ASSOCIATED_WITH,
    "linked to": RelationType.ASSOCIATED_WITH,
    "correlated with": RelationType.ASSOCIATED_WITH,
    "associated": RelationType.ASSOCIATED_WITH,

    # Approved by variants
    "approved by": RelationType.APPROVED_BY,
    "authorized by": RelationType.APPROVED_BY,
    "approved": RelationType.APPROVED_BY,

    # Recommended by variants
    "recommended by": RelationType.RECOMMENDED_BY,
    "advised by": RelationType.RECOMMENDED_BY,
    "endorsed by": RelationType.RECOMMENDED_BY,
    "recommends": RelationType.RECOMMENDED_BY,
    "recommend": RelationType.RECOMMENDED_BY,
    "recommending": RelationType.RECOMMENDED_BY,

    # Confirms/States variants - map to ASSOCIATED_WITH for attribution
    "confirms": RelationType.ASSOCIATED_WITH,
    "confirm": RelationType.ASSOCIATED_WITH,
    "confirmed": RelationType.ASSOCIATED_WITH,
    "states": RelationType.ASSOCIATED_WITH,
    "state": RelationType.ASSOCIATED_WITH,
    "stated": RelationType.ASSOCIATED_WITH,

    # Contradicts variants
    "contradicts": RelationType.CONTRADICTS,
    "refutes": RelationType.CONTRADICTS,
    "disputes": RelationType.CONTRADICTS,

    # Transmits variants
    "transmits": RelationType.TRANSMITS,
    "transmit": RelationType.TRANSMITS,
    "spreads": RelationType.TRANSMITS,

    # Diagnoses variants
    "diagnoses": RelationType.DIAGNOSES,
    "diagnose": RelationType.DIAGNOSES,

    # Symptom of variants
    "symptom of": RelationType.SYMPTOM_OF,
    "is a symptom of": RelationType.SYMPTOM_OF,
    "symptoms of": RelationType.SYMPTOM_OF,

    # Risk factor for variants
    "risk factor for": RelationType.RISK_FACTOR_FOR,
    "risk factor": RelationType.RISK_FACTOR_FOR,

    # Contraindicated for variants
    "contraindicated for": RelationType.CONTRAINDICATED_FOR,

    # Interacts with variants
    "interacts with": RelationType.INTERACTS_WITH,
    "interact with": RelationType.INTERACTS_WITH,

    # Reduces variants
    "reduces": RelationType.REDUCES,

    # Enhances variants
    "enhances": RelationType.ENHANCES,

    # Inhibits variants
    "inhibits": RelationType.INHIBITS,

    # Activates variants
    "activates": RelationType.ACTIVATES,

    # Binds to variants
    "binds to": RelationType.BINDS_TO,

    # Expresses variants
    "expresses": RelationType.EXPRESSES,

    # Regulates variants
    "regulates": RelationType.REGULATES,
    "regulate": RelationType.REGULATES,

    # Political/healthcare-policy relations
    "blames": RelationType.BLAMES,
    "blamed": RelationType.BLAMES,
    "attacks": RelationType.ATTACKS,
    "attacked": RelationType.ATTACKS,
    "criticizes": RelationType.CRITICIZES,
    "criticized": RelationType.CRITICIZES,
    "accuses": RelationType.ACCUSES,
    "accused": RelationType.ACCUSES,
    "claimed": RelationType.CLAIMED,
    "claims": RelationType.CLAIMED,
    "stated": RelationType.CLAIMED,
    "said": RelationType.CLAIMED,
    "endorsed": RelationType.ENDORSED,
    "endorses": RelationType.ENDORSED,
    "supported": RelationType.SUPPORTED,
    "supports": RelationType.SUPPORTED,
    "compared": RelationType.COMPARED,
    "compares": RelationType.COMPARED,
    "cited": RelationType.CITED,
    "cites": RelationType.CITED,
    "voted for": RelationType.VOTED_FOR,
    "voted against": RelationType.VOTED_AGAINST,
    "prevented": RelationType.PREVENTED,
    "proposed": RelationType.PROPOSED,
    "proposes": RelationType.PROPOSED,
    "gut": RelationType.GUT,
    "gutted": RelationType.GUT,
    "cost": RelationType.COST,
    "costs": RelationType.COST,
    "estimated": RelationType.ESTIMATED,
    "estimates": RelationType.ESTIMATED,
    "projected": RelationType.ESTIMATED,
    "broke": RelationType.BROKE,
    "breaks": RelationType.BROKE,
    "left": RelationType.LEFT,
    "leaves": RelationType.LEFT,
    "spread": RelationType.SPREAD,
    "spreads": RelationType.SPREAD,
    "would have allowed": RelationType.WOULD_ALLOW,
    "would allow": RelationType.WOULD_ALLOW,
    "would cost": RelationType.WOULD_COST,
    "upheld": RelationType.UPHELD,
    "lowered": RelationType.LOWERED,
    "projects": RelationType.PROJECTS,
    "allows": RelationType.ALLOWS,
    "negotiate": RelationType.NEGOTIATE,
    "enrolled": RelationType.ENROLLED,
    "grew": RelationType.GREW,
    "shows": RelationType.SHOWS,
    "reported": RelationType.REPORTED,
}


def normalize_predicate(predicate: str) -> RelationType:
    """Normalize natural language predicate to RelationType enum."""
    pred_lower = predicate.lower().strip()

    # Direct match
    if pred_lower in PREDICATE_NORMALIZATION:
        return PREDICATE_NORMALIZATION[pred_lower]

    # Try partial matches
    for key, value in PREDICATE_NORMALIZATION.items():
        if key in pred_lower or pred_lower in key:
            return value

    # Default
    return RelationType.ASSOCIATED_WITH


# Valid subject-object type combinations for each relation
RELATION_CONSTRAINTS = {
    RelationType.CAUSES: [
        (EntityType.VIRUS, EntityType.DISEASE),
        (EntityType.DISEASE, EntityType.SYMPTOM),
        (EntityType.DISEASE, EntityType.MEDICAL_CONDITION),
        (EntityType.DISEASE, EntityType.DISEASE),
        (EntityType.MEDICATION, EntityType.SYMPTOM),
        (EntityType.EXPOSURE, EntityType.DISEASE),
        (EntityType.VACCINE, EntityType.MEDICAL_CONDITION),  # For claims like "vaccines cause X"
        (EntityType.VACCINE, EntityType.DISEASE),
    ],
    RelationType.PREVENTS: [
        (EntityType.VACCINE, EntityType.DISEASE),
        (EntityType.VACCINE, EntityType.MEDICAL_CONDITION),
        (EntityType.MEDICATION, EntityType.DISEASE),
        (EntityType.PUBLIC_POLICY, EntityType.DISEASE),
        (EntityType.TREATMENT, EntityType.DISEASE),
        (EntityType.DISEASE, EntityType.DISEASE),  # Allow for LLM misclassification
        (EntityType.TREATMENT, EntityType.SYMPTOM),  # Masks prevent spread
        (EntityType.TREATMENT, EntityType.TRANSMISSION),
        (EntityType.MEDICATION, EntityType.SYMPTOM),
        (EntityType.MEDICATION, EntityType.TRANSMISSION),
    ],
    RelationType.TREATS: [
        (EntityType.MEDICATION, EntityType.DISEASE),
        (EntityType.MEDICATION, EntityType.MEDICAL_CONDITION),
        (EntityType.TREATMENT, EntityType.DISEASE),
        (EntityType.THERAPY, EntityType.DISEASE),
    ],
    RelationType.INCREASES: [
        (EntityType.RISK_FACTOR, EntityType.DISEASE),
        (EntityType.MEDICATION, EntityType.RISK),
        (EntityType.CONDITION, EntityType.RISK),
    ],
    RelationType.DECREASES: [
        (EntityType.TREATMENT, EntityType.RISK),
        (EntityType.MEDICATION, EntityType.SYMPTOM),
        (EntityType.VACCINE, EntityType.TRANSMISSION),
    ],
    RelationType.ASSOCIATED_WITH: [
        (EntityType.DISEASE, EntityType.DISEASE),
        (EntityType.GENE, EntityType.DISEASE),
        (EntityType.PROTEIN, EntityType.DISEASE),
        (EntityType.RISK_FACTOR, EntityType.DISEASE),
        (EntityType.MEDICAL_CONDITION, EntityType.VACCINE),
        (EntityType.VACCINE, EntityType.MEDICAL_CONDITION),
        # Political/healthcare policy relations
        (EntityType.HEALTH_ORGANIZATION, EntityType.PUBLIC_POLICY),
        (EntityType.HEALTH_ORGANIZATION, EntityType.PERSON),
        (EntityType.PERSON, EntityType.PUBLIC_POLICY),
        (EntityType.PUBLIC_POLICY, EntityType.QUANTITY),
        (EntityType.PUBLIC_POLICY, EntityType.PUBLIC_POLICY),
    ],
    RelationType.APPROVED_BY: [
        (EntityType.MEDICATION, EntityType.HEALTH_ORGANIZATION),
        (EntityType.VACCINE, EntityType.HEALTH_ORGANIZATION),
        (EntityType.TREATMENT, EntityType.HEALTH_ORGANIZATION),
    ],
    RelationType.RECOMMENDED_BY: [
        (EntityType.VACCINE, EntityType.HEALTH_ORGANIZATION),
        (EntityType.TREATMENT, EntityType.HEALTH_ORGANIZATION),
        (EntityType.PUBLIC_POLICY, EntityType.HEALTH_ORGANIZATION),
    ],
    RelationType.CONTRADICTS: [
        (EntityType.CLAIM, EntityType.CLAIM),
        (EntityType.STUDY, EntityType.STUDY),
        (EntityType.POLICY, EntityType.POLICY),
    ],
    RelationType.SYMPTOM_OF: [
        (EntityType.SYMPTOM, EntityType.DISEASE),
        (EntityType.SYMPTOM, EntityType.MEDICAL_CONDITION),
        (EntityType.DISEASE, EntityType.DISEASE),  # Allow for LLM misclassification
        (EntityType.MEDICAL_CONDITION, EntityType.DISEASE),
    ],
    RelationType.RISK_FACTOR_FOR: [
        (EntityType.CONDITION, EntityType.DISEASE),
        (EntityType.BEHAVIOR, EntityType.DISEASE),
        (EntityType.GENE, EntityType.DISEASE),
    ],
    RelationType.TRANSMITS: [
        (EntityType.DISEASE, EntityType.TRANSMISSION),
        (EntityType.VIRUS, EntityType.TRANSMISSION),
        (EntityType.PERSON, EntityType.DISEASE),
    ],

    # Political/healthcare-policy relation constraints
    RelationType.BLAMES: [
        (EntityType.HEALTH_ORGANIZATION, EntityType.PERSON),
        (EntityType.HEALTH_ORGANIZATION, EntityType.PUBLIC_POLICY),
        (EntityType.PERSON, EntityType.PERSON),
        (EntityType.PERSON, EntityType.PUBLIC_POLICY),
    ],
    RelationType.ATTACKS: [
        (EntityType.HEALTH_ORGANIZATION, EntityType.PERSON),
        (EntityType.PERSON, EntityType.PERSON),
        (EntityType.PERSON, EntityType.PUBLIC_POLICY),
    ],
    RelationType.CRITICIZES: [
        (EntityType.HEALTH_ORGANIZATION, EntityType.PERSON),
        (EntityType.PERSON, EntityType.PERSON),
        (EntityType.PERSON, EntityType.PUBLIC_POLICY),
    ],
    RelationType.ACCUSES: [
        (EntityType.HEALTH_ORGANIZATION, EntityType.PERSON),
        (EntityType.PERSON, EntityType.PERSON),
        (EntityType.PERSON, EntityType.PUBLIC_POLICY),
    ],
    RelationType.CLAIMED: [
        (EntityType.HEALTH_ORGANIZATION, EntityType.PUBLIC_POLICY),
        (EntityType.HEALTH_ORGANIZATION, EntityType.QUANTITY),
        (EntityType.PERSON, EntityType.PUBLIC_POLICY),
        (EntityType.PERSON, EntityType.QUANTITY),
        (EntityType.PERSON, EntityType.PERSON),
    ],
    RelationType.ENDORSED: [
        (EntityType.PERSON, EntityType.PERSON),
        (EntityType.PERSON, EntityType.PUBLIC_POLICY),
        (EntityType.HEALTH_ORGANIZATION, EntityType.PERSON),
    ],
    RelationType.SUPPORTED: [
        (EntityType.PERSON, EntityType.PUBLIC_POLICY),
        (EntityType.PERSON, EntityType.PERSON),
        (EntityType.HEALTH_ORGANIZATION, EntityType.PUBLIC_POLICY),
    ],
    RelationType.COMPARED: [
        (EntityType.PERSON, EntityType.PUBLIC_POLICY),
        (EntityType.PERSON, EntityType.QUANTITY),
    ],
    RelationType.CITED: [
        (EntityType.PERSON, EntityType.PUBLIC_POLICY),
        (EntityType.PERSON, EntityType.QUANTITY),
    ],
    RelationType.VOTED_FOR: [
        (EntityType.PERSON, EntityType.PUBLIC_POLICY),
        (EntityType.PERSON, EntityType.PERSON),
    ],
    RelationType.VOTED_AGAINST: [
        (EntityType.PERSON, EntityType.PUBLIC_POLICY),
        (EntityType.PERSON, EntityType.PERSON),
    ],
    RelationType.PREVENTED: [
        (EntityType.PERSON, EntityType.PUBLIC_POLICY),
        (EntityType.PUBLIC_POLICY, EntityType.PUBLIC_POLICY),
    ],
    RelationType.PROPOSED: [
        (EntityType.PERSON, EntityType.PUBLIC_POLICY),
        (EntityType.PUBLIC_POLICY, EntityType.PUBLIC_POLICY),
    ],
    RelationType.GUT: [
        (EntityType.PERSON, EntityType.PUBLIC_POLICY),
        (EntityType.PERSON, EntityType.PERSON),
    ],
    RelationType.COST: [
        (EntityType.PUBLIC_POLICY, EntityType.QUANTITY),
        (EntityType.PERSON, EntityType.QUANTITY),
    ],
    RelationType.ESTIMATED: [
        (EntityType.PUBLIC_POLICY, EntityType.QUANTITY),
        (EntityType.PERSON, EntityType.QUANTITY),
    ],
    RelationType.BROKE: [
        (EntityType.PERSON, EntityType.PUBLIC_POLICY),
        (EntityType.PERSON, EntityType.PERSON),
    ],
    RelationType.LEFT: [
        (EntityType.PERSON, EntityType.PUBLIC_POLICY),
        (EntityType.PERSON, EntityType.QUANTITY),
    ],
    RelationType.SPREAD: [
        (EntityType.VIRUS, EntityType.DISEASE),
        (EntityType.PERSON, EntityType.DISEASE),
        (EntityType.DISEASE, EntityType.TRANSMISSION),
    ],
    RelationType.WOULD_ALLOW: [
        (EntityType.PUBLIC_POLICY, EntityType.PUBLIC_POLICY),
        (EntityType.PERSON, EntityType.PUBLIC_POLICY),
    ],
    RelationType.WOULD_COST: [
        (EntityType.PUBLIC_POLICY, EntityType.QUANTITY),
        (EntityType.PERSON, EntityType.QUANTITY),
    ],
    RelationType.UPHELD: [
        (EntityType.HEALTH_ORGANIZATION, EntityType.PUBLIC_POLICY),
        (EntityType.PERSON, EntityType.PUBLIC_POLICY),
    ],
    RelationType.LOWERED: [
        (EntityType.PUBLIC_POLICY, EntityType.QUANTITY),
        (EntityType.PERSON, EntityType.QUANTITY),
    ],
    RelationType.PROJECTS: [
        (EntityType.HEALTH_ORGANIZATION, EntityType.QUANTITY),
        (EntityType.PUBLIC_POLICY, EntityType.QUANTITY),
    ],
    RelationType.ALLOWS: [
        (EntityType.PUBLIC_POLICY, EntityType.PUBLIC_POLICY),
        (EntityType.HEALTH_ORGANIZATION, EntityType.PUBLIC_POLICY),
    ],
    RelationType.NEGOTIATE: [
        (EntityType.PUBLIC_POLICY, EntityType.PUBLIC_POLICY),
        (EntityType.HEALTH_ORGANIZATION, EntityType.PUBLIC_POLICY),
    ],
    RelationType.ENROLLED: [
        (EntityType.QUANTITY, EntityType.PUBLIC_POLICY),
        (EntityType.PUBLIC_POLICY, EntityType.QUANTITY),
    ],
    RelationType.GREW: [
        (EntityType.PUBLIC_POLICY, EntityType.QUANTITY),
        (EntityType.HEALTH_ORGANIZATION, EntityType.QUANTITY),
    ],
    RelationType.SHOWS: [
        (EntityType.PUBLIC_POLICY, EntityType.QUANTITY),
        (EntityType.STUDY, EntityType.QUANTITY),
    ],
    RelationType.REPORTED: [
        (EntityType.HEALTH_ORGANIZATION, EntityType.QUANTITY),
        (EntityType.PERSON, EntityType.QUANTITY),
        (EntityType.PUBLIC_POLICY, EntityType.QUANTITY),
    ],
}


def build_relation_extraction_prompt(
    text: str,
    entities: List[ExtractedEntity],
    examples: List[Dict],
    relation_types: List[RelationType],
) -> str:
    """Build few-shot prompt for relation extraction."""
    type_list = ", ".join([t.value for t in relation_types])
    entity_list = ", ".join([f"{e.text} ({e.entity_type.value})" for e in entities])

    prompt = f"""Extract relations between entities from the text as subject-predicate-object triples.
IMPORTANT: The SUBJECT is the cause/agent/source, the OBJECT is the effect/outcome/target.
Examples of correct direction:
- "SARS-CoV-2 causes COVID-19" → subject="SARS-CoV-2" (virus), predicate="causes", object="COVID-19" (disease)
- "Vaccines prevent disease" → subject="Vaccines", predicate="prevents", object="disease"
- "Treatment reduces symptoms" → subject="Treatment", predicate="reduces", object="symptoms"

Relation types: {type_list}
Entities in text: {entity_list}

Return a JSON object with "triples" array. Each triple must have: subject, predicate, object, source_text, source_id, confidence.

Examples:
"""
    for i, ex in enumerate(examples):
        # Convert relations to triples format for examples
        triples = []
        for rel in ex.get('relations', []):
            triples.append({
                "subject": rel.get("subject", ""),
                "predicate": rel.get("predicate", ""),
                "object": rel.get("object", ""),
                "source_text": ex['text'],
                "source_id": f"example_{i}",
                "confidence": 0.9
            })
        prompt += f"\nExample {i+1}:\nText: {ex['text']}\nTriples: {json.dumps(triples)}\n"

    prompt += f"\nNow extract from:\nText: {text}\nTriples:"
    return prompt


class RelationExtractor:
    """Relation extractor using few-shot LLM prompting."""

    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.config = config or ExtractionConfig()
        self._client: Optional[SyncOllamaClient] = None
        self._examples = RELATION_EXTRACTION_EXAMPLES[:self.config.few_shot_examples]

    def _get_client(self) -> SyncOllamaClient:
        if self._client is None:
            settings = get_settings()
            client_config = OllamaClientConfig(
                base_url=settings.llm.base_url,
                model=settings.llm.model,
                temperature=settings.llm.temperature,
            )
            self._client = SyncOllamaClient(client_config)
        return self._client

    def extract_relations(
        self,
        text: str,
        entities: List[ExtractedEntity],
        source_id: str = "unknown",
        validate: bool = True,
        use_llm: Optional[bool] = None,
    ) -> List[ExtractedRelation]:
        """Extract relations between entities in text.

        Args:
            text: Source text
            entities: Extracted entities
            source_id: Source identifier
            validate: Whether to validate relations against type constraints (True for KB construction, False for query processing)
        """
        relations = []

        # Determine whether to use LLM (override config if provided)
        use_llm_extraction = self.config.use_llm if use_llm is None else use_llm

        if use_llm_extraction:
            try:
                relations = self._extract_with_llm(text, entities, source_id)
            except Exception as e:
                logger.warning(f"LLM relation extraction failed: {e}")
                if self.config.fallback_to_patterns:
                    relations = self._extract_with_patterns(text, entities, source_id)
        else:
            relations = self._extract_with_patterns(text, entities, source_id)

        # Filter by confidence
        relations = [r for r in relations if r.confidence >= self.config.min_confidence]

        # Validate relation constraints (skip for query processing)
        if validate:
            relations = self._validate_relations(relations, entities)
        else:
            logger.debug("Skipping relation validation for query processing")

        # Limit count
        relations = relations[:self.config.max_relations_per_text]

        # Deduplicate
        relations = self._deduplicate_relations(relations)

        return relations

    def _extract_with_llm(
        self,
        text: str,
        entities: List[ExtractedEntity],
        source_id: str,
    ) -> List[ExtractedRelation]:
        """Extract using LLM."""
        client = self._get_client()

        prompt = build_relation_extraction_prompt(
            text, entities, self._examples, self.config.relation_types
        )

        # Use triple_extraction schema
        response = client.generate_structured("triple_extraction", prompt)

        logger.debug(f"Triple extraction response: success={response.success}, data={response.data}")

        if not response.success or response.data is None:
            raise RuntimeError(f"LLM extraction failed: {response.error}")

        relations = []
        # Use validated response.data (dict) instead of raw_response
        data = response.data
        if isinstance(data, dict) and "triples" in data:
            logger.debug(f"Found {len(data['triples'])} triples in response")
            for triple in data["triples"]:
                # Find entity types for subject and object
                subj_ent = self._find_entity(triple.get("subject", ""), entities)
                obj_ent = self._find_entity(triple.get("object", ""), entities)

                raw_predicate = triple.get("predicate", "associated_with")
                normalized_predicate = normalize_predicate(raw_predicate)

                logger.debug(f"Triple: {triple.get('subject')} --{raw_predicate}->{normalized_predicate.value}-- {triple.get('object')}")

                relations.append(ExtractedRelation(
                    subject=triple.get("subject", ""),
                    subject_type=subj_ent.entity_type if subj_ent else EntityType.DISEASE,
                    predicate=normalized_predicate,
                    object=triple.get("object", ""),
                    object_type=obj_ent.entity_type if obj_ent else EntityType.DISEASE,
                    confidence=triple.get("confidence", 0.8),
                    source_text=text,
                    source_span=(0, len(text)),
                    raw_predicate=raw_predicate,
                ))
        else:
            logger.warning(f"Unexpected response format: {data}")

        return relations

    def _find_entity(self, text: str, entities: List[ExtractedEntity]) -> Optional[ExtractedEntity]:
        """Find entity by text (fuzzy match)."""
        text_lower = text.lower()
        for ent in entities:
            if ent.text.lower() in text_lower or text_lower in ent.text.lower():
                return ent
        return None

    def _extract_with_patterns(
        self,
        text: str,
        entities: List[ExtractedEntity],
        source_id: str,
    ) -> List[ExtractedRelation]:
        """Fallback pattern-based relation extraction."""
        relations = []

        # Create entity lookup by position
        entities_by_pos = {ent.start: ent for ent in entities}

        # Simple pattern-based extraction
        # Pattern: Entity1 relation Entity2 (with optional connecting words like "that", "which", etc.)
        # Use more flexible patterns that can handle 1-4 word entities and optional connecting words
        # Object pattern: 1-4 words but not ending with common prepositions (using word boundary)
        relation_patterns = [
            # Fixed lookahead: allow word boundary at end of string without requiring trailing whitespace
            (r'(\S+(?:\s+\S+){0,3})\s+(?:that\s+)?(?:causes?|cause|led\s+to|results?\s+in)\s+(?:that\s+)?(\S+(?:\s+\S+){0,3}?)(?=\s+(?:by|in|on|at|for|with|from|to|of|the|a|an|that|which|\.|,)|$)', RelationType.CAUSES),
            (r'(\S+(?:\s+\S+){0,3})\s+(?:that\s+)?(?:prevents?|prevent|protects?\s+against|reduces?\s+risk\s+of)\s+(?:that\s+)?(\S+(?:\s+\S+){0,3}?)(?=\s+(?:by|in|on|at|for|with|from|to|of|the|a|an|that|which|\.|,)|$)', RelationType.PREVENTS),
            (r'(\S+(?:\s+\S+){0,3})\s+(?:that\s+)?(?:treats?|treat|cures?|therapy\s+for)\s+(?:that\s+)?(\S+(?:\s+\S+){0,3}?)(?=\s+(?:by|in|on|at|for|with|from|to|of|the|a|an|that|which|\.|,)|$)', RelationType.TREATS),
            (r'(\S+(?:\s+\S+){0,3})\s+(?:that\s+)?(?:increases?|raises?|elevates?)\s+(?:that\s+)?(\S+(?:\s+\S+){0,3}?)(?=\s+(?:by|in|on|at|for|with|from|to|of|the|a|an|that|which|\.|,)|$)', RelationType.INCREASES),
            (r'(\S+(?:\s+\S+){0,3})\s+(?:that\s+)?(?:decreases?|reduces?|reduced|lowers?|lowered)\s+(?:that\s+)?(\S+(?:\s+\S+){0,3}?)(?=\s+(?:by|in|on|at|for|with|from|to|of|the|a|an|that|which|\.|,)|$)', RelationType.DECREASES),
            (r'(\S+(?:\s+\S+){0,3})\s+(?:that\s+)?(?:associated\s+with|linked\s+to|correlated\s+with)\s+(?:that\s+)?(\S+(?:\s+\S+){0,3}?)(?=\s+(?:by|in|on|at|for|with|from|to|of|the|a|an|that|which|\.|,)|$)', RelationType.ASSOCIATED_WITH),
            (r'(\S+(?:\s+\S+){0,3})\s+(?:that\s+)?(?:approved\s+by|authorized\s+by)\s+(?:that\s+)?(\S+(?:\s+\S+){0,3}?)(?=\s+(?:by|in|on|at|for|with|from|to|of|the|a|an|that|which|\.|,)|$)', RelationType.APPROVED_BY),
            (r'(\S+(?:\s+\S+){0,3})\s+(?:that\s+)?(?:recommended\s+by|advised\s+by)\s+(?:that\s+)?(\S+(?:\s+\S+){0,3}?)(?=\s+(?:by|in|on|at|for|with|from|to|of|the|a|an|that|which|\.|,)|$)', RelationType.RECOMMENDED_BY),
            (r'(\S+(?:\s+\S+){0,3})\s+(?:that\s+)?(?:contradicts?|refutes?|disputes?)\s+(?:that\s+)?(\S+(?:\s+\S+){0,3}?)(?=\s+(?:by|in|on|at|for|with|from|to|of|the|a|an|that|which|\.|,)|$)', RelationType.CONTRADICTS),
            # Negation patterns (e.g., "do not alter", "do not cause", "do not change")
            (r'(\S+(?:\s+\S+){0,3})\s+(?:that\s+)?(?:do\s+not|does\s+not|did\s+not)\s+(?:alter|change|affect|cause|impact|modify)\s+(?:that\s+)?(\S+(?:\s+\S+){0,3}?)(?=\s+(?:by|in|on|at|for|with|from|to|of|the|a|an|that|which|\.|,)|$)', RelationType.CONTRADICTS),
            (r'(\S+(?:\s+\S+){0,3})\s+(?:that\s+)?(?:do\s+not|does\s+not|did\s+not)\s+(?:increase|decrease|raise|lower|reduce)\s+(?:that\s+)?(\S+(?:\s+\S+){0,3}?)(?=\s+(?:by|in|on|at|for|with|from|to|of|the|a|an|that|which|\.|,)|$)', RelationType.CONTRADICTS),
            # Healthcare policy relation patterns
            (r'(\S+(?:\s+\S+){0,3})\s+(?:that\s+)?(?:projects?|projected|projecting)\s+(?:that\s+)?(\S+(?:\s+\S+){0,3}?)(?=\s+(?:by|in|on|at|for|with|from|to|of|the|a|an|that|which|\.|,)|$)', RelationType.ASSOCIATED_WITH),
            # Political relation patterns - expanded (with flexible object capture)
            # Use simpler lookahead that only stops at punctuation or end of string
            # Increase subject capture to 5 words to handle compound subjects like "Karen Handel and Lucy McBath"
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:blames?|blamed|blaming)\s+(?:that\s+)?(.+?)(?=\s*[.!?]|$)', RelationType.BLAMES),
            # Special pattern for "X attacks Y on Z" structure
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:attacks?|attacked|attacking)\s+(?:that\s+)?(.+?)\s+on\s+(.+?)(?=\s*[.!?]|$)', RelationType.ATTACKS),
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:attacks?|attacked|attacking)\s+(?:that\s+)?(.+?)(?=\s*(?:on\s+|[.!?])|$)', RelationType.ATTACKS),
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:criticizes?|criticized|criticizing)\s+(?:that\s+)?(.+?)(?=\s*[.!?]|$)', RelationType.CRITICIZES),
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:accuses?|accused|accusing)\s+(?:that\s+)?(.+?)(?=\s*[.!?]|$)', RelationType.ACCUSES),
            # Special pattern for "X compares A to B" structure
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:compares?|compared|comparing)\s+(?:that\s+)?(.+?)\s+to\s+(.+?)(?=\s*[.!?]|$)', RelationType.COMPARED),
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:says?|said|saying)\s+(?:that\s+)?(.+?)(?=\s*[.!?]|$)', RelationType.CLAIMED),
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:voted\s+against)\s+(?:that\s+)?(.+?)(?=\s*[.!?]|$)', RelationType.VOTED_AGAINST),
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:voted\s+for)\s+(?:that\s+)?(.+?)(?=\s*[.!?]|$)', RelationType.VOTED_FOR),
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:endorses?|endorsed|endorsing)\s+(?:that\s+)?(.+?)(?=\s*[.!?]|$)', RelationType.ENDORSED),
            # Special pattern for "X contradicts Y on Z" structure
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:contradicts?|contradicted|contradicting)\s+(?:that\s+)?(.+?)\s+on\s+(.+?)(?=\s*[.!?]|$)', RelationType.CONTRADICTS),
            # Generic contradicts pattern (fallback) - stop at "on" or punctuation
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:contradicts?|contradicted|contradicting)\s+(?:that\s+)?(.+?)(?=\s*(?:on\s+|[.!?])|$)', RelationType.CONTRADICTS),
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:for)\s+(?:that\s+)?(.+?)(?=\s*[.!?]|$)', RelationType.ASSOCIATED_WITH),
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:upheld|upholds|upheld)\s+(?:that\s+)?(.+?)(?=\s*[.!?]|$)', RelationType.UPHELD),
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:would\s+cost|costs?|cost)\s+(?:that\s+)?(.+?)(?=\s*[.!?]|$)', RelationType.WOULD_COST),
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:allows?|allowed|allow)\s+(?:that\s+)?(.+?)(?=\s*[.!?]|$)', RelationType.ALLOWS),
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:negotiate[sd]?|negotiating)\s+(?:that\s+)?(.+?)(?=\s*[.!?]|$)', RelationType.NEGOTIATE),
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:enrolled?|enrolls?|enrolling)\s+(?:that\s+)?(.+?)(?=\s*[.!?]|$)', RelationType.ENROLLED),
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:grew|grow|grows|grown)\s+(?:that\s+)?(.+?)(?=\s*[.!?]|$)', RelationType.GREW),
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:shows?|showed|shown)\s+(?:that\s+)?(.+?)(?=\s*[.!?]|$)', RelationType.SHOWS),
            (r'(\S+(?:\S+){0,5})\s+(?:that\s+)?(?:reported?|reports?|reporting)\s+(?:that\s+)?(.+?)(?=\s*[.!?]|$)', RelationType.REPORTED),
            # Additional political patterns for claims - with longer object capture
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:claims?|claimed|claiming)\s+(?:that\s+)?(.+?)(?=\s*[.!?]|$)', RelationType.CLAIMED),
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:wrong\s+that|wrongly\s+said|incorrectly\s+said)\s+(?:that\s+)?(.+?)(?=\s*[.!?]|$)', RelationType.CONTRADICTS),
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:misleads?|misled|misleading)\s+(?:that\s+)?(.+?)(?=\s*[.!?]|$)', RelationType.CONTRADICTS),
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:concludes?|concluded|concluding)\s+(?:that\s+)?(.+?)(?=\s*[.!?]|$)', RelationType.CLAIMED),
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:argues?|argued|arguing)\s+(?:that\s+)?(.+?)(?=\s*[.!?]|$)', RelationType.CLAIMED),
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:states?|stated|stating)\s+(?:that\s+)?(.+?)(?=\s*[.!?]|$)', RelationType.CLAIMED),
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:asserts?|asserted|asserting)\s+(?:that\s+)?(.+?)(?=\s*[.!?]|$)', RelationType.CLAIMED),
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:alleges?|alleged|alleging)\s+(?:that\s+)?(.+?)(?=\s*[.!?]|$)', RelationType.CLAIMED),
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:errs\s+on|erred\s+on|erring\s+on)\s+(?:that\s+)?(.+?)(?=\s*[.!?]|$)', RelationType.CONTRADICTS),
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:tying)\s+(?:that\s+)?(.+?)(?=\s*[.!?]|$)', RelationType.ASSOCIATED_WITH),
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:claim\s+about)\s+(?:that\s+)?(.+?)(?=\s*[.!?]|$)', RelationType.CLAIMED),
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:attack\s+on)\s+(?:that\s+)?(.+?)(?=\s*[.!?]|$)', RelationType.ATTACKS),
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:demonstrated|demonstrates|demonstrating)\s+(?:that\s+)?(.+?)(?=\s*[.!?]|$)', RelationType.SHOWS),
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:won\'t\s+lead\s+to|will\s+not\s+lead\s+to|won\'t\s+cause|will\s+not\s+cause)\s+(?:that\s+)?(.+?)(?=\s*[.!?]|$)', RelationType.CONTRADICTS),
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:leads\s+to|causes?|caused)\s+(?:that\s+)?(.+?)(?=\s*[.!?]|$)', RelationType.CAUSES),
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:result\s+of|resulted\s+from|caused\s+by)\s+(?:that\s+)?(.+?)(?=\s*[.!?]|$)', RelationType.CAUSES),
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:gut|gutted|gutting)\s+(?:that\s+)?(.+?)(?=\s*[.!?]|$)', RelationType.VOTED_AGAINST),
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:voted\s+to)\s+(?:that\s+)?(.+?)(?=\s*[.!?]|$)', RelationType.VOTED_FOR),
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:supports?|supported|supporting)\s+(?:that\s+)?(.+?)(?=\s*[.!?]|$)', RelationType.ENDORSED),
            (r'(\S+(?:\s+\S+){0,5})\s+(?:that\s+)?(?:opposes?|opposed|opposing)\s+(?:that\s+)?(.+?)(?=\s*[.!?]|$)', RelationType.CONTRADICTS),
        ]

        for pattern, rel_type in relation_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                subj_text = match.group(1).strip()

                # Handle special "compares X to Y" pattern with 3 groups
                if rel_type == RelationType.COMPARED and len(match.groups()) >= 3:
                    # Combine group 2 and group 3 as "X to Y"
                    obj_text = f"{match.group(2).strip()} to {match.group(3).strip()}"
                # Handle special "contradicts X on Y" pattern with 3 groups - use only X as object
                elif rel_type == RelationType.CONTRADICTS and len(match.groups()) >= 3:
                    # Use only group 2 (X) as object, ignore group 3 (Y after "on")
                    obj_text = match.group(2).strip()
                # Handle special "attacks X on Y" pattern with 3 groups - use only X as object
                elif rel_type == RelationType.ATTACKS and len(match.groups()) >= 3:
                    # Use only group 2 (X) as object, ignore group 3 (Y after "on")
                    obj_text = match.group(2).strip()
                else:
                    obj_text = match.group(2).strip()

                # Find matching entities
                subj_ent = self._find_entity_fuzzy(subj_text, entities)
                obj_ent = self._find_entity_fuzzy(obj_text, entities)

                if subj_ent and obj_ent:
                    # Use entity's normalized_form (or text) for subject/object to ensure consistency with KB
                    subj_normalized = subj_ent.normalized_form if subj_ent.normalized_form else subj_ent.text
                    obj_normalized = obj_ent.normalized_form if obj_ent.normalized_form else obj_ent.text

                    # Extract the matched predicate text for raw_predicate
                    matched_text = match.group(0)
                    # Try to extract the predicate part (middle part)
                    raw_pred = rel_type.value  # default to normalized
                    for pat, _ in relation_patterns:
                        m = re.search(pat, matched_text, re.IGNORECASE)
                        if m and len(m.groups()) >= 2:
                            # The predicate is between group 1 and group 2
                            full = m.group(0)
                            subj_part = m.group(1)
                            obj_part = m.group(2)
                            pred_part = full[len(subj_part):-len(obj_part)].strip()
                            if pred_part:
                                raw_pred = pred_part
                            break

                    relations.append(ExtractedRelation(
                        subject=subj_normalized,
                        subject_type=subj_ent.entity_type,
                        predicate=rel_type,
                        object=obj_normalized,
                        object_type=obj_ent.entity_type,
                        confidence=0.7,
                        source_text=text,
                        source_span=(match.start(), match.end()),
                        raw_predicate=raw_pred,
                    ))

        # Deduplicate relations by (subject, predicate, object)
        seen = set()
        unique_relations = []
        for rel in relations:
            key = (rel.subject, rel.predicate, rel.object)
            if key not in seen:
                seen.add(key)
                unique_relations.append(rel)

        return unique_relations

    def _find_entity_fuzzy(self, text: str, entities: List[ExtractedEntity]) -> Optional[ExtractedEntity]:
        """Find entity with fuzzy matching."""
        text_lower = text.lower()
        best_match = None
        best_score = 0

        for ent in entities:
            ent_lower = ent.text.lower()
            # Simple containment score
            if text_lower in ent_lower or ent_lower in text_lower:
                score = min(len(text_lower), len(ent_lower)) / max(len(text_lower), len(ent_lower))
                if score > best_score:
                    best_score = score
                    best_match = ent
            # Also check if entity text is part of the query text (for compound entities)
            elif ent_lower in text_lower:
                score = len(ent_lower) / len(text_lower)
                if score > best_score:
                    best_score = score
                    best_match = ent

        return best_match if best_score > 0.15 else None  # Lower threshold for query processing

    def _validate_relations(
        self,
        relations: List[ExtractedRelation],
        entities: List[ExtractedEntity],
    ) -> List[ExtractedRelation]:
        """Validate relations against type constraints."""
        valid_relations = []

        for rel in relations:
            # Check if subject/object types are compatible with relation
            constraints = RELATION_CONSTRAINTS.get(rel.predicate, [])
            if not constraints:
                # No constraints defined, allow
                valid_relations.append(rel)
                continue

            valid = False
            for subj_type, obj_type in constraints:
                if rel.subject_type == subj_type and rel.object_type == obj_type:
                    valid = True
                    break

            if valid:
                valid_relations.append(rel)
            else:
                logger.debug(f"Relation filtered: {rel.subject} ({rel.subject_type}) {rel.predicate} {rel.object} ({rel.object_type})")

        return valid_relations

    def _deduplicate_relations(self, relations: List[ExtractedRelation]) -> List[ExtractedRelation]:
        """Remove duplicate relations."""
        seen: Set[Tuple[str, str, str]] = set()
        unique = []
        for rel in relations:
            key = (rel.subject.lower(), rel.predicate.value, rel.object.lower())
            if key not in seen:
                seen.add(key)
                unique.append(rel)
        return unique

    def extract_batch(
        self,
        texts: List[str],
        entities_list: List[List[ExtractedEntity]],
        source_ids: Optional[List[str]] = None,
    ) -> List[List[ExtractedRelation]]:
        """Extract relations from multiple texts."""
        if source_ids is None:
            source_ids = [f"doc_{i}" for i in range(len(texts))]
        return [
            self.extract_relations(text, entities, sid)
            for text, entities, sid in zip(texts, entities_list, source_ids)
        ]