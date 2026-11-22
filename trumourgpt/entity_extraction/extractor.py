"""
Entity Extraction for TrumorGPT.
Uses few-shot LLM prompting to extract health entities from text.
Paper Section III-A, III-D: Entity extraction for semantic health knowledge graph.
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum
from loguru import logger

from trumourgpt.config.settings import get_settings
from trumourgpt.llm.client import SyncOllamaClient, OllamaClientConfig
from trumourgpt.llm.schemas import (
    ClaimDecomposition,
    AtomicClaim,
    EntityExtraction,
    EntityExtractionItem,
    TASK_SCHEMAS,
    TASK_MODELS,
)

# Entity name normalization mappings for matching query entities to KB entities
# (Copied from graphrag.py to avoid circular import)
ENTITY_NORMALIZATION = {
    # Vaccine variations
    "covid-19 vaccines": "COVID-19 vaccine",
    "covid-19 vaccine": "COVID-19 vaccine",
    "covid vaccines": "COVID-19 vaccine",
    "covid vaccine": "COVID-19 vaccine",
    "coronavirus vaccine": "COVID-19 vaccine",
    "coronavirus vaccines": "COVID-19 vaccine",
    "sars-cov-2 vaccine": "COVID-19 vaccine",
    "sars-cov-2 vaccines": "COVID-19 vaccine",
    "pfizer vaccine": "Comirnaty",
    "pfizer-biontech": "Pfizer-BioNTech",
    "moderna vaccine": "Spikevax",
    "mrna vaccine": "mRNA vaccine",
    "mrna vaccines": "mRNA vaccine",

    # Disease variations
    "covid-19": "COVID-19",
    "covid": "COVID-19",
    "coronavirus": "COVID-19",
    "sars-cov-2": "SARS-CoV-2",
    "sars cov 2": "SARS-CoV-2",
    "the coronavirus": "COVID-19",
    "the virus": "SARS-CoV-2",

    # Transmission/spread variations
    "the spread of covid-19": "transmission",
    "the spread of covid": "transmission",
    "the spread of the virus": "transmission",
    "spread of covid-19": "transmission",
    "spread of covid": "transmission",
    "spread of the virus": "transmission",
    "covid-19 transmission": "transmission",
    "covid transmission": "transmission",
    "virus transmission": "transmission",
    "spread": "transmission",
    "spreading": "transmission",

    # Prevention variations
    "preventing the spread": "prevents",
    "prevent spread": "prevents",
    "prevents the spread": "prevents",
    "effective at preventing": "prevents",
    "effective in preventing": "prevents",

    # Mask variations
    "masks": "mask",
    "face masks": "mask",
    "face mask": "mask",
    "surgical masks": "mask",
    "n95 masks": "mask",
    "mask wearing": "mask",
    "masking": "mask",

    # Infertility/fertility
    "infertility": "infertility",
    "fertility problems": "infertility",
    "fertility issues": "infertility",
    "reproductive problems": "infertility",

    # General health terms
    "vaccination": "vaccination",
    "vaccines": "vaccination",
    "vaccine": "vaccination",
    "immunization": "vaccination",
    "hospitalization": "hospitalization",
    "hospitalisation": "hospitalization",
    "death": "death",
    "mortality": "death",
    "long covid": "Long COVID",
    "long-haul covid": "Long COVID",
    "post-covid": "Long COVID",

    # Organizations
    "cdc": "CDC",
    "who": "WHO",
    "fda": "FDA",
    "nih": "NIH",

    # Variants
    "delta variant": "Delta",
    "omicron variant": "Omicron",
    "delta": "Delta",
    "omicron": "Omicron",

    # Symptoms
    "fever": "fever",
    "cough": "cough",
    "fatigue": "fatigue",
    "brain fog": "brain fog",

    # Flu/Influenza variations
    "flu": "flu vaccine",
    "influenza": "flu vaccine",
    "flu vaccine": "flu vaccine",
    "influenza vaccine": "flu vaccine",
    "flu shot": "flu vaccine",
    "flu vaccination": "flu vaccination",
    "annual flu vaccination": "flu vaccination",
    "flu illness": "influenza illness",
    "influenza illness": "influenza illness",
    "flu-related hospitalization": "flu-related hospitalization",

    # ACA/Healthcare reform variations
    "affordable care act": "Affordable Care Act",
    "aca": "ACA",
    "obamacare": "ACA",
    "obama care": "ACA",
    "health care law": "ACA",
    "healthcare reform": "ACA",
    "uninsured rate": "uninsured rate",
    "uninsured rates": "uninsured rate",
    "number of uninsured": "number of uninsured Americans",
    "uninsured americans": "number of uninsured Americans",

    # Medicaid variations
    "medicaid expansion": "Medicaid expansion",
    "medicaid": "Medicaid expansion",
    "expanded medicaid": "Medicaid expansion",
    "access to care": "access to care",
    "improved access": "access to care",
    "insurance coverage": "insurance coverage",

    # Mental health parity
    "mental health parity": "mental health parity",
    "mhpaea": "MHPAEA",
    "mental health parity law": "mental health parity",
    "equal coverage": "equal coverage",
    "parity": "equal coverage",

    # Childhood vaccines
    "childhood vaccines": "childhood vaccines",
    "childhood immunizations": "childhood vaccines",
    "child vaccines": "childhood vaccines",
    "vaccines prevent": "vaccines",
    "immunization saves": "immunization",
    "millions of deaths": "millions of deaths annually",

    # Vaccine safety / myths
    "autism": "autism",
    "cause autism": "autism",
    "vaccines cause autism": "autism",
    "mmr vaccine": "MMR vaccine",
    "mmr": "MMR vaccine",
    "thimerosal": "thimerosal",
    "ivermectin": "ivermectin",
    "microchips": "microchips",
    "tracking devices": "tracking devices",
    "microchip": "microchips",
    "tracking": "tracking devices",
    "mrna alter dna": "mRNA vaccines",
    "alter dna": "mRNA vaccines",
    "alter human dna": "mRNA vaccines",
    "dna": "mRNA vaccines",

    # Death panels
    "death panels": "death panels",
    "death panel": "death panels",

    # Medicare for All
    "medicare for all": "Medicare for All",
    "single payer": "Medicare for All",

    # Pre-existing conditions
    "pre-existing conditions": "pre-existing conditions",
    "preexisting conditions": "pre-existing conditions",
    "pre existing conditions": "pre-existing conditions",
    "covered under aca": "pre-existing conditions",

    # Healthcare costs
    "healthcare costs": "healthcare costs",
    "health care costs": "healthcare costs",
    "cost growth": "healthcare cost growth",
    "medicare spending": "Medicare spending",

    # Political figures
    "chabot": "Rep. Steve Chabot",
    "steve chabot": "Rep. Steve Chabot",
    "rep chabot": "Rep. Steve Chabot",
    "brat": "Rep. Dave Brat",
    "dave brat": "Rep. Dave Brat",
    "rep brat": "Rep. Dave Brat",
    "trump": "Donald Trump",
    "donald trump": "Donald Trump",
    "obama": "Barack Obama",
    "barack obama": "Barack Obama",
    "biden": "Joe Biden",
    "joe biden": "Joe Biden",
    "pelosi": "Nancy Pelosi",
    "nancy pelosi": "Nancy Pelosi",
    "mcconnell": "Mitch McConnell",
    "mitch mcconnell": "Mitch McConnell",
    "schumer": "Chuck Schumer",
    "chuck schumer": "Chuck Schumer",
    "ryan": "Paul Ryan",
    "paul ryan": "Paul Ryan",
    "boehner": "John Boehner",
    "john boehner": "John Boehner",
    "romney": "Mitt Romney",
    "mitt romney": "Mitt Romney",
    "sanders": "Bernie Sanders",
    "bernie sanders": "Bernie Sanders",
    "warren": "Elizabeth Warren",
    "elizabeth warren": "Elizabeth Warren",
    "harris": "Kamala Harris",
    "kamala harris": "Kamala Harris",
    "pence": "Mike Pence",
    "mike pence": "Mike Pence",
    "rubio": "Marco Rubio",
    "marco rubio": "Marco Rubio",
    "crist": "Charlie Crist",
    "charlie crist": "Charlie Crist",
    "toomey": "Pat Toomey",
    "pat toomey": "Pat Toomey",
    "runyan": "Jon Runyan",
    "jon runyan": "Jon Runyan",
    "shaheen": "Jeanne Shaheen",
    "jeanne shaheen": "Jeanne Shaheen",
    "mccaskill": "Claire McCaskill",
    "claire mccaskill": "Claire McCaskill",
    "edwards": "Donna Edwards",
    "donna edwards": "Donna Edwards",
    "schultz": "Ed Schultz",
    "ed schultz": "Ed Schultz",
    "handel": "Karen Handel",
    "karen handel": "Karen Handel",
    "mcbath": "Lucy McBath",
    "lucy mcbath": "Lucy McBath",

    # Political organizations
    "rnc": "Republican National Committee",
    "republican national committee": "Republican National Committee",
    "dnc": "Democratic National Committee",
    "democratic national committee": "Democratic National Committee",
    "gop": "Republican Party",
    "republican party": "Republican Party",
    "democratic party": "Democratic Party",
    "congress": "Congress",
    "senate": "Senate",
    "house of representatives": "House of Representatives",
    "house": "House of Representatives",

    # Political speech
    "ad": "ad",
    "advertisement": "ad",
    "campaign ad": "ad",
    "blames": "blames",
    "attacked": "attacks",
    "attacks": "attacks",
    "criticized": "criticizes",
    "criticizes": "criticizes",
    "accused": "accuses",
    "accuses": "accuses",
    "claimed": "claimed",
    "claims": "claimed",
    "said": "claimed",
    "stated": "claimed",
    "voted for": "voted for",
    "voted against": "voted against",
    "endorsed": "endorsed",
    "endorses": "endorsed",
    "supported": "supported",
    "supports": "supported",
    "gut": "gut",
    "gutting": "gut",
}


def normalize_entity_name(entity: str) -> str:
    """
    Normalize entity name for matching with KB entities.
    Applies lowercase, strips punctuation, applies known mappings.
    """
    if not entity:
        return entity

    # Lowercase and strip
    normalized = entity.lower().strip()

    # Remove articles and common words
    normalized = re.sub(r'^(the|a|an)\s+', '', normalized)
    normalized = re.sub(r'\s+(the|a|an)$', '', normalized)

    # Remove punctuation
    normalized = re.sub(r'[^\w\s-]', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    # Apply known mappings
    if normalized in ENTITY_NORMALIZATION:
        return ENTITY_NORMALIZATION[normalized]

    # Try partial matches for longer phrases
    for key, value in ENTITY_NORMALIZATION.items():
        if key in normalized or normalized in key:
            return value

    # Return original with proper capitalization for known entities
    # Check if original matches any known entity (case-insensitive)
    for key, value in ENTITY_NORMALIZATION.items():
        if entity.lower() == key.lower():
            return value

    return entity


class EntityType(str, Enum):
    """Health domain entity types."""
    DISEASE = "DISEASE"
    MEDICATION = "MEDICATION"
    VACCINE = "VACCINE"
    HEALTH_ORGANIZATION = "HEALTH_ORGANIZATION"
    PUBLIC_POLICY = "PUBLIC_POLICY"
    MEDICAL_CONDITION = "MEDICAL_CONDITION"
    QUANTITY = "QUANTITY"
    DATE = "DATE"
    LOCATION = "LOCATION"
    PERSON = "PERSON"
    VIRUS = "VIRUS"
    SYMPTOM = "SYMPTOM"
    TREATMENT = "TREATMENT"
    GENE = "GENE"
    PROTEIN = "PROTEIN"
    EXPOSURE = "EXPOSURE"
    THERAPY = "THERAPY"
    RISK_FACTOR = "RISK_FACTOR"
    RISK = "RISK"
    CONDITION = "CONDITION"
    TRANSMISSION = "TRANSMISSION"
    CLAIM = "CLAIM"
    STUDY = "STUDY"
    POLICY = "POLICY"
    BEHAVIOR = "BEHAVIOR"


@dataclass
class ExtractedEntity:
    """Extracted entity with metadata."""
    text: str
    entity_type: EntityType
    start: int
    end: int
    confidence: float
    source_text: str
    normalized_form: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "entity_type": self.entity_type.value,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "source_text": self.source_text,
            "normalized_form": self.normalized_form,
        }


@dataclass
class ExtractionConfig:
    """Configuration for entity extraction."""
    few_shot_examples: int = 5
    max_entities_per_text: int = 50
    min_confidence: float = 0.5
    use_llm: bool = True
    fallback_to_patterns: bool = True

    # Entity types to extract
    entity_types: List[EntityType] = field(default_factory=lambda: [
        EntityType.DISEASE, EntityType.MEDICATION, EntityType.VACCINE,
        EntityType.HEALTH_ORGANIZATION, EntityType.PUBLIC_POLICY,
        EntityType.MEDICAL_CONDITION, EntityType.QUANTITY, EntityType.DATE,
        EntityType.LOCATION, EntityType.VIRUS, EntityType.SYMPTOM,
        EntityType.TREATMENT,
    ])


# Stop words to filter out from entity extraction
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
    'as', 'is', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall',
    'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their', 'we', 'us', 'our',
    'he', 'him', 'his', 'she', 'her', 'hers', 'i', 'me', 'my', 'mine', 'you', 'your', 'yours',
    'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how',
    'if', 'then', 'than', 'so', 'such', 'very', 'too', 'also', 'just', 'only', 'not', 'no',
    'yes', 'any', 'all', 'some', 'each', 'every', 'both', 'either', 'neither',
    'into', 'onto', 'upon', 'over', 'under', 'above', 'below', 'between', 'among',
    'through', 'during', 'before', 'after', 'since', 'until', 'while', 'because',
    'there', 'here', 'where', 'when', 'how', 'why',
    'ad', 'ads', 'advertisement', 'commercial', 'campaign', 'speech', 'statement', 'claim',
    'accusation', 'allegation', 'charge', 'assertion', 'factcheck', 'fact-checking',
    'blames', 'blamed', 'blaming', 'attacks', 'attacked', 'attacking', 'criticizes', 'criticized', 'criticizing', 'misleads', 'misled', 'misleading',
    'accuses', 'accused', 'accusing', 'says', 'said', 'saying', 'tells', 'told', 'telling',
    'according', 'to', 'based', 'on', 'reported', 'reports', 'reporting',
    'claims', 'claimed', 'claiming',
    'for', 'from', 'about', 'over', 'regarding', 'concerning',
    'would', 'will', 'could', 'should', 'might', 'must',
    'save', 'cost', 'costs', 'costing', 'spike', 'spikes', 'spiking',
    'tie', 'ties', 'tying', 'tied', 'back', 'backer', 'backers',
    'err', 'errs', 'erred', 'erring', 'on', 'by',
    'repeal', 'repeals', 'repealed', 'repealing',
    'say', 'says', 'said', 'saying',
    'talk', 'talks', 'talked', 'talking',
    'vote', 'votes', 'voted', 'voting',
    'contradict', 'contradicts', 'contradicted', 'contradicting',
    'nation', 'nationwide', 'national',
}

# Few-shot examples for entity extraction
ENTITY_EXTRACTION_EXAMPLES = [
    {
        "text": "COVID-19 vaccines prevent severe disease caused by SARS-CoV-2.",
        "entities": [
            {"text": "COVID-19", "type": "DISEASE", "start": 0, "end": 8},
            {"text": "vaccines", "type": "VACCINE", "start": 9, "end": 17},
            {"text": "severe disease", "type": "MEDICAL_CONDITION", "start": 26, "end": 40},
            {"text": "SARS-CoV-2", "type": "VIRUS", "start": 48, "end": 58},
        ]
    },
    {
        "text": "The CDC recommends Pfizer and Moderna vaccines for everyone aged 6 months and older.",
        "entities": [
            {"text": "CDC", "type": "HEALTH_ORGANIZATION", "start": 4, "end": 7},
            {"text": "Pfizer", "type": "VACCINE", "start": 19, "end": 25},
            {"text": "Moderna", "type": "VACCINE", "start": 30, "end": 37},
            {"text": "vaccines", "type": "VACCINE", "start": 38, "end": 46},
            {"text": "6 months", "type": "QUANTITY", "start": 57, "end": 65},
        ]
    },
    {
        "text": "Governor Ron DeSantis signed legislation blocking vaccine mandates in Florida.",
        "entities": [
            {"text": "Ron DeSantis", "type": "PERSON", "start": 9, "end": 21},
            {"text": "legislation", "type": "PUBLIC_POLICY", "start": 22, "end": 33},
            {"text": "vaccine mandates", "type": "PUBLIC_POLICY", "start": 39, "end": 55},
            {"text": "Florida", "type": "LOCATION", "start": 59, "end": 66},
        ]
    },
    {
        "text": "Clinical trials show 95% efficacy for mRNA vaccines against symptomatic infection.",
        "entities": [
            {"text": "Clinical trials", "type": "TREATMENT", "start": 0, "end": 15},
            {"text": "95%", "type": "QUANTITY", "start": 19, "end": 22},
            {"text": "efficacy", "type": "MEDICAL_CONDITION", "start": 23, "end": 31},
            {"text": "mRNA vaccines", "type": "VACCINE", "start": 35, "end": 48},
            {"text": "symptomatic infection", "type": "DISEASE", "start": 52, "end": 71},
        ]
    },
    {
        "text": "Long COVID causes fatigue, brain fog, and shortness of breath in patients.",
        "entities": [
            {"text": "Long COVID", "type": "DISEASE", "start": 0, "end": 10},
            {"text": "fatigue", "type": "SYMPTOM", "start": 16, "end": 23},
            {"text": "brain fog", "type": "SYMPTOM", "start": 25, "end": 34},
            {"text": "shortness of breath", "type": "SYMPTOM", "start": 36, "end": 55},
            {"text": "patients", "type": "PERSON", "start": 59, "end": 67},
        ]
    },
]


def build_entity_extraction_prompt(text: str, examples: List[Dict], entity_types: List[EntityType]) -> str:
    """Build few-shot prompt for entity extraction."""
    type_list = ", ".join([t.value for t in entity_types])

    prompt = f"""Extract health-related entities from the text. Entity types: {type_list}.

Return a JSON object with "entities" array. Each entity has: text, type, start, end, confidence.

Examples:
"""
    for i, ex in enumerate(examples):
        prompt += f"\nExample {i+1}:\nText: {ex['text']}\nEntities: {json.dumps(ex['entities'])}\n"

    prompt += f"\nNow extract from:\nText: {text}\nEntities:"
    return prompt


class EntityExtractor:
    """Entity extractor using few-shot LLM prompting."""

    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.config = config or ExtractionConfig()
        self._client: Optional[SyncOllamaClient] = None
        self._examples = ENTITY_EXTRACTION_EXAMPLES[:self.config.few_shot_examples]

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

    def extract_entities(self, text: str, source_id: str = "unknown", use_patterns_only: bool = False) -> List[ExtractedEntity]:
        """Extract entities from text.

        Args:
            text: Source text
            source_id: Source identifier
            use_patterns_only: If True, skip LLM and use pattern-based extraction only (recommended for query processing)
        """
        entities = []

        if use_patterns_only:
            entities = self._extract_with_patterns(text, source_id)
        elif self.config.use_llm:
            try:
                entities = self._extract_with_llm(text, source_id)
            except Exception as e:
                logger.warning(f"LLM entity extraction failed: {e}")
                if self.config.fallback_to_patterns:
                    entities = self._extract_with_patterns(text, source_id)
        else:
            entities = self._extract_with_patterns(text, source_id)

        # Filter by confidence
        entities = [e for e in entities if e.confidence >= self.config.min_confidence]

        # Limit count
        entities = entities[:self.config.max_entities_per_text]

        # Deduplicate by text span
        entities = self._deduplicate_entities(entities)

        # Apply entity normalization
        for ent in entities:
            ent.normalized_form = normalize_entity_name(ent.text)

        return entities

    def _extract_with_llm(self, text: str, source_id: str) -> List[ExtractedEntity]:
        """Extract using LLM."""
        client = self._get_client()

        prompt = build_entity_extraction_prompt(
            text, self._examples, self.config.entity_types
        )

        # Use entity_extraction schema
        response = client.generate_structured("entity_extraction", prompt)

        logger.debug(f"LLM entity extraction response: success={response.success}, error={response.error}, data={response.data}")

        if not response.success or response.data is None:
            raise RuntimeError(f"LLM extraction failed: {response.error}")

        # Parse entities from response (response.data is a dict)
        entities = []
        data = response.data
        if isinstance(data, dict):
            entities_data = data.get('entities', [])
        else:
            entities_data = getattr(data, 'entities', [])

        logger.debug(f"Found {len(entities_data)} entities in response")
        for ent in entities_data:
            try:
                # Handle both dict and object
                if isinstance(ent, dict):
                    ent_text = ent.get('text', '').strip()
                    ent_type = ent.get('type', 'DISEASE')
                    ent_start = ent.get('start', 0)
                    ent_end = ent.get('end', 0)
                    ent_conf = ent.get('confidence', 0.8)
                else:
                    ent_text = ent.text.strip()
                    ent_type = ent.type
                    ent_start = ent.start
                    ent_end = ent.end
                    ent_conf = ent.confidence

                # Filter out stop words and empty text
                if not ent_text or ent_text.lower() in STOP_WORDS:
                    continue

                # Skip invalid entity types (don't default to DISEASE)
                try:
                    entity_type = EntityType(ent_type)
                except ValueError:
                    logger.debug(f"Skipping entity with invalid type: {ent_text} ({ent_type})")
                    continue

                entities.append(ExtractedEntity(
                    text=ent_text,
                    entity_type=entity_type,
                    start=ent_start,
                    end=ent_end,
                    confidence=ent_conf,
                    source_text=text,
                ))
            except (ValueError, AttributeError) as e:
                logger.debug(f"Skipping malformed entity: {e}")
                continue

        logger.debug(f"Returning {len(entities)} entities")
        return entities

    def _extract_with_patterns(self, text: str, source_id: str) -> List[ExtractedEntity]:
        """Fallback pattern-based extraction."""
        entities = []

        # Disease patterns
        disease_patterns = [
            (r'\b(?:COVID-?19|SARS-CoV-2|coronavirus|influenza|flu|diabetes|cancer|hypertension|asthma|pneumonia|tuberculosis|malaria|HIV|AIDS|hepatitis|ebola|zika|dengue|measles|mumps|rubella|polio|tetanus|diphtheria|pertussis|varicella|shingles|meningitis|encephalitis|sepsis)\b', EntityType.DISEASE),
            (r'\b(?:disease|syndrome|disorder|infection|illness|pathology)\b', EntityType.DISEASE),
        ]

        # Medical condition patterns (includes infertility, sterility, etc.)
        medical_condition_patterns = [
            (r'\b(?:infertility|sterility|reproductive\s+issues?|birth\s+defects?|miscarriage|stillbirth|condition|complication)\b', EntityType.MEDICAL_CONDITION),
            (r'\b(?:human\s+DNA|DNA)\b', EntityType.MEDICAL_CONDITION),
        ]

        # Vaccine/medication patterns
        med_patterns = [
            (r'\b(?:vaccine|vaccines|vaccination|immunization|shot|dose|booster)\b', EntityType.VACCINE),
            (r'\b(?:Pfizer|Moderna|AstraZeneca|Johnson\s*&?\s*Johnson|Novavax|Sinovac|Sinopharm|Covaxin|Sputnik)\b', EntityType.VACCINE),
            (r'\b(?:mRNA|adenovirus|vector|protein\s+subunit|inactivated|live\s+attenuated)\s+vaccine\b', EntityType.VACCINE),
            (r'\b(?:mRNA\s+vaccines?|COVID-19\s+vaccines?)\b', EntityType.VACCINE),
            (r'\b(?:medication|drug|medicine|treatment|therapy|antiviral|antibiotic|antibody|monoclonal)\b', EntityType.MEDICATION),
            (r'\b(?:Remdesivir|Paxlovid|Molnupiravir|Dexamethasone|Tocilizumab)\b', EntityType.MEDICATION),
        ]

        # Organization patterns
        org_patterns = [
            (r'\b(?:CDC|FDA|WHO|NIH|EMA|NHS|CDPH|HHS)\b', EntityType.HEALTH_ORGANIZATION),
            (r'\b(?:World Health Organization|Centers for Disease Control|Food and Drug Administration|National Institutes of Health)\b', EntityType.HEALTH_ORGANIZATION),
        ]

        # Healthcare policy organization patterns
        healthcare_policy_org_patterns = [
            (r'\b(?:CBO|KFF|CMS|ASPE|GAO|MACPAC|MedPAC)\b', EntityType.HEALTH_ORGANIZATION),
            (r'\b(?:Congressional Budget Office|Kaiser Family Foundation|Centers for Medicare and Medicaid Services|Assistant Secretary for Planning and Evaluation|Government Accountability Office|Medicaid and CHIP Payment and Access Commission|Medicare Payment Advisory Commission)\b', EntityType.HEALTH_ORGANIZATION),
            (r'\b(?:Department of Health and Human Services|HHS)\b', EntityType.HEALTH_ORGANIZATION),
            (r'\b(?:Supreme Court|SCOTUS|U\.S\. Supreme Court)\b', EntityType.HEALTH_ORGANIZATION),
        ]

        # Political organization patterns
        political_org_patterns = [
            (r'\b(?:RNC|DNC|GOP|Republican Party|Democratic Party|Congress|Senate|House of Representatives)\b', EntityType.HEALTH_ORGANIZATION),
            (r'\b(?:Republican National Committee|Democratic National Committee|Grand Old Party)\b', EntityType.HEALTH_ORGANIZATION),
        ]

        # Political figure patterns - expanded
        political_figure_patterns = [
            (r'\b(?:Obama|Trump|Biden|Clinton|Bush|Pelosi|McConnell|Schumer|Ryan|Boehner|Romney|Sanders|Warren|Harris|Pence|Chabot|Brat|Handel|McBath|Runyan|Shaheen|McCaskill|Edwards|Schultz|Toomey|Crist|Rubio|Shaheen|DeSantis|Fleming|Shea-Porter|Sullivan|Desantis)\b', EntityType.PERSON),
            (r'\b(?:Barack Obama|Donald Trump|Joe Biden|Hillary Clinton|Bill Clinton|George Bush|Nancy Pelosi|Mitch McConnell|Chuck Schumer|Paul Ryan|John Boehner|Mitt Romney|Bernie Sanders|Elizabeth Warren|Kamala Harris|Mike Pence|Steve Chabot|Dave Brat|Karen Handel|Lucy McBath|Jon Runyan|Jeanne Shaheen|Claire McCaskill|Donna Edwards|Ed Schultz|Pat Toomey|Charlie Crist|Marco Rubio|Ron DeSantis|John Fleming|Carol Shea-Porter|Jim Sullivan)\b', EntityType.PERSON),
        ]

        # Political speech/claim patterns
        political_speech_patterns = [
            (r'\b(?:ad|ads|advertisement|commercial|campaign|speech|statement|claim|accusation|allegation|charge|assertion)\b', EntityType.PUBLIC_POLICY),
            (r'\b(?:blames?|blamed|blaming|attacks?|attacked|attacking|criticizes?|criticized|criticizing|accuses?|accused|accusing)\b', EntityType.PUBLIC_POLICY),
            (r'\b(?:fact.?check|factcheck|politifact|fact-checking)\b', EntityType.PUBLIC_POLICY),
        ]

        # Healthcare legislation/act patterns
        healthcare_legislation_patterns = [
            (r'\b(?:Affordable Care Act|ACA|Obamacare)\b', EntityType.PUBLIC_POLICY),
            (r'\b(?:Medicare|Medicaid|CHIP|Children\'s Health Insurance Program)\b', EntityType.PUBLIC_POLICY),
            (r'\b(?:Social Security Act|Public Health Service Act|Health Insurance Portability and Accountability Act|HIPAA)\b', EntityType.PUBLIC_POLICY),
            (r'\b(?:American Rescue Plan|ARPA|Inflation Reduction Act|IRA|CARES Act)\b', EntityType.PUBLIC_POLICY),
            (r'\b(?:federal statute|federal law|congressional act|supreme court decision|supreme court ruling)\b', EntityType.PUBLIC_POLICY),
        ]

        # Healthcare policy concept patterns - expanded
        healthcare_policy_concept_patterns = [
            (r'\b(?:uninsured rate|uninsured rates|uninsurance rate|uninsurance rates)\b', EntityType.QUANTITY),
            (r'\b(?:Medicaid expansion|Medicaid eligibility|Medicaid enrollment)\b', EntityType.PUBLIC_POLICY),
            (r'\b(?:health insurance marketplace|marketplace|exchange|exchanges|healthcare\.gov)\b', EntityType.PUBLIC_POLICY),
            (r'\b(?:individual mandate|employer mandate|mandate penalty)\b', EntityType.PUBLIC_POLICY),
            (r'\b(?:essential health benefits|EHB)\b', EntityType.PUBLIC_POLICY),
            (r'\b(?:pre-?existing condition|pre-?existing conditions)\b', EntityType.MEDICAL_CONDITION),
            (r'\b(?:premium|premiums|premium tax credit|premium tax credits|premium subsidies|subsidy|subsidies|tax credit)\b', EntityType.QUANTITY),
            (r'\b(?:Premium tax credits?)\b', EntityType.PUBLIC_POLICY),
            (r'\b(?:federal poverty level|FPL|% of poverty|percent of poverty)\b', EntityType.QUANTITY),
            (r'\b(?:federal spending|federal funding|federal outlays|federal budget|CBO estimate|CBO projection|CBO score)\b', EntityType.QUANTITY),
            (r'\b(?:enrollment|enrollees|coverage|covered|uninsured|insured)\b', EntityType.QUANTITY),
            (r'\b(?:benchmark plan|silver plan|bronze plan|gold plan|platinum plan|catastrophic plan)\b', EntityType.PUBLIC_POLICY),
            (r'\b(?:cost-?sharing reduction|CSR|out-of-pocket|deductible|copay|coinsurance)\b', EntityType.QUANTITY),
            (r'\b(?:risk adjustment|reinsurance|risk corridor|payment model|delivery reform)\b', EntityType.PUBLIC_POLICY),
            # Healthcare cost/premium patterns
            (r'\b(?:health\s+care\s+premium|health\s+care\s+premiums|healthcare\s+premium|healthcare\s+premiums)\b', EntityType.QUANTITY),
            (r'\b(?:health\s+care\s+cost|health\s+care\s+costs|healthcare\s+cost|healthcare\s+costs)\b', EntityType.QUANTITY),
            (r'\b(?:health\s+care\s+talk|health\s+care\s+plan|health\s+care\s+reform)\b', EntityType.PUBLIC_POLICY),
            (r'\b(?:family\s+health\s+care\s+premium|family\s+health\s+care\s+cost|family\s+healthcare\s+premium|family\s+healthcare\s+cost)\b', EntityType.QUANTITY),
            (r'\b(?:spike\s+in\s+premium|spike\s+in\s+cost|spike\s+in\s+health\s+care)\b', EntityType.QUANTITY),
            (r'\b(?:state\s+and\s+local\s+aid|coronavirus\s+relief\s+bill|tax\s+cuts)\b', EntityType.PUBLIC_POLICY),
            (r'\b(?:medicare\s+for\s+all|single\s+payer|public\s+option)\b', EntityType.PUBLIC_POLICY),
            (r'\b(?:repeal|repealing|repealed)\b', EntityType.PUBLIC_POLICY),
            (r'\b(?:nation|nationwide|national)\b', EntityType.LOCATION),
            # Additional healthcare entities
            (r'\b(?:Hydroxychloroquine|hydroxychloroquine|HCQ)\b', EntityType.MEDICATION),
            (r'\b(?:chloroquine|chloroquine\s+phosphate)\b', EntityType.MEDICATION),
            (r'\b(?:ivermectin)\b', EntityType.MEDICATION),
            (r'\b(?:mask|masks|face\s+mask|surgical\s+mask|N95|respirator|ventilator|ventilators)\b', EntityType.TREATMENT),
            (r'\b(?:ventilation|oxygen\s+therapy|respiratory\s+support)\b', EntityType.TREATMENT),
            (r'\b(?:social\s+distancing|physical\s+distancing|distancing|lockdown|quarantine|isolation)\b', EntityType.PUBLIC_POLICY),
            (r'\b(?:contact\s+tracing|contact\s+trace)\b', EntityType.PUBLIC_POLICY),
            # NEW: Budget and legislative terms for political healthcare claims
            (r'\b(?:budget|budgets|appropriation|appropriations|spending bill|funding bill)\b', EntityType.PUBLIC_POLICY),
            (r'\b(?:health\s+care\s+bill|healthcare\s+bill|health\s+bill|medical\s+bill)\b', EntityType.PUBLIC_POLICY),
            (r'\b(?:ration|rationing|rationed)\b', EntityType.PUBLIC_POLICY),
            (r'\b(?:Republican|Democrat|Democratic|bipartisan|partisan)\b', EntityType.PUBLIC_POLICY),
            (r'\b(?:zombie|zombie\s+budget)\b', EntityType.PUBLIC_POLICY),
            (r'\b(?:health\s+care|healthcare)\b', EntityType.PUBLIC_POLICY),
            (r'\b(?:herd\s+immunity|herd\s+immunization)\b', EntityType.MEDICAL_CONDITION),
            (r'\b(?:asymptomatic|symptomatic|pre-?symptomatic)\b', EntityType.MEDICAL_CONDITION),
            (r'\b(?:long\s+COVID|long-?COVID|post-?COVID|post-COVID\s+condition)\b', EntityType.DISEASE),
            (r'\b(?:variant|variants|Delta|Omicron|Alpha|Beta|Gamma)\b', EntityType.VIRUS),
            (r'\b(?:mutations?|spike\s+protein)\b', EntityType.MEDICAL_CONDITION),
            (r'\b(?:mRNA|messenger\s+RNA|DNA|genetic\s+material|genome|genomes)\b', EntityType.MEDICAL_CONDITION),
            (r'\b(?:infertility|sterility|reproductive\s+issues?|birth\s+defects?|miscarriage|stillbirth)\b', EntityType.MEDICAL_CONDITION),
            (r'\b(?:blood\s+clots?|thrombosis|myocarditis|pericarditis|anaphylaxis)\b', EntityType.MEDICAL_CONDITION),
            (r'\b(?:VAERS|vaccine\s+adverse\s+event|adverse\s+event|side\s+effect|side\s+effects)\b', EntityType.MEDICAL_CONDITION),
            (r'\b(?:VA|Veterans\s+Affairs|Veterans\s+Administration)\b', EntityType.HEALTH_ORGANIZATION),
            (r'\b(?:hospital|hospitals|clinic|clinics|emergency\s+room|ICU|intensive\s+care)\b', EntityType.HEALTH_ORGANIZATION),
            (r'\b(?:doctor|doctors|physician|physicians|nurse|nurses|healthcare\s+worker|health\s+care\s+worker)\b', EntityType.PERSON),
            (r'\b(?:patient|patients|enrollee|enrollees|beneficiary|beneficiaries)\b', EntityType.PERSON),
            (r'\b(?:provider|providers|insurer|insurers|payer|payers)\b', EntityType.HEALTH_ORGANIZATION),
            (r'\b(?:rural\s+hospital|rural\s+health|community\s+health\s+center|FQHC)\b', EntityType.HEALTH_ORGANIZATION),
            (r'\b(?:telehealth|telemedicine|virtual\s+care)\b', EntityType.TREATMENT),
            (r'\b(?:opioid|opioids|fentanyl|heroin|overdose|addiction|substance\s+use\s+disorder)\b', EntityType.MEDICAL_CONDITION),
            (r'\b(?:mental\s+health|behavioral\s+health|psychiatry|psychology|depression|anxiety)\b', EntityType.MEDICAL_CONDITION),
            (r'\b(?:cancer|cancers|tumor|tumors|oncology|chemotherapy|radiation)\b', EntityType.DISEASE),
            (r'\b(?:heart\s+disease|cardiovascular|stroke|hypertension|high\s+blood\s+pressure)\b', EntityType.DISEASE),
            (r'\b(?:diabetes|diabetic|blood\s+sugar|insulin|glucose)\b', EntityType.DISEASE),
            (r'\b(?:Alzheimer|dementia|cognitive\s+decline|memory\s+loss)\b', EntityType.DISEASE),
        ]

        # Policy patterns
        policy_patterns = [
            (r'\b(?:mandate|mandates|mandatory|requirement|policy|legislation|law|regulation|guideline|recommendation|advisory|order|directive)\b', EntityType.PUBLIC_POLICY),
            (r'\b(?:vaccine\s+mandate|mask\s+mandate|lockdown|quarantine|isolation|distancing|restriction)\b', EntityType.PUBLIC_POLICY),
        ]

        # Quantity patterns
        quantity_patterns = [
            (r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|percent|percentage)\b', EntityType.QUANTITY),
            (r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|thousand|people|patients|cases|deaths)\b', EntityType.QUANTITY),
            (r'\b\d+(?:\.\d+)?\s*(?:mg|ml|mcg|IU|units?)\b', EntityType.QUANTITY),
        ]

        # Date patterns
        date_patterns = [
            (r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', EntityType.DATE),
            (r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', EntityType.DATE),
            (r'\b(?:2020|2021|2022|2023|2024|2025)\b', EntityType.DATE),
        ]

        # Location patterns
        location_patterns = [
            (r'\b(?:USA|US|United States|China|India|Brazil|Russia|UK|United Kingdom|Europe|Africa|Asia)\b', EntityType.LOCATION),
            (r'\b(?:California|Texas|Florida|New York|New Jersey|Illinois|Pennsylvania|Ohio|Georgia|North Carolina)\b', EntityType.LOCATION),
        ]

        # Virus patterns
        virus_patterns = [
            (r'\b(?:SARS-CoV-2|coronavirus|influenza\s+virus|HIV|HPV|HBV|HCV|HSV|VZV|RSV|MERS-CoV|SARS-CoV)\b', EntityType.VIRUS),
        ]

        # Symptom patterns
        symptom_patterns = [
            (r'\b(?:fever|cough|fatigue|shortness\s+of\s+breath|headache|nausea|vomiting|diarrhea|sore\s+throat|runny\s+nose|congestion|body\s+aches|chills|loss\s+of\s+taste|loss\s+of\s+smell)\b', EntityType.SYMPTOM),
        ]

        # Treatment/device patterns (masks, ventilators, etc.)
        treatment_patterns = [
            (r'\b(?:mask|masks|face\s+mask|surgical\s+mask|N95|respirator|ventilator|ventilators)\b', EntityType.TREATMENT),
            (r'\b(?:ventilation|oxygen\s+therapy|respiratory\s+support)\b', EntityType.TREATMENT),
        ]

        # Transmission patterns
        transmission_patterns = [
            (r'\b(?:spread|transmission|transmissibility|infectiousness|contagious)\b', EntityType.TRANSMISSION),
        ]

        all_patterns = (
            disease_patterns + medical_condition_patterns + med_patterns + org_patterns + healthcare_policy_org_patterns +
            political_org_patterns + political_figure_patterns + political_speech_patterns +
            healthcare_legislation_patterns + healthcare_policy_concept_patterns + policy_patterns +
            quantity_patterns + date_patterns + location_patterns + virus_patterns + symptom_patterns +
            treatment_patterns + transmission_patterns
        )

        for pattern, ent_type in all_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matched_text = match.group()
                # Filter out stop words
                if matched_text.lower() in STOP_WORDS:
                    continue
                entities.append(ExtractedEntity(
                    text=matched_text,
                    entity_type=ent_type,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.7,
                    source_text=text,
                ))

        return entities

    def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove duplicate entities by text span. Prefers longer spans for nested entities."""
        # Sort by start position, then by length (descending) so longer spans come first
        entities_sorted = sorted(entities, key=lambda e: (e.start, -(e.end - e.start)))

        unique = []
        for ent in entities_sorted:
            # Check if this entity is contained within an already added entity
            contained = False
            for existing in unique:
                if existing.start <= ent.start and existing.end >= ent.end:
                    # New entity is contained in existing, skip it
                    contained = True
                    break
            if not contained:
                unique.append(ent)

        # Sort by position
        unique.sort(key=lambda e: e.start)
        return unique

    def extract_batch(self, texts: List[str], source_ids: Optional[List[str]] = None) -> List[List[ExtractedEntity]]:
        """Extract entities from multiple texts."""
        if source_ids is None:
            source_ids = [f"doc_{i}" for i in range(len(texts))]
        return [self.extract_entities(text, sid) for text, sid in zip(texts, source_ids)]