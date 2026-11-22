"""
Text preprocessing for TrumorGPT.
Implements: sentence segmentation, Unicode normalization, whitespace cleanup,
entity-preserving tokenization, claim normalization, duplicate detection,
negation preservation, date and numerical expression preservation.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple, Dict, Any
from functools import lru_cache

import nltk
from loguru import logger

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing."""
    # Unicode normalization
    unicode_form: str = "NFKC"  # NFKC for compatibility decomposition

    # Whitespace
    normalize_whitespace: bool = True
    strip_leading_trailing: bool = True

    # Sentence segmentation
    sentence_tokenizer: str = "nltk"  # nltk or spacy
    min_sentence_length: int = 3
    max_sentence_length: int = 500

    # Entity preservation
    preserve_entities: bool = True
    entity_types: List[str] = field(default_factory=lambda: [
        "DISEASE", "MEDICATION", "VACCINE", "ORGANIZATION",
        "PERSON", "LOCATION", "DATE", "QUANTITY", "PERCENT"
    ])

    # Negation
    preserve_negation: bool = True
    negation_markers: List[str] = field(default_factory=lambda: [
        "not", "no", "never", "none", "nothing", "neither", "nor",
        "without", "lack", "absent", "denies", "refutes", "contradicts",
        "false", "incorrect", "wrong", "misleading", "unfounded"
    ])

    # Numerical expressions
    preserve_numerical: bool = True
    numerical_patterns: List[str] = field(default_factory=lambda: [
        r'\b\d+(?:,\d{3})*(?:\.\d+)?\b',  # Numbers with commas
        r'\b\d+(?:\.\d+)?%\b',             # Percentages
        r'\b\d+(?:\.\d+)?\s*(?:million|billion|trillion|thousand)\b',  # Large numbers
        r'\$\d+(?:,\d{3})*(?:\.\d+)?\b',   # Currency
    ])

    # Dates
    preserve_dates: bool = True
    date_patterns: List[str] = field(default_factory=lambda: [
        r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b',
        r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',
        r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
        r'\b(?:Q[1-4]\s+\d{4})\b',
        r'\b\d{4}\b',  # Year only
    ])

    # Claim normalization
    lowercase: bool = False  # Preserve case for entity recognition
    remove_urls: bool = True
    remove_emails: bool = True
    normalize_quotes: bool = True

    # Deduplication
    similarity_threshold: float = 0.95

    def __post_init__(self):
        # Compile regex patterns
        self._compiled_numerical = [re.compile(p, re.IGNORECASE) for p in self.numerical_patterns]
        self._compiled_dates = [re.compile(p, re.IGNORECASE) for p in self.date_patterns]
        self._negation_set = set(m.lower() for m in self.negation_markers)


class ProcessedText:
    """Container for preprocessed text with metadata."""

    def __init__(
        self,
        original: str,
        normalized: str,
        sentences: List[str],
        entities: List[Dict[str, Any]],
        negation_spans: List[Tuple[int, int]],
        numerical_expressions: List[Tuple[int, int, str]],
        dates: List[Tuple[int, int, str]],
        metadata: Dict[str, Any]
    ):
        self.original = original
        self.normalized = normalized
        self.sentences = sentences
        self.entities = entities
        self.negation_spans = negation_spans
        self.numerical_expressions = numerical_expressions
        self.dates = dates
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original": self.original,
            "normalized": self.normalized,
            "sentences": self.sentences,
            "entities": self.entities,
            "negation_spans": self.negation_spans,
            "numerical_expressions": self.numerical_expressions,
            "dates": self.dates,
            "metadata": self.metadata,
        }


class TextPreprocessor:
    """Main text preprocessing pipeline."""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self._sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self._seen_hashes: Set[str] = set()

    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode to NFKC form."""
        return unicodedata.normalize(self.config.unicode_form, text)

    def clean_whitespace(self, text: str) -> str:
        """Clean up whitespace while preserving structure."""
        if not self.config.normalize_whitespace:
            return text

        # Replace multiple spaces with single space
        text = re.sub(r'[ \t]+', ' ', text)
        # Replace multiple newlines with double newline (paragraph break)
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Strip leading/trailing
        if self.config.strip_leading_trailing:
            text = text.strip()
        return text

    def remove_urls_emails(self, text: str) -> str:
        """Remove URLs and email addresses."""
        if self.config.remove_urls:
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
        if self.config.remove_emails:
            text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        return text

    def normalize_quotes(self, text: str) -> str:
        """Normalize quote characters."""
        if not self.config.normalize_quotes:
            return text
        text = text.replace('“', '"').replace('”', '"')
        text = text.replace('‘', "'").replace('’', "'")
        return text

    def segment_sentences(self, text: str) -> List[str]:
        """Segment text into sentences using NLTK."""
        sentences = self._sentence_tokenizer.tokenize(text)

        # Filter by length
        filtered = []
        for sent in sentences:
            sent = sent.strip()
            if self.config.min_sentence_length <= len(sent) <= self.config.max_sentence_length:
                filtered.append(sent)

        return filtered

    def detect_negation(self, text: str) -> List[Tuple[int, int]]:
        """Detect negation spans in text."""
        if not self.config.preserve_negation:
            return []

        spans = []
        text_lower = text.lower()
        for marker in self.config._negation_set:
            # Find all occurrences
            start = 0
            while True:
                idx = text_lower.find(marker, start)
                if idx == -1:
                    break
                # Check word boundaries
                before_ok = idx == 0 or not text_lower[idx - 1].isalnum()
                after_idx = idx + len(marker)
                after_ok = after_idx >= len(text_lower) or not text_lower[after_idx].isalnum()
                if before_ok and after_ok:
                    spans.append((idx, after_idx))
                start = idx + 1

        # Merge overlapping spans
        if spans:
            spans.sort()
            merged = [spans[0]]
            for start, end in spans[1:]:
                if start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            return merged
        return []

    def extract_numerical_expressions(self, text: str) -> List[Tuple[int, int, str]]:
        """Extract numerical expressions with positions."""
        if not self.config.preserve_numerical:
            return []

        expressions = []
        for pattern in self.config._compiled_numerical:
            for match in pattern.finditer(text):
                expressions.append((match.start(), match.end(), match.group()))

        # Sort by position
        expressions.sort(key=lambda x: x[0])
        return expressions

    def extract_dates(self, text: str) -> List[Tuple[int, int, str]]:
        """Extract date expressions with positions."""
        if not self.config.preserve_dates:
            return []

        dates = []
        for pattern in self.config._compiled_dates:
            for match in pattern.finditer(text):
                dates.append((match.start(), match.end(), match.group()))

        dates.sort(key=lambda x: x[0])
        return dates

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using pattern matching (placeholder for NER)."""
        if not self.config.preserve_entities:
            return []

        entities = []

        # Simple pattern-based entity detection for health domain
        # In production, use spaCy NER or similar

        # Disease/condition patterns
        disease_patterns = [
            r'\b(?:COVID-?19|SARS-CoV-2|coronavirus|influenza|diabetes|cancer|hypertension|asthma)\b',
            r'\b(?:disease|syndrome|disorder|condition|infection|virus|bacteria)\b',
        ]

        # Medication/vaccine patterns
        med_patterns = [
            r'\b(?:vaccine|medication|drug|treatment|therapy|medicine|pill|tablet)\b',
            r'\b(?:Pfizer|Moderna|AstraZeneca|Johnson\s+&?\s*Johnson|Novavax)\b',
        ]

        # Organization patterns
        org_patterns = [
            r'\b(?:CDC|FDA|WHO|NIH|EMA|NHS)\b',
            r'\b(?:World Health Organization|Centers for Disease Control|Food and Drug Administration)\b',
        ]

        all_patterns = {
            "DISEASE": disease_patterns,
            "MEDICATION": med_patterns,
            "ORGANIZATION": org_patterns,
        }

        for entity_type, patterns in all_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entities.append({
                        "text": match.group(),
                        "type": entity_type,
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 0.8
                    })

        # Sort by position
        entities.sort(key=lambda x: x["start"])
        return entities

    def compute_hash(self, text: str) -> str:
        """Compute hash for deduplication."""
        import hashlib
        normalized = self.normalize_unicode(text)
        normalized = self.clean_whitespace(normalized)
        normalized = normalized.lower()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def is_duplicate(self, text: str) -> bool:
        """Check if text is duplicate of previously seen."""
        text_hash = self.compute_hash(text)
        if text_hash in self._seen_hashes:
            return True
        self._seen_hashes.add(text_hash)
        return False

    def reset_dedup(self) -> None:
        """Reset deduplication tracker."""
        self._seen_hashes.clear()

    def process(self, text: str, claim_id: Optional[str] = None) -> ProcessedText:
        """Full preprocessing pipeline."""
        original = text

        # Step 1: Unicode normalization
        text = self.normalize_unicode(text)

        # Step 2: Remove URLs/emails
        text = self.remove_urls_emails(text)

        # Step 3: Normalize quotes
        text = self.normalize_quotes(text)

        # Step 4: Clean whitespace
        text = self.clean_whitespace(text)

        normalized = text

        # Step 5: Extract metadata before sentence segmentation
        negation_spans = self.detect_negation(text)
        numerical_expressions = self.extract_numerical_expressions(text)
        dates = self.extract_dates(text)
        entities = self.extract_entities(text)

        # Step 6: Sentence segmentation
        sentences = self.segment_sentences(text)

        # Metadata
        metadata = {
            "claim_id": claim_id,
            "original_length": len(original),
            "normalized_length": len(normalized),
            "num_sentences": len(sentences),
            "num_entities": len(entities),
            "num_negations": len(negation_spans),
            "num_numerical": len(numerical_expressions),
            "num_dates": len(dates),
            "is_duplicate": self.is_duplicate(normalized) if claim_id else False,
        }

        return ProcessedText(
            original=original,
            normalized=normalized,
            sentences=sentences,
            entities=entities,
            negation_spans=negation_spans,
            numerical_expressions=numerical_expressions,
            dates=dates,
            metadata=metadata,
        )

    def process_batch(self, texts: List[str], claim_ids: Optional[List[str]] = None) -> List[ProcessedText]:
        """Process multiple texts."""
        if claim_ids is None:
            claim_ids = [None] * len(texts)
        return [self.process(text, cid) for text, cid in zip(texts, claim_ids)]


# Standalone functions for convenience
def normalize_unicode(text: str, form: str = "NFKC") -> str:
    return unicodedata.normalize(form, text)


def clean_whitespace(text: str) -> str:
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def segment_sentences(text: str, min_len: int = 3, max_len: int = 500) -> List[str]:
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(text)
    return [s.strip() for s in sentences if min_len <= len(s.strip()) <= max_len]


def preserve_entities(text: str) -> List[Dict[str, Any]]:
    """Extract entities using pattern matching."""
    entities = []
    patterns = {
        "DISEASE": [r'\b(?:COVID-?19|SARS-CoV-2|coronavirus|influenza|diabetes|cancer)\b'],
        "MEDICATION": [r'\b(?:vaccine|medication|drug|treatment|therapy)\b'],
        "ORGANIZATION": [r'\b(?:CDC|FDA|WHO|NIH)\b'],
    }
    for etype, pats in patterns.items():
        for pat in pats:
            for m in re.finditer(pat, text, re.IGNORECASE):
                entities.append({"text": m.group(), "type": etype, "start": m.start(), "end": m.end()})
    entities.sort(key=lambda x: x["start"])
    return entities


def detect_negation(text: str, markers: Optional[List[str]] = None) -> List[Tuple[int, int]]:
    if markers is None:
        markers = ["not", "no", "never", "none", "nothing", "without", "lack", "false", "incorrect"]
    spans = []
    text_lower = text.lower()
    for marker in markers:
        start = 0
        while True:
            idx = text_lower.find(marker, start)
            if idx == -1:
                break
            before_ok = idx == 0 or not text_lower[idx - 1].isalnum()
            after_idx = idx + len(marker)
            after_ok = after_idx >= len(text_lower) or not text_lower[after_idx].isalnum()
            if before_ok and after_ok:
                spans.append((idx, after_idx))
            start = idx + 1
    if spans:
        spans.sort()
        merged = [spans[0]]
        for s, e in spans[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        return merged
    return []


def extract_numerical_expressions(text: str) -> List[Tuple[int, int, str]]:
    patterns = [
        r'\b\d+(?:,\d{3})*(?:\.\d+)?\b',
        r'\b\d+(?:\.\d+)?%\b',
        r'\b\d+(?:\.\d+)?\s*(?:million|billion|trillion|thousand)\b',
    ]
    expressions = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            expressions.append((m.start(), m.end(), m.group()))
    expressions.sort(key=lambda x: x[0])
    return expressions


def extract_dates(text: str) -> List[Tuple[int, int, str]]:
    patterns = [
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',
        r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',
        r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
    ]
    dates = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            dates.append((m.start(), m.end(), m.group()))
    dates.sort(key=lambda x: x[0])
    return dates