"""
Dataset handling for TrumorGPT.
Paper Section IV-B: PolitiFact dataset with 600 claims (300 TRUE, 300 FALSE).
Binary label mapping from 6-category PolitiFact ratings.
"""

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from enum import Enum
from loguru import logger


class PolitiFactLabel(str, Enum):
    """Original PolitiFact 6-category labels."""
    TRUE = "True"
    MOSTLY_TRUE = "Mostly True"
    HALF_TRUE = "Half True"
    MOSTLY_FALSE = "Mostly False"
    FALSE = "False"
    PANTS_ON_FIRE = "Pants on Fire"

    @classmethod
    def from_raw(cls, raw_label: str) -> "PolitiFactLabel":
        """Convert raw lowercase label (e.g., 'half-true') to enum value."""
        # Mapping from raw data format to enum values
        # Paper binary mapping: TRUE={True, Mostly True, Half True}, FALSE={Mostly False, False, Pants on Fire}
        mapping = {
            "true": cls.TRUE,
            "mostly-true": cls.MOSTLY_TRUE,
            "half-true": cls.HALF_TRUE,
            "barely-true": cls.MOSTLY_FALSE,  # Barely true = mostly false (paper mapping)
            "mostly-false": cls.MOSTLY_FALSE,
            "false": cls.FALSE,
            "pants-on-fire": cls.PANTS_ON_FIRE,
            "pants on fire": cls.PANTS_ON_FIRE,
            "pants-fire": cls.PANTS_ON_FIRE,  # Alias
            "full-flop": cls.FALSE,  # Full flop = false claim (flip-flop)
        }
        normalized = raw_label.lower().strip()
        if normalized in mapping:
            return mapping[normalized]
        # Try direct match (already capitalized)
        try:
            return cls(raw_label)
        except ValueError:
            raise ValueError(f"Unknown PolitiFact label: {raw_label}")


class BinaryLabel(str, Enum):
    """Binary labels (Paper mapping)."""
    TRUE = "TRUE"
    FALSE = "FALSE"


@dataclass
class LabelMapping:
    """Binary label mapping from PolitiFact 6 categories."""
    true_labels: List[PolitiFactLabel] = field(default_factory=lambda: [
        PolitiFactLabel.TRUE,
        PolitiFactLabel.MOSTLY_TRUE,
        PolitiFactLabel.HALF_TRUE,
    ])
    false_labels: List[PolitiFactLabel] = field(default_factory=lambda: [
        PolitiFactLabel.MOSTLY_FALSE,
        PolitiFactLabel.FALSE,
        PolitiFactLabel.PANTS_ON_FIRE,
    ])

    def to_binary(self, label: PolitiFactLabel) -> BinaryLabel:
        """Map PolitiFact label to binary."""
        if label in self.true_labels:
            return BinaryLabel.TRUE
        elif label in self.false_labels:
            return BinaryLabel.FALSE
        else:
            raise ValueError(f"Unknown label: {label}")

    @classmethod
    def from_config(cls, config: Dict[str, List[str]]) -> "LabelMapping":
        """Create from config dict."""
        return cls(
            true_labels=[PolitiFactLabel(l) for l in config.get("TRUE", [])],
            false_labels=[PolitiFactLabel(l) for l in config.get("FALSE", [])],
        )


@dataclass
class Claim:
    """Single claim with metadata."""
    claim_id: str
    text: str
    original_label: PolitiFactLabel
    binary_label: BinaryLabel
    category: str
    source_url: str
    publication_date: Optional[str] = None
    speaker: Optional[str] = None
    context: Optional[str] = None
    split: Optional[str] = None  # dev, val, test

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "original_label": self.original_label.value,
            "binary_label": self.binary_label.value,
            "category": self.category,
            "source_url": self.source_url,
            "publication_date": self.publication_date,
            "speaker": self.speaker,
            "context": self.context,
            "split": self.split,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], label_mapping: LabelMapping) -> "Claim":
        original = PolitiFactLabel.from_raw(data["original_label"])
        return cls(
            claim_id=data["claim_id"],
            text=data["text"],
            original_label=original,
            binary_label=label_mapping.to_binary(original),
            category=data["category"],
            source_url=data["source_url"],
            publication_date=data.get("publication_date"),
            speaker=data.get("speaker"),
            context=data.get("context"),
            split=data.get("split"),
        )


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    target_size: int = 600
    true_count: int = 300
    false_count: int = 300
    categories: List[str] = field(default_factory=lambda: ["Health Care", "Coronavirus"])
    splits: Dict[str, int] = field(default_factory=lambda: {
        "dev": 100, "val": 100, "test": 400
    })
    raw_path: str = "data/raw/politifact_claims.json"
    processed_path: str = "data/processed/politifact_processed.json"
    splits_path: str = "data/processed/splits.json"
    label_mapping: LabelMapping = field(default_factory=LabelMapping)


class PolitiFactDataset:
    """
    PolitiFact dataset loader and processor.
    Paper: "600 statements from the 'Health Care' and 'Coronavirus' categories on PolitiFact,
    evenly split with 300 true and 300 false claims."
    """

    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig()
        self.claims: List[Claim] = []
        self._splits: Dict[str, List[Claim]] = {}

    def load_raw(self, path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load raw PolitiFact data."""
        path = Path(path or self.config.raw_path)

        if not path.exists():
            logger.warning(f"Raw data not found at {path}. Creating sample data.")
            return self._create_sample_data()

        with open(path) as f:
            return json.load(f)

    def _create_sample_data(self) -> List[Dict[str, Any]]:
        """Create sample PolitiFact data for testing."""
        # Sample claims covering both categories
        sample_claims = [
            # TRUE claims
            {"text": "COVID-19 vaccines are effective at preventing severe disease.", "label": "True", "category": "Coronavirus"},
            {"text": "The CDC recommends COVID-19 vaccination for everyone 6 months and older.", "label": "True", "category": "Coronavirus"},
            {"text": "mRNA vaccines do not alter human DNA.", "label": "True", "category": "Coronavirus"},
            {"text": "The number of COVID-19 deaths in 2021 surpassed the 2020 total.", "label": "True", "category": "Coronavirus"},
            {"text": "Vaccines reduce the risk of long COVID.", "label": "Mostly True", "category": "Coronavirus"},
            {"text": "Flu vaccination reduces risk of flu illness by 40-60%.", "label": "True", "category": "Health Care"},
            {"text": "The Affordable Care Act reduced uninsured rates.", "label": "Mostly True", "category": "Health Care"},
            {"text": "Medicaid expansion improved access to care.", "label": "True", "category": "Health Care"},
            {"text": "Mental health parity law requires equal coverage.", "label": "True", "category": "Health Care"},
            {"text": "Childhood vaccines prevent millions of deaths annually.", "label": "True", "category": "Health Care"},

            # FALSE claims
            {"text": "COVID-19 vaccines cause infertility.", "label": "False", "category": "Coronavirus"},
            {"text": "mRNA vaccines alter human DNA.", "label": "Pants on Fire", "category": "Coronavirus"},
            {"text": "Florida had a vaccine mandate for everyone.", "label": "False", "category": "Coronavirus"},
            {"text": "Vaccines contain microchips for tracking.", "label": "Pants on Fire", "category": "Coronavirus"},
            {"text": "Ivermectin is a proven cure for COVID-19.", "label": "False", "category": "Coronavirus"},
            {"text": "The ACA created death panels.", "label": "Pants on Fire", "category": "Health Care"},
            {"text": "Obamacare caused healthcare costs to skyrocket.", "label": "Mostly False", "category": "Health Care"},
            {"text": "Medicare for All would bankrupt the country.", "label": "Half True", "category": "Health Care"},
            {"text": "Pre-existing conditions are not covered under ACA.", "label": "False", "category": "Health Care"},
            {"text": "Vaccines cause autism.", "label": "Pants on Fire", "category": "Health Care"},
        ]

        # Expand to target size by duplication with variation
        expanded = []
        for i, claim in enumerate(sample_claims):
            for j in range(max(1, self.config.target_size // len(sample_claims))):
                claim_id = f"{claim['category'].lower().replace(' ', '_')}_{claim['label'].lower().replace(' ', '_')}_{i}_{j}"
                expanded.append({
                    "claim_id": claim_id,
                    "text": claim["text"],
                    "original_label": claim["label"],
                    "category": claim["category"],
                    "source_url": f"https://www.politifact.com/factchecks/{claim_id}/",
                    "publication_date": "2024-01-01",
                    "speaker": "Various",
                    "context": f"PolitiFact {claim['category']} fact-check",
                })

        return expanded[:self.config.target_size]

    def process(self, raw_data: Optional[List[Dict]] = None) -> List[Claim]:
        """Process raw data into Claim objects."""
        if raw_data is None:
            raw_data = self.load_raw()

        claims = []
        for item in raw_data:
            try:
                # Ensure required fields
                if "claim_id" not in item:
                    text = item.get("text", "")
                    item["claim_id"] = hashlib.md5(text.encode()).hexdigest()[:12]

                if "original_label" not in item:
                    logger.warning(f"Missing label for claim {item.get('claim_id')}")
                    continue

                # Use from_raw to handle lowercase labels from dataset
                original_label = PolitiFactLabel.from_raw(item["original_label"])
                binary_label = self.config.label_mapping.to_binary(original_label)

                # Use context as claim text if available (concise claim statement),
                # otherwise fall back to text (full article)
                claim_text = item.get("context", "").strip()
                if not claim_text or len(claim_text) < 20:
                    claim_text = item["text"]

                claim = Claim(
                    claim_id=item["claim_id"],
                    text=claim_text,
                    original_label=original_label,
                    binary_label=binary_label,
                    category=item.get("category", "Health Care"),
                    source_url=item.get("source_url", ""),
                    publication_date=item.get("publication_date"),
                    speaker=item.get("speaker"),
                    context=item.get("context"),
                )
                claims.append(claim)

            except Exception as e:
                logger.warning(f"Failed to process claim: {e}")

        self.claims = claims
        logger.info(f"Processed {len(claims)} claims")

        # Create and save splits
        self.create_splits()
        self.save_splits()

        return claims

    def create_splits(self, claims: Optional[List[Claim]] = None, seed: int = 42) -> Dict[str, List[Claim]]:
        """Create train/dev/test splits with stratification."""
        import random
        random.seed(seed)

        claims = claims or self.claims
        if not claims:
            return {}

        # Separate by binary label
        true_claims = [c for c in claims if c.binary_label == BinaryLabel.TRUE]
        false_claims = [c for c in claims if c.binary_label == BinaryLabel.FALSE]

        # Shuffle
        random.shuffle(true_claims)
        random.shuffle(false_claims)

        # Calculate split sizes per class
        total_dev = self.config.splits.get("dev", 100)
        total_val = self.config.splits.get("val", 100)
        total_test = self.config.splits.get("test", 400)

        dev_per_class = total_dev // 2
        val_per_class = total_val // 2
        test_per_class = total_test // 2

        # Assign splits
        splits = {"dev": [], "val": [], "test": []}

        for label_claims, label in [(true_claims, "TRUE"), (false_claims, "FALSE")]:
            idx = 0
            for split_name, count in [("dev", dev_per_class), ("val", val_per_class), ("test", test_per_class)]:
                split_claims = label_claims[idx:idx + count]
                for claim in split_claims:
                    claim.split = split_name
                splits[split_name].extend(split_claims)
                idx += count

        # Shuffle splits
        for split_claims in splits.values():
            random.shuffle(split_claims)

        self._splits = splits

        # Save splits
        self.save_splits()

        logger.info(f"Created splits: dev={len(splits['dev'])}, val={len(splits['val'])}, test={len(splits['test'])}")
        return splits

    def get_split(self, split_name: str) -> List[Claim]:
        """Get claims for a specific split."""
        if not self._splits:
            self.create_splits()
        return self._splits.get(split_name, [])

    def save_processed(self, path: Optional[str] = None) -> None:
        """Save processed claims."""
        path = Path(path or self.config.processed_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [c.to_dict() for c in self.claims]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved processed claims to {path}")

    def load_processed(self, path: Optional[str] = None) -> List[Claim]:
        """Load processed claims."""
        path = Path(path or self.config.processed_path)

        if not path.exists():
            return self.process()

        with open(path) as f:
            data = json.load(f)

        claims = [Claim.from_dict(item, self.config.label_mapping) for item in data]
        self.claims = claims
        logger.info(f"Loaded {len(claims)} processed claims")

        # Load split assignments
        self.load_splits()

        return claims

    def save_splits(self, path: Optional[str] = None) -> None:
        """Save split assignments."""
        path = Path(path or self.config.splits_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {split: [c.claim_id for c in claims] for split, claims in self._splits.items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_splits(self, path: Optional[str] = None) -> Dict[str, List[str]]:
        """Load split assignments."""
        path = Path(path or self.config.splits_path)

        if not path.exists():
            return self.create_splits()

        with open(path) as f:
            data = json.load(f)

        # Reconstruct splits - handle claim_id type mismatch (float vs string)
        # Convert claim_ids to strings for consistent lookup
        claim_map = {str(c.claim_id): c for c in self.claims}
        self._splits = {}
        for split_name, claim_ids in data.items():
            # Ensure claim_ids are strings for lookup
            str_claim_ids = [str(cid) for cid in claim_ids]
            self._splits[split_name] = [claim_map[cid] for cid in str_claim_ids if cid in claim_map]
            for claim in self._splits[split_name]:
                claim.split = split_name

        return self._splits

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self.claims:
            return {}

        total = len(self.claims)
        true_count = sum(1 for c in self.claims if c.binary_label == BinaryLabel.TRUE)
        false_count = sum(1 for c in self.claims if c.binary_label == BinaryLabel.FALSE)

        categories = Counter(c.category for c in self.claims)
        original_labels = Counter(c.original_label.value for c in self.claims)

        splits = {}
        for split_name in ["dev", "val", "test"]:
            split_claims = self.get_split(split_name)
            splits[split_name] = {
                "total": len(split_claims),
                "true": sum(1 for c in split_claims if c.binary_label == BinaryLabel.TRUE),
                "false": sum(1 for c in split_claims if c.binary_label == BinaryLabel.FALSE),
            }

        return {
            "total_claims": total,
            "true_count": true_count,
            "false_count": false_count,
            "balance": true_count / total if total > 0 else 0,
            "categories": dict(categories),
            "original_labels": dict(original_labels),
            "splits": splits,
        }

    def check_leakage(self) -> Dict[str, Any]:
        """Check for data leakage between splits."""
        if not self._splits:
            self.create_splits()

        # Check for duplicate texts across splits
        split_texts = {}
        for split_name, claims in self._splits.items():
            texts = set(c.text.lower().strip() for c in claims)
            split_texts[split_name] = texts

        overlaps = {}
        split_names = list(split_texts.keys())
        for i, split1 in enumerate(split_names):
            for split2 in split_names[i+1:]:
                overlap = split_texts[split1] & split_texts[split2]
                if overlap:
                    overlaps[f"{split1}_vs_{split2}"] = len(overlap)

        return {
            "has_leakage": len(overlaps) > 0,
            "overlaps": overlaps,
        }

    def deduplicate(self, similarity_threshold: float = 0.95) -> int:
        """Remove near-duplicate claims."""
        from difflib import SequenceMatcher

        unique = []
        seen_texts = []
        removed = 0

        for claim in self.claims:
            is_dup = False
            for seen_text in seen_texts:
                similarity = SequenceMatcher(None, claim.text.lower(), seen_text).ratio()
                if similarity >= similarity_threshold:
                    is_dup = True
                    removed += 1
                    break

            if not is_dup:
                unique.append(claim)
                seen_texts.append(claim.text.lower())

        self.claims = unique
        logger.info(f"Deduplicated: removed {removed} claims, {len(unique)} remaining")
        return removed


from collections import Counter