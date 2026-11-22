"""Relation extraction module for TrumorGPT."""

from trumourgpt.relation_extraction.extractor import (
    RelationExtractor,
    ExtractionConfig,
    ExtractedRelation,
    RelationType,
)

__all__ = [
    "RelationExtractor",
    "ExtractionConfig",
    "ExtractedRelation",
    "RelationType",
]