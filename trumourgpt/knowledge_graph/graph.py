"""
Semantic Health Knowledge Graph for TrumorGPT.
Paper Section III-A: G = {E, R, F} where E=entities, R=relations, F=triples.
Each fact is a triple (h, r, t) with provenance tracking.
"""

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple, Iterator
from collections import defaultdict
from loguru import logger

import networkx as nx

from trumourgpt.config.settings import get_settings
from trumourgpt.entity_extraction.extractor import ExtractedEntity, EntityType
from trumourgpt.relation_extraction.extractor import ExtractedRelation, RelationType


@dataclass
class Triple:
    """Knowledge graph triple with provenance (Paper: (h, r, t))."""
    head: str
    relation: str
    tail: str
    head_type: str
    tail_type: str
    confidence: float
    source_id: str
    source_text: str
    source_url: Optional[str] = None
    extraction_method: str = "llm"
    timestamp: float = field(default_factory=lambda: __import__('time').time())

    def to_tuple(self) -> Tuple[str, str, str]:
        """Return as (head, relation, tail)."""
        return (self.head, self.relation, self.tail)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "head": self.head,
            "relation": self.relation,
            "tail": self.tail,
            "head_type": self.head_type,
            "tail_type": self.tail_type,
            "confidence": self.confidence,
            "source_id": self.source_id,
            "source_text": self.source_text,
            "source_url": self.source_url,
            "extraction_method": self.extraction_method,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Triple":
        return cls(**data)

    def __hash__(self):
        return hash((self.head.lower(), self.relation.lower(), self.tail.lower()))

    def __eq__(self, other):
        if not isinstance(other, Triple):
            return False
        return (self.head.lower() == other.head.lower() and
                self.relation.lower() == other.relation.lower() and
                self.tail.lower() == other.tail.lower())


@dataclass
class GraphStats:
    """Knowledge graph statistics."""
    num_entities: int
    num_relations: int
    num_triples: int
    entity_types: Dict[str, int]
    relation_types: Dict[str, int]
    sources: int
    avg_degree: float
    density: float
    connected_components: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_entities": self.num_entities,
            "num_relations": self.num_relations,
            "num_triples": self.num_triples,
            "entity_types": self.entity_types,
            "relation_types": self.relation_types,
            "sources": self.sources,
            "avg_degree": self.avg_degree,
            "density": self.density,
            "connected_components": self.connected_components,
        }


@dataclass
class KnowledgeGraphConfig:
    """Configuration for knowledge graph."""
    backend: str = "networkx"
    storage_path: str = "data/cache/knowledge_graph"

    # Entity and relation types
    entity_types: List[str] = field(default_factory=lambda: [
        "DISEASE", "MEDICATION", "VACCINE", "HEALTH_ORGANIZATION",
        "PUBLIC_POLICY", "MEDICAL_CONDITION", "QUANTITY", "DATE", "LOCATION",
        "PERSON", "VIRUS", "SYMPTOM", "TREATMENT"
    ])

    relation_types: List[str] = field(default_factory=lambda: [
        "causes", "prevents", "treats", "increases", "decreases",
        "associated_with", "approved_by", "recommended_by", "contradicts",
        "transmits", "symptom_of", "risk_factor_for", "contraindicated_for",
        "interacts_with", "reduces", "enhances", "inhibits", "activates",
        "binds_to", "expresses", "regulates"
    ])

    # Quality filters
    min_confidence: float = 0.5
    deduplicate: bool = True
    normalize_entities: bool = True

    # Few-shot for LLM-based KG construction
    few_shot_examples: int = 5


class SemanticHealthKnowledgeGraph:
    """
    Semantic Health Knowledge Graph (Paper Section III-A).
    G = {E, R, F} where E=entities, R=relations, F=triples.
    Each fact is a triple (h, r, t) with provenance.
    """

    def __init__(self, config: Optional[KnowledgeGraphConfig] = None):
        self.config = config or self._load_config()
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._entity_info: Dict[str, Dict[str, Any]] = {}
        self._triples: List[Triple] = []
        self._triple_index: Dict[Tuple[str, str, str], Triple] = {}
        self._source_index: Dict[str, List[Triple]] = defaultdict(list)

        # Initialize storage
        self._storage_path = Path(self.config.storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> KnowledgeGraphConfig:
        settings = get_settings()
        return KnowledgeGraphConfig(
            backend=settings.knowledge_graph.backend,
            storage_path=settings.knowledge_graph.storage_path,
            entity_types=settings.knowledge_graph.entity_types,
            relation_types=settings.knowledge_graph.relation_types,
            min_confidence=settings.knowledge_graph.min_confidence,
            deduplicate=settings.knowledge_graph.deduplicate,
        )

    def add_triple(self, triple: Triple) -> bool:
        """
        Add a triple to the knowledge graph.
        Returns True if added, False if duplicate or below confidence.
        """
        # Check confidence
        if triple.confidence < self.config.min_confidence:
            return False

        # Check duplicate
        triple_key = (triple.head.lower(), triple.relation.lower(), triple.tail.lower())
        if self.config.deduplicate and triple_key in self._triple_index:
            # Update confidence if higher
            existing = self._triple_index[triple_key]
            if triple.confidence > existing.confidence:
                self._update_triple(existing, triple)
            return False

        # Add to graph
        self._graph.add_edge(
            triple.head,
            triple.tail,
            key=triple.relation,
            relation=triple.relation,
            confidence=triple.confidence,
            source_id=triple.source_id,
            source_text=triple.source_text,
        )

        # Update entity info
        for entity, ent_type in [(triple.head, triple.head_type), (triple.tail, triple.tail_type)]:
            if entity not in self._entity_info:
                self._entity_info[entity] = {"type": ent_type, "count": 0}
            self._entity_info[entity]["count"] += 1

        # Store triple
        self._triples.append(triple)
        self._triple_index[triple_key] = triple
        self._source_index[triple.source_id].append(triple)

        return True

    def _update_triple(self, existing: Triple, new: Triple) -> None:
        """Update existing triple with new info."""
        existing.confidence = max(existing.confidence, new.confidence)
        # Update edge in graph
        for u, v, key, data in self._graph.edges(existing.head, data=True, keys=True):
            if v == existing.tail and data.get("relation") == existing.relation:
                data["confidence"] = existing.confidence
                break

    def add_triples_from_extraction(
        self,
        relations: List[ExtractedRelation],
        entities: List[ExtractedEntity],
        source_id: str,
        source_text: str,
        source_url: Optional[str] = None,
        use_raw_predicate: bool = False,
    ) -> int:
        """Add triples from extracted relations and entities.

        Args:
            relations: Extracted relations
            entities: Extracted entities
            source_id: Source identifier
            source_text: Source text
            source_url: Optional source URL
            use_raw_predicate: If True, use raw_predicate field instead of normalized predicate.
                              Useful for query KG building to preserve negation (e.g., "do not alter").
        """
        entity_map = {e.text.lower(): e for e in entities}
        added = 0

        for rel in relations:
            # Get entity types
            subj_ent = entity_map.get(rel.subject.lower())
            obj_ent = entity_map.get(rel.object.lower())

            head_type = subj_ent.entity_type.value if subj_ent else rel.subject_type.value
            tail_type = obj_ent.entity_type.value if obj_ent else rel.object_type.value

            # Use raw predicate if requested (for query processing to preserve negation)
            relation_str = rel.raw_predicate if (use_raw_predicate and rel.raw_predicate) else rel.predicate.value

            triple = Triple(
                head=rel.subject,
                relation=relation_str,
                tail=rel.object,
                head_type=head_type,
                tail_type=tail_type,
                confidence=rel.confidence,
                source_id=source_id,
                source_text=source_text,
                source_url=source_url,
            )

            if self.add_triple(triple):
                added += 1

        return added

    def get_triples(self, head: Optional[str] = None, relation: Optional[str] = None, tail: Optional[str] = None) -> List[Triple]:
        """Query triples by pattern."""
        results = []
        for triple in self._triples:
            if head and triple.head.lower() != head.lower():
                continue
            if relation and triple.relation.lower() != relation.lower():
                continue
            if tail and triple.tail.lower() != tail.lower():
                continue
            results.append(triple)
        return results

    def get_entity_triples(self, entity: str) -> List[Triple]:
        """Get all triples involving an entity."""
        return self.get_triples(head=entity) + self.get_triples(tail=entity)

    def get_neighbors(self, entity: str, relation: Optional[str] = None) -> List[Tuple[str, str, str]]:
        """Get neighbor entities."""
        neighbors = []
        if entity not in self._graph:
            return neighbors

        for _, neighbor, data in self._graph.out_edges(entity, data=True):
            rel = data.get("relation", "")
            if relation is None or rel == relation:
                neighbors.append((entity, rel, neighbor))

        for neighbor, _, data in self._graph.in_edges(entity, data=True):
            rel = data.get("relation", "")
            if relation is None or rel == relation:
                neighbors.append((neighbor, rel, entity))

        return neighbors

    def has_path(self, source: str, target: str, max_length: int = 3) -> bool:
        """Check if path exists between entities (Paper: short path indicates truth)."""
        try:
            return nx.has_path(self._graph, source, target) and \
                   nx.shortest_path_length(self._graph, source, target) <= max_length
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return False

    def shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find shortest path between entities."""
        try:
            return nx.shortest_path(self._graph, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def get_consecutive_triples(self) -> List[Tuple[str, str, str]]:
        """
        Get set of consecutive triples T_i = {(e1, r, e2) | path e1 -r-> e2 exists}.
        Used for GraphRAG Jaccard similarity (Paper Equation 5).
        """
        consecutive = set()
        for u, v, data in self._graph.edges(data=True):
            relation = data.get("relation", "")
            consecutive.add((u, relation, v))
        return list(consecutive)

    def compute_stats(self) -> GraphStats:
        """Compute graph statistics."""
        num_entities = self._graph.number_of_nodes()
        num_triples = len(self._triples)

        # Entity type distribution
        entity_types = defaultdict(int)
        for info in self._entity_info.values():
            entity_types[info["type"]] += 1

        # Relation type distribution
        relation_types = defaultdict(int)
        for triple in self._triples:
            relation_types[triple.relation] += 1

        # Sources
        sources = len(self._source_index)

        # Average degree
        if num_entities > 0:
            total_degree = sum(dict(self._graph.degree()).values())
            avg_degree = total_degree / num_entities
        else:
            avg_degree = 0.0

        # Density
        if num_entities > 1:
            max_edges = num_entities * (num_entities - 1)
            density = num_triples / max_edges if max_edges > 0 else 0.0
        else:
            density = 0.0

        # Connected components (undirected)
        undirected = self._graph.to_undirected()
        components = nx.number_connected_components(undirected)

        return GraphStats(
            num_entities=num_entities,
            num_relations=len(relation_types),
            num_triples=num_triples,
            entity_types=dict(entity_types),
            relation_types=dict(relation_types),
            sources=sources,
            avg_degree=avg_degree,
            density=density,
            connected_components=components,
        )

    def save(self, path: Optional[str] = None) -> None:
        """Save knowledge graph to disk."""
        path = Path(path or self._storage_path / "kg.pkl")

        data = {
            "triples": [t.to_dict() for t in self._triples],
            "entity_info": self._entity_info,
            "config": {
                "entity_types": self.config.entity_types,
                "relation_types": self.config.relation_types,
                "min_confidence": self.config.min_confidence,
            }
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Knowledge graph saved to {path} ({len(self._triples)} triples)")

    def load(self, path: Optional[str] = None) -> bool:
        """Load knowledge graph from disk."""
        path = Path(path or self._storage_path / "kg.pkl")

        if not path.exists():
            logger.warning(f"Knowledge graph file not found: {path}")
            return False

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            # Rebuild graph
            self._graph = nx.MultiDiGraph()
            self._entity_info = data.get("entity_info", {})
            self._triples = [Triple.from_dict(t) for t in data.get("triples", [])]
            self._triple_index = {}
            self._source_index = defaultdict(list)

            for triple in self._triples:
                self._graph.add_edge(
                    triple.head,
                    triple.tail,
                    key=triple.relation,
                    relation=triple.relation,
                    confidence=triple.confidence,
                    source_id=triple.source_id,
                    source_text=triple.source_text,
                )
                triple_key = (triple.head.lower(), triple.relation.lower(), triple.tail.lower())
                self._triple_index[triple_key] = triple
                self._source_index[triple.source_id].append(triple)

            logger.info(f"Knowledge graph loaded from {path} ({len(self._triples)} triples)")
            return True
        except Exception as e:
            logger.error(f"Failed to load knowledge graph: {e}")
            return False

    def export_json(self, path: Optional[str] = None) -> None:
        """Export as JSON for inspection."""
        path = Path(path or self._storage_path / "kg.json")

        data = {
            "nodes": [
                {"id": node, **info}
                for node, info in self._entity_info.items()
            ],
            "edges": [t.to_dict() for t in self._triples],
            "stats": self.compute_stats().to_dict(),
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Knowledge graph exported to {path}")

    def clear(self) -> None:
        """Clear the knowledge graph."""
        self._graph.clear()
        self._entity_info.clear()
        self._triples.clear()
        self._triple_index.clear()
        self._source_index.clear()
        logger.info("Knowledge graph cleared")

    def __len__(self) -> int:
        return len(self._triples)

    def __contains__(self, triple: Triple) -> bool:
        key = (triple.head.lower(), triple.relation.lower(), triple.tail.lower())
        return key in self._triple_index

    def __iter__(self) -> Iterator[Triple]:
        return iter(self._triples)


def create_kg_from_texts(
    texts: List[str],
    source_ids: List[str],
    entity_extractor,
    relation_extractor,
    config: Optional[KnowledgeGraphConfig] = None,
) -> SemanticHealthKnowledgeGraph:
    """
    Create knowledge graph from texts using extractors.
    Paper Section III-D: Few-shot LLM for KG construction.
    """
    kg = SemanticHealthKnowledgeGraph(config)
    total_relations = 0

    for text, source_id in zip(texts, source_ids):
        # Extract entities
        entities = entity_extractor.extract_entities(text, source_id)

        # Extract relations
        relations = relation_extractor.extract_relations(text, entities, source_id)

        # Add to KG
        added = kg.add_triples_from_extraction(
            relations, entities, source_id, text
        )
        total_relations += added

        logger.debug(f"Source {source_id}: {len(entities)} entities, {len(relations)} relations, {added} added to KG")

    logger.info(f"Created KG with {len(kg)} triples from {len(texts)} texts")
    return kg