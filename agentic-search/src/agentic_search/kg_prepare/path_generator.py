"""
Path Generator Module for Knowledge Graph

Strategy 1: Node-based Path Expansion
- Sample seed nodes meeting quality criteria
- Generate paths via random walk (2-4 hops)
- Avoid cycles and stoplist relations
- Control answer type at path endpoints
"""

import random
from typing import Optional, List, Dict, Any, Set, Tuple
from dataclasses import dataclass, field

from .node_filter import (
    NodeFilter, NodeFilterConfig,
    is_relation_valid, classify_answer_type,
    is_category_node, is_type_lift_node
)


@dataclass
class PathConfig:
    """Configuration for path generation."""
    min_hops: int = 2
    max_hops: int = 4
    max_attempts_per_node: int = 10
    max_paths_per_seed: int = 5
    max_answer_frequency: int = 5
    strict_path_filter: bool = False
    include_attributes: bool = False
    max_attributes_per_node: int = 8
    
    # Node filter config
    node_filter_config: NodeFilterConfig = field(default_factory=NodeFilterConfig)
    
    # Answer type constraints
    allowed_answer_types: Set[str] = field(default_factory=lambda: {
        "person", "organization", "location", "date", "number", "other"
    })
    strict_answer_types: Set[str] = field(default_factory=lambda: {
        "person", "organization", "location", "date", "number"
    })


@dataclass
class PathNode:
    """A node in a knowledge graph path."""
    title: str
    text: str = ""
    answer_type: str = "other"
    attributes: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        out = {
            "title": self.title,
            "text": self.text[:500] if self.text else "",
            "answer_type": self.answer_type
        }
        if self.attributes is not None:
            out["attributes"] = self.attributes
        return out


@dataclass
class PathEdge:
    """An edge (relation) in a knowledge graph path."""
    source: str
    target: str
    anchor_text: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.anchor_text
        }


@dataclass
class KGPath:
    """A path in the knowledge graph."""
    nodes: List[PathNode]
    edges: List[PathEdge]
    seed_node: str
    
    @property
    def start_node(self) -> str:
        return self.nodes[0].title if self.nodes else ""
    
    @property
    def end_node(self) -> str:
        return self.nodes[-1].title if self.nodes else ""
    
    @property
    def length(self) -> int:
        return len(self.edges)
    
    @property
    def node_titles(self) -> List[str]:
        return [n.title for n in self.nodes]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed_node": self.seed_node,
            "start_node": self.start_node,
            "end_node": self.end_node,
            "length": self.length,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "path": " -> ".join(self.node_titles)
        }


class PathGenerator:
    """
    Generate knowledge graph paths from seed nodes.
    
    Strategy 1: Node-based path expansion
    - Select high-quality seed nodes
    - Random walk to generate paths
    - Filter paths based on quality criteria
    """
    
    def __init__(
        self,
        client,
        config: Optional[PathConfig] = None
    ):
        self.client = client
        self.config = config or PathConfig()
        self.node_filter = NodeFilter(self.config.node_filter_config)
    
    def sample_seed_nodes(
        self,
        n: int = 100,
        min_degree: int = None,
        max_degree: int = None
    ) -> List[str]:
        """
        Sample high-quality seed nodes for path generation.
        
        Criteria:
        - Out-degree in specified range
        - Relation entropy above threshold
        - Label quality passes checks
        """
        min_degree = min_degree or self.config.node_filter_config.min_out_degree
        max_degree = max_degree or self.config.node_filter_config.max_out_degree
        
        # Get candidate entities by degree range
        candidates = self.client.get_entities_by_degree_range(
            min_degree=min_degree,
            max_degree=max_degree,
            limit=n * 5  # Over-sample to account for filtering
        )
        
        valid_seeds = []
        for entity in candidates:
            title = entity.get("wikipedia_title", "")
            out_degree = entity.get("anchor_count", 0)
            
            # Check label quality first (cheap check)
            if not self.node_filter.check_label_quality(title):
                continue

            if self.config.strict_path_filter:
                if is_category_node(title) or is_type_lift_node(title):
                    continue
            
            # Check entropy (more expensive)
            entropy = self.client.compute_relation_entropy(title)
            if not self.node_filter.check_entropy_threshold(entropy):
                continue
            
            valid_seeds.append(title)
            if len(valid_seeds) >= n:
                break
        
        return valid_seeds
    
    def _get_valid_next_nodes(
        self,
        current_title: str,
        visited: Set[str]
    ) -> List[Tuple[str, str, str]]:
        """
        Get valid next nodes from current node.
        
        Returns list of (target_id, target_label, anchor_text) tuples.
        target_id is the QID for internal operations, target_label is for display.
        """
        relations = self.client.query_relations(current_title)
        valid_next = []
        
        for rel in relations:
            target_id = rel.get("href", "")
            target_label = rel.get("title", target_id)  # Use label, fallback to ID
            anchor_text = rel.get("text", "")
            property_id = rel.get("property_id", "")
            
            if not target_id:
                continue
            
            # Skip visited nodes (prevent cycles) - use ID for deduplication
            if target_id.lower() in visited:
                continue
            
            # Check relation validity
            filter_title = target_label if self.config.strict_path_filter else target_id
            if not is_relation_valid(
                anchor_text,
                filter_title,
                property_id=property_id,
                strict=self.config.strict_path_filter,
                exclude_category_nodes=self.config.strict_path_filter,
                exclude_type_lift_nodes=self.config.strict_path_filter
            ):
                continue
            
            # Check if target exists in database
            if not self.client.entity_exists(target_id):
                continue
            
            valid_next.append((target_id, target_label, anchor_text))
        
        return valid_next

    def _get_node_attributes(self, title: str) -> Optional[List[Dict[str, Any]]]:
        if not self.config.include_attributes:
            return None
        getter = getattr(self.client, "get_attributes", None)
        if callable(getter):
            return getter(title, max_count=self.config.max_attributes_per_node)
        return []

    def _get_answer_count(
        self,
        answer: str,
        answer_counts: Dict[str, int],
        answer_count_lock: Any = None
    ) -> int:
        if answer_count_lock is not None:
            with answer_count_lock:
                return answer_counts.get(answer, 0)
        return answer_counts.get(answer, 0)

    def _try_reserve_answer_slot(
        self,
        answer: str,
        answer_counts: Dict[str, int],
        answer_count_lock: Any = None
    ) -> bool:
        if answer_count_lock is not None:
            with answer_count_lock:
                current = answer_counts.get(answer, 0)
                if current >= self.config.max_answer_frequency:
                    return False
                answer_counts[answer] = current + 1
                return True

        current = answer_counts.get(answer, 0)
        if current >= self.config.max_answer_frequency:
            return False
        answer_counts[answer] = current + 1
        return True
    
    def generate_path_from_seed(
        self,
        seed: str,
        target_hops: int = None,
        answer_counts: Optional[Dict[str, int]] = None,
        max_endpoint_resamples: int = 20,
        answer_count_lock: Any = None
    ) -> Optional[KGPath]:
        """
        Generate a single path starting from a seed node.
        
        Uses random walk with:
        - No repeated nodes (cycle prevention)
        - Valid relations only
        - Controlled path length
        
        Note: seed is already a human-readable label (from sample_seed_nodes).
        We track both the QID (for internal operations) and label (for output).
        """
        if target_hops is None:
            target_hops = random.randint(self.config.min_hops, self.config.max_hops)
        
        nodes = []
        edges = []
        visited = set()
        
        # Seed is already the label; get its entity data
        current_label = seed
        # For cycle detection, we use the label (lowercase) since seed is already a label
        current_id = seed  # Will be used for entity_exists and query_text
        visited.add(current_id.lower())
        
        # Get seed node info
        seed_text = self.client.query_text(current_id)
        seed_type = classify_answer_type(seed_text, current_label)
        seed_attrs = self._get_node_attributes(current_id)
        nodes.append(PathNode(
            title=current_label,
            text=seed_text,
            answer_type=seed_type,
            attributes=seed_attrs
        ))
        
        for hop in range(target_hops):
            valid_next = self._get_valid_next_nodes(current_id, visited)
            
            if not valid_next:
                break
            
            # Random selection - now returns (target_id, target_label, anchor_text)
            # On the final hop, enforce answer frequency cap with up to N resamples.
            if hop == target_hops - 1 and answer_counts is not None:
                selected_next = None
                for _ in range(max_endpoint_resamples):
                    candidate = random.choice(valid_next)
                    candidate_label = candidate[1]
                    count = self._get_answer_count(
                        candidate_label,
                        answer_counts,
                        answer_count_lock=answer_count_lock
                    )
                    if count < self.config.max_answer_frequency:
                        selected_next = candidate
                        break
                if selected_next is None:
                    return None
                next_id, next_label, anchor_text = selected_next
            else:
                next_id, next_label, anchor_text = random.choice(valid_next)
            
            # Get node info using QID (for database lookup)
            next_text = self.client.query_text(next_id)
            next_type = classify_answer_type(next_text, next_label)
            next_attrs = self._get_node_attributes(next_id)
            
            # Add edge and node using human-readable labels
            edges.append(PathEdge(
                source=current_label,
                target=next_label,
                anchor_text=anchor_text
            ))
            nodes.append(PathNode(
                title=next_label,
                text=next_text,
                answer_type=next_type,
                attributes=next_attrs
            ))
            
            # Update state - use QID for visited set (for cycle detection)
            visited.add(next_id.lower())
            current_id = next_id
            current_label = next_label
        
        # Check if path meets minimum length
        if len(edges) < self.config.min_hops:
            return None
        
        # Check if final answer type is allowed
        allowed_types = (
            self.config.strict_answer_types
            if self.config.strict_path_filter
            else self.config.allowed_answer_types
        )
        if nodes[-1].answer_type not in allowed_types:
            return None
        
        return KGPath(nodes=nodes, edges=edges, seed_node=seed)
    
    def generate_paths_from_seed(
        self,
        seed: str,
        n_paths: int = None,
        answer_counts: Optional[Dict[str, int]] = None,
        answer_count_lock: Any = None
    ) -> List[KGPath]:
        """Generate multiple paths from a single seed node."""
        n_paths = n_paths or self.config.max_paths_per_seed
        paths = []
        attempts = 0
        max_attempts = n_paths * self.config.max_attempts_per_node
        
        while len(paths) < n_paths and attempts < max_attempts:
            attempts += 1
            path = self.generate_path_from_seed(
                seed,
                answer_counts=answer_counts,
                answer_count_lock=answer_count_lock
            )
            if path:
                # Check for duplicate paths
                path_key = tuple(path.node_titles)
                if not any(tuple(p.node_titles) == path_key for p in paths):
                    if answer_counts is not None:
                        if not self._try_reserve_answer_slot(
                            path.end_node,
                            answer_counts,
                            answer_count_lock=answer_count_lock
                        ):
                            continue
                    paths.append(path)
        
        return paths
    
    def generate_paths_batch(
        self,
        n_seeds: int = 100,
        paths_per_seed: int = 3,
        answer_counts: Optional[Dict[str, int]] = None,
        answer_count_lock: Any = None
    ) -> List[KGPath]:
        """
        Generate paths from multiple seed nodes.
        
        Args:
            n_seeds: Number of seed nodes to sample
            paths_per_seed: Number of paths to generate per seed
            
        Returns:
            List of generated paths
        """
        seeds = self.sample_seed_nodes(n_seeds)
        all_paths = []
        
        for seed in seeds:
            paths = self.generate_paths_from_seed(
                seed,
                n_paths=paths_per_seed,
                answer_counts=answer_counts,
                answer_count_lock=answer_count_lock
            )
            all_paths.extend(paths)
        
        return all_paths



