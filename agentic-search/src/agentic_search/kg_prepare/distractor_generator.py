"""
Distractor Generator Module for Knowledge Graph QA

Strategy 3: Shared Subpath Distractor Generation
- Find paths that share prefix with the main path
- Find paths that share intermediate nodes
- Generate distractors from divergent endpoints
- Ensure type consistency between answer and distractors
"""

import random
from typing import Optional, List, Dict, Any, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from .node_filter import classify_answer_type, is_same_answer_type, is_relation_valid
from .path_generator import KGPath, PathNode, PathEdge, PathConfig, PathGenerator


@dataclass
class DistractorConfig:
    """Configuration for distractor generation."""
    # Shared prefix options
    # 至少需要4步共同路径才能生成 distractor
    min_shared_prefix: int = 4
    max_shared_prefix: int = 5
    
    # Distractor count
    # 允许没有 distractor (min_distractors = 0)
    min_distractors: int = 0
    max_distractors: int = 4
    
    # Search depth for alternative paths
    max_divergent_hops: int = 3
    max_search_attempts: int = 50
    
    # Type matching strictness
    require_same_type: bool = True
    # Apply strict path filters (same as path generation)
    strict_path_filter: bool = False


@dataclass
class DistractorCandidate:
    """A candidate distractor answer."""
    title: str
    text: str
    answer_type: str
    shared_path: List[str]  # Nodes shared with main path
    divergent_path: List[str]  # Nodes after divergence
    divergence_point: str  # Node where paths diverge
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.title,
            "text": self.text[:300] if self.text else "",
            "answer_type": self.answer_type,
            "shared_nodes": self.shared_path,
            "divergent_nodes": self.divergent_path,
            "divergence_point": self.divergence_point
        }


@dataclass
class QAWithDistractors:
    """A QA pair with distractors."""
    main_path: KGPath
    correct_answer: str
    answer_type: str
    distractors: List[DistractorCandidate]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.main_path.to_dict(),
            "correct_answer": self.correct_answer,
            "answer_type": self.answer_type,
            "distractors": [d.to_dict() for d in self.distractors],
        }


class DistractorGenerator:
    """
    Generate distractors for KG paths using shared subpath strategy.
    
    Strategy 3: Shared Subpath Distractor Generation
    - Shared prefix: e→...→y' where first 1-2 hops match main path
    - Shared middle: paths that share intermediate nodes
    - Type consistency: distractors match correct answer type
    """
    
    def __init__(
        self,
        kilt_client: Any,
        config: Optional[DistractorConfig] = None
    ):
        self.client = kilt_client
        self.config = config or DistractorConfig()
    
    def _find_divergent_paths(
        self,
        start_node: str,
        visited: Set[str],
        target_type: str,
        max_hops: int
    ) -> List[Tuple[str, List[str]]]:
        """
        Find paths from start_node that lead to different endpoints.
        
        Returns list of (endpoint, path) tuples.
        """
        results = []
        queue = [(start_node, [start_node], 0)]
        seen = set(visited)
        seen.add(start_node.lower())
        
        while queue and len(results) < self.config.max_search_attempts:
            current, path, depth = queue.pop(0)
            
            if depth > 0:
                # Check if this could be an endpoint
                text = self.client.query_text(current)
                node_type = classify_answer_type(text, current)
                
                if is_same_answer_type(node_type, target_type) or not self.config.require_same_type:
                    results.append((current, path, text, node_type))
            
            if depth >= max_hops:
                continue
            
            # Explore neighbors
            relations = self.client.query_relations(current)
            random.shuffle(relations)
            
            for rel in relations[:10]:  # Limit exploration
                target_id = rel.get("href", "")
                target_label = rel.get("title", target_id)  # Use label for display
                anchor = rel.get("text", "")
                property_id = rel.get("property_id", "")
                
                if not target_id or target_id.lower() in seen:
                    continue
                
                filter_title = target_label if self.config.strict_path_filter else target_id
                if not is_relation_valid(
                    anchor,
                    filter_title,
                    property_id=property_id,
                    strict=self.config.strict_path_filter,
                    exclude_category_nodes=self.config.strict_path_filter,
                    exclude_type_lift_nodes=self.config.strict_path_filter
                ):
                    continue
                
                if not self.client.entity_exists(target_id):
                    continue
                
                seen.add(target_id.lower())
                # Store label in path for display
                queue.append((target_label, path + [target_label], depth + 1))
        
        return results
    
    def find_shared_prefix_distractors(
        self,
        main_path: KGPath,
        n_distractors: int = 3
    ) -> List[DistractorCandidate]:
        """
        Find distractors by exploring alternative paths from shared prefix.
        
        Strategy: Share first 1-2 hops of main path, then diverge.
        """
        if main_path.length < 2:
            return []
        
        distractors = []
        correct_answer = main_path.end_node
        correct_type = main_path.nodes[-1].answer_type
        main_nodes_set = set(n.lower() for n in main_path.node_titles)
        
        # Try different prefix lengths
        for prefix_len in range(self.config.min_shared_prefix, 
                                min(self.config.max_shared_prefix + 1, main_path.length)):
            
            # Get the divergence point (last node of shared prefix)
            divergence_node = main_path.nodes[prefix_len].title
            shared_path = main_path.node_titles[:prefix_len + 1]
            
            # Find alternative paths from divergence point
            visited = set(main_nodes_set)
            visited.add(correct_answer.lower())
            
            alternatives = self._find_divergent_paths(
                start_node=divergence_node,
                visited=visited,
                target_type=correct_type,
                max_hops=self.config.max_divergent_hops
            )
            
            for endpoint, path, text, node_type in alternatives:
                if endpoint.lower() == correct_answer.lower():
                    continue
                
                # Type matching check
                if self.config.require_same_type and not is_same_type(node_type, correct_type):
                    continue
                
                distractor = DistractorCandidate(
                    title=endpoint,
                    text=text,
                    answer_type=node_type,
                    shared_path=shared_path,
                    divergent_path=path[1:],  # Exclude divergence point
                    divergence_point=divergence_node
                )
                distractors.append(distractor)
                
                if len(distractors) >= n_distractors:
                    return distractors
        
        return distractors
    
    def find_shared_middle_distractors(
        self,
        main_path: KGPath,
        n_distractors: int = 3
    ) -> List[DistractorCandidate]:
        """
        Find distractors by exploring from middle nodes of main path.
        
        Strategy: Share intermediate node, different final hop.
        "Looks similar but last hop is different"
        """
        if main_path.length < 3:
            return []
        
        distractors = []
        correct_answer = main_path.end_node
        correct_type = main_path.nodes[-1].answer_type
        main_nodes_set = set(n.lower() for n in main_path.node_titles)
        
        # Try middle nodes (exclude first and last)
        middle_indices = list(range(1, main_path.length))
        random.shuffle(middle_indices)
        
        for idx in middle_indices:
            middle_node = main_path.nodes[idx].title
            shared_path = main_path.node_titles[:idx + 1]
            
            # Get alternative targets from middle node
            relations = self.client.query_relations(middle_node)
            random.shuffle(relations)
            
            for rel in relations:
                target_id = rel.get("href", "")
                target_label = rel.get("title", target_id)  # Use label for display
                anchor = rel.get("text", "")
                property_id = rel.get("property_id", "")
                
                if not target_id or target_id.lower() in main_nodes_set:
                    continue
                
                if target_label.lower() == correct_answer.lower():
                    continue
                
                filter_title = target_label if self.config.strict_path_filter else target_id
                if not is_relation_valid(
                    anchor,
                    filter_title,
                    property_id=property_id,
                    strict=self.config.strict_path_filter,
                    exclude_category_nodes=self.config.strict_path_filter,
                    exclude_type_lift_nodes=self.config.strict_path_filter
                ):
                    continue
                
                if not self.client.entity_exists(target_id):
                    continue
                
                # Check type - use target_id for query, target_label for classification
                text = self.client.query_text(target_id)
                node_type = classify_answer_type(text, target_label)
                
                if self.config.require_same_type and not is_same_answer_type(node_type, correct_type):
                    continue
                
                distractor = DistractorCandidate(
                    title=target_label,  # Use label for display
                    text=text,
                    answer_type=node_type,
                    shared_path=shared_path,
                    divergent_path=[target_label],  # Use label for display
                    divergence_point=middle_node
                )
                distractors.append(distractor)
                
                if len(distractors) >= n_distractors:
                    return distractors
        
        return distractors
    
    def generate_distractors(
        self,
        main_path: KGPath,
        n_distractors: int = None
    ) -> List[DistractorCandidate]:
        """
        Generate distractors for a main path using multiple strategies.
        
        Combines:
        - Shared prefix distractors
        - Shared middle distractors
        
        Returns deduplicated list of distractors.
        """
        n_distractors = n_distractors or self.config.max_distractors
        
        all_distractors = []
        seen_titles = set()
        
        # Strategy 1: Shared prefix
        prefix_distractors = self.find_shared_prefix_distractors(
            main_path, n_distractors
        )
        for d in prefix_distractors:
            if d.title.lower() not in seen_titles:
                all_distractors.append(d)
                seen_titles.add(d.title.lower())
        
        # Strategy 2: Shared middle (if we need more)
        if len(all_distractors) < n_distractors:
            middle_distractors = self.find_shared_middle_distractors(
                main_path, n_distractors - len(all_distractors)
            )
            for d in middle_distractors:
                if d.title.lower() not in seen_titles:
                    all_distractors.append(d)
                    seen_titles.add(d.title.lower())
        
        return all_distractors[:n_distractors]
    
    def create_qa_with_distractors(
        self,
        main_path: KGPath,
        n_distractors: int = None
    ) -> Optional[QAWithDistractors]:
        """
        Create a QA pair with distractors from a main path.
        
        Note: 如果路径长度不足 min_shared_prefix，可能无法生成 distractor。
        当 min_distractors = 0 时，即使没有 distractor 也会返回结果。
        """
        n_distractors = n_distractors or self.config.max_distractors
        
        # 只有路径长度足够时才尝试生成 distractor
        if main_path.length >= self.config.min_shared_prefix:
            distractors = self.generate_distractors(main_path, n_distractors)
        else:
            distractors = []
        
        # 如果 distractor 数量不足且 min_distractors > 0，返回 None
        if len(distractors) < self.config.min_distractors:
            return None
        
        return QAWithDistractors(
            main_path=main_path,
            correct_answer=main_path.end_node,
            answer_type=main_path.nodes[-1].answer_type,
            distractors=distractors
        )


def is_same_type(type1: str, type2: str) -> bool:
    """Wrapper for type comparison."""
    return is_same_answer_type(type1, type2)


class FullPipelineGenerator:
    """
    Complete pipeline: Generate paths + distractors.
    
    Combines Strategy 1 (path generation) and Strategy 3 (distractor generation).
    """
    
    def __init__(
        self,
        kilt_client: Any,
        path_config: Optional[PathConfig] = None,
        distractor_config: Optional[DistractorConfig] = None
    ):
        self.client = kilt_client
        self.path_generator = PathGenerator(kilt_client, path_config)
        self.distractor_generator = DistractorGenerator(kilt_client, distractor_config)
    
    def generate_qa_batch(
        self,
        n_seeds: int = 100,
        paths_per_seed: int = 3,
        n_distractors: int = 3
    ) -> List[QAWithDistractors]:
        """
        Generate QA pairs with distractors in batch.
        """
        # Generate paths
        paths = self.path_generator.generate_paths_batch(
            n_seeds=n_seeds,
            paths_per_seed=paths_per_seed
        )
        
        # Generate distractors for each path
        qa_pairs = []
        for path in paths:
            qa = self.distractor_generator.create_qa_with_distractors(
                path, n_distractors
            )
            if qa:
                qa_pairs.append(qa)
        
        return qa_pairs



