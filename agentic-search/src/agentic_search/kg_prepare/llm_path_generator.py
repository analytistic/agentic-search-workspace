"""
LLM-led Path Generator Module for Knowledge Graph

Extends the base PathGenerator to use LLM for intelligent node selection
instead of random selection. This produces more semantically coherent
paths for generating difficult multi-hop reasoning questions.

Strategy: LLM-guided Path Expansion
- Sample seed nodes meeting quality criteria (same as base)
- At each hop, use LLM to select which relationship to expand
- LLM considers semantic coherence, information richness, and reasoning potential
"""

import os
import re
import json
import random
import time
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field

from jinja2 import Environment, FileSystemLoader

from .path_generator import PathGenerator, PathConfig, PathNode, PathEdge, KGPath
from .node_filter import NodeFilterConfig, classify_answer_type
from .llm_caller import LLMCaller


@dataclass
class LLMPathConfig(PathConfig):
    """Configuration for LLM-led path generation."""
    # LLM settings
    llm_model: str = "direct_dsr1"
    llm_max_retries: int = 3
    llm_timeout: int = 60
    
    # Fallback settings
    fallback_to_random: bool = True  # Fall back to random if LLM fails
    
    # Candidate limits
    max_candidates_for_llm: int = 10  # Max candidates to show to LLM
    
    # Rate limiting
    llm_call_delay: float = 0.05  # Small delay between LLM calls (seconds)


class LLMPathGenerator(PathGenerator):
    """
    LLM-led path generator that uses LLM to select which relationship to expand.
    
    Instead of random selection, the LLM considers:
    - Semantic coherence with current path
    - Information richness of target nodes
    - Potential for multi-hop reasoning
    """
    
    def __init__(
        self,
        kilt_client,
        config: Optional[LLMPathConfig] = None,
        template_path: Optional[str] = None,
        llm_trace_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """
        Initialize the LLM path generator.
        
        Args:
            kilt_client: Knowledge graph client (Wikidata, KILT, etc.)
            config: LLM path generation configuration
            template_path: Path to Jinja2 template file (optional)
        """
        self.llm_config = config or LLMPathConfig()
        super().__init__(kilt_client, self.llm_config)
        self._llm_trace_callback = llm_trace_callback
        
        # Initialize LLM caller
        self._llm_caller = None
        
        # Load Jinja2 template
        if template_path is None:
            template_dir = os.path.join(os.path.dirname(__file__), "templates")
            template_path = "node_select.jinja2"
        else:
            template_dir = os.path.dirname(template_path)
            template_path = os.path.basename(template_path)
        
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        self.template = self.env.get_template(template_path)
        
        # Statistics
        self.llm_calls = 0
        self.llm_failures = 0
        self.random_fallbacks = 0

    def _emit_llm_trace(self, payload: Dict[str, Any]) -> None:
        """Emit one LLM selection trace record if callback is configured."""
        if self._llm_trace_callback is None:
            return
        try:
            self._llm_trace_callback(payload)
        except Exception:
            # Trace logging must never interrupt generation.
            pass
    
    @property
    def llm_caller(self):
        """Lazy initialization of LLM caller."""
        if self._llm_caller is None:
            caller = LLMCaller()
            self._llm_caller = caller[self.llm_config.llm_model]
        return self._llm_caller
    
    def _format_candidates_for_llm(
        self,
        candidates: List[Tuple[str, str, str]],
        answer_counts: Optional[Dict[str, int]] = None,
        answer_count_lock: Any = None,
        enforce_answer_frequency: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Format candidate relationships for LLM prompt.
        
        Args:
            candidates: List of (target_id, target_label, anchor_text) tuples
            
        Returns:
            List of formatted candidate dicts with full node info
        """
        if enforce_answer_frequency and answer_counts is not None:
            candidates = [
                c for c in candidates
                if self._get_answer_count(c[1], answer_counts, answer_count_lock)
                < self.config.max_answer_frequency
            ]
            if not candidates:
                return []

        # Limit candidates if too many
        if len(candidates) > self.llm_config.max_candidates_for_llm:
            candidates = random.sample(candidates, self.llm_config.max_candidates_for_llm)
        
        formatted = []
        for target_id, target_label, anchor_text in candidates:
            # Get target node info
            target_text = self.client.query_text(target_id)
            target_attrs = self._get_node_attributes(target_id)
            
            formatted.append({
                "relation": anchor_text,
                "target_id": target_id,
                "target_title": target_label,
                "target_description": target_text[:300] if target_text else "",
                "target_attributes": target_attrs[:5] if target_attrs else None,
            })
        
        return formatted
    
    def _format_current_node(self, node: PathNode) -> Dict[str, Any]:
        """Format current node for LLM prompt."""
        return {
            "title": node.title,
            "description": node.text[:300] if node.text else "",
            "attributes": node.attributes[:5] if node.attributes else None,
        }
    
    def _parse_llm_selection(self, response: str, num_candidates: int) -> int:
        """
        Parse LLM response to extract selected candidate index.
        
        Args:
            response: LLM response text
            num_candidates: Number of candidates available
            
        Returns:
            0-based index of selected candidate, or -1 if the model rejected all candidates
            
        Raises:
            ValueError: If response cannot be parsed
        """
        if not response:
            raise ValueError("Empty LLM response")
        
        # Clean response
        response = response.strip()

        def _to_index(num: int) -> Optional[int]:
            if num == 0:
                return -1
            idx = num - 1
            if 0 <= idx < num_candidates:
                return idx
            return None

        # 1) Preferred: JSON response with {"think": "...", "answer": <number>}
        # Support outputs wrapped with <think>...</think> by parsing the segment after it first.
        json_candidates: List[str] = [response]
        if "</think>" in response:
            json_candidates.insert(0, response.rsplit("</think>", 1)[-1].strip())
        if "</thinking>" in response:
            json_candidates.insert(0, response.rsplit("</thinking>", 1)[-1].strip())

        for text in json_candidates:
            # Remove common markdown fences if present.
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE | re.DOTALL).strip()

            # Try full text as JSON.
            try:
                payload = json.loads(cleaned)
                if isinstance(payload, dict) and "answer" in payload:
                    ans = payload["answer"]
                    num = int(str(ans).strip())
                    idx = _to_index(num)
                    if idx is not None:
                        return idx
            except Exception:
                pass

            # Try first JSON object embedded in text.
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    payload = json.loads(match.group(0))
                    if isinstance(payload, dict) and "answer" in payload:
                        ans = payload["answer"]
                        num = int(str(ans).strip())
                        idx = _to_index(num)
                        if idx is not None:
                            return idx
                except Exception:
                    pass

        # 2) Backward-compatible fallback for non-JSON outputs.
        tail = response
        if "</think>" in response:
            tail = response.rsplit("</think>", 1)[-1].strip()
        elif "</thinking>" in response:
            tail = response.rsplit("</thinking>", 1)[-1].strip()

        if tail.isdigit():
            idx = _to_index(int(tail))
            if idx is not None:
                return idx

        lines = [line.strip() for line in tail.splitlines() if line.strip()]
        if lines:
            last_line = lines[-1]
            if last_line.isdigit():
                idx = _to_index(int(last_line))
                if idx is not None:
                    return idx

            end_match = re.search(
                r'(?:option|选项|answer|final answer|答案)?\s*[:：]?\s*(\d+)\s*$',
                last_line,
                re.IGNORECASE,
            )
            if end_match:
                idx = _to_index(int(end_match.group(1)))
                if idx is not None:
                    return idx

        nums = re.findall(r'\b(\d+)\b', tail)
        if nums:
            idx = _to_index(int(nums[-1]))
            if idx is not None:
                return idx
        
        raise ValueError(f"Cannot parse LLM response: {response[:100]}")
    
    def _llm_select_next_node(
        self,
        current_node: PathNode,
        candidates: List[Tuple[str, str, str]],
        path_history: Optional[str] = None,
        answer_counts: Optional[Dict[str, int]] = None,
        answer_count_lock: Any = None,
        enforce_answer_frequency: bool = False
    ) -> Optional[Tuple[str, str, str]]:
        """
        Use LLM to select which candidate relationship to expand.
        
        Args:
            current_node: Current node in the path
            candidates: List of (target_id, target_label, anchor_text) tuples
            path_history: Optional string describing path so far
            
        Returns:
            Selected (target_id, target_label, anchor_text) tuple, or None if LLM rejected all candidates
            
        Raises:
            RuntimeError: If LLM call fails and fallback is disabled
        """
        # Format candidates for LLM
        formatted_candidates = self._format_candidates_for_llm(
            candidates,
            answer_counts=answer_counts,
            answer_count_lock=answer_count_lock,
            enforce_answer_frequency=enforce_answer_frequency
        )
        if not formatted_candidates:
            return None
        
        # Create mapping from formatted candidates back to original tuples
        # Since we may have sampled, we need to track which originals correspond
        formatted_to_original = {}
        for fc in formatted_candidates:
            for orig in candidates:
                if orig[0] == fc["target_id"]:
                    formatted_to_original[fc["target_id"]] = orig
                    break
        
        # Format current node
        formatted_node = self._format_current_node(current_node)
        
        # Render prompt
        prompt = self.template.render(
            current_node=formatted_node,
            candidates=formatted_candidates,
            path_history=path_history,
        )
        
        # Call LLM
        self.llm_calls += 1
        messages = [{"role": "user", "content": prompt}]
        
        # Debug log
        import threading
        thread_id = threading.current_thread().name
        print(f"[{thread_id}] LLM call #{self.llm_calls}: selecting from {len(formatted_candidates)} candidates for node '{current_node.title}'", flush=True)
        
        try:
            call_start = time.time()
            response, _ = self.llm_caller.call_llm(
                messages,
                max_retry=self.llm_config.llm_max_retries
            )
            call_duration = time.time() - call_start
            print(f"[{thread_id}] LLM call completed in {call_duration:.1f}s", flush=True)
            
            # Parse response
            selected_idx = self._parse_llm_selection(response, len(formatted_candidates))
            
            # Check if LLM rejected all candidates
            if selected_idx == -1:
                self._emit_llm_trace({
                    "event": "llm_select",
                    "thread": thread_id,
                    "call_id": self.llm_calls,
                    "node_title": current_node.title,
                    "path_history": path_history,
                    "num_candidates": len(formatted_candidates),
                    "candidate_titles": [c["target_title"] for c in formatted_candidates],
                    "duration_sec": round(call_duration, 4),
                    "response": response,
                    "selected_index": -1,
                    "selected_target_id": None,
                    "selected_target_title": None,
                    "status": "rejected_all",
                })
                print(f"[{thread_id}] LLM rejected all candidates (selected 0)", flush=True)
                return None
            
            selected_target_id = formatted_candidates[selected_idx]["target_id"]
            selected_title = formatted_candidates[selected_idx]["target_title"]
            self._emit_llm_trace({
                "event": "llm_select",
                "thread": thread_id,
                "call_id": self.llm_calls,
                "node_title": current_node.title,
                "path_history": path_history,
                "num_candidates": len(formatted_candidates),
                "candidate_titles": [c["target_title"] for c in formatted_candidates],
                "duration_sec": round(call_duration, 4),
                "response": response,
                "selected_index": selected_idx,
                "selected_target_id": selected_target_id,
                "selected_target_title": selected_title,
                "status": "selected",
            })
            print(f"[{thread_id}] Selected: {selected_title}", flush=True)
            
            # Small delay for rate limiting
            if self.llm_config.llm_call_delay > 0:
                time.sleep(self.llm_config.llm_call_delay)
            
            return formatted_to_original[selected_target_id]
            
        except Exception as e:
            self.llm_failures += 1
            self._emit_llm_trace({
                "event": "llm_select",
                "thread": thread_id,
                "call_id": self.llm_calls,
                "node_title": current_node.title,
                "path_history": path_history,
                "num_candidates": len(formatted_candidates),
                "candidate_titles": [c["target_title"] for c in formatted_candidates],
                "response": None,
                "selected_index": None,
                "selected_target_id": None,
                "selected_target_title": None,
                "status": "error",
                "error": str(e),
            })
            print(f"[{thread_id}] LLM selection failed: {e}", flush=True)
            
            if self.llm_config.fallback_to_random:
                self.random_fallbacks += 1
                # Fall back to random selection from formatted candidates
                selected = random.choice(formatted_candidates)
                return formatted_to_original[selected["target_id"]]
            else:
                raise RuntimeError(f"LLM selection failed: {e}") from e
    
    def generate_path_from_seed(
        self,
        seed: str,
        target_hops: int = None,
        answer_counts: Optional[Dict[str, int]] = None,
        answer_count_lock: Any = None
    ) -> Optional[KGPath]:
        """
        Generate a path using LLM-guided node selection.
        
        Args:
            seed: Seed node (human-readable label)
            target_hops: Target path length (random if None)
            
        Returns:
            Generated KGPath or None if path doesn't meet criteria
        """
        if target_hops is None:
            target_hops = random.randint(self.config.min_hops, self.config.max_hops)
        
        nodes = []
        edges = []
        visited = set()
        path_titles = []  # For path history
        
        # Seed is already the label
        current_label = seed
        current_id = seed
        visited.add(current_id.lower())
        
        # Get seed node info
        seed_text = self.client.query_text(current_id)
        seed_type = classify_answer_type(seed_text, current_label)
        seed_attrs = self._get_node_attributes(current_id)
        seed_node = PathNode(
            title=current_label,
            text=seed_text,
            answer_type=seed_type,
            attributes=seed_attrs
        )
        nodes.append(seed_node)
        path_titles.append(current_label)
        
        llm_rejected = False  # Track if LLM rejected all candidates
        
        for hop in range(target_hops):
            valid_next = self._get_valid_next_nodes(current_id, visited)
            
            if not valid_next:
                break
            
            # Build path history string
            path_history = " -> ".join(path_titles)
            
            # Use LLM to select next node
            try:
                selection = self._llm_select_next_node(
                    nodes[-1],  # Current node
                    valid_next,
                    path_history=path_history,
                    answer_counts=answer_counts,
                    answer_count_lock=answer_count_lock,
                    enforce_answer_frequency=(
                        hop == target_hops - 1 and answer_counts is not None
                    )
                )
                
                # Check if LLM rejected all candidates
                if selection is None:
                    llm_rejected = True
                    break
                
                next_id, next_label, anchor_text = selection
                
            except RuntimeError:
                # If LLM fails and fallback is disabled, stop path generation
                break
            
            # Get node info
            next_text = self.client.query_text(next_id)
            next_type = classify_answer_type(next_text, next_label)
            next_attrs = self._get_node_attributes(next_id)
            
            # Add edge and node
            edges.append(PathEdge(
                source=current_label,
                target=next_label,
                anchor_text=anchor_text
            ))
            next_node = PathNode(
                title=next_label,
                text=next_text,
                answer_type=next_type,
                attributes=next_attrs
            )
            nodes.append(next_node)
            path_titles.append(next_label)
            
            # Update state
            visited.add(next_id.lower())
            current_id = next_id
            current_label = next_label
        
        # 如果 LLM 拒绝了所有候选项:
        # - 如果当前长度 < min_hops: 放弃这条链 (返回 None)
        # - 如果当前长度 >= min_hops: 到此结束 (返回当前路径)
        if llm_rejected:
            if len(edges) < self.config.min_hops:
                import threading
                thread_id = threading.current_thread().name
                print(f"[{thread_id}] Path abandoned: LLM rejected at length {len(edges)} < min_hops {self.config.min_hops}", flush=True)
                return None
            else:
                import threading
                thread_id = threading.current_thread().name
                print(f"[{thread_id}] Path ended early: LLM rejected at length {len(edges)} >= min_hops {self.config.min_hops}", flush=True)
        
        # Check minimum length
        if len(edges) < self.config.min_hops:
            return None
        
        # Check final answer type
        allowed_types = (
            self.config.strict_answer_types
            if self.config.strict_path_filter
            else self.config.allowed_answer_types
        )
        if nodes[-1].answer_type not in allowed_types:
            return None
        
        return KGPath(nodes=nodes, edges=edges, seed_node=seed)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get LLM usage statistics."""
        return {
            "llm_calls": self.llm_calls,
            "llm_failures": self.llm_failures,
            "random_fallbacks": self.random_fallbacks,
            "success_rate": (
                (self.llm_calls - self.llm_failures) / self.llm_calls
                if self.llm_calls > 0 else 0.0
            ),
        }
