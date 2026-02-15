import hashlib
import json
from typing import List, Dict, Any, Optional, Set
from difflib import SequenceMatcher
from agenttest.models.llm_call_detail import LLMCallDetail

from agenttest.models.comparison import *
from agenttest.alignment import align_by_lcs, AlignStatus
from agenttest.cascade import mark_cascades

class Comparison:
    def __init__(self, similarity_threshold: float = 0.85, ignore_fields: Optional[List[str]] = None, semantic_model = None):
        self.threshold = similarity_threshold
        self.ignore_fields = ignore_fields or []
        self.semantic_model = semantic_model

    def compare_recordings(self, baseline_details: List[LLMCallDetail], replay_details: List[LLMCallDetail]) -> ComparisonResult:
        aligned_pairs = align_by_lcs(baseline_details, replay_details)
        step_comparisons = []

        for i, pair in enumerate(aligned_pairs):
            step_comp = self._compare_pair(pair, i)
            step_comparisons.append(step_comp)

        total_steps = len(step_comparisons)
        matched_steps = sum(1 for sc in step_comparisons if sc.status == StepStatus.MATCH)
        mismatched_steps = sum(1 for sc in step_comparisons if sc.status == StepStatus.DIVERGE)
        added_steps = sum(1 for sc in step_comparisons if sc.status == StepStatus.ADD)
        removed_steps = sum(1 for sc in step_comparisons if sc.status == StepStatus.REMOVE)

        root_cause_index = None
        for i, sc in enumerate(step_comparisons):
            if sc.status in (StepStatus.DIVERGE, StepStatus.ADD, StepStatus.REMOVE):
                if root_cause_index is None:
                    root_cause_index = i
                    break

        mark_cascades(step_comparisons, root_cause_index)

        cascade_steps = sum(1 for sc in step_comparisons if sc.status == StepStatus.CASCADE)

        overall_pass = mismatched_steps == 0 and added_steps == 0 and removed_steps == 0

        return ComparisonResult(
            comparison_id=f"cmp_{hashlib.sha256(str(baseline_details[0].recording_id).encode()).hexdigest()[:12]}",
            baseline_recording_id=baseline_details[0].recording_id if baseline_details else "",
            replay_recording_id=replay_details[0].recording_id if replay_details else "",
            overall_pass=overall_pass,
            total_steps=total_steps,
            matched_steps=matched_steps,
            mismatched_steps=mismatched_steps,
            added_steps=added_steps,
            removed_steps=removed_steps,
            cascade_steps=cascade_steps,
            root_cause_index=root_cause_index,
            step_comparisons=step_comparisons
        )

    def _compare_pair(self, pair, si: int) -> StepComparison:
        if pair.status == AlignStatus.REMOVED:
            return StepComparison(
                step_index = si,
                baseline_node_id = pair.baseline_detail.node_id if pair.baseline_detail else None,
                replay_node_id =  None,
                status = StepStatus.REMOVE,
                baseline_detail_id=pair.baseline_detail.id if pair.baseline_detail else None,
                replay_detail_id=None,
                match_type=None,
                similarity_score=0,
                diff_summary = "Step removed in replay"
            )

        elif pair.status == AlignStatus.ADDED:
            return StepComparison(
                step_index = si,
                baseline_node_id = None,
                replay_node_id = pair.replay_detail.node_id if pair.replay_detail else None,
                status = StepStatus.ADD,
                baseline_detail_id = None,
                replay_detail_id = pair.replay_detail.id if pair.replay_detail else None,
                match_type = None,
                similarity_score = 0,
                diff_summary = "Step added in replay"
            )
        
        
        baseline = pair.baseline_detail
        replay = pair.replay_detail
        exact_score = self._exact_match(baseline, replay)
        if exact_score == 1:
            return StepComparison(
                step_index = si,
                baseline_node_id = baseline.node_id,
                replay_node_id = replay.node_id,
                status = StepStatus.MATCH,
                baseline_detail_id = baseline.id,
                replay_detail_id = replay.id,
                match_type = MatchType.EXACT,
                similarity_score = 1,
                diff_summary = None
            )

        structural_score = self._structural_similarity(baseline, replay)
        semantic_score = self._semantic_similarity(baseline.response_data, replay.response_data)

        combined_score = min(structural_score, semantic_score)

        if combined_score >= self.threshold:
            return StepComparison(
                step_index=si,
                baseline_node_id=baseline.node_id,
                replay_node_id=replay.node_id,
                status=StepStatus.MATCH,
                baseline_detail_id=baseline.id,
                replay_detail_id=replay.id,
                match_type=MatchType.SIMILAR,
                similarity_score=combined_score,
                diff_summary=None
            )

        return StepComparison(
            step_index=si,
            baseline_node_id=baseline.node_id,
            replay_node_id=replay.node_id,
            status=StepStatus.DIVERGE,
            baseline_detail_id=baseline.id,
            replay_detail_id=replay.id,
            match_type=None,
            similarity_score=combined_score,
            diff_summary=f"Similarity {combined_score:.2f} below threshold {self.threshold}"
        )


    def _exact_match(self, baseline: LLMCallDetail, replay: LLMCallDetail) -> float:
        if baseline.fingerprint != replay.fingerprint:
            return 0.0
        baseline_hash = self._hash_response(baseline.response_data)
        replay_hash = self._hash_response(replay.response_data)
        return 1.0 if baseline_hash == replay_hash else 0.0

    def _hash_response(self, response_data: Dict[str, Any]) -> str:
        canonical_json = json.dumps(response_data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()

    def _structural_similarity(self, baseline: LLMCallDetail, replay: LLMCallDetail) -> float:
        baseline_filter = self._filter_response(baseline.response_data)
        replay_filter = self._filter_response(replay.response_data)
        baseline_keys = self._extract_keys(baseline_filter)
        replay_keys = self._extract_keys(replay_filter)
        
        if not baseline_keys and not replay_keys:
            return 1
        
        if not baseline_keys or not replay_keys:
            return 0
        
        intersection = len(baseline_keys & replay_keys)
        union = len(baseline_keys | replay_keys)
        key_score = intersection / union if union > 0 else 0

        common_keys = baseline_keys & replay_keys
        if common_keys:
            
            type_m = sum(1 for k in common_keys if self._get_type(baseline_filter, k) == self._get_type(replay_filter, k))
            type_score = type_m / len(common_keys)
        else:
            type_score = 0
        
        return 0.6 * key_score + 0.4 * type_score

    def _semantic_similarity(self, baseline_resp: Dict[str, Any], replay_resp: Dict[str, Any]) -> float:
        baseline_text = self._extract_text(baseline_resp)
        replay_text = self._extract_text(replay_resp)

        if not baseline_text or not replay_text:
            return 0.0

        # Option 1: Sentence transformers (if model provided)
        if self.semantic_model:
            try:
                from sentence_transformers import util
                baseline_emb = self.semantic_model.encode(baseline_text)
                replay_emb = self.semantic_model.encode(replay_text)
                score = util.cos_sim(baseline_emb, replay_emb).item()
                return float(score)
            except Exception:
                pass  # Fall back to difflib

        # Option 2: Difflib (stdlib)
        return SequenceMatcher(None, baseline_text, replay_text).ratio()
        
        
    def _filter_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k : v for k, v in data.items() if k not in self.ignore_fields
        }

    def _extract_keys(self, data: Any, prefix: str = "") -> Set[str]:
        keys = set()
        if isinstance(data, dict):
            for k, v in data.items():
                key_path = f"{prefix}.{k}" if prefix else k
                keys.add(key_path)
                keys.update(self._extract_keys(v, key_path))
        elif isinstance(data, list):
            for i, v in enumerate(data):
                keys.update(self._extract_keys(v, f"{prefix}[{i}]"))

        return keys

    def _get_type(self, data: Dict[str, Any], key: str) -> str:
        try:
            value = data
            for part in key.split('.'):
                value = value[part]
            return type(value).__name__
        except (KeyError, TypeError):
            return "unknown"

    def _extract_text(self, data: Any) -> str:
        texts = []

        if isinstance(data, str):
            texts.append(data)
        elif isinstance(data, dict):
            # Extract from common text fields
            if "content" in data:
                texts.append(str(data["content"]))
            if "text" in data:
                texts.append(str(data["text"]))
            # Recurse for nested dicts
            for v in data.values():
                if isinstance(v, (dict, list)):
                    texts.append(self._extract_text(v))
        elif isinstance(data, list):
            for item in data:
                texts.append(self._extract_text(item))

        return " ".join(filter(None, texts))
