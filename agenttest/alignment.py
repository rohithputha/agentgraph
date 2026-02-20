

from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from agenttest.models.llm_call_detail import LLMCallDetail


class AlignStatus(Enum):
    MATCHED = "matched"  
    ADDED = "added"     
    REMOVED = "removed" 


@dataclass
class AlignedPair:
    baseline_detail: Optional[LLMCallDetail]
    replay_detail: Optional[LLMCallDetail]
    status: AlignStatus
    baseline_index: Optional[int] = None
    replay_index: Optional[int] = None


def compute_lcs(seq1: List[str], seq2: List[str]) -> List[Tuple[int, int]]:
    m, n = len(seq1), len(seq2)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack to find the actual LCS matches
    matches = []
    i, j = m, n

    while i > 0 and j > 0:
        if seq1[i - 1] == seq2[j - 1]:
            matches.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    # Reverse to get correct order
    matches.reverse()

    return matches


def align_by_lcs(
    baseline_details: List[LLMCallDetail],
    replay_details: List[LLMCallDetail]
) -> List[AlignedPair]:

    baseline_fng = [d.fingerprint for d in baseline_details]
    replay_fng = [d.fingerprint for d in replay_details]

    lcs_matches = compute_lcs(baseline_fng, replay_fng)
    lcs_match_set = set(lcs_matches)

    baseline_to_replay = {b: r for b, r in lcs_matches}
    replay_to_baseline = {r: b for b, r in lcs_matches}

    aligned_pairs = []
    baseline_idx = 0
    replay_idx = 0

    while baseline_idx < len(baseline_details) or replay_idx < len(replay_details):
        
        if (baseline_idx, replay_idx) in lcs_match_set:
            # Matched pair
            aligned_pairs.append(AlignedPair(
                baseline_detail=baseline_details[baseline_idx],
                replay_detail=replay_details[replay_idx],
                status=AlignStatus.MATCHED,
                baseline_index=baseline_idx,
                replay_index=replay_idx
            ))
            baseline_idx += 1
            replay_idx += 1

        elif baseline_idx < len(baseline_details) and \
             (baseline_idx not in baseline_to_replay or baseline_to_replay[baseline_idx] < replay_idx):
            
            aligned_pairs.append(AlignedPair(
                baseline_detail=baseline_details[baseline_idx],
                replay_detail=None,
                status=AlignStatus.REMOVED,
                baseline_index=baseline_idx,
                replay_index=None
            ))
            baseline_idx += 1

        elif replay_idx < len(replay_details) and \
             (replay_idx not in replay_to_baseline or replay_to_baseline[replay_idx] < baseline_idx):
            
            aligned_pairs.append(AlignedPair(
                baseline_detail=None,
                replay_detail=replay_details[replay_idx],
                status=AlignStatus.ADDED,
                baseline_index=None,
                replay_index=replay_idx
            ))
            replay_idx += 1

        else:          
            baseline_match_pos = baseline_to_replay.get(baseline_idx, float('inf'))
            replay_match_pos = replay_to_baseline.get(replay_idx, float('inf'))

            if baseline_idx < len(baseline_details) and baseline_match_pos <= replay_match_pos:
                baseline_idx += 1
            elif replay_idx < len(replay_details):
                replay_idx += 1
            else:
                # Safety: advance both if we somehow get here
                if baseline_idx < len(baseline_details):
                    baseline_idx += 1
                if replay_idx < len(replay_details):
                    replay_idx += 1

    return aligned_pairs


def get_alignment_summary(aligned_pairs: List[AlignedPair]) -> dict:

    matched = sum(1 for p in aligned_pairs if p.status == AlignStatus.MATCHED)
    added = sum(1 for p in aligned_pairs if p.status == AlignStatus.ADDED)
    removed = sum(1 for p in aligned_pairs if p.status == AlignStatus.REMOVED)

    return {
        "total_pairs": len(aligned_pairs),
        "matched": matched,
        "added": added,
        "removed": removed
    }
