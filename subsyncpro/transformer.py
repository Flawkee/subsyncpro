"""Apply timing transforms to subtitle event lists.

Supports:
  - Constant offset  (shift all timestamps by N ms)
  - Linear transform (scale × time + offset, for frame-rate drift correction)

Both are lossless: the text and all metadata are preserved unchanged.
"""

from __future__ import annotations

from copy import deepcopy

from subsyncpro.aligner import AlignResult
from subsyncpro.parser import SubtitleEvent


def _transform_ms(t: int, scale: float, offset_ms: float) -> int:
    """Apply ref_time = scale * unsync_time + offset_ms, clamped to ≥ 0."""
    return max(0, round(scale * t + offset_ms))


def apply_transform(
    events: list[SubtitleEvent],
    result: AlignResult,
) -> list[SubtitleEvent]:
    """Return a new list of SubtitleEvents with timestamps adjusted.

    The original list is not modified.
    """
    scale = result.scale
    offset_ms = result.offset_ms
    synced: list[SubtitleEvent] = []
    for ev in events:
        new_ev = deepcopy(ev)
        new_ev.start_ms = _transform_ms(ev.start_ms, scale, offset_ms)
        new_ev.end_ms = _transform_ms(ev.end_ms, scale, offset_ms)
        # Guarantee minimum 1 ms duration
        if new_ev.end_ms <= new_ev.start_ms:
            new_ev.end_ms = new_ev.start_ms + max(1, ev.end_ms - ev.start_ms)
        synced.append(new_ev)
    # Re-sort in case scale caused reordering (shouldn't happen for scale≈1)
    synced.sort(key=lambda e: e.start_ms)
    return synced


def compute_delta_summary(
    original: list[SubtitleEvent],
    synced: list[SubtitleEvent],
) -> dict:
    """Return a human-readable summary of what changed."""
    if len(original) != len(synced):
        return {}
    deltas = [
        s.start_ms - o.start_ms
        for o, s in zip(original, synced)
    ]
    import statistics
    return {
        "min_delta_ms": min(deltas),
        "max_delta_ms": max(deltas),
        "mean_delta_ms": statistics.mean(deltas),
        "stdev_delta_ms": statistics.stdev(deltas) if len(deltas) > 1 else 0,
        "n_events": len(deltas),
    }
