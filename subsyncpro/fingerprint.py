"""Convert subtitle event lists into binary time-series fingerprints.

The fingerprint is a 1-D numpy array where each sample represents a fixed time
window (resolution_ms).  A value of 1 means at least one subtitle was active
during that window; 0 means silence.

Resolution: 33 ms (≈ one video frame at 30 fps).  This is fine-grained enough
for accurate alignment while keeping arrays small enough for fast FFT math.
"""

from __future__ import annotations

import numpy as np

from subsyncpro.parser import SubtitleEvent

RESOLUTION_MS: int = 33  # milliseconds per sample


def make_fingerprint(
    events: list[SubtitleEvent],
    duration_ms: int,
    resolution_ms: int = RESOLUTION_MS,
) -> np.ndarray:
    """Return a float32 array of shape (n_samples,) using onset-only impulses.

    Each subtitle event contributes a single +1 at its start sample.

    Onset impulses are always sparse regardless of subtitle density, making
    cross-correlation reliable even for dialogue-heavy content.  Using only
    positive values avoids the negative cross-correlation artefacts that occur
    with bipolar (+1/-1) fingerprints when one subtitle merges events that
    appear separately in the other (common in translations).
    """
    n = max(1, (duration_ms + resolution_ms - 1) // resolution_ms)
    fp = np.zeros(n, dtype=np.float32)

    for ev in events:
        i_start = max(0, ev.start_ms // resolution_ms)
        if i_start < n:
            fp[i_start] += 1.0

    return fp


def fingerprint_duration_ms(events: list[SubtitleEvent], padding_ms: int = 120_000) -> int:
    """Return an appropriate fingerprint length for a set of events."""
    if not events:
        return padding_ms
    return max(e.end_ms for e in events) + padding_ms


def speech_density(fp: np.ndarray) -> float:
    """Fraction of samples that have subtitle content."""
    return float(fp.mean())


def make_binary_fingerprint(
    events: list[SubtitleEvent],
    duration_ms: int,
    resolution_ms: int = RESOLUTION_MS,
) -> np.ndarray:
    """Return a float32 binary occupancy array: 1 where subtitle is active, 0 elsewhere.

    Unlike onset impulses, this captures the full speech-activity pattern (when
    dialogue is on screen).  It is more robust for cross-language alignment where
    event boundaries differ because translations merge or split events differently.
    """
    n = max(1, (duration_ms + resolution_ms - 1) // resolution_ms)
    fp = np.zeros(n, dtype=np.float32)
    for ev in events:
        i_start = max(0, ev.start_ms // resolution_ms)
        i_end = min(n, (ev.end_ms + resolution_ms - 1) // resolution_ms)
        if i_start < i_end:
            fp[i_start:i_end] = 1.0
    return fp


def window_has_content(window: np.ndarray, min_events: int = 2) -> bool:
    """True if the window contains enough subtitle onset events for reliable matching."""
    return int(window.sum()) >= min_events
