"""Core subtitle alignment engine.

Pipeline
--------
1. Windowed anchor search   — slide 30 s windows over unsync_fp; for each,
                              find the best-matching position anywhere in ref_fp
                              using FFT cross-correlation.
2. RANSAC                   — robustly fit ref_ms = scale*unsync_ms + offset
                              to the (unsync_ms, ref_ms) anchor pairs, rejecting
                              outliers.
3. Confidence scoring       — inlier fraction + anchor count.

Why this beats a plain cross-correlation (ffsubsync's approach)
---------------------------------------------------------------
Global cross-correlation finds the *one* offset that maximises total overlap.
When the reference contains content not in the unsync file ("Previously on…",
extra episodes, bonus scenes), that extra content pulls the global peak to the
wrong position.

The windowed approach is immune: each short window independently finds where it
best matches in the reference.  Outlier windows (those matching extra content)
produce inconsistent offset estimates and are discarded by RANSAC.  The dominant
cluster — the true episode content — survives, giving the correct offset.
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

from subsyncpro.fingerprint import (
    RESOLUTION_MS,
    fingerprint_duration_ms,
    make_binary_fingerprint,
    make_fingerprint,
    window_has_content,
)
from subsyncpro.parser import SubtitleEvent

log = logging.getLogger(__name__)

# ── tunable constants ─────────────────────────────────────────────────────────

_WIN_MS: int = 30_000       # sliding window length (30 s)
_STEP_MS: int = 5_000       # step between windows (5 s)
_MIN_SCORE: float = 0.15    # minimum score after zero-mean Pearson-like normalisation
_RANSAC_ITERS: int = 500
_INLIER_THRESH_MS: int = 1_500   # 1.5 s residual threshold for RANSAC inlier
_DRIFT_INLIER_THRESH_MS: int = 1_500
_MIN_INLIERS: int = 4
_MAX_DRIFT: float = 0.05    # reject scale outside [1-MAX_DRIFT, 1+MAX_DRIFT]

# Segmented binary alignment (fallback for cross-language / gradual-drift cases)
_SEG_MS: int = 300_000      # 5-minute segments
_SEG_SEARCH_MS: int = 45_000  # ±45 s search window per segment
_SEG_INLIER_MS: int = 3_000  # 3 s residual threshold for segment-level fit

# Common frame-rate conversion ratios.  Free-fitting RANSAC over hundreds of
# noisy event pairs can land within ~0.04% of the true ratio but rarely lands
# exactly on it.  Over a 2-hour movie a 0.04% slope error → ~300 ms drift,
# which is the residual we are trying to eliminate.  When the fitted scale is
# close to a known conversion ratio, snap to it and re-fit the offset.
#
# Restricted to ratios inside the global ±_MAX_DRIFT = 5% sanity bound.
_FRAMERATE_RATIOS: tuple[float, ...] = tuple(sorted({
    23.976 / 24.0,      # 0.999000   NTSC film → true 24
    23.976 / 25.0,      # 0.959040   PAL → NTSC film  ← Test 4
    24.0 / 23.976,      # 1.001001   true 24 → NTSC film
    24.0 / 25.0,        # 0.960000   true 24 → PAL
    25.0 / 24.0,        # 1.041667   PAL → true 24
    29.97 / 30.0,       # 0.999000   NTSC video → true 30
    30.0 / 29.97,       # 1.001001   true 30 → NTSC video
}))
# Snap when the fitted scale is within this distance of a whitelist ratio.
# 0.0005 = 0.05% = ~30 ms over 60 s; tight enough that we never snap when
# the data clearly disagrees, loose enough to catch noisy free-fit landings.
_FRAMERATE_SNAP_THRESHOLD: float = 0.0005


def _snap_to_framerate_ratio(scale: float) -> float | None:
    """Return the nearest whitelist ratio if within snap threshold, else None."""
    best: float | None = None
    best_dist = _FRAMERATE_SNAP_THRESHOLD
    for r in _FRAMERATE_RATIOS:
        d = abs(scale - r)
        if d < best_dist:
            best_dist = d
            best = r
    return best


# ── SDH event filtering ───────────────────────────────────────────────────────

# Strips brackets, parens, asterisk-groups, music notes, HTML tags, and
# punctuation.  Anything that survives is "real dialogue".
_SDH_STRIP = re.compile(
    r"\[[^\]]*\]"           # [bracket content] — sound descriptors
    r"|\([^)]*\)"           # (paren content) — stage directions
    r"|<[^>]+>"             # <html tags>
    r"|\*[^*]*\*"           # *asterisk content* — music/sound markers
    r"|\*"                  # lone *
    r"|♪[^♪]*♪"             # ♪lyrics♪
    r"|♪"                   # lone ♪
    r"|[-–—_,. !?:;'\"]"   # punctuation and whitespace
    r"|\s+",
    re.IGNORECASE | re.DOTALL,
)


def _is_sdh_event(text: str) -> bool:
    """Return True when an event contains only sound descriptors, no dialogue."""
    return len(_SDH_STRIP.sub("", text)) == 0


def _filter_dialogue_only(events: list[SubtitleEvent]) -> list[SubtitleEvent]:
    """Remove sound-descriptor-only events from a subtitle list."""
    return [e for e in events if not _is_sdh_event(e.text)]


# ── boundary time exclusion ───────────────────────────────────────────────────

# Translator/sync credits live in the first ~60 s of a translated subtitle.
# Cast and crew credits roll in the last ~90 s of a movie.  Neither has a
# counterpart in the original-language reference, so they generate spurious
# anchors that bias the offset estimate.
#
# Time-based exclusion is symmetric (same wall-clock window for both files),
# density-independent (works for sparse and dense subtitles alike), and
# never accidentally discards real dialogue — translator credits are always
# well within the first 90 s.
_SKIP_HEAD_MS: int = 90_000    # ignore first 90 s of each subtitle's span
_SKIP_TAIL_MS: int = 120_000   # ignore last 2 min of each subtitle's span


def _skip_boundary_time(
    events: list[SubtitleEvent],
    head_ms: int = _SKIP_HEAD_MS,
    tail_ms: int = _SKIP_TAIL_MS,
) -> list[SubtitleEvent]:
    """Return only events that fall outside the head/tail boundary windows.

    Events whose *start* is within the first *head_ms* or the last *tail_ms*
    of the subtitle's own time span are excluded from fingerprinting.  The
    full event list is still passed to the writer — the exclusion is only for
    alignment voting.

    If the subtitle is shorter than head + tail + 60 s the full list is
    returned unchanged so short content is not over-excluded.
    """
    if not events:
        return events
    t_first = events[0].start_ms
    t_last = events[-1].start_ms
    if t_last - t_first < head_ms + tail_ms + 60_000:
        return events
    lo = t_first + head_ms
    hi = t_last - tail_ms
    return [e for e in events if lo <= e.start_ms <= hi]


# ── result type ───────────────────────────────────────────────────────────────

@dataclass
class Segment:
    """One piece of a piecewise-linear timing model.

    *start_ms* is the inclusive lower bound on the unsync time domain;
    *end_ms* is the exclusive upper bound (use ±inf for the outer edges).
    Within [start_ms, end_ms) the transform is ref = scale * unsync + offset_ms.
    """
    start_ms: float
    end_ms: float
    scale: float
    offset_ms: float


@dataclass
class AlignResult:
    offset_ms: float
    """Milliseconds to *add* to every unsync timestamp."""

    scale: float
    """Multiplicative scale applied before the offset. 1.0 = no drift."""

    confidence: float
    """0–1 reliability estimate."""

    n_anchors: int
    n_inliers: int
    coarse_offset_ms: float
    mode_used: str

    # Optional piecewise-linear model.  When set, the writer applies the
    # per-segment (scale, offset) instead of the global pair.  The global
    # scale/offset above remain populated with the best-effort single-segment
    # values for display and backwards compatibility.
    segments: list[Segment] | None = None

    @property
    def is_reliable(self) -> bool:
        # For segmented alignment n_anchors is small (5-10 segments), so
        # require at least 2 inliers instead of the windowed-mode minimum.
        min_in = 2 if self.n_anchors <= 15 else _MIN_INLIERS
        return self.confidence >= 0.4 and self.n_inliers >= min_in

    def describe(self) -> str:
        sign = "+" if self.offset_ms >= 0 else ""
        s = f"offset={sign}{self.offset_ms/1000:.3f}s"
        if abs(self.scale - 1.0) > 1e-4:
            s += f"  scale={self.scale:.6f}"
        s += f"  confidence={self.confidence:.0%}"
        s += f"  anchors={self.n_inliers}/{self.n_anchors}"
        if self.segments and len(self.segments) > 1:
            s += f"  [piecewise: {len(self.segments)} segments]"
        return s


# ── sub-bin peak interpolation ────────────────────────────────────────────────

def _parabolic_peak(corr: np.ndarray, k: int) -> float:
    """Return the interpolated peak index near integer argmax *k*.

    Fits a parabola through (k-1, k, k+1) and returns k + δ where δ ∈ [-0.5,
    0.5] is the sub-bin offset of the parabola's apex.  Falls back to *k*
    at array boundaries or when curvature is flat.

    With a 33 ms fingerprint resolution, quantisation alone introduces ±16 ms
    of noise at every cross-correlation peak; parabolic interpolation cuts
    this to ~1-2 ms.  Applied at every argmax in the pipeline, the integrated
    accuracy gain is tens of milliseconds.
    """
    if k <= 0 or k >= len(corr) - 1:
        return float(k)
    y0 = float(corr[k - 1])
    y1 = float(corr[k])
    y2 = float(corr[k + 1])
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return float(k)
    delta = 0.5 * (y0 - y2) / denom
    if delta < -0.5:
        delta = -0.5
    elif delta > 0.5:
        delta = 0.5
    return float(k) + delta


# ── coarse FFT alignment ──────────────────────────────────────────────────────

def _coarse_offset(ref_fp: np.ndarray, uns_fp: np.ndarray, resolution_ms: int) -> float:
    """Global cross-correlation via FFT. Returns best offset in ms.

    Uses fftconvolve(ref, uns_reversed, 'full') so that result[k] is the
    dot-product of ref starting at position (k - len_uns + 1) with uns.
    Peak at k0 → offset_samples = k0 - (len_uns - 1).
    """
    corr = fftconvolve(ref_fp, uns_fp[::-1], mode="full")
    best_k_int = int(np.argmax(corr))
    best_k = _parabolic_peak(corr, best_k_int)
    offset_samples = best_k - (len(uns_fp) - 1)
    # Clamp to reasonable range to avoid wrap-around artefacts
    max_samples = max(len(ref_fp), len(uns_fp))
    offset_samples = float(np.clip(offset_samples, -max_samples, max_samples))
    return float(offset_samples * resolution_ms)


# ── windowed anchor search ────────────────────────────────────────────────────

def _find_anchors(
    ref_fp: np.ndarray,
    uns_fp: np.ndarray,
    resolution_ms: int,
) -> list[dict]:
    """Slide a window across uns_fp and find its best match in ref_fp.

    fftconvolve(ref_fp, window[::-1], mode='valid')[k]
        = dot(ref_fp[k : k+win_n], window)

    So argmax gives ref_start directly (0-indexed).

    Returns list of anchor dicts: {unsync_ms, ref_ms, offset_ms, score}.
    """
    win_n = _WIN_MS // resolution_ms
    step_n = _STEP_MS // resolution_ms

    if len(ref_fp) < win_n:
        return []

    anchors: list[dict] = []

    for pos in range(0, len(uns_fp) - win_n + 1, step_n):
        window = uns_fp[pos : pos + win_n]
        if not window_has_content(window):
            continue

        # Zero-mean the window before correlating.  This removes the DC
        # component so that windows with uniformly distributed events produce
        # near-zero correlation at wrong offsets, while genuine matches stand
        # out clearly.  Without this, positive-only onset impulses create
        # spurious high-scoring matches wherever the reference is dense.
        window_zm = window - window.mean()
        corr = fftconvolve(ref_fp, window_zm[::-1], mode="valid")
        ref_start_int = int(np.argmax(corr))
        ref_start = _parabolic_peak(corr, ref_start_int)

        # Normalise by the energy of the zero-mean window (L2 norm).
        # This gives a Pearson-like score in [0, 1] when both signals align.
        window_energy = float(np.dot(window_zm, window_zm))
        if window_energy < 1e-9:
            continue
        score = float(corr[ref_start_int]) / window_energy

        if score < _MIN_SCORE:
            continue

        ref_center_ms = (ref_start + win_n / 2) * resolution_ms
        uns_center_ms = (pos + win_n / 2) * resolution_ms

        anchors.append(
            {
                "unsync_ms": float(uns_center_ms),
                "ref_ms": float(ref_center_ms),
                "offset_ms": float(ref_center_ms - uns_center_ms),
                "score": score,
            }
        )

    return anchors


# ── RANSAC: constant offset ───────────────────────────────────────────────────

def _count_inliers(
    offsets: np.ndarray,
    scores: np.ndarray,
    candidate: float,
    threshold_ms: int,
) -> tuple[int, float]:
    """Return (inlier_count, weighted_mean_offset) for a candidate offset."""
    mask = np.abs(offsets - candidate) < threshold_ms
    n = int(mask.sum())
    if n == 0:
        return 0, candidate
    refined = float(np.average(offsets[mask], weights=scores[mask]))
    return n, refined


def _ransac_offset(
    anchors: list[dict],
    n_iter: int = _RANSAC_ITERS,
    threshold_ms: int = _INLIER_THRESH_MS,
    seed_offsets: list[float] | None = None,
) -> tuple[float, list[dict], float]:
    """RANSAC for a constant offset.

    *seed_offsets* are additional candidate values evaluated unconditionally
    (e.g. the coarse FFT estimate).  This prevents RANSAC from ignoring the
    globally optimal solution when anchors are noisy.
    """
    if not anchors:
        return 0.0, [], 0.0

    offsets = np.array([a["offset_ms"] for a in anchors])
    scores = np.array([a["score"] for a in anchors])
    weights = scores / scores.sum()
    rng = random.Random(42)

    best_n = 0
    best_offset = float(np.median(offsets))

    # Evaluate forced seed candidates first
    for seed in (seed_offsets or []):
        n, refined = _count_inliers(offsets, scores, seed, threshold_ms)
        if n > best_n:
            best_n, best_offset = n, refined

    # Random sampling
    for _ in range(n_iter):
        idx = rng.choices(range(len(anchors)), weights=weights.tolist())[0]
        candidate = offsets[idx]
        n, refined = _count_inliers(offsets, scores, candidate, threshold_ms)
        if n > best_n:
            best_n, best_offset = n, refined

    # Re-evaluate the winner with a second pass (the refinement may have moved
    # the centre, pulling in a few extra inliers)
    _, best_offset = _count_inliers(offsets, scores, best_offset, threshold_ms)

    final_mask = np.abs(offsets - best_offset) < threshold_ms
    inliers = [a for a, m in zip(anchors, final_mask) if m]
    confidence = len(inliers) / len(anchors)
    return best_offset, inliers, confidence


# ── RANSAC: linear (offset + drift) ──────────────────────────────────────────

def _ransac_linear(
    anchors: list[dict],
    n_iter: int = _RANSAC_ITERS,
    threshold_ms: int = _DRIFT_INLIER_THRESH_MS,
    seed_offset: float | None = None,
) -> tuple[float, float, list[dict], float]:
    """RANSAC for ref_ms = scale * unsync_ms + offset.

    Starts with the offset-only solution (scale=1, seed_offset) as an initial
    hypothesis so the linear search cannot do worse than the constant model.
    """
    if len(anchors) < 2:
        offset, inliers, conf = _ransac_offset(anchors, seed_offsets=[seed_offset] if seed_offset is not None else None)
        return 1.0, offset, inliers, conf

    xs = np.array([a["unsync_ms"] for a in anchors])
    ys = np.array([a["ref_ms"] for a in anchors])
    scores = np.array([a["score"] for a in anchors])
    weights = scores / scores.sum()
    rng = random.Random(42)

    # Bootstrap with constant-offset hypothesis (scale = 1)
    init_off = seed_offset if seed_offset is not None else float(np.median(ys - xs))
    init_residuals = np.abs(ys - (xs + init_off))
    init_mask = init_residuals < threshold_ms
    best_inliers = [a for a, m in zip(anchors, init_mask) if m]
    best_scale, best_offset = 1.0, init_off

    indices = list(range(len(anchors)))

    for _ in range(n_iter):
        i, j = rng.choices(indices, weights=weights.tolist(), k=2)
        if i == j:
            continue
        dx = xs[j] - xs[i]
        if abs(dx) < 5_000:
            continue
        scale = (ys[j] - ys[i]) / dx
        if not (1.0 - _MAX_DRIFT <= scale <= 1.0 + _MAX_DRIFT):
            continue
        offset = ys[i] - scale * xs[i]
        residuals = np.abs(ys - (scale * xs + offset))
        inlier_mask = residuals < threshold_ms
        n_in = int(inlier_mask.sum())
        if n_in > len(best_inliers):
            best_inliers = [a for a, m in zip(anchors, inlier_mask) if m]
            best_scale, best_offset = scale, offset

    if len(best_inliers) >= 2:
        bx = np.array([a["unsync_ms"] for a in best_inliers])
        by = np.array([a["ref_ms"] for a in best_inliers])
        bw = np.array([a["score"] for a in best_inliers])
        coeffs = np.polyfit(bx, by, 1, w=bw)
        if abs(coeffs[0] - 1.0) <= _MAX_DRIFT:
            best_scale, best_offset = float(coeffs[0]), float(coeffs[1])
            # Snap scale to the nearest standard frame-rate ratio when very
            # close, and re-fit the offset with the snapped slope held fixed.
            # Removes the ~0.04% slope error free-RANSAC leaves behind on PAL/
            # NTSC conversions, which compounds to ~300 ms over a 2-hour film.
            snapped = _snap_to_framerate_ratio(best_scale)
            if snapped is not None:
                weights = bw / bw.sum()
                best_offset = float(np.sum(weights * (by - snapped * bx)))
                best_scale = snapped

    confidence = len(best_inliers) / len(anchors) if anchors else 0.0
    return best_scale, best_offset, best_inliers, confidence


# ── segmented binary alignment (cross-language / gradual-drift fallback) ─────

def _segmented_binary_alignment(
    ref_fp_bin: np.ndarray,
    uns_fp_bin: np.ndarray,
    resolution_ms: int,
    max_offset_ms: float = 600_000,
    seed_scale: float | None = None,
    seed_offset_ms: float | None = None,
    refine_radius_ms: float = 3_000,
) -> list[tuple[float, float, float]]:
    """Segment-by-segment binary cross-correlation against the full reference.

    Divides the unsync fingerprint into 5-minute segments and cross-correlates
    each against the ENTIRE reference fingerprint using zero-mean binary
    occupancy.  No seed is required — the FFT scans every possible offset so
    the result is immune to a bad coarse-offset estimate.

    Binary occupancy (1 = subtitle active, 0 = silent) captures the speech-
    activity pattern that is similar across languages even when one subtitle
    merges events the other keeps separate.  Zero-mean cross-correlation
    removes the DC so high-density windows don't produce spurious peaks.

    When *seed_scale* / *seed_offset_ms* are provided the search is narrowed
    to ±refine_radius_ms around the predicted offset for each segment.  This
    second-pass mode is much faster and more precise than the full-range scan.

    Returns a list of (uns_center_ms, local_offset_ms, snr) triples — one per
    segment that has enough speech content.  *snr* is peak / mean-of-window;
    higher means a sharper, more reliable peak.
    """
    seg_n = _SEG_MS // resolution_ms
    narrow = seed_scale is not None and seed_offset_ms is not None
    radius_n = max(1, int(refine_radius_ms / resolution_ms))
    results: list[tuple[float, float, float]] = []

    for seg_start in range(0, len(uns_fp_bin) - seg_n + 1, seg_n):
        seg_end = seg_start + seg_n
        uns_seg = uns_fp_bin[seg_start:seg_end]

        density = float(uns_seg.mean())
        if density < 0.03 or density > 0.97:
            continue  # too sparse or fully saturated — no useful structure

        uns_zm = uns_seg - density
        energy = float(np.dot(uns_zm, uns_zm))
        if energy < 1.0:
            continue

        uns_center_ms = float((seg_start + seg_end) / 2 * resolution_ms)

        if narrow:
            # Predict where this segment lands and search only a narrow window.
            predicted_start_ms = seed_scale * (seg_start * resolution_ms) + seed_offset_ms  # type: ignore[operator]
            pred_k = int(predicted_start_ms / resolution_ms)
            lo = max(0, pred_k - radius_n)
            hi = min(len(ref_fp_bin) - seg_n, pred_k + radius_n)
            if hi < lo:
                continue
            ref_slice = ref_fp_bin[lo : hi + seg_n]
            corr = fftconvolve(ref_slice, uns_zm[::-1], mode="valid")
            if len(corr) == 0:
                continue
            local_idx_int = int(np.argmax(corr))
            local_idx = _parabolic_peak(corr, local_idx_int)
            best_k = lo + local_idx
        else:
            corr = fftconvolve(ref_fp_bin, uns_zm[::-1], mode="valid")
            if len(corr) == 0:
                continue
            local_idx_int = int(np.argmax(corr))
            local_idx = _parabolic_peak(corr, local_idx_int)
            best_k = local_idx

        local_off_ms = float((best_k - seg_start) * resolution_ms)

        if not narrow and abs(local_off_ms) > max_offset_ms:
            continue  # plausibility filter only needed for full-range pass

        # SNR: correlation peak value divided by mean of ±radius neighbourhood.
        lo_w = max(0, local_idx_int - radius_n)
        hi_w = min(len(corr), local_idx_int + radius_n + 1)
        noise = float(np.mean(np.abs(corr[lo_w:hi_w])))
        snr = float(corr[local_idx_int]) / (noise + 1e-9)

        results.append((uns_center_ms, local_off_ms, snr))

    return results


def _fit_segment_offsets(
    seg_offsets: list[tuple[float, float, float]],
) -> tuple[float, float, float, int]:
    """Fit a linear drift model to per-segment local offsets.

    Each segment provides one (unsync_time, offset_ms, snr) observation.  The
    model is offset_ms(t) = (scale - 1) * t + const, equivalently
    ref_ms = scale * unsync_ms + const.

    Uses exhaustive pairwise RANSAC to get a robust slope estimate, then
    SNR-weighted polyfit so high-confidence segments pull harder on the result.

    Returns (scale, offset_ms, confidence, n_inliers).
    """
    n = len(seg_offsets)
    if n == 0:
        return 1.0, 0.0, 0.0, 0
    if n == 1:
        return 1.0, seg_offsets[0][1], 0.3, 1

    times = np.array([t for t, _, _ in seg_offsets], dtype=np.float64)
    offsets = np.array([o for _, o, _ in seg_offsets], dtype=np.float64)
    snrs = np.array([s for _, _, s in seg_offsets], dtype=np.float64)

    best_n_in = 0
    best_slope = 0.0
    best_intercept = float(np.median(offsets))

    for i in range(n):
        for j in range(i + 1, n):
            dt = times[j] - times[i]
            if abs(dt) < _SEG_MS * 0.5:
                continue
            slope = (offsets[j] - offsets[i]) / dt
            if abs(slope) > _MAX_DRIFT:
                continue
            intercept = offsets[i] - slope * times[i]
            residuals = np.abs(offsets - (slope * times + intercept))
            n_in = int((residuals < _SEG_INLIER_MS).sum())
            if n_in > best_n_in:
                best_n_in = n_in
                best_slope = slope
                best_intercept = intercept

    # Refine using all inliers, weighted by per-segment SNR
    predicted = best_slope * times + best_intercept
    inlier_mask = np.abs(offsets - predicted) < _SEG_INLIER_MS
    n_inliers = int(inlier_mask.sum())

    if n_inliers >= 2:
        in_t = times[inlier_mask]
        in_o = offsets[inlier_mask]
        in_w = snrs[inlier_mask]
        in_w = in_w / (in_w.sum() + 1e-9)  # normalise weights
        coeffs = np.polyfit(in_t, in_o, 1, w=in_w)
        best_slope = float(np.clip(coeffs[0], -_MAX_DRIFT, _MAX_DRIFT))
        best_intercept = float(coeffs[1])

    scale = float(np.clip(1.0 + best_slope, 1.0 - _MAX_DRIFT, 1.0 + _MAX_DRIFT))
    # Snap to the nearest standard frame-rate ratio (re-fits intercept too).
    # Without this the seed handed to event-pair refinement has a small slope
    # error that the ±5 s match tolerance smears across many anchors.
    snapped = _snap_to_framerate_ratio(scale)
    if snapped is not None and n_inliers >= 2:
        in_t = times[inlier_mask]
        in_o = offsets[inlier_mask]
        in_w = snrs[inlier_mask]
        wsum = float(in_w.sum()) + 1e-9
        snapped_slope = snapped - 1.0
        best_intercept = float(np.sum(in_w * (in_o - snapped_slope * in_t)) / wsum)
        scale = snapped
    confidence = n_inliers / n if n > 0 else 0.0
    return scale, float(best_intercept), confidence, n_inliers


# ── event-pair refinement ─────────────────────────────────────────────────────

# Duration-aware pair matching weights.  When the rough model is off by 2–4 s
# (PAL/NTSC drift), two adjacent reference events can both fall inside the
# search tolerance — picking the closer one by time alone locks onto the wrong
# neighbour and biases RANSAC.  Cross-language translations preserve subtitle
# duration fairly reliably even when wording is reorganised, so a duration-
# similarity term breaks ambiguous ties on the correct candidate.
_DUR_WEIGHT: float = 0.30        # share of match score from duration similarity
_DUR_TOLERANCE: float = 0.30     # |Δdur|/dur at which the duration bonus → 0


def _refine_with_event_pairs(
    ref_events: list[SubtitleEvent],
    uns_events: list[SubtitleEvent],
    rough_scale: float,
    rough_offset_ms: float,
    tolerance_ms: float = 5_000.0,
) -> tuple[float, float, list[dict], float] | None:
    """Refine a rough (scale, offset) by matching individual subtitle events.

    Predicts the reference-time position of each unsync event using the rough
    model, picks the ref event within *tolerance_ms* whose start AND duration
    best match, then runs _ransac_linear() on the resulting anchor pairs.

    This is more precise than segment-level binary cross-correlation because:
    - Each pair is a point measurement — no spatial smearing from a 5-min window.
    - RANSAC over hundreds of pairs provides a much tighter linear fit than the
      ~10–25 segment-level estimates that the binary pass produces.

    Returns (scale, offset_ms, inliers, confidence) or None when too few pairs.
    """
    if not ref_events or not uns_events:
        return None

    ref_starts = np.array([e.start_ms for e in ref_events], dtype=np.float64)
    ref_durs = np.array(
        [max(1, e.end_ms - e.start_ms) for e in ref_events], dtype=np.float64
    )
    anchors: list[dict] = []

    for ev in uns_events:
        pred = rough_scale * ev.start_ms + rough_offset_ms
        uns_dur = max(1.0, float(ev.end_ms - ev.start_ms))

        # Look at every ref event whose start lies within ±tolerance of the
        # prediction (typically 1–3 candidates).  searchsorted gives an
        # O(log n) window so this stays cheap.
        lo = int(np.searchsorted(ref_starts, pred - tolerance_ms))
        hi = int(np.searchsorted(ref_starts, pred + tolerance_ms))

        best_score = -1.0
        best_ref_ms: float | None = None

        for ci in range(max(0, lo - 1), min(len(ref_starts), hi + 1)):
            dist = abs(ref_starts[ci] - pred)
            if dist > tolerance_ms:
                continue
            time_score = 1.0 - dist / tolerance_ms
            dratio = abs(ref_durs[ci] - uns_dur) / uns_dur
            dur_score = max(0.0, 1.0 - dratio / _DUR_TOLERANCE)
            score = (1.0 - _DUR_WEIGHT) * time_score + _DUR_WEIGHT * dur_score
            if score > best_score:
                best_score = score
                best_ref_ms = float(ref_starts[ci])

        if best_ref_ms is not None:
            anchors.append({
                "unsync_ms": float(ev.start_ms),
                "ref_ms": best_ref_ms,
                "offset_ms": best_ref_ms - float(ev.start_ms),
                # Floor at 0.1 so weak matches still contribute to RANSAC —
                # they may be correct pairs whose neighbour has an unusual
                # duration.
                "score": max(0.1, best_score),
            })

    if len(anchors) < max(_MIN_INLIERS * 2, 8):
        return None

    return _ransac_linear(anchors, seed_offset=rough_offset_ms)


# ── piecewise-linear fit ─────────────────────────────────────────────────────

# Tiny segments overfit noise — any segment must span at least this much time.
_PIECEWISE_MIN_SEG_MS: int = 600_000   # 10 minutes
# Below this anchor count the per-segment scale/offset are dominated by noise.
_PIECEWISE_MIN_ANCHORS_PER_SEG: int = 30
_PIECEWISE_MAX_DRIFT: float = _MAX_DRIFT
# A real edit cut produces a fractional-second jump.  Larger discontinuities
# usually indicate overfitting and trigger a fallback to single-segment.
_PIECEWISE_MAX_JUMP_MS: float = 2_500.0
# Raftery's "very strong evidence" threshold — ΔBIC > 10 to escalate k → k+1.
_PIECEWISE_BIC_MARGIN: float = 10.0


def _weighted_polyfit(xs: np.ndarray, ys: np.ndarray, ws: np.ndarray) -> tuple[float, float]:
    coeffs = np.polyfit(xs, ys, 1, w=ws)
    return float(coeffs[0]), float(coeffs[1])


def _bic(rss: float, n_params: int, n_obs: int) -> float:
    if n_obs <= 0 or rss <= 0:
        return float("inf")
    return n_obs * float(np.log(rss / n_obs)) + n_params * float(np.log(n_obs))


def _fit_pieces(
    xs: np.ndarray,
    ys: np.ndarray,
    ws: np.ndarray,
    breakpoints: list[float],
) -> tuple[list[tuple[float, float, float, float]] | None, float]:
    """Fit one weighted-linear model per segment between *breakpoints*.

    Returns (segments, weighted_rss) or (None, inf) when constraints fail.
    Each segment tuple is (start_ms, end_ms, scale, offset).
    """
    edges = [float(xs.min()) - 1.0] + list(breakpoints) + [float(xs.max()) + 1.0]
    segs: list[tuple[float, float, float, float]] = []
    total_rss = 0.0
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (xs > lo) & (xs <= hi)
        n = int(mask.sum())
        if n < _PIECEWISE_MIN_ANCHORS_PER_SEG:
            return None, float("inf")
        xseg, yseg, wseg = xs[mask], ys[mask], ws[mask]
        if float(xseg.max() - xseg.min()) < _PIECEWISE_MIN_SEG_MS:
            return None, float("inf")
        scale, offset = _weighted_polyfit(xseg, yseg, wseg)
        if abs(scale - 1.0) > _PIECEWISE_MAX_DRIFT:
            return None, float("inf")
        pred = scale * xseg + offset
        total_rss += float(np.sum(wseg * (yseg - pred) ** 2))
        seg_start = float("-inf") if i == 0 else float(breakpoints[i - 1])
        seg_end = float("inf") if i == len(edges) - 2 else float(breakpoints[i])
        segs.append((seg_start, seg_end, scale, offset))
    return segs, total_rss


def _piecewise_linear_fit(
    anchors: list[dict],
    *,
    verbose: bool = False,
) -> tuple[list[tuple[float, float, float, float]] | None, dict]:
    """Fit 1, 2 or 3 linear pieces to anchor pairs; pick by BIC.

    Returns the list of (start_ms, end_ms, scale, offset) segments when
    piecewise is preferred, or None when single-linear wins.  Diagnostics
    (BIC values, residuals) come back in a dict.
    """
    diag: dict = {}
    if len(anchors) < _PIECEWISE_MIN_ANCHORS_PER_SEG * 2:
        diag["reason"] = "too few anchors"
        return None, diag

    xs = np.array([a["unsync_ms"] for a in anchors], dtype=np.float64)
    ys = np.array([a["ref_ms"] for a in anchors], dtype=np.float64)
    ws = np.array([a["score"] for a in anchors], dtype=np.float64)
    order = np.argsort(xs)
    xs, ys, ws = xs[order], ys[order], ws[order]
    n_obs = len(xs)

    scale1, offset1 = _weighted_polyfit(xs, ys, ws)
    if abs(scale1 - 1.0) > _PIECEWISE_MAX_DRIFT:
        diag["reason"] = "single-segment scale out of range"
        return None, diag
    pred1 = scale1 * xs + offset1
    rss1 = float(np.sum(ws * (ys - pred1) ** 2))
    bic1 = _bic(rss1, 2, n_obs)
    diag["bic"] = {1: bic1}
    diag["single"] = (scale1, offset1)

    # Candidate breakpoints at quantile positions of the unsync domain.
    n_candidates = 60
    qs = np.linspace(0.08, 0.92, n_candidates)
    cand_bps = np.quantile(xs, qs)

    best2_rss = float("inf")
    best2_segs = None
    best2_bp: float | None = None
    for bp in cand_bps:
        segs, rss = _fit_pieces(xs, ys, ws, [float(bp)])
        if rss < best2_rss:
            best2_rss = rss
            best2_segs = segs
            best2_bp = float(bp)
    bic2 = _bic(best2_rss, 5, n_obs) if best2_segs else float("inf")
    diag["bic"][2] = bic2
    diag["bp2"] = best2_bp

    best3_rss = float("inf")
    best3_segs = None
    best3_bps: list[float] | None = None
    if bic2 < bic1:
        for i, b1 in enumerate(cand_bps):
            for b2 in cand_bps[i + 1:]:
                if b2 - b1 < _PIECEWISE_MIN_SEG_MS:
                    continue
                segs, rss = _fit_pieces(xs, ys, ws, [float(b1), float(b2)])
                if rss < best3_rss:
                    best3_rss = rss
                    best3_segs = segs
                    best3_bps = [float(b1), float(b2)]
    bic3 = _bic(best3_rss, 8, n_obs) if best3_segs else float("inf")
    diag["bic"][3] = bic3
    diag["bp3"] = best3_bps

    best_k = 1
    if bic2 + _PIECEWISE_BIC_MARGIN < bic1:
        best_k = 2
        if bic3 + _PIECEWISE_BIC_MARGIN < bic2:
            best_k = 3
    diag["best_k"] = best_k

    if best_k == 1:
        return None, diag

    chosen_segs = best2_segs if best_k == 2 else best3_segs
    chosen_bps = [best2_bp] if best_k == 2 else best3_bps or []

    # Safety: bound the time discontinuity at each breakpoint.  Edit cuts
    # produce fractional-second jumps; bigger ones usually mean overfitting.
    jumps_ms = []
    for bp in chosen_bps:
        prev_seg = None
        next_seg = None
        for s in chosen_segs:  # type: ignore[union-attr]
            if s[1] == bp:
                prev_seg = s
            if s[0] == bp:
                next_seg = s
        if prev_seg is None or next_seg is None:
            continue
        pred_prev = prev_seg[2] * bp + prev_seg[3]
        pred_next = next_seg[2] * bp + next_seg[3]
        jumps_ms.append(abs(pred_next - pred_prev))
    diag["jumps_ms"] = jumps_ms
    if jumps_ms and max(jumps_ms) > _PIECEWISE_MAX_JUMP_MS:
        diag["reason"] = "breakpoint jump exceeds safety cap"
        return None, diag

    return chosen_segs, diag


# ── public API ────────────────────────────────────────────────────────────────

def align(
    ref_events: list[SubtitleEvent],
    unsync_events: list[SubtitleEvent],
    *,
    mode: str = "auto",
    max_offset_s: float = 600.0,
    resolution_ms: int = RESOLUTION_MS,
    verbose: bool = False,
) -> AlignResult:
    """Synchronise *unsync_events* against *ref_events*.

    Parameters
    ----------
    mode : 'auto' | 'offset' | 'linear'
    max_offset_s :
        Maximum expected timing difference. Increase for very large gaps.
    """
    if not ref_events:
        raise ValueError("Reference subtitle has no events.")
    if not unsync_events:
        raise ValueError("Unsynchronised subtitle has no events.")

    max_offset_ms = max_offset_s * 1000

    # ── 1. Build fingerprints ────────────────────────────────────────────
    # Exclude the first 90 s and last 2 min of each subtitle from fingerprinting.
    # Translator/sync credits live in the first ~60 s of a translated subtitle;
    # cast/crew credits roll in the last ~90 s of a movie.  Neither has a
    # counterpart in the reference, so they bias the offset estimate.
    # Time-based exclusion is symmetric and density-independent.
    # The full event lists are still used by the writer — exclusion is alignment-only.
    ref_fp_events = _skip_boundary_time(ref_events)
    uns_fp_events = _skip_boundary_time(unsync_events)
    if verbose and (len(ref_fp_events) != len(ref_events)
                    or len(uns_fp_events) != len(unsync_events)):
        log.info(
            "Boundary skip: ref %d→%d events, unsync %d→%d events "
            "(first 90 s and last 2 min excluded from alignment voting)",
            len(ref_events), len(ref_fp_events),
            len(unsync_events), len(uns_fp_events),
        )

    ref_dur = fingerprint_duration_ms(ref_fp_events)
    uns_dur = fingerprint_duration_ms(uns_fp_events)
    # ref is padded to cover the full search range; unsync keeps its natural size.
    # We add max_offset_ms to ref's right so that windows at the end of unsync
    # can still find a match even if ref ends earlier.
    ref_padded_dur = ref_dur + int(max_offset_ms) + 60_000

    ref_fp = make_fingerprint(ref_fp_events, ref_padded_dur, resolution_ms)
    uns_fp = make_fingerprint(uns_fp_events, uns_dur, resolution_ms)

    if verbose:
        log.info(
            "Fingerprints: ref=%d samples (%.1f min), unsync=%d samples (%.1f min)",
            len(ref_fp), ref_dur / 60_000,
            len(uns_fp), uns_dur / 60_000,
        )

    # ── 2. Coarse FFT offset (informational only; not used to limit search) ─
    try:
        # Use a shorter version of both fingerprints for speed
        trim = min(len(ref_fp), len(uns_fp), int(max_offset_ms * 3 / resolution_ms))
        coarse_ms = _coarse_offset(ref_fp[:trim], uns_fp[:trim], resolution_ms)
        coarse_ms = float(np.clip(coarse_ms, -max_offset_ms, max_offset_ms))
    except Exception:
        coarse_ms = 0.0

    if verbose:
        log.info("Coarse FFT offset estimate: %+.3f s", coarse_ms / 1000)

    # ── 3. Windowed anchor search (full ref search — no restriction) ──────
    anchors = _find_anchors(ref_fp, uns_fp, resolution_ms)

    if verbose:
        log.info("Found %d candidate anchor pairs", len(anchors))

    if not anchors:
        log.warning("No anchor points found — falling back to coarse FFT offset.")
        return AlignResult(
            offset_ms=coarse_ms,
            scale=1.0,
            confidence=0.0,
            n_anchors=0,
            n_inliers=0,
            coarse_offset_ms=coarse_ms,
            mode_used="offset",
        )

    # ── 4. RANSAC ─────────────────────────────────────────────────────────
    # Always run offset-only RANSAC first, seeded with the coarse FFT estimate.
    # This guarantees we can never do worse than the global FFT answer.
    off_ms, off_inliers, off_conf = _ransac_offset(
        anchors, seed_offsets=[coarse_ms]
    )
    scale, offset_ms, inliers, conf = 1.0, off_ms, off_inliers, off_conf
    mode_used = "offset"

    # Try linear model if requested (it is initialised from the offset solution,
    # so it cannot return fewer inliers than the constant-offset model).
    if mode in ("linear", "auto"):
        lin_scale, lin_off, lin_inliers, lin_conf = _ransac_linear(
            anchors, seed_offset=coarse_ms
        )
        drift_significant = abs(lin_scale - 1.0) >= 0.0005
        # Accept linear only when it genuinely improves on the offset model
        if len(lin_inliers) > len(off_inliers) and drift_significant:
            scale, offset_ms, inliers, conf = lin_scale, lin_off, lin_inliers, lin_conf
            mode_used = "linear"
        elif mode == "auto" and not drift_significant:
            # Linear found negligible drift — stay with offset result
            pass
        elif mode == "linear":
            # Caller explicitly asked for linear; honour even if drift is small
            scale, offset_ms, inliers, conf = lin_scale, lin_off, lin_inliers, lin_conf
            mode_used = "linear"

    # Last-resort: loosen threshold with 3× window if still very few inliers
    if len(inliers) < _MIN_INLIERS and len(anchors) >= _MIN_INLIERS:
        off2, in2, conf2 = _ransac_offset(
            anchors,
            threshold_ms=_INLIER_THRESH_MS * 3,
            seed_offsets=[coarse_ms, offset_ms],
        )
        if len(in2) > len(inliers):
            scale, offset_ms, inliers, conf, mode_used = 1.0, off2, in2, conf2, "offset"

    if verbose:
        log.info(
            "RANSAC: %d/%d inliers, offset=%+.3f s, scale=%.6f, confidence=%.0f%%",
            len(inliers), len(anchors), offset_ms / 1000, scale, conf * 100,
        )

    # ── 5. Segmented binary fallback (cross-language / gradual drift) ─────────
    # Onset impulses fail when subtitles are translated and events are merged
    # (e.g. Hebrew merging multiple English lines into one).  Binary occupancy
    # captures the overall speech-activity pattern, which is similar across
    # languages, and is better at detecting gradual linear drift.
    if conf < 0.3 or len(inliers) < _MIN_INLIERS:
        if verbose:
            log.info(
                "Low confidence (%.0f%%). Trying segmented binary alignment…", conf * 100
            )

        # Strip sound-descriptor-only events (SDH markers like "* *",
        # "[phone chimes]") before building binary fingerprints.  SDH tracks
        # contain 5-7 second sound-effect blocks that have no counterpart in a
        # translated subtitle; leaving them in creates 10× stronger false peaks
        # at wrong offsets that swamp the true dialogue-based correlation signal.
        ref_dial = _filter_dialogue_only(ref_fp_events)
        uns_dial = _filter_dialogue_only(uns_fp_events)
        ref_fp_bin = make_binary_fingerprint(
            ref_dial if ref_dial else ref_fp_events, ref_padded_dur, resolution_ms
        )
        uns_fp_bin = make_binary_fingerprint(
            uns_dial if uns_dial else uns_fp_events, uns_dur, resolution_ms
        )

        # Pass 1: full-range search — finds the right neighbourhood for each segment.
        seg_offsets = _segmented_binary_alignment(
            ref_fp_bin, uns_fp_bin, resolution_ms, max_offset_ms
        )

        if verbose:
            log.info("Segmented alignment pass 1: %d segments", len(seg_offsets))
            for t_ms, o_ms, snr in seg_offsets:
                log.info("  t=%5.0fs  local_offset=%+.3fs  snr=%.1f", t_ms / 1000, o_ms / 1000, snr)

        if len(seg_offsets) >= 2:
            seg_scale, seg_off_ms, seg_conf, seg_n_in = _fit_segment_offsets(seg_offsets)
            seg_n_total = len(seg_offsets)

            if verbose:
                log.info(
                    "Segmented fit pass 1: scale=%.6f, offset=%+.3fs, confidence=%.0f%%",
                    seg_scale, seg_off_ms / 1000, seg_conf * 100,
                )

            # Dynamic search radius for pass 2.
            #
            # When pass 1 captures some false-peak segments (e.g. sparse early
            # content in a movie) the fitted intercept can be off by several
            # seconds even though the scale is correct.  If we fix the radius at
            # 3 s, pass 2's search window for those segments will miss the true
            # peak entirely — perpetuating the error.
            #
            # We measure how far each pass-1 segment deviates from the fitted
            # line (residual).  The 90th-percentile residual tells us the worst
            # reliable deviation without being dominated by a single outlier.
            # The pass-2 radius is then set to cover that spread plus 2 s margin,
            # clamped between 3 s (normal case) and 20 s (very uncertain pass 1).
            _slope_p1 = seg_scale - 1.0
            _p1_residuals = [
                abs(o - (_slope_p1 * t + seg_off_ms))
                for t, o, _ in seg_offsets
            ]
            _p90_residual = float(np.percentile(_p1_residuals, 90)) if _p1_residuals else 3_000.0
            refine_radius_p2 = max(3_000, min(20_000, int(_p90_residual * 1.5) + 2_000))

            if verbose:
                log.info(
                    "Pass 2 search radius: %.1f s (p90 residual=%.1f s)",
                    refine_radius_p2 / 1000, _p90_residual / 1000,
                )

            # Pass 2: narrow search around each segment's predicted position.
            # This eliminates ambiguity from weak false peaks that pulled the
            # full-range estimate away from truth.
            seg_offsets2 = _segmented_binary_alignment(
                ref_fp_bin, uns_fp_bin, resolution_ms, max_offset_ms,
                seed_scale=seg_scale, seed_offset_ms=seg_off_ms,
                refine_radius_ms=float(refine_radius_p2),
            )
            if len(seg_offsets2) >= 2:
                seg_scale2, seg_off_ms2, seg_conf2, seg_n_in2 = _fit_segment_offsets(seg_offsets2)
                if verbose:
                    log.info(
                        "Segmented fit pass 2: scale=%.6f, offset=%+.3fs, confidence=%.0f%%",
                        seg_scale2, seg_off_ms2 / 1000, seg_conf2 * 100,
                    )
                    for t_ms, o_ms, snr in seg_offsets2:
                        log.info("  t=%5.0fs  local_offset=%+.3fs  snr=%.1f", t_ms / 1000, o_ms / 1000, snr)
                # Accept pass-2 result only if it's at least as good
                if seg_conf2 >= seg_conf:
                    seg_scale, seg_off_ms, seg_conf, seg_n_in = seg_scale2, seg_off_ms2, seg_conf2, seg_n_in2
                    seg_n_total = len(seg_offsets2)

            # Pass 3: event-pair refinement.
            #
            # The segmented binary model is accurate to within a few seconds.
            # We use it to predict each unsync event's reference position and
            # match to the nearest reference event within ±5 s.  RANSAC over
            # these point-level pairs (typically hundreds) gives a much tighter
            # (scale, offset) than the ~10–25 segment estimates above.
            #
            # Critical for PAL/NTSC drift: binary cross-correlation of a
            # 5-min Hebrew segment against a time-compressed English reference
            # finds a biased peak (±1–4 s per segment), causing the fitted
            # intercept to be off by up to 2.5 s.  Event-pair matching avoids
            # this smearing because each pair is a single onset measurement.
            ref_pairs = ref_dial if ref_dial else ref_fp_events
            uns_pairs = uns_dial if uns_dial else uns_fp_events
            refined = _refine_with_event_pairs(ref_pairs, uns_pairs, seg_scale, seg_off_ms)
            seg_segments: list[Segment] | None = None
            if refined is not None:
                r_scale, r_off, r_inliers, r_conf = refined
                n_matched = int(len(r_inliers) / r_conf) if r_conf > 0 else 0
                if verbose:
                    log.info(
                        "Event-pair refinement: scale=%.6f, offset=%+.3fs, "
                        "confidence=%.0f%%, %d/%d matched pairs",
                        r_scale, r_off / 1000, r_conf * 100,
                        len(r_inliers), n_matched,
                    )
                if len(r_inliers) >= 8:
                    seg_scale, seg_off_ms, seg_conf, seg_n_in = r_scale, r_off, r_conf, len(r_inliers)
                    seg_n_total = n_matched

                    # Piecewise-linear check on the high-quality event-pair
                    # inliers.  These are point matches (hundreds of them) so
                    # they are the best signal we ever get for detecting
                    # structural breaks (edit cuts, frame drops, VFR).
                    pw_segs, pw_diag = _piecewise_linear_fit(r_inliers, verbose=verbose)
                    if verbose:
                        log.info(
                            "Piecewise-linear search: BIC %s, best_k=%s",
                            pw_diag.get("bic"), pw_diag.get("best_k"),
                        )
                        if pw_diag.get("reason"):
                            log.info("  rejected: %s", pw_diag["reason"])
                    if pw_segs:
                        seg_segments = [
                            Segment(start_ms=s, end_ms=e, scale=sc, offset_ms=of)
                            for (s, e, sc, of) in pw_segs
                        ]
                        if verbose:
                            for i, seg in enumerate(seg_segments):
                                lo_s = "-inf" if seg.start_ms == float("-inf") else f"{seg.start_ms/1000:.1f}s"
                                hi_s = "+inf" if seg.end_ms == float("inf") else f"{seg.end_ms/1000:.1f}s"
                                log.info(
                                    "  segment %d: [%s, %s)  scale=%.6f  offset=%+.3fs",
                                    i + 1, lo_s, hi_s, seg.scale, seg.offset_ms / 1000,
                                )

            if seg_conf > conf or len(inliers) < _MIN_INLIERS:
                seg_mode = "linear" if abs(seg_scale - 1.0) > 1e-4 else "offset"
                if seg_segments and len(seg_segments) > 1:
                    seg_mode = "piecewise"
                return AlignResult(
                    offset_ms=float(seg_off_ms),
                    scale=float(seg_scale),
                    confidence=float(seg_conf),
                    n_anchors=seg_n_total,
                    n_inliers=seg_n_in,
                    coarse_offset_ms=coarse_ms,
                    mode_used=seg_mode,
                    segments=seg_segments,
                )

    # ── 6. Piecewise check on the main-path linear inliers ──────────────────
    # For tests that never enter the segmented fallback (Tests 1-3) we still
    # want to detect structural breaks.  The main windowed anchors are coarser
    # than event-pair anchors, so this check is conservative — it only fires
    # with plenty of inliers and a strong BIC win.
    final_segments: list[Segment] | None = None
    if mode_used == "linear" and len(inliers) >= _PIECEWISE_MIN_ANCHORS_PER_SEG * 2:
        pw_segs, pw_diag = _piecewise_linear_fit(inliers, verbose=verbose)
        if verbose:
            log.info(
                "Piecewise-linear search (main path): BIC %s, best_k=%s",
                pw_diag.get("bic"), pw_diag.get("best_k"),
            )
            if pw_diag.get("reason"):
                log.info("  rejected: %s", pw_diag["reason"])
        if pw_segs:
            final_segments = [
                Segment(start_ms=s, end_ms=e, scale=sc, offset_ms=of)
                for (s, e, sc, of) in pw_segs
            ]
            mode_used = "piecewise"
            if verbose:
                for i, seg in enumerate(final_segments):
                    lo_s = "-inf" if seg.start_ms == float("-inf") else f"{seg.start_ms/1000:.1f}s"
                    hi_s = "+inf" if seg.end_ms == float("inf") else f"{seg.end_ms/1000:.1f}s"
                    log.info(
                        "  segment %d: [%s, %s)  scale=%.6f  offset=%+.3fs",
                        i + 1, lo_s, hi_s, seg.scale, seg.offset_ms / 1000,
                    )

    return AlignResult(
        offset_ms=float(offset_ms),
        scale=float(scale),
        confidence=float(conf),
        n_anchors=len(anchors),
        n_inliers=len(inliers),
        coarse_offset_ms=coarse_ms,
        mode_used=mode_used,
        segments=final_segments,
    )
