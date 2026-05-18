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


# ── result type ───────────────────────────────────────────────────────────────

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
        return s


# ── coarse FFT alignment ──────────────────────────────────────────────────────

def _coarse_offset(ref_fp: np.ndarray, uns_fp: np.ndarray, resolution_ms: int) -> float:
    """Global cross-correlation via FFT. Returns best offset in ms.

    Uses fftconvolve(ref, uns_reversed, 'full') so that result[k] is the
    dot-product of ref starting at position (k - len_uns + 1) with uns.
    Peak at k0 → offset_samples = k0 - (len_uns - 1).
    """
    corr = fftconvolve(ref_fp, uns_fp[::-1], mode="full")
    best_k = int(np.argmax(corr))
    offset_samples = best_k - (len(uns_fp) - 1)
    # Clamp to reasonable range to avoid wrap-around artefacts
    max_samples = max(len(ref_fp), len(uns_fp))
    offset_samples = int(np.clip(offset_samples, -max_samples, max_samples))
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
        ref_start = int(np.argmax(corr))

        # Normalise by the energy of the zero-mean window (L2 norm).
        # This gives a Pearson-like score in [0, 1] when both signals align.
        window_energy = float(np.dot(window_zm, window_zm))
        if window_energy < 1e-9:
            continue
        score = float(corr[ref_start]) / window_energy

        if score < _MIN_SCORE:
            continue

        ref_center_ms = (ref_start + win_n // 2) * resolution_ms
        uns_center_ms = (pos + win_n // 2) * resolution_ms

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
            local_idx = int(np.argmax(corr))
            best_k = lo + local_idx
        else:
            corr = fftconvolve(ref_fp_bin, uns_zm[::-1], mode="valid")
            if len(corr) == 0:
                continue
            local_idx = int(np.argmax(corr))
            best_k = local_idx

        local_off_ms = float((best_k - seg_start) * resolution_ms)

        if not narrow and abs(local_off_ms) > max_offset_ms:
            continue  # plausibility filter only needed for full-range pass

        # SNR: correlation peak value divided by mean of ±radius neighbourhood.
        lo_w = max(0, local_idx - radius_n)
        hi_w = min(len(corr), local_idx + radius_n + 1)
        noise = float(np.mean(np.abs(corr[lo_w:hi_w])))
        snr = float(corr[local_idx]) / (noise + 1e-9)

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
    confidence = n_inliers / n if n > 0 else 0.0
    return scale, float(best_intercept), confidence, n_inliers


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
    ref_dur = fingerprint_duration_ms(ref_events)
    uns_dur = fingerprint_duration_ms(unsync_events)
    # ref is padded to cover the full search range; unsync keeps its natural size.
    # We add max_offset_ms to ref's right so that windows at the end of unsync
    # can still find a match even if ref ends earlier.
    ref_padded_dur = ref_dur + int(max_offset_ms) + 60_000

    ref_fp = make_fingerprint(ref_events, ref_padded_dur, resolution_ms)
    uns_fp = make_fingerprint(unsync_events, uns_dur, resolution_ms)

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
        ref_dial = _filter_dialogue_only(ref_events)
        uns_dial = _filter_dialogue_only(unsync_events)
        ref_fp_bin = make_binary_fingerprint(
            ref_dial if ref_dial else ref_events, ref_padded_dur, resolution_ms
        )
        uns_fp_bin = make_binary_fingerprint(
            uns_dial if uns_dial else unsync_events, uns_dur, resolution_ms
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

            if verbose:
                log.info(
                    "Segmented fit pass 1: scale=%.6f, offset=%+.3fs, confidence=%.0f%%",
                    seg_scale, seg_off_ms / 1000, seg_conf * 100,
                )

            # Pass 2: narrow search (±3 s) around each segment's predicted position.
            # This eliminates ambiguity from weak false peaks that pulled the
            # full-range estimate away from truth.
            seg_offsets2 = _segmented_binary_alignment(
                ref_fp_bin, uns_fp_bin, resolution_ms, max_offset_ms,
                seed_scale=seg_scale, seed_offset_ms=seg_off_ms,
                refine_radius_ms=3_000,
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

            if seg_conf > conf or len(inliers) < _MIN_INLIERS:
                seg_mode = "linear" if abs(seg_scale - 1.0) > 1e-4 else "offset"
                return AlignResult(
                    offset_ms=float(seg_off_ms),
                    scale=float(seg_scale),
                    confidence=float(seg_conf),
                    n_anchors=len(seg_offsets),
                    n_inliers=seg_n_in,
                    coarse_offset_ms=coarse_ms,
                    mode_used=seg_mode,
                )

    return AlignResult(
        offset_ms=float(offset_ms),
        scale=float(scale),
        confidence=float(conf),
        n_anchors=len(anchors),
        n_inliers=len(inliers),
        coarse_offset_ms=coarse_ms,
        mode_used=mode_used,
    )
