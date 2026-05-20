"""Command-line interface for SubSyncPro."""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

# ── rich (optional but strongly recommended) ──────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.table import Table
    from rich.text import Text
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False

from subsyncpro import __version__
from subsyncpro.aligner import AlignResult, align
from subsyncpro.extractor import (
    extract_best_subtitle,
    format_track_table,
    list_subtitle_tracks,
)
from subsyncpro.parser import parse_subtitle_file
from subsyncpro.transformer import apply_transform, compute_delta_summary
from subsyncpro.writer import default_output_path, write_subtitle

# ── console setup ─────────────────────────────────────────────────────────────
if _HAS_RICH:
    console = Console(stderr=False)
    err_console = Console(stderr=True)
else:
    class _FallbackConsole:
        def print(self, *args, **kw):
            print(*args)
        def log(self, *args, **kw):
            print(*args, file=sys.stderr)
    console = err_console = _FallbackConsole()


def _print(msg: str, style: str = "") -> None:
    if _HAS_RICH and style:
        console.print(msg, style=style)
    else:
        print(msg)


def _err(msg: str, style: str = "bold red") -> None:
    if _HAS_RICH:
        err_console.print(f"[{style}]ERROR:[/{style}] {msg}")
    else:
        print(f"ERROR: {msg}", file=sys.stderr)


def _warn(msg: str) -> None:
    if _HAS_RICH:
        err_console.print(f"[bold yellow]WARNING:[/bold yellow] {msg}")
    else:
        print(f"WARNING: {msg}", file=sys.stderr)


# ── argument parser ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="subsyncpro",
        description=(
            "SubSyncPro — synchronise subtitles using a reference subtitle or MKV file.\n"
            "No audio required. Works even when the reference has extra content\n"
            "(previously-on segments, bonus scenes, etc.)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EXAMPLES,
    )

    p.add_argument("ref", metavar="REF",
                   help="Reference subtitle (.srt/.ass/.vtt) or video file (.mkv/.mp4/…) "
                        "with embedded subtitles.")
    p.add_argument("unsync", metavar="UNSYNC",
                   help="Unsynchronised subtitle to fix (.srt/.ass/.vtt).")

    # Output
    out = p.add_argument_group("Output")
    out.add_argument("-o", "--output", metavar="PATH",
                     help="Output path.  Default: UNSYNC with '.synced' inserted before extension.")
    out.add_argument("--overwrite", action="store_true",
                     help="Overwrite the original UNSYNC file in-place instead of creating "
                          "a new '.synced' copy alongside it.")
    out.add_argument("-f", "--format", metavar="FMT",
                     choices=["srt", "ass", "vtt", "auto"], default="auto",
                     help="Output format (default: auto — same as UNSYNC).")
    out.add_argument("--encoding", metavar="ENC",
                     help="Character encoding for reading subtitle files "
                          "(default: auto-detect).  Example: utf-8, cp1252.")
    out.add_argument("--output-encoding", metavar="ENC", default="utf-8",
                     help="Character encoding for writing the output file (default: utf-8).")

    # Alignment
    alg = p.add_argument_group("Alignment")
    alg.add_argument("-m", "--mode",
                     choices=["auto", "offset", "linear"], default="auto",
                     help="Alignment mode:\n"
                          "  offset  — constant shift only (fastest, most common)\n"
                          "  linear  — shift + slight speed correction (frame-rate drift)\n"
                          "  auto    — try linear; use offset if drift is negligible (default)")
    alg.add_argument("--max-offset", metavar="SECONDS", type=float, default=600.0,
                     help="Maximum expected timing difference in seconds (default: 600 = 10 min). "
                          "Increase for long-form content or when the reference covers multiple episodes.")
    alg.add_argument("--passes", metavar="N", type=int, default=3,
                     help="Maximum sync passes (default: 3).  After the first sync the tool "
                          "re-aligns the result to shave off any residual offset, stopping early "
                          "once a pass finds nothing worth applying.  Set 1 for a single pass.")
    alg.add_argument("--offset-hint", metavar="MS", type=float, default=None,
                     help="Rough offset hint in milliseconds.  Speeds up search when you already "
                          "know the approximate delay (e.g. from a previous run).")
    alg.add_argument("--warp", choices=["auto", "on", "off"], default="auto",
                     help="Dense piecewise-linear warp for non-linear drift (edit cuts).\n"
                          "  auto  — build a warp and keep it only if hold-out cross-validation\n"
                          "          shows it beats a single line (default; self-selects per file)\n"
                          "  on    — force the warp whenever one can be built\n"
                          "  off   — always use a single (scale, offset)")
    alg.add_argument("--lead-bias-ms", metavar="MS", type=float, default=0.0,
                     help="Constant bias (ms) added to every aligned timestamp AFTER alignment. "
                          "Use a negative value (e.g. -150) when the translated subtitle was "
                          "deliberately timed to appear before the audio for reading time, and "
                          "you want to remove that lead.  Default: 0 (keep translator intent).")

    # MKV options
    mkv = p.add_argument_group("MKV / reference video options")
    mkv.add_argument("--list-tracks", action="store_true",
                     help="List subtitle tracks in REF (if it is a video file) and exit.")
    mkv.add_argument("--ref-track", metavar="IDX", type=int, default=None,
                     help="Stream index of the subtitle track to use from a video file. "
                          "Use --list-tracks to see available indices.")
    mkv.add_argument("--ref-lang", metavar="LANG", default=None,
                     help="Preferred language code when auto-selecting from a video file "
                          "(e.g. 'eng', 'jpn').  Default: English.")
    mkv.add_argument("--ffprobe-timeout", metavar="SEC", type=int, default=120,
                     help="Seconds to wait for ffprobe when reading a video file (default: 120). "
                          "Increase on slow HDD servers with large files.")
    mkv.add_argument("--ffmpeg-timeout", metavar="SEC", type=int, default=300,
                     help="Seconds to wait for ffmpeg when extracting a subtitle track (default: 300). "
                          "Increase on slow HDD servers with large files.")

    # Performance
    perf = p.add_argument_group("Performance")
    perf.add_argument("--workers", metavar="N", type=int, default=0,
                      help="CPU threads for the FFT correlation stages "
                           "(default: 0 = auto, use all logical cores).  Set to 1 to force "
                           "single-threaded, or a specific number to cap CPU usage.")

    # Behaviour
    beh = p.add_argument_group("Behaviour")
    beh.add_argument("-n", "--dry-run", action="store_true",
                     help="Compute alignment and display results without writing any file.")
    beh.add_argument("-v", "--verbose", action="store_true",
                     help="Detailed progress output.")
    beh.add_argument("--version", action="version", version=f"SubSyncPro {__version__}")
    beh.add_argument("--confidence-threshold", metavar="0-1", type=float, default=0.2,
                     help="Warn (but still write) when confidence is below this value (default: 0.3).")

    return p


_EXAMPLES = """
examples:
  # Basic usage — sync using a reference SRT
  subsyncpro reference.srt unsynced.srt

  # Use an MKV file as reference (auto-selects best subtitle track)
  subsyncpro episode.mkv unsynced.srt

  # List subtitle tracks inside an MKV before choosing
  subsyncpro --list-tracks episode.mkv unsynced.srt

  # Pick a specific track (e.g. track 3) and use ASS format for output
  subsyncpro --ref-track 3 episode.mkv unsynced.srt -f ass

  # Overwrite the original file in-place
  subsyncpro reference.srt unsynced.srt --overwrite

  # Dry run — show what offset would be applied without writing
  subsyncpro reference.srt unsynced.srt --dry-run --verbose

  # Allow up to 30-minute offset (for content with very long gaps)
  subsyncpro reference.srt unsynced.srt --max-offset 1800

  # Force frame-rate drift correction
  subsyncpro reference.srt unsynced.srt --mode linear

  # Specify output path and encoding
  subsyncpro reference.srt unsynced.srt -o fixed.srt --encoding cp1252
"""


# ── external tool validation ──────────────────────────────────────────────────

def _check_ffmpeg() -> None:
    """Verify ffprobe and ffmpeg are available.  Exit with clear instructions if not."""
    import shutil
    missing = [t for t in ("ffprobe", "ffmpeg") if shutil.which(t) is None]
    if not missing:
        return
    tools = " and ".join(missing)
    _err(
        f"{tools} not found in PATH.\n\n"
        "SubSyncPro needs FFmpeg to read subtitle tracks from video files.\n\n"
        "  Install FFmpeg:\n"
        "    Windows : winget install Gyan.FFmpeg\n"
        "              or download from https://ffmpeg.org/download.html\n"
        "    macOS   : brew install ffmpeg\n"
        "    Linux   : sudo apt install ffmpeg   (or your distro's package manager)\n\n"
        "After installing, open a new terminal so the updated PATH takes effect,\n"
        "then run SubSyncPro again."
    )
    sys.exit(1)


# ── reference file handling ───────────────────────────────────────────────────

_VIDEO_EXTS = {".mkv", ".mp4", ".avi", ".m4v", ".ts", ".m2ts", ".mov", ".webm"}

_SDH_MARKERS = {"sdh", "hi", "hearing.impaired", "hearing_impaired", "cc"}


def _is_video(path: Path) -> bool:
    return path.suffix.lower() in _VIDEO_EXTS


def _is_sdh_subtitle(path: Path) -> bool:
    """Return True if the filename signals that this is an SDH/HI subtitle.

    Matches common naming patterns: movie.sdh.srt, movie.hi.srt,
    movie.hearing.impaired.srt, movie.en.sdh.srt, etc.
    """
    # Strip all extensions and check parts of the stem
    stem = path.stem.lower()
    for marker in _SDH_MARKERS:
        if f".{marker}." in f".{stem}." or stem.endswith(f".{marker}") or stem.endswith(f"_{marker}"):
            return True
    return False


def _load_ref(args: argparse.Namespace) -> tuple[list, str, dict, Optional[Path]]:
    """Load reference subtitle events.  Returns (events, fmt, meta, tmp_dir)."""
    ref_path = Path(args.ref)
    if not ref_path.exists():
        _err(f"Reference file not found: {ref_path}")
        sys.exit(1)

    tmp_dir: Optional[Path] = None

    if _is_video(ref_path):
        unsync_is_sdh = _is_sdh_subtitle(Path(args.unsync))
        tmp_dir = Path(tempfile.mkdtemp(prefix="subsyncpro_ref_"))
        try:
            extracted, track = extract_best_subtitle(
                ref_path,
                preferred_lang=args.ref_lang,
                preferred_index=args.ref_track,
                output_dir=tmp_dir,
                prefer_sdh=unsync_is_sdh,
            )
        except RuntimeError as exc:
            _err(str(exc))
            sys.exit(1)

        sdh_note = " [SDH]" if track.hearing_impaired else ""
        _print(
            f"  Using embedded subtitle track {track.index} "
            f"({track.codec}, lang={track.language}{sdh_note})"
            + (f", title='{track.title}'" if track.title else ""),
            style="dim",
        )
        ref_path = extracted

    events, fmt, meta = parse_subtitle_file(ref_path, encoding=args.encoding)
    return events, fmt, meta, tmp_dir


# ── display helpers ───────────────────────────────────────────────────────────

def _render_result(result: AlignResult, delta: dict, dry_run: bool) -> None:
    if not _HAS_RICH:
        sign = "+" if result.offset_ms >= 0 else ""
        print(f"\n  Offset:     {sign}{result.offset_ms/1000:.3f} s")
        if abs(result.scale - 1.0) > 1e-4:
            print(f"  Scale:      {result.scale:.6f}")
        print(f"  Confidence: {result.confidence:.0%}  ({result.n_inliers}/{result.n_anchors} anchors)")
        print(f"  Mode:       {result.mode_used}")
        if delta:
            print(f"  Events:     {delta['n_events']}")
        return

    tbl = Table(show_header=False, box=None, padding=(0, 2))
    tbl.add_column(style="bold cyan", width=14)
    tbl.add_column()

    sign = "+" if result.offset_ms >= 0 else ""
    tbl.add_row("Offset", f"{sign}{result.offset_ms/1000:.3f} s")
    if abs(result.scale - 1.0) > 1e-4:
        tbl.add_row("Scale", f"{result.scale:.6f}  (speed drift correction)")
    tbl.add_row(
        "Confidence",
        f"[{'green' if result.confidence >= 0.6 else 'yellow' if result.confidence >= 0.3 else 'red'}]"
        f"{result.confidence:.0%}[/]  ({result.n_inliers}/{result.n_anchors} anchor pairs)"
    )
    tbl.add_row("Mode", result.mode_used)
    if delta:
        tbl.add_row("Events", str(delta["n_events"]))

    title = "[bold]Dry-run result[/bold]" if dry_run else "[bold green]Synchronised[/bold green]"
    console.print(Panel(tbl, title=title, border_style="blue"))


# ── iterative (multi-pass) sync ───────────────────────────────────────────────

# A refinement pass re-aligns the already-synced subtitle against the reference.
# After the first pass the timing is close, so the windowed anchor search locks
# on with high confidence and can shave off any residual the first pass left
# (common when the first pass had to use the low-confidence cross-language
# fallback).  Passes stop early once a pass finds nothing worth applying.
_CONVERGE_OFFSET_MS: float = 60.0    # residual shift below this = converged
_CONVERGE_SCALE: float = 5e-5        # residual scale within this of 1 = converged
_REFINE_MIN_CONF: float = 0.5        # don't trust a refinement below this confidence
_REFINE_MAX_OFFSET_MS: float = 5_000.0  # a refinement shouldn't find a huge new shift


def _shift_events(events: list, ms: float):
    """Apply a constant millisecond shift (used for --lead-bias-ms)."""
    if not ms:
        return events
    bias = AlignResult(offset_ms=float(ms), scale=1.0, confidence=1.0,
                       n_anchors=0, n_inliers=0, coarse_offset_ms=0.0,
                       mode_used="lead-bias")
    return apply_transform(events, bias)


def _sync_passes(
    ref_events: list,
    unsync_events: list,
    *,
    mode: str,
    max_offset_s: float,
    verbose: bool,
    workers: int,
    warp: str,
    max_passes: int,
):
    """Align, then re-align the synced result up to *max_passes* times.

    Returns (synced_events, [AlignResult per applied pass]).  The first pass is
    always applied; each subsequent pass is applied only if it finds a
    trustworthy, non-trivial, plausibly-sized correction — otherwise iteration
    stops (the result has converged).

    A non-linear warp (if any) is only considered on the first pass; refinement
    passes are pure linear clean-ups of the residual offset / frame-rate error.
    """
    working = unsync_events
    applied: list[AlignResult] = []
    for p in range(max(1, max_passes)):
        warp_this_pass = warp if p == 0 else "off"
        r = align(ref_events, working, mode=mode, max_offset_s=max_offset_s,
                  verbose=verbose, workers=workers, warp=warp_this_pass)
        has_warp = bool(r.segments and len(r.segments) > 1)

        if p > 0:
            resid_off = abs(r.offset_ms)
            resid_scale = abs(r.scale - 1.0)
            if not has_warp and resid_off < _CONVERGE_OFFSET_MS and resid_scale < _CONVERGE_SCALE:
                if verbose:
                    logging.info("Pass %d: residual %+.0f ms / scale %.6f within tolerance — converged.",
                                 p + 1, r.offset_ms, r.scale)
                break
            if r.confidence < _REFINE_MIN_CONF:
                if verbose:
                    logging.info("Pass %d: confidence %.0f%% too low to trust a refinement — stopping.",
                                 p + 1, r.confidence * 100)
                break
            if not has_warp and resid_off > _REFINE_MAX_OFFSET_MS:
                if verbose:
                    logging.info("Pass %d: implausibly large new shift %+.0f ms — stopping.",
                                 p + 1, r.offset_ms)
                break
            if verbose:
                logging.info("Pass %d: applying refinement offset=%+.0f ms scale=%.6f conf=%.0f%%",
                             p + 1, r.offset_ms, r.scale, r.confidence * 100)

        working = apply_transform(working, r)
        applied.append(r)

    return working, applied


def _composed_result(orig: list, synced: list, passes: list[AlignResult]) -> AlignResult:
    """Build a display AlignResult describing the *net* transform across passes.

    The written file is *synced* itself; this is only for the result panel, so
    the effective (scale, offset) is recovered by a least-squares fit of the
    final timestamps against the originals.
    """
    import numpy as np
    base = passes[0]
    x = np.array([e.start_ms for e in orig], dtype=float)
    y = np.array([e.start_ms for e in synced], dtype=float)
    if len(x) >= 2 and float(x.max() - x.min()) > 0:
        a = np.vstack([x, np.ones_like(x)]).T
        s, o = np.linalg.lstsq(a, y, rcond=None)[0]
        s, o = float(s), float(o)
    else:
        s, o = 1.0, (float(np.median(y - x)) if len(x) else 0.0)
    n = len(passes)
    mode = base.mode_used + (f"  ({n} passes)" if n > 1 else "")
    return AlignResult(offset_ms=o, scale=s, confidence=base.confidence,
                       n_anchors=base.n_anchors, n_inliers=base.n_inliers,
                       coarse_offset_ms=base.coarse_offset_ms, mode_used=mode,
                       segments=None)


# ── core sync function (reused by __init__.align_subtitles) ──────────────────

def _run_sync(
    ref_path: str,
    unsync_path: str,
    output_path: Optional[str],
    *,
    mode: str,
    max_offset_s: float,
    ref_lang: Optional[str],
    ref_track: Optional[int],
    encoding: Optional[str],
    verbose: bool,
    dry_run: bool,
    output_format: str,
    lead_bias_ms: float = 0.0,
    workers: int = 0,
    warp: str = "auto",
    passes: int = 3,
) -> dict:
    """Internal sync runner shared by CLI and programmatic API."""
    # ── Load reference ────────────────────────────────────────────────────
    ref_p = Path(ref_path)
    tmp_dir = None
    if _is_video(ref_p):
        tmp_dir = Path(tempfile.mkdtemp(prefix="subsyncpro_ref_"))
        try:
            extracted, track = extract_best_subtitle(
                ref_p, preferred_lang=ref_lang,
                preferred_index=ref_track, output_dir=tmp_dir,
            )
            ref_p = extracted
        except RuntimeError as exc:
            raise RuntimeError(f"MKV extraction failed: {exc}") from exc

    ref_events, _, _ = parse_subtitle_file(ref_p, encoding=encoding)
    unsync_events, unsync_fmt, unsync_meta = parse_subtitle_file(unsync_path, encoding=encoding)

    if verbose:
        logging.info(
            "Loaded %d ref events, %d unsync events",
            len(ref_events), len(unsync_events),
        )

    # ── Align (iteratively) ───────────────────────────────────────────────
    synced_events, pass_results = _sync_passes(
        ref_events, unsync_events,
        mode=mode,
        max_offset_s=max_offset_s,
        verbose=verbose,
        workers=workers,
        warp=warp,
        max_passes=passes,
    )
    if lead_bias_ms:
        synced_events = _shift_events(synced_events, lead_bias_ms)
    result = _composed_result(unsync_events, synced_events, pass_results)

    # ── Write output ──────────────────────────────────────────────────────
    out_fmt = output_format if output_format != "auto" else unsync_fmt
    if output_path is None:
        output_path = str(default_output_path(unsync_path, out_fmt))

    if not dry_run:
        write_subtitle(synced_events, output_path, fmt=out_fmt, metadata=unsync_meta)

    # Cleanup temp dir
    if tmp_dir and tmp_dir.exists():
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "offset_ms": result.offset_ms,
        "scale": result.scale,
        "confidence": result.confidence,
        "n_anchors": result.n_anchors,
        "n_inliers": result.n_inliers,
        "mode_used": result.mode_used,
        "output_path": output_path,
        "dry_run": dry_run,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ── Logging ───────────────────────────────────────────────────────────
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        format="%(levelname)s %(name)s: %(message)s",
        level=log_level,
        stream=sys.stderr,
    )

    # ── Validate mutually exclusive flags ────────────────────────────────────
    if args.overwrite and args.output:
        _err("--overwrite and --output are mutually exclusive.")
        sys.exit(1)

    # ── Check for required external tools before touching video files ─────────
    ref_path = Path(args.ref)
    if _is_video(ref_path):
        _check_ffmpeg()

    # ── --list-tracks shortcut (does not need UNSYNC to exist) ───────────────
    if args.list_tracks:
        if not _is_video(ref_path):
            _err("--list-tracks requires REF to be a video file.")
            sys.exit(1)
        if not ref_path.exists():
            _err(f"File not found: {ref_path}")
            sys.exit(1)
        _print(f"Subtitle tracks in [bold]{ref_path.name}[/bold]:" if _HAS_RICH else
               f"Subtitle tracks in {ref_path.name}:")
        try:
            tracks = list_subtitle_tracks(ref_path, timeout=args.ffprobe_timeout)
        except RuntimeError as e:
            _err(str(e))
            sys.exit(1)
        print(format_track_table(tracks))
        sys.exit(0)

    # ── Validate UNSYNC now (after --list-tracks which doesn't need it) ──────
    unsync_path = Path(args.unsync)
    if not unsync_path.exists():
        _err(f"Unsynchronised subtitle not found: {unsync_path}")
        sys.exit(1)

    # ── Banner ─────────────────────────────────────────────────────────────
    if _HAS_RICH:
        console.print(
            f"[bold blue]SubSyncPro[/bold blue] [dim]v{__version__}[/dim]  "
            f"[dim]ref=[/dim][cyan]{ref_path.name}[/cyan]  "
            f"[dim]unsync=[/dim][cyan]{unsync_path.name}[/cyan]"
        )
    else:
        print(f"SubSyncPro v{__version__}  ref={ref_path.name}  unsync={unsync_path.name}")

    t0 = time.perf_counter()

    # ── Progress indicator ─────────────────────────────────────────────────
    progress_ctx = None
    if _HAS_RICH and not args.verbose:
        progress_ctx = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=30),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        )

    try:
        # ── Load reference ────────────────────────────────────────────────
        if progress_ctx:
            progress_ctx.start()
            task = progress_ctx.add_task("Loading reference…", total=None)

        ref_events, _, _ = _load_ref_raw(args)

        if progress_ctx:
            progress_ctx.update(task, description="Parsing unsync subtitle…")

        unsync_events, unsync_fmt, unsync_meta = parse_subtitle_file(
            unsync_path, encoding=args.encoding
        )
        _print(
            f"  Reference: [green]{len(ref_events)}[/green] events  |  "
            f"Unsync: [green]{len(unsync_events)}[/green] events"
            if _HAS_RICH else
            f"  Reference: {len(ref_events)} events  |  Unsync: {len(unsync_events)} events",
            style="dim",
        )

        if not ref_events:
            _err("Reference subtitle appears to be empty.")
            sys.exit(1)
        if not unsync_events:
            _err("Unsynchronised subtitle appears to be empty.")
            sys.exit(1)

        # ── Align (iteratively) ───────────────────────────────────────────
        if progress_ctx:
            progress_ctx.update(task, description="Aligning… (fingerprinting + RANSAC)")

        synced_events, pass_results = _sync_passes(
            ref_events, unsync_events,
            mode=args.mode,
            max_offset_s=args.max_offset,
            verbose=args.verbose,
            workers=args.workers,
            warp=args.warp,
            max_passes=args.passes,
        )
        if args.lead_bias_ms:
            synced_events = _shift_events(synced_events, args.lead_bias_ms)
        result = _composed_result(unsync_events, synced_events, pass_results)

        # ── Write ──────────────────────────────────────────────────────────
        if progress_ctx:
            progress_ctx.update(task, description="Writing output…")

        delta = compute_delta_summary(unsync_events, synced_events)

        out_fmt = args.format if args.format != "auto" else unsync_fmt
        if args.overwrite:
            out_path = unsync_path
        elif args.output:
            out_path = Path(args.output)
        else:
            out_path = default_output_path(unsync_path, out_fmt)

        if not args.dry_run:
            if progress_ctx:
                progress_ctx.update(task, description="Writing output…")
            write_subtitle(synced_events, out_path, fmt=out_fmt, metadata=unsync_meta,
                           encoding=args.output_encoding)

        if progress_ctx:
            progress_ctx.stop()
            progress_ctx = None

    except KeyboardInterrupt:
        if progress_ctx:
            progress_ctx.stop()
        print("\nAborted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        if progress_ctx:
            progress_ctx.stop()
        _err(str(exc))
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    elapsed = time.perf_counter() - t0

    # ── Confidence warning ────────────────────────────────────────────────
    if result.confidence < args.confidence_threshold:
        _warn(
            f"Low confidence ({result.confidence:.0%}).  The alignment may be incorrect.\n"
            "  Try: --max-offset (larger value), --mode linear, or check that the reference\n"
            "  subtitle actually matches this episode."
        )
    elif not result.is_reliable:
        _warn(
            f"Only {result.n_inliers} anchor points found.  Result may be unreliable.\n"
            "  Try --verbose to inspect the alignment quality."
        )

    # ── Result display ────────────────────────────────────────────────────
    _render_result(result, delta, args.dry_run)

    if not args.dry_run:
        _print(
            f"  [dim]Written →[/dim] [bold]{out_path}[/bold]  "
            f"[dim]({elapsed:.1f} s)[/dim]"
            if _HAS_RICH else
            f"  Written → {out_path}  ({elapsed:.1f} s)",
            style="",
        )
    else:
        _print(
            "  [dim italic]Dry run — no file written.[/dim italic]" if _HAS_RICH
            else "  Dry run — no file written.",
        )


def _load_ref_raw(args: argparse.Namespace):
    """Load reference and return (events, fmt, meta)."""
    ref_path = Path(args.ref)
    if not ref_path.exists():
        _err(f"Reference file not found: {ref_path}")
        sys.exit(1)

    if _is_video(ref_path):
        # Match the MKV track type to the target subtitle: if the target is SDH
        # prefer an SDH track; otherwise prefer the standard (non-SDH) track so
        # event density is comparable.
        unsync_is_sdh = _is_sdh_subtitle(Path(args.unsync))
        tmp_dir = Path(tempfile.mkdtemp(prefix="subsyncpro_ref_"))
        try:
            extracted, track, tool_used = extract_best_subtitle(
                ref_path,
                preferred_lang=args.ref_lang,
                preferred_index=args.ref_track,
                output_dir=tmp_dir,
                prefer_sdh=unsync_is_sdh,
                ffprobe_timeout=args.ffprobe_timeout,
                ffmpeg_timeout=args.ffmpeg_timeout,
            )
        except RuntimeError as exc:
            _err(str(exc))
            sys.exit(1)
        sdh_note = " [SDH]" if track.hearing_impaired else ""
        _print(
            f"  MKV track {track.index}: [cyan]{track.codec}[/cyan] "
            f"lang=[cyan]{track.language}[/cyan]{sdh_note}"
            + (f" title='{track.title}'" if track.title else "")
            + f" [dim](via {tool_used})[/dim]"
            if _HAS_RICH else
            f"  MKV track {track.index}: {track.codec} lang={track.language}{sdh_note}"
            + (f" title='{track.title}'" if track.title else "")
            + f" (via {tool_used})",
            style="dim",
        )
        ref_path = extracted

    return parse_subtitle_file(ref_path, encoding=args.encoding)


if __name__ == "__main__":
    main()
