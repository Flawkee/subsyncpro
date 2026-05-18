"""MKV subtitle track extractor.

Uses ffprobe (to list tracks) and ffmpeg (to extract) from the system PATH.
Both are part of the standard FFmpeg distribution.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ── data ─────────────────────────────────────────────────────────────────────

@dataclass
class SubtitleTrack:
    index: int
    codec: str
    language: str
    title: str
    forced: bool
    default: bool
    hearing_impaired: bool
    event_count: int

    @property
    def is_text_based(self) -> bool:
        return self.codec.lower() in ("subrip", "srt", "ass", "ssa", "webvtt", "mov_text", "text")

    @property
    def format_hint(self) -> str:
        c = self.codec.lower()
        if c in ("ass", "ssa"):
            return "ass"
        if c in ("webvtt",):
            return "vtt"
        return "srt"

    def score(self, preferred_lang: Optional[str] = None, prefer_sdh: bool = False) -> int:
        """Higher is better.

        *prefer_sdh* should be True only when the unsynchronised target subtitle
        is itself an SDH/HI file; otherwise a standard (non-SDH) track is a
        better reference because it has the same event density as the target.
        """
        s = 0
        # Prefer text-based over image-based (PGS, DVDSUB)
        if self.is_text_based:
            s += 100
        # Prefer non-forced (full subtitles — forced tracks cover only foreign dialogue)
        if not self.forced:
            s += 30
        # SDH preference depends on the target subtitle type.
        if prefer_sdh:
            if self.hearing_impaired:
                s += 25  # target is SDH — match it
        else:
            if not self.hearing_impaired:
                s += 20  # target is standard — avoid SDH's extra sound-effect events
        # Prefer default track
        if self.default:
            s += 10
        # Language match
        if preferred_lang:
            lang = preferred_lang.lower()
            if self.language.lower() in (lang, lang[:2], lang[:3]):
                s += 200
        elif self.language.lower() in ("eng", "en", "english"):
            s += 50
        # Prefer more events (denser subtitle tracks)
        s += min(self.event_count, 1000) // 50
        return s


# ── ffprobe / ffmpeg detection ───────────────────────────────────────────────

def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(
            f"'{name}' not found in PATH.\n"
            "Install FFmpeg from https://ffmpeg.org/download.html and ensure it is on your PATH."
        )
    return path


# ── track listing ─────────────────────────────────────────────────────────────

def list_subtitle_tracks(mkv_path: str | Path) -> list[SubtitleTrack]:
    """Return all subtitle tracks found in the container."""
    ffprobe = _require_binary("ffprobe")
    cmd = [
        ffprobe, "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "s",
        str(mkv_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffprobe timed out while reading the MKV file.")

    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{result.stderr.strip()}")

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []

    tracks: list[SubtitleTrack] = []
    for stream in data.get("streams", []):
        if stream.get("codec_type") != "subtitle":
            continue

        tags = stream.get("tags", {})
        disp = stream.get("disposition", {})

        language = (
            tags.get("language")
            or tags.get("LANGUAGE")
            or "und"
        ).lower()
        title = tags.get("title") or tags.get("TITLE") or ""
        codec = stream.get("codec_name", "unknown")

        # Estimate event count from nb_read_frames / nb_frames if available
        event_count = int(stream.get("nb_read_frames") or stream.get("nb_frames") or 0)

        tracks.append(SubtitleTrack(
            index=stream["index"],
            codec=codec,
            language=language,
            title=title,
            forced=bool(disp.get("forced")),
            default=bool(disp.get("default")),
            hearing_impaired=bool(disp.get("hearing_impaired")) or _is_sdh(title),
            event_count=event_count,
        ))

    return tracks


def _is_sdh(title: str) -> bool:
    t = title.lower()
    return any(kw in t for kw in ("sdh", "hearing impaired", "hi ", "[hi]", "(hi)"))


def select_best_track(
    tracks: list[SubtitleTrack],
    preferred_lang: Optional[str] = None,
    preferred_index: Optional[int] = None,
    prefer_sdh: bool = False,
) -> Optional[SubtitleTrack]:
    """Pick the best subtitle track, optionally honouring a user preference."""
    if not tracks:
        return None
    if preferred_index is not None:
        for t in tracks:
            if t.index == preferred_index:
                return t
        raise ValueError(
            f"Track index {preferred_index} not found. "
            f"Available: {[t.index for t in tracks]}"
        )
    text_tracks = [t for t in tracks if t.is_text_based]
    pool = text_tracks if text_tracks else tracks
    return max(pool, key=lambda t: t.score(preferred_lang, prefer_sdh))


# ── extraction ────────────────────────────────────────────────────────────────

def extract_subtitle_track(
    mkv_path: str | Path,
    track_index: int,
    output_path: str | Path,
) -> Path:
    """Extract a specific subtitle track to *output_path* using ffmpeg."""
    ffmpeg = _require_binary("ffmpeg")
    out = Path(output_path)
    cmd = [
        ffmpeg, "-v", "warning",
        "-i", str(mkv_path),
        "-map", f"0:{track_index}",
        "-c:s", "copy",
        "-y",           # overwrite
        str(out),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffmpeg timed out while extracting subtitle track.")

    if result.returncode != 0:
        # Some codecs need explicit conversion
        cmd2 = [
            ffmpeg, "-v", "warning",
            "-i", str(mkv_path),
            "-map", f"0:{track_index}",
            "-y",
            str(out),
        ]
        result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=120)
        if result2.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed to extract track {track_index}:\n{result2.stderr.strip()}"
            )

    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(
            f"ffmpeg produced an empty file for track {track_index}. "
            "The track may be image-based (PGS/DVDSUB) and cannot be used as reference."
        )
    return out


def extract_best_subtitle(
    mkv_path: str | Path,
    preferred_lang: Optional[str] = None,
    preferred_index: Optional[int] = None,
    output_dir: Optional[str | Path] = None,
    prefer_sdh: bool = False,
) -> tuple[Path, SubtitleTrack]:
    """Find and extract the best subtitle track from an MKV.

    Returns (extracted_path, track_info).
    Caller is responsible for cleaning up the file if *output_dir* is a temp dir.
    """
    mkv = Path(mkv_path)
    if not mkv.exists():
        raise FileNotFoundError(f"MKV file not found: {mkv}")

    tracks = list_subtitle_tracks(mkv)
    if not tracks:
        raise RuntimeError(f"No subtitle tracks found in {mkv.name}")

    track = select_best_track(tracks, preferred_lang, preferred_index, prefer_sdh)
    if track is None:
        raise RuntimeError("Could not select a subtitle track.")

    if not track.is_text_based:
        raise RuntimeError(
            f"Selected track {track.index} is image-based ({track.codec}). "
            "SubSyncPro requires a text-based subtitle track (SRT, ASS, VTT).\n"
            f"Available text tracks: {[t.index for t in tracks if t.is_text_based] or 'none'}"
        )

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="subsyncpro_")

    ext_map = {"ass": ".ass", "vtt": ".vtt", "srt": ".srt"}
    ext = ext_map.get(track.format_hint, ".srt")
    out_path = Path(output_dir) / f"ref_track{track.index}{ext}"

    extracted = extract_subtitle_track(mkv, track.index, out_path)
    return extracted, track


# ── human-readable track table ────────────────────────────────────────────────

def format_track_table(tracks: list[SubtitleTrack]) -> str:
    if not tracks:
        return "  (no subtitle tracks found)"
    lines = [
        f"  {'IDX':>3}  {'CODEC':<12}  {'LANG':<6}  {'DEF':>3}  {'FORCED':>6}  {'SDH':>4}  TITLE",
        "  " + "-" * 70,
    ]
    for t in tracks:
        lines.append(
            f"  {t.index:>3}  {t.codec:<12}  {t.language:<6}  "
            f"{'yes' if t.default else 'no':>3}  "
            f"{'yes' if t.forced else 'no':>6}  "
            f"{'yes' if t.hearing_impaired else 'no':>4}  "
            f"{t.title or '-'}"
        )
    return "\n".join(lines)
