"""Subtitle file parser — SRT, ASS/SSA, VTT.

Returns a unified list of SubtitleEvent objects with millisecond timestamps.
All text is kept as-is (HTML/ASS tags preserved) so the writer can round-trip.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import chardet


@dataclass
class SubtitleEvent:
    start_ms: int
    end_ms: int
    text: str
    index: int = 0
    raw_extras: dict = field(default_factory=dict)


# ── timestamp helpers ───────────────────────────────────────────────────────

_SRT_TS = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)
_VTT_TS = re.compile(
    r"(?:(\d{1,2}):)?(\d{2}):(\d{2})\.(\d{1,3})"
)


def _srt_ts_to_ms(h: str, m: str, s: str, ms: str) -> int:
    frac = int(ms.ljust(3, "0"))
    return int(h) * 3_600_000 + int(m) * 60_000 + int(s) * 1_000 + frac


def ms_to_srt_ts(ms: int) -> str:
    ms = max(0, ms)
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, frac = divmod(rem, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{frac:03d}"


def ms_to_vtt_ts(ms: int) -> str:
    ms = max(0, ms)
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, frac = divmod(rem, 1_000)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}.{frac:03d}"
    return f"{m:02d}:{s:02d}.{frac:03d}"


def ms_to_ass_ts(ms: int) -> str:
    ms = max(0, ms)
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, frac = divmod(rem, 1_000)
    cs = frac // 10
    return f"{h:01d}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_ts_to_ms(ts: str) -> int:
    parts = ts.strip().split(":")
    if len(parts) == 3:
        h, m, s_cs = int(parts[0]), int(parts[1]), parts[2]
    else:
        h, m, s_cs = 0, int(parts[0]), parts[1]
    s, cs = s_cs.split(".")
    return h * 3_600_000 + m * 60_000 + int(s) * 1_000 + int(cs.ljust(3, "0")[:3])


# ── encoding detection ──────────────────────────────────────────────────────

def detect_encoding(raw: bytes) -> str:
    # Strip UTF BOM markers first
    for bom, enc in [
        (b"\xef\xbb\xbf", "utf-8-sig"),
        (b"\xff\xfe", "utf-16-le"),
        (b"\xfe\xff", "utf-16-be"),
    ]:
        if raw.startswith(bom):
            return enc
    result = chardet.detect(raw)
    enc = result.get("encoding") or "utf-8"
    # chardet sometimes returns "ascii" for valid utf-8 content
    if enc.lower() in ("ascii", "iso-8859-1"):
        try:
            raw.decode("utf-8")
            return "utf-8"
        except UnicodeDecodeError:
            pass
    return enc


def read_text(path: Path, encoding: Optional[str] = None) -> str:
    raw = path.read_bytes()
    if encoding is None:
        encoding = detect_encoding(raw)
    try:
        return raw.decode(encoding, errors="replace")
    except (LookupError, UnicodeDecodeError):
        return raw.decode("utf-8", errors="replace")


# ── format detection ────────────────────────────────────────────────────────

def detect_format(path: Path, text: str) -> str:
    suffix = path.suffix.lower()
    if suffix in (".srt",):
        return "srt"
    if suffix in (".ass", ".ssa"):
        return "ass"
    if suffix in (".vtt",):
        return "vtt"
    # Sniff content
    head = text[:512]
    if head.lstrip().startswith("WEBVTT"):
        return "vtt"
    if "[Script Info]" in head or "[V4+ Styles]" in head or "[Events]" in head:
        return "ass"
    if _SRT_TS.search(head):
        return "srt"
    return "srt"


# ── SRT parser ──────────────────────────────────────────────────────────────

_SRT_BLOCK = re.compile(
    r"(\d+)\r?\n"
    r"(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})[^\r\n]*\r?\n"
    r"([\s\S]*?)(?=\r?\n\s*\r?\n\d+\r?\n|\Z)",
    re.MULTILINE,
)


def parse_srt(text: str) -> list[SubtitleEvent]:
    # Normalise line endings; strip BOM
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿")
    events: list[SubtitleEvent] = []

    for m in _SRT_BLOCK.finditer(text):
        idx = int(m.group(1))
        ts_start = _SRT_TS.match(m.group(2))
        ts_end = _SRT_TS.match(m.group(3))
        if not ts_start or not ts_end:
            continue
        start_ms = _srt_ts_to_ms(*ts_start.groups())
        end_ms = _srt_ts_to_ms(*ts_end.groups())
        content = m.group(4).strip()
        if end_ms <= start_ms:
            end_ms = start_ms + 1000
        events.append(SubtitleEvent(start_ms=start_ms, end_ms=end_ms, text=content, index=idx))

    # Fallback: split on double newline if regex found nothing
    if not events:
        events = _parse_srt_fallback(text)

    events.sort(key=lambda e: e.start_ms)
    return events


def _parse_srt_fallback(text: str) -> list[SubtitleEvent]:
    blocks = re.split(r"\n{2,}", text.strip())
    events: list[SubtitleEvent] = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue
        # Find the timestamp line
        ts_line = None
        text_start = 0
        for i, line in enumerate(lines):
            if "-->" in line:
                ts_line = line
                text_start = i + 1
                break
        if ts_line is None:
            continue
        parts = ts_line.split("-->")
        ts_start = _SRT_TS.search(parts[0])
        ts_end = _SRT_TS.search(parts[1])
        if not ts_start or not ts_end:
            continue
        start_ms = _srt_ts_to_ms(*ts_start.groups())
        end_ms = _srt_ts_to_ms(*ts_end.groups())
        content = "\n".join(lines[text_start:]).strip()
        if end_ms <= start_ms:
            end_ms = start_ms + 1000
        events.append(SubtitleEvent(start_ms=start_ms, end_ms=end_ms, text=content))
    return events


# ── ASS / SSA parser ─────────────────────────────────────────────────────────

def parse_ass(text: str) -> tuple[list[SubtitleEvent], dict]:
    """Parse ASS/SSA. Returns (events, metadata) where metadata contains
    the raw header sections for round-tripping."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿")

    # Split into sections
    sections: dict[str, list[str]] = {}
    current = "__pre__"
    for line in text.splitlines():
        m = re.match(r"^\[(.+)\]", line)
        if m:
            current = m.group(1)
            sections[current] = []
        else:
            sections.setdefault(current, []).append(line)

    events_section = sections.get("Events", [])

    # Parse Format line
    fmt_line = next((l for l in events_section if l.startswith("Format:")), None)
    if fmt_line:
        fmt_cols = [c.strip() for c in fmt_line[len("Format:"):].split(",")]
    else:
        fmt_cols = ["Layer", "Start", "End", "Style", "Name", "MarginL", "MarginR", "MarginV", "Effect", "Text"]

    def col(row_parts: list[str], name: str) -> str:
        try:
            return row_parts[fmt_cols.index(name)]
        except (ValueError, IndexError):
            return ""

    events: list[SubtitleEvent] = []
    for i, line in enumerate(events_section):
        if not line.startswith("Dialogue:"):
            continue
        raw = line[len("Dialogue:"):].strip()
        # Split on commas but the Text field may contain commas
        n_fields = len(fmt_cols)
        parts = raw.split(",", n_fields - 1)
        if len(parts) < n_fields:
            parts += [""] * (n_fields - len(parts))

        try:
            start_ms = _ass_ts_to_ms(col(parts, "Start"))
            end_ms = _ass_ts_to_ms(col(parts, "End"))
        except (ValueError, IndexError):
            continue

        text_val = col(parts, "Text")
        # Strip override tags for display but keep raw
        display = re.sub(r"\{[^}]*\}", "", text_val).replace("\\N", "\n").replace("\\n", "\n")
        if end_ms <= start_ms:
            end_ms = start_ms + 1000

        events.append(SubtitleEvent(
            start_ms=start_ms,
            end_ms=end_ms,
            text=display.strip(),
            index=i,
            raw_extras={"ass_raw_text": text_val, "ass_parts": parts, "ass_fmt": fmt_cols},
        ))

    events.sort(key=lambda e: e.start_ms)

    metadata = {
        "format": "ass",
        "sections": sections,
        "fmt_cols": fmt_cols,
    }
    return events, metadata


# ── VTT parser ───────────────────────────────────────────────────────────────

_VTT_TS_LINE = re.compile(
    r"((?:\d{1,2}:)?\d{2}:\d{2}\.\d{1,3})\s*-->\s*((?:\d{1,2}:)?\d{2}:\d{2}\.\d{1,3})"
)


def _parse_vtt_ts(ts: str) -> int:
    m = _VTT_TS.match(ts.strip())
    if not m:
        raise ValueError(f"Bad VTT timestamp: {ts!r}")
    h, mi, s, frac = m.groups()
    h = int(h) if h else 0
    return h * 3_600_000 + int(mi) * 60_000 + int(s) * 1_000 + int(frac.ljust(3, "0"))


def parse_vtt(text: str) -> list[SubtitleEvent]:
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿")
    lines = text.splitlines()

    # Drop WEBVTT header and any NOTE blocks
    in_note = False
    cleaned: list[str] = []
    for line in lines:
        if line.startswith("NOTE"):
            in_note = True
            continue
        if in_note:
            if line.strip() == "":
                in_note = False
            continue
        if line.startswith("WEBVTT"):
            continue
        cleaned.append(line)

    blocks = re.split(r"\n{2,}", "\n".join(cleaned).strip())
    events: list[SubtitleEvent] = []
    for i, block in enumerate(blocks):
        block_lines = block.strip().splitlines()
        if not block_lines:
            continue
        ts_line_idx = None
        for j, bl in enumerate(block_lines):
            if _VTT_TS_LINE.search(bl):
                ts_line_idx = j
                break
        if ts_line_idx is None:
            continue
        ts_match = _VTT_TS_LINE.search(block_lines[ts_line_idx])
        try:
            start_ms = _parse_vtt_ts(ts_match.group(1))
            end_ms = _parse_vtt_ts(ts_match.group(2))
        except (ValueError, AttributeError):
            continue
        content = "\n".join(block_lines[ts_line_idx + 1:]).strip()
        if end_ms <= start_ms:
            end_ms = start_ms + 1000
        events.append(SubtitleEvent(start_ms=start_ms, end_ms=end_ms, text=content, index=i))

    events.sort(key=lambda e: e.start_ms)
    return events


# ── public entry point ────────────────────────────────────────────────────────

def parse_subtitle_file(
    path: str | Path,
    encoding: Optional[str] = None,
) -> tuple[list[SubtitleEvent], str, dict]:
    """Parse a subtitle file.

    Returns (events, format_str, metadata).
    format_str is one of 'srt', 'ass', 'vtt'.
    metadata carries format-specific data for lossless round-tripping.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Subtitle file not found: {p}")

    text = read_text(p, encoding)
    fmt = detect_format(p, text)

    if fmt == "ass":
        events, meta = parse_ass(text)
        return events, "ass", meta
    if fmt == "vtt":
        events = parse_vtt(text)
        return events, "vtt", {}
    # Default: srt
    events = parse_srt(text)
    return events, "srt", {}
