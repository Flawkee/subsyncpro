"""Write SubtitleEvent lists to SRT, ASS, or VTT files.

For ASS files the original header sections (styles, script info, etc.) are
preserved verbatim — only the Dialogue timestamps are modified.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from subsyncpro.parser import (
    SubtitleEvent,
    ms_to_ass_ts,
    ms_to_srt_ts,
    ms_to_vtt_ts,
)


# ── SRT writer ────────────────────────────────────────────────────────────────

def write_srt(
    events: list[SubtitleEvent],
    output_path: str | Path,
    encoding: str = "utf-8",
) -> Path:
    out = Path(output_path)
    lines: list[str] = []
    for i, ev in enumerate(events, 1):
        lines.append(str(i))
        lines.append(f"{ms_to_srt_ts(ev.start_ms)} --> {ms_to_srt_ts(ev.end_ms)}")
        lines.append(ev.text)
        lines.append("")
    out.write_text("\n".join(lines), encoding=encoding)
    return out


# ── ASS writer ────────────────────────────────────────────────────────────────

def write_ass(
    events: list[SubtitleEvent],
    output_path: str | Path,
    metadata: Optional[dict] = None,
    encoding: str = "utf-8",
) -> Path:
    """Write ASS.  If *metadata* contains original section data, reuse it to
    preserve styles, fonts, and script info exactly.  Otherwise emit a minimal
    ASS file."""
    out = Path(output_path)

    if metadata and "sections" in metadata:
        _write_ass_from_original(events, out, metadata, encoding)
    else:
        _write_ass_minimal(events, out, encoding)
    return out


def _write_ass_from_original(
    events: list[SubtitleEvent],
    out: Path,
    metadata: dict,
    encoding: str,
) -> None:
    """Reconstruct the ASS file, replacing only Dialogue timestamps."""
    sections = metadata["sections"]
    fmt_cols = metadata.get("fmt_cols", [])

    # Build a lookup: original index → new event
    idx_map = {ev.index: ev for ev in events}

    output_lines: list[str] = []

    def emit_section(name: str, lines: list[str]) -> None:
        output_lines.append(f"[{name}]")
        output_lines.extend(lines)
        output_lines.append("")

    section_order = [
        "Script Info", "V4+ Styles", "V4 Styles", "Fonts", "Graphics", "Events"
    ]
    remaining = set(sections.keys()) - {"__pre__"} - set(section_order)

    for name in section_order + sorted(remaining):
        if name not in sections:
            continue
        raw_lines = sections[name]

        if name == "Events":
            new_lines: list[str] = []
            dialogue_idx = 0
            for line in raw_lines:
                if not line.startswith("Dialogue:"):
                    new_lines.append(line)
                    continue
                # Find the matching (possibly reordered) event
                # Events were indexed by line order during parsing
                ev = idx_map.get(dialogue_idx)
                if ev is None:
                    new_lines.append(line)
                    dialogue_idx += 1
                    continue

                # Replace Start and End fields in the original line
                raw_rest = line[len("Dialogue:"):].strip()
                n_fields = len(fmt_cols)
                parts = raw_rest.split(",", n_fields - 1)
                if len(parts) >= n_fields:
                    try:
                        start_i = fmt_cols.index("Start")
                        end_i = fmt_cols.index("End")
                        parts[start_i] = ms_to_ass_ts(ev.start_ms)
                        parts[end_i] = ms_to_ass_ts(ev.end_ms)
                    except ValueError:
                        pass
                new_lines.append("Dialogue: " + ",".join(parts))
                dialogue_idx += 1
            emit_section(name, new_lines)
        else:
            emit_section(name, raw_lines)

    out.write_text("\n".join(output_lines), encoding=encoding)


def _write_ass_minimal(
    events: list[SubtitleEvent],
    out: Path,
    encoding: str,
) -> None:
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "Collisions: Normal\n"
        "PlayDepth: 0\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
        "0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = [header]
    for ev in events:
        text = ev.text.replace("\n", "\\N")
        lines.append(
            f"Dialogue: 0,{ms_to_ass_ts(ev.start_ms)},{ms_to_ass_ts(ev.end_ms)},"
            f"Default,,0000,0000,0000,,{text}"
        )
    out.write_text("\n".join(lines) + "\n", encoding=encoding)


# ── VTT writer ────────────────────────────────────────────────────────────────

def write_vtt(
    events: list[SubtitleEvent],
    output_path: str | Path,
    encoding: str = "utf-8",
) -> Path:
    out = Path(output_path)
    lines = ["WEBVTT", ""]
    for i, ev in enumerate(events, 1):
        lines.append(str(i))
        lines.append(f"{ms_to_vtt_ts(ev.start_ms)} --> {ms_to_vtt_ts(ev.end_ms)}")
        lines.append(ev.text)
        lines.append("")
    out.write_text("\n".join(lines), encoding=encoding)
    return out


# ── unified writer ────────────────────────────────────────────────────────────

def write_subtitle(
    events: list[SubtitleEvent],
    output_path: str | Path,
    fmt: str = "auto",
    metadata: Optional[dict] = None,
    encoding: str = "utf-8",
) -> Path:
    """Write subtitle events to *output_path* in the requested format.

    Parameters
    ----------
    fmt : 'srt' | 'ass' | 'vtt' | 'auto'
        'auto' infers from the output file extension, defaulting to 'srt'.
    """
    out = Path(output_path)
    if fmt == "auto":
        fmt = out.suffix.lower().lstrip(".")
        if fmt not in ("srt", "ass", "ssa", "vtt"):
            fmt = "srt"

    if fmt in ("ass", "ssa"):
        return write_ass(events, out, metadata=metadata, encoding=encoding)
    if fmt == "vtt":
        return write_vtt(events, out, encoding=encoding)
    return write_srt(events, out, encoding=encoding)


def default_output_path(unsync_path: str | Path, fmt: str = "srt") -> Path:
    """Derive a sensible output path from the unsync input path."""
    p = Path(unsync_path)
    ext = f".{fmt}" if fmt != "auto" else p.suffix
    return p.with_name(p.stem + ".synced" + ext)
