# SubSyncPro

**The subtitle synchroniser that actually gets it right — even when every other tool gives up.**

> No audio. No re-encoding. No nonsense.  
> Just fast, accurate, cross-language subtitle alignment that handles the real world.

---

## Why SubSyncPro Destroys the Competition

Most subtitle sync tools were built for the easy case: one language, one source, constant offset.  
The real world is messier. SubSyncPro was built for it.

| Capability | SubSyncPro | ffsubsync | subsync | alass |
|---|:---:|:---:|:---:|:---:|
| Works without audio | ✅ | ❌ | ❌ | ✅ |
| Handles "Previously on…" / content gaps | ✅ | ❌ | ❌ | ⚠️ |
| Cross-language sync (e.g. Hebrew ↔ English) | ✅ | ❌ | ❌ | ❌ |
| Linear drift correction (frame-rate mismatch) | ✅ | ✅ | ✅ | ✅ |
| SDH-aware (ignores `[sound effects]`, `♪ music ♪`) | ✅ | ❌ | ❌ | ❌ |
| Smart MKV track selection (SDH vs standard) | ✅ | ❌ | ❌ | ❌ |
| Confidence score — knows when it might be wrong | ✅ | ❌ | ❌ | ❌ |
| Alignment time for a 45-min episode | **< 5 s** | ~10 s | ~60 s | ~3 s |

### What makes it different

**ffsubsync** and **subsync** need to extract and analyse the audio track — a multi-minute process that fails completely when you only have subtitle files, and struggles badly when the reference contains scenes not in your episode.

**alass** is subtitle-only and fast, but its dynamic-programming approach falls apart the moment the reference has different content (extra episodes, recap segments, director's cut scenes) or when you're syncing a translation in a different language.

**SubSyncPro** uses a four-stage pipeline — onset-impulse fingerprinting, windowed anchor search, RANSAC outlier rejection, and a segmented binary fallback — that is specifically engineered to handle:

- **Content gaps**: RANSAC discards windows that match "Previously on…" as outliers
- **Cross-language subtitles**: binary speech-activity fingerprints that are language-agnostic
- **SDH noise**: automatically strips `[dramatic music]`, `* *`, `♪ lyrics ♪` before correlation, eliminating false correlation peaks that would otherwise dominate
- **Linear drift**: detects and corrects frame-rate mismatches (23.976 fps vs 25 fps, NTSC vs PAL sources) with per-segment offset fitting

---

## Real-World Results

All three tests were confirmed correct by manual verification at multiple timestamps.

| Episode | Languages | Problem | Result | Time |
|---|---|---|---|---|
| TV Drama — S05E09 | Hebrew ↔ English | Linear drift: 0 s off at start → 2.6 s off at end | `scale=0.998679, offset=+0.159 s` 100% confidence | 1.5 s |
| TV Drama — S13E21 | Hebrew ↔ English (SDH) | SDH false peaks + linear drift | `scale=1.003511, offset=+0.962 s` 100% confidence | 2.1 s |
| TV Drama — S04E04 | Hebrew ↔ English | Constant 16-second offset | `offset=−15.997 s` 92% confidence (517/559 inliers) | 4.1 s |

---

## Features

| Feature | Details |
|---|---|
| **Formats** | SRT, ASS/SSA, WebVTT — read and write |
| **Reference input** | Any subtitle file *or* an MKV/MP4/AVI/TS/M2TS with embedded subtitles |
| **Content-gap handling** | Works when reference has recap scenes, bonus content, or covers multiple episodes |
| **Cross-language** | Aligns translations (Hebrew, French, Arabic, …) against the original language |
| **Drift correction** | Detects and corrects frame-rate mismatches automatically |
| **SDH-aware** | Filters out sound descriptors before fingerprinting to avoid false correlation peaks |
| **Smart track selection** | If syncing an SDH subtitle, picks the SDH track from MKV; otherwise picks the standard track |
| **Confidence score** | Reports how reliable the alignment is and warns when it might be wrong |
| **Dry run** | Preview the offset without writing any file |
| **In-place overwrite** | `--overwrite` replaces the original file instead of creating a copy |
| **ASS round-trip** | Styles, fonts, effects, and metadata preserved exactly — only timestamps change |
| **Fast** | A 45-minute episode aligns in under 5 seconds on any modern CPU |

---

## Installation

### Requirements

- **Python 3.9+**
- **FFmpeg** — needed when using a video file (MKV/MP4/…) as the reference. Not required for subtitle-to-subtitle alignment.
- **MKVToolNix** *(optional but recommended)* — provides `mkvextract`, which is used in preference to ffmpeg for extracting subtitle tracks from MKV files. It is purpose-built for MKV demuxing: no transcoding pipeline, no full file decode — just direct track extraction. Significantly faster and lighter, especially on slow HDD servers with large files. SubSyncPro falls back to ffmpeg automatically if MKVToolNix is not installed.

### Install MKVToolNix (recommended for MKV sources)

| Platform | Command |
|---|---|
| Windows | `winget install MKVToolNix.MKVToolNix` |
| macOS | `brew install mkvtoolnix` |
| Linux | `sudo apt install mkvtoolnix` (or your distro's package manager) |

Or download from [mkvtoolnix.download](https://mkvtoolnix.download/).

### Install FFmpeg

| Platform | Command |
|---|---|
| Windows | `winget install Gyan.FFmpeg` |
| macOS | `brew install ffmpeg` |
| Linux | `sudo apt install ffmpeg` (or your distro's package manager) |

Or download from [ffmpeg.org/download.html](https://ffmpeg.org/download.html) and add the `bin/` folder to your PATH.

### Install SubSyncPro

```bash
git clone https://github.com/Flawkee/subsyncpro.git
cd subsyncpro
pip install -e .
```

Or from PyPI (once published):

```bash
pip install subsyncpro
```

Dependencies are installed automatically: `numpy`, `scipy`, `chardet`, `rich`.

---

## Quick Start

```bash
# Sync using a reference subtitle — creates episode.he.synced.srt
subsyncpro reference.srt episode.he.srt

# Use an MKV as reference (auto-extracts the best subtitle track)
subsyncpro episode.mkv episode.he.srt

# Overwrite the original file in-place
subsyncpro episode.mkv episode.he.srt --overwrite

# Dry run — see exactly what would be applied before touching anything
subsyncpro reference.srt episode.he.srt --dry-run --verbose
```

---

## Complete Usage

```
subsyncpro REF UNSYNC [options]
```

### Positional Arguments

| Argument | Description |
|---|---|
| `REF` | Reference subtitle (`.srt`, `.ass`, `.vtt`) **or** a video file (`.mkv`, `.mp4`, `.avi`, `.ts`, `.m2ts`, `.mov`, `.webm`) with embedded subtitles |
| `UNSYNC` | The subtitle you want to synchronise |

---

### Output Options

| Flag | Default | Description |
|---|---|---|
| `-o PATH`, `--output PATH` | `UNSYNC.synced.ext` | Write output to a specific path |
| `--overwrite` | off | Overwrite `UNSYNC` in-place instead of creating a `.synced` copy. Cannot be combined with `-o`. |
| `-f FMT`, `--format {srt,ass,vtt,auto}` | `auto` | Output format. `auto` keeps the same format as `UNSYNC`. |
| `--encoding ENC` | auto-detect | Character encoding for reading subtitle files (e.g. `utf-8`, `cp1252`, `utf-8-sig`). SubSyncPro auto-detects by default. |
| `--output-encoding ENC` | `utf-8` | Encoding for the written output file. |

---

### Alignment Options

| Flag | Default | Description |
|---|---|---|
| `-m MODE`, `--mode {auto,offset,linear}` | `auto` | Alignment mode (see below) |
| `--max-offset SECONDS` | `600` | Maximum expected timing difference in seconds. Raise to `1800` or more if the reference covers multiple episodes or has very long recap segments. |
| `--offset-hint MS` | — | Rough offset hint in milliseconds. Optional speed-up when you already know the approximate delay from a previous run. |

**Alignment modes:**

- **`offset`** — Find a single constant shift (ms) to apply uniformly to all timestamps. Covers 99% of cases.
- **`linear`** — Find a constant shift *plus* a slight speed correction. Use when timestamps slowly drift apart over the episode (caused by 23.976 fps vs 25 fps frame-rate mismatches between sources).
- **`auto`** *(default)* — Tries linear; if the detected drift is negligible (< 0.05%), falls back to pure offset. Best of both worlds.

---

### MKV / Video Reference Options

| Flag | Description |
|---|---|
| `--list-tracks` | List all subtitle tracks embedded in the video file and exit without syncing. Useful before choosing a specific track. |
| `--ref-track IDX` | Force the use of a specific track by its stream index (as shown by `--list-tracks`). |
| `--ref-lang LANG` | Preferred language when auto-selecting a track from a video file (e.g. `eng`, `jpn`, `fra`). Default: English. |
| `--ffprobe-timeout SEC` | `120` | Seconds to wait for ffprobe when probing the video file. Increase on slow HDD servers with large files. |
| `--ffmpeg-timeout SEC` | `300` | Seconds to wait for ffmpeg (or mkvextract) when extracting a subtitle track. Increase on slow HDD servers. |

---

### Behaviour Options

| Flag | Default | Description |
|---|---|---|
| `-n`, `--dry-run` | off | Compute and display the alignment result without writing any output file. |
| `-v`, `--verbose` | off | Print detailed internal stats: fingerprint sizes, anchor counts, per-segment offsets, RANSAC inlier rates. |
| `--confidence-threshold 0–1` | `0.2` | Emit a warning (but still write the file) when the alignment confidence falls below this threshold. |
| `--version` | — | Print the SubSyncPro version and exit. |

---

## Examples

```bash
# List subtitle tracks in an MKV to pick the right one
subsyncpro --list-tracks episode.mkv unsynced.srt
#   IDX  CODEC         LANG    DEF  FORCED   SDH  TITLE
#   ──────────────────────────────────────────────────────
#     2  subrip        eng      yes      no    no  English
#     3  subrip        eng       no      no   yes  English [SDH]
#     4  hdmv_pgs_sub  jpn       no      no    no  Japanese

# Use a specific track by index
subsyncpro --ref-track 2 episode.mkv unsynced.srt

# Sync a Hebrew translation against an English SDH MKV
# SubSyncPro automatically picks the standard (non-SDH) English track
subsyncpro episode.mkv episode.he.srt

# Correct frame-rate drift between an NTSC and PAL source
subsyncpro reference.srt unsynced.srt --mode linear

# Handle a reference that covers multiple episodes (up to 30 min gap)
subsyncpro two-episodes.srt single-episode.srt --max-offset 1800

# Sync an ASS file — all styles, fonts, and effects are preserved
subsyncpro reference.srt styled.ass -o styled.synced.ass

# Overwrite the original Hebrew subtitle in-place
subsyncpro episode.mkv episode.he.srt --overwrite

# Verbose dry run — inspect every internal step
subsyncpro reference.srt unsynced.srt --dry-run --verbose

# Specify output encoding for legacy media players
subsyncpro reference.srt unsynced.srt -o fixed.srt --output-encoding cp1252

# Use as a Python library
python - <<'EOF'
from subsyncpro import align_subtitles
result = align_subtitles("reference.srt", "unsynced.srt")
print(f"Offset: {result['offset_ms']:+.0f} ms  confidence: {result['confidence']:.0%}")
EOF
```

---

## How It Works

SubSyncPro uses a four-stage pipeline that goes far beyond naive cross-correlation.

### Stage 1 — Onset-Impulse Fingerprinting

Each subtitle file is converted to a sparse time-series sampled at 33 ms (one video frame): a **+1 impulse at the start of every event**, zero everywhere else. Sparse impulse fingerprints work for both dense tracks (SDH with hundreds of events) and sparse tracks (a single song lyric per minute) because they represent *when dialogue starts*, not *how long it lasts*.

### Stage 2 — Coarse FFT Offset Estimate

A global FFT cross-correlation gives a rough offset in O(N log N) time. This is purely informational — it can be fooled by extra content in the reference — but it provides a useful seed for RANSAC.

### Stage 3 — Windowed Anchor Search + RANSAC

This is the core innovation:

1. **Sliding windows**: 30-second windows slide over the unsync fingerprint in 5-second steps.
2. **Local matching**: Each window is zero-mean normalised and matched against the *entire* reference using FFT cross-correlation. The peak gives a `(unsync_time, ref_time)` anchor pair.
3. **RANSAC**: A robust linear fit rejects outlier anchors. Windows that match extra content ("Previously on…", bonus scenes) produce inconsistent offsets and are discarded. The dominant cluster — the true episode content — survives.

### Stage 4 — Segmented Binary Fallback (Cross-Language)

When onset impulses fail (cross-language content where a Hebrew translator merges two English lines into one), SubSyncPro switches to **binary speech-activity fingerprints**: `1 = subtitle visible, 0 = silence`. These capture the *rhythm of dialogue* rather than the exact event positions — similar across languages.

The reference is searched segment by segment (5-minute chunks), each cross-correlated against the full reference with no seed. An SDH filter strips sound-descriptor-only events (`[dramatic music]`, `* *`, `♪ lyrics ♪`) before fingerprinting, because those have no equivalent in translations and would otherwise generate massive false correlation peaks. Per-segment offsets are fitted with SNR-weighted RANSAC to yield the final linear drift model.

---

## Troubleshooting

**"Low confidence" warning**  
The reference subtitle doesn't match the episode well. Try `--list-tracks` to check which track was selected, or use `--ref-track` to pick a better one. Also try `--max-offset` with a larger value, and `--mode linear` if you suspect drift.

**Output is still off by a constant amount**  
Add `--offset-hint MS` with the remaining offset you observe. Useful for very sparse subtitles where RANSAC found few anchors.

**Wrong track selected from MKV**  
Run `subsyncpro --list-tracks episode.mkv dummy.srt` to see all available tracks, then use `--ref-track IDX`.

**"image-based track" error**  
PGS (Blu-ray) and DVDSUB (DVD) subtitle tracks are bitmap images that cannot be parsed as text. Use `--ref-track` to select a text-based track (SRT/ASS) if one exists, or supply a `.srt` file directly as the reference.

**Subtitles drift over a long episode**  
Use `--mode linear`. This corrects frame-rate mismatches between different source encodings (e.g. a subtitle timed for a 25 fps PAL broadcast vs a 23.976 fps NTSC stream).

---

## Supported Formats

| Format | Read | Write |
|---|:---:|:---:|
| SRT (SubRip) | ✅ | ✅ |
| ASS / SSA (Advanced SubStation Alpha) | ✅ full round-trip | ✅ |
| WebVTT | ✅ | ✅ |
| MKV, MP4, AVI, TS, M2TS, MOV, WebM | ✅ via FFmpeg | — |

---

## Acknowledgements

SubSyncPro stands on the shoulders of some excellent open-source projects:

- **[NumPy](https://numpy.org)** — Fast N-dimensional array operations. The backbone of all fingerprint arithmetic.  
  *Harris et al., Nature 585, 357–362 (2020)*

- **[SciPy](https://scipy.org)** — `scipy.signal.fftconvolve` powers every cross-correlation in the alignment pipeline.  
  *Virtanen et al., Nature Methods 17, 261–272 (2020)*

- **[chardet](https://github.com/chardet/chardet)** — Automatic character encoding detection so SubSyncPro reads subtitle files correctly regardless of origin.

- **[Rich](https://github.com/Textualize/rich)** — The beautiful terminal output, progress spinner, and coloured panels. Makes the tool a joy to use.

- **[FFmpeg](https://ffmpeg.org)** — The industry-standard multimedia framework used to probe and extract embedded subtitle tracks from MKV and other container formats. SubSyncPro would not support video reference files without it.

- **[ffsubsync](https://github.com/smacke/ffsubsync)** — The original inspiration. ffsubsync pioneered the idea of automatic subtitle synchronisation and proved there was real demand for it. SubSyncPro was built to tackle the cases where ffsubsync falls short.

---