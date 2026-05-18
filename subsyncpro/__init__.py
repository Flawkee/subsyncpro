"""SubSyncPro — subtitle synchronization using a reference subtitle or MKV file."""

__version__ = "1.0.0"
__all__ = ["align_subtitles"]


def align_subtitles(
    ref_path: str,
    unsync_path: str,
    output_path: str | None = None,
    *,
    mode: str = "auto",
    max_offset_s: float = 600.0,
    ref_lang: str | None = None,
    ref_track: int | None = None,
    encoding: str | None = None,
    verbose: bool = False,
    lead_bias_ms: float = 0.0,
) -> dict:
    """Programmatic entry point — synchronize *unsync_path* using *ref_path*.

    Returns a result dict with keys: offset_ms, scale, confidence, output_path.
    """
    from subsyncpro.cli import _run_sync

    return _run_sync(
        ref_path=ref_path,
        unsync_path=unsync_path,
        output_path=output_path,
        mode=mode,
        max_offset_s=max_offset_s,
        ref_lang=ref_lang,
        ref_track=ref_track,
        encoding=encoding,
        verbose=verbose,
        dry_run=False,
        output_format="auto",
        lead_bias_ms=lead_bias_ms,
    )
