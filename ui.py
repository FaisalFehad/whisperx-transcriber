"""Shared Rich-based UI components for whisperx-transcriber."""

from collections import deque

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

# Waveform block characters — 8 levels from silence to peak
_WAVE_CHARS = " ▁▂▃▄▅▆▇"


def info_panel(title, rows, *, subtitle=None):
    """Display an info box with label: value rows.

    Args:
        title: Panel title (e.g. "Recording", "Transcription")
        rows: list of (label, value) tuples
        subtitle: optional text below the border
    """
    table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    table.add_column("label", style="dim", no_wrap=True, min_width=10)
    table.add_column("value")
    for label, value in rows:
        table.add_row(label, str(value))
    console.print(Panel(
        table,
        title=f"[bold]{title}[/bold]",
        subtitle=subtitle,
        border_style="dim",
        expand=False,
        width=54,
    ))


def success_panel(title, rows):
    """Display a green-bordered completion box."""
    table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    table.add_column("label", style="dim", no_wrap=True, min_width=10)
    table.add_column("value")
    for label, value in rows:
        table.add_row(label, str(value))
    console.print(Panel(
        table,
        title=f"[bold green]{title}[/bold green]",
        border_style="green",
        expand=False,
        width=54,
    ))


# ── Recording display ────────────────────────────────────────────────────────

def _waveform(rms_history, width=24):
    """Render RMS history as a unicode waveform string."""
    if not rms_history:
        return " " * width
    chars = []
    for rms in rms_history:
        level = min(1.0, rms * 15)  # same scaling as level_meter()
        idx = int(level * (len(_WAVE_CHARS) - 1))
        chars.append(_WAVE_CHARS[idx])
    return "".join(chars).ljust(width)


def _quality_indicator(rms):
    """Return colored quality label for current RMS."""
    if rms > 0.01:
        return "[green]● Good[/green]"
    elif rms > 0.003:
        return "[yellow]● Low[/yellow]"
    else:
        return "[red]● Silent[/red]"


def _format_duration(seconds):
    """Format seconds as MM:SS or HH:MM:SS."""
    if seconds < 0:
        return "--:--"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class RecordingDisplay:
    """Live-updating recording meter using Rich.

    Usage:
        display = RecordingDisplay()
        with display:
            while recording:
                display.update(rms, elapsed, silence_elapsed)
                time.sleep(0.25)
    """

    def __init__(self, wave_width=24):
        self._rms_history = deque(maxlen=wave_width)
        self._live = Live("", console=console, auto_refresh=False)
        self._warnings = []

    def __enter__(self):
        self._live.__enter__()
        return self

    def __exit__(self, *args):
        self._live.__exit__(*args)

    def set_warning(self, msg):
        """Set a warning message (or None to clear)."""
        self._warnings = [msg] if msg else []

    def update(self, rms, elapsed, silence_elapsed):
        """Refresh the display with current audio state."""
        self._rms_history.append(rms)

        wave = _waveform(self._rms_history)
        quality = _quality_indicator(rms)
        time_str = _format_duration(elapsed)
        silence_str = f"{int(silence_elapsed)}s" if silence_elapsed > 0 else ""

        # Build the status line
        parts = [f"  ⏺ {time_str}  {wave}  {quality}"]
        if silence_str:
            parts[0] += f"  [dim]silence {silence_str}[/dim]"

        # Add warnings below if any
        for w in self._warnings:
            parts.append(f"  [yellow]⚠ {w}[/yellow]")

        text = Text.from_markup("\n".join(parts))
        self._live.update(text)
        self._live.refresh()
