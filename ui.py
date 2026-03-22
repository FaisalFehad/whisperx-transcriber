"""Shared Rich-based UI components for whisperx-transcriber."""

import select
import sys
import termios
import tty
from collections import deque
from contextlib import contextmanager

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


# ── Keyboard handling ─────────────────────────────────────────────────────────

@contextmanager
def raw_keys():
    """Context manager for single-keypress input (no echo, no Enter needed).

    Usage:
        with raw_keys():
            while True:
                key = poll_key()
                if key == "p":
                    paused = not paused
    """
    if not sys.stdin.isatty():
        yield
        return
    old = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        yield
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)


def poll_key():
    """Non-blocking single key read. Returns the key or None.

    Must be called inside a raw_keys() context.
    """
    if not sys.stdin.isatty():
        return None
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None

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


# ── System health ─────────────────────────────────────────────────────────────

def check_system_health(threshold=85):
    """Check CPU and RAM usage. Returns warning string or None.

    On Apple Silicon, GPU memory IS system RAM (unified architecture),
    so RAM monitoring effectively covers GPU memory too.
    """
    try:
        import psutil
        warnings = []
        cpu = psutil.cpu_percent(interval=None)  # non-blocking (uses cached value)
        ram = psutil.virtual_memory().percent
        if cpu > threshold:
            warnings.append(f"CPU {int(cpu)}%")
        if ram > threshold:
            warnings.append(f"RAM {int(ram)}%")
        return " │ ".join(warnings) if warnings else None
    except ImportError:
        return None


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


def format_duration(seconds):
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

    def update(self, rms, elapsed, silence_elapsed, paused=False,
               mic_rms=None, sys_rms=None):
        """Refresh the display with current audio state.

        Args:
            rms: combined RMS (max of mic and system)
            elapsed: seconds since recording started
            silence_elapsed: seconds of continuous silence
            paused: whether recording is paused
            mic_rms: mic channel RMS (optional, for per-channel display)
            sys_rms: system channel RMS (optional, for per-channel display)
        """
        if not paused:
            self._rms_history.append(rms)

        time_str = format_duration(elapsed)

        if paused:
            parts = [f"  [yellow]⏸ {time_str}  PAUSED[/yellow]  [dim](P to resume)[/dim]"]
        else:
            wave = _waveform(self._rms_history)
            quality = _quality_indicator(rms)
            parts = [f"  ⏺ {time_str}  {wave}  {quality}"]
            if silence_elapsed > 0:
                parts[0] += f"  [dim]silence {int(silence_elapsed)}s[/dim]"
            # Per-channel levels when both available
            if mic_rms is not None and sys_rms is not None:
                mic_q = "[green]●[/green]" if mic_rms > 0.003 else "[red]●[/red]"
                sys_q = "[green]●[/green]" if sys_rms > 0.003 else "[red]●[/red]"
                parts.append(f"  [dim]Mic {mic_q} {mic_rms:.3f}  System {sys_q} {sys_rms:.3f}[/dim]")

        for w in self._warnings:
            parts.append(f"  [yellow]⚠ {w}[/yellow]")

        text = Text.from_markup("\n".join(parts))
        self._live.update(text)
        self._live.refresh()
