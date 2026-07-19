"""Terminal styling for delv-e runs.

Zero dependencies (plain ANSI). Presentation only — nothing here affects the
investigation. All styling auto-disables when stdout is not a TTY (piped to a
file, CI, etc.) so logs stay clean. Keep it lean; this is not a full TUI.
"""
import itertools
import os
import shutil
import sys
import textwrap
import threading
import time

ENABLED = (sys.stdout.isatty()
           and os.environ.get("NO_COLOR") is None
           and os.environ.get("TERM") != "dumb")

# Windows consoles and pipes often default to a legacy codepage (cp1252) that
# cannot encode the banner's box glyphs or the status marks — a run launched
# through eval/run_matrix.py on Windows crashed in banner() exactly this way.
# Prefer reconfiguring the stream to UTF-8; where that is impossible, _SAFE
# marks the stream as unable to carry non-ASCII and printing falls back.
def _utf8_stream():
    enc = (getattr(sys.stdout, "encoding", None) or "").lower()
    if "utf" in enc:
        return True
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        return True
    except (AttributeError, ValueError, OSError):
        return False

_SAFE = _utf8_stream()


def _print(s):
    """Print that never dies on a legacy codepage: non-encodable characters
    degrade to '?' instead of raising UnicodeEncodeError mid-run."""
    try:
        print(s)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(s.encode(enc, errors="replace").decode(enc))

VERSION = "0.2.0"
TAGLINE = "Deep Exploratory Learning & Visualization Engine"

# Set to "verify" by run_core in --verify mode: the iteration banner relabels
# EXPLORING as VERIFYING and the run recolors magenta, so an audit pass is
# visually unmistakable from a primary investigation.
MODE = None

_C = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "white": "\033[97m", "cyan": "\033[96m", "green": "\033[32m",
    "yellow": "\033[33m", "red": "\033[31m", "blue": "\033[34m", "gray": "\033[90m",
    "magenta": "\033[95m",
}


def c(s, *styles):
    if not ENABLED or not styles:
        return s
    return "".join(_C[x] for x in styles) + s + _C["reset"]


def _width(default=72):
    try:
        return min(shutil.get_terminal_size().columns, 100)
    except Exception:
        return default


# ── block-letter logo (generated from a glyph table so it always aligns) ──────
_LOGO = (
    "██████╗ ███████╗██╗    ██╗   ██╗    ███████╗\n"
    "██╔══██╗██╔════╝██║    ██║   ██║    ██╔════╝\n"
    "██║  ██║█████╗  ██║    ██║   ██║ ██ █████╗  \n"
    "██║  ██║██╔══╝  ██║    ╚██╗ ██╔╝    ██╔══╝  \n"
    "██████╔╝███████╗███████╗╚████╔╝     ███████╗\n"
    "╚═════╝ ╚══════╝╚══════╝ ╚═══╝      ╚══════╝"
)


def banner():
    print()
    if _SAFE:
        art = "\n".join("  " + ln for ln in _LOGO.split("\n"))
        _print(c(art, "white", "bold"))
    else:
        _print(c("  D E L V - E", "white", "bold"))
    print()
    _print("  " + c(f"{VERSION} - {TAGLINE}", "dim"))
    print()


def _wrap(text, width, indent="    "):
    out = []
    for line in (text.splitlines() or [""]):
        out.append(textwrap.fill(line, width, initial_indent=indent,
                                 subsequent_indent=indent) or indent)
    return "\n".join(out)


def run_header(seed, rows, cols, iterations, code_model, brain_model, output,
               compute=False):
    def row(k, v, *st):
        print(f"  {c(k.ljust(8), 'dim')}{c(v, *st)}")
    if compute:
        row("Data", "computation-only (no dataset)")
    else:
        row("Data", f"{rows:,} rows × {cols} cols")
    row("Loop", f"{iterations} iterations")
    row("Code", code_model, "cyan")
    row("Brain", brain_model, "cyan")
    row("Output", output)
    print()
    print("  " + c("Question", "bold"))
    print(_wrap(seed, _width() - 6))
    print()


def iteration(step, max_steps, status="EXPLORING"):
    w = _width()
    if MODE == "verify" and status == "EXPLORING":
        status = "VERIFYING"
    tone = "magenta" if MODE == "verify" else "cyan"
    left = f"— Iteration {step}/{max_steps} "
    right = f" {status} —"
    fill = max(3, w - len(left) - len(right) - 2)
    print()
    print(c(left, tone, "bold") + c("─" * fill, "gray") + c(right, tone, "bold"))
    print()


def agent(label, model):
    print(c(f"  [{label}] ", "blue") + c(model, "dim"))


def question(text):
    body = " ".join(text.split())
    indent = "    "
    wrapped = textwrap.fill(body, _width() - len(indent),
                            initial_indent=indent, subsequent_indent=indent)
    # Light weight: a subtle marker, body in the terminal's default text weight.
    _print(c("  ▸ ", "cyan") + wrapped[len(indent):])


def _truncate(s, n):
    return s if len(s) <= n else s[:n - 1] + "…"


def executed(entry, artifact_path=None):
    """Post-execution status: ✓/✗ and a pointer to the step's full analysis.md
    (where the move, code, and raw output live). No inline result preview —
    arbitrary executor output does not reduce to a reliable one-liner."""
    err = entry.get("error")
    if err:
        _print(c(f"  ✗ failed after {entry.get('attempts','?')} attempt(s)", "red")
              + c("  " + _truncate(str(err), 100), "dim"))
    else:
        n_lines = len((entry.get("code") or "").splitlines())
        extra = c(f" ({entry['attempts']} attempts)", "dim") if (entry.get("attempts") or 1) > 1 else ""
        _print(c(f"  ✓ {n_lines} lines, executed OK", "green") + extra)
    if artifact_path:
        _print(c(f"    📄 {artifact_path}", "dim"))


def searched(query, artifact_path=None):
    """Post-search status: the query and a pointer to the saved findings."""
    _print(c("  \U0001F50D searched: ", "cyan") + c(_truncate(query, _width() - 16), "dim"))
    if artifact_path:
        _print(c(f"    \U0001F4C4 {artifact_path}", "dim"))


def synthesis(verdict, g1=None, reason=None):
    if verdict == "FINAL":
        _print(c("  ◆ synthesis → FINAL", "green", "bold")
              + (c(f"  (G1={g1})", "dim") if g1 is not None else ""))
    else:
        _print(c("  ◆ synthesis → needs more work", "yellow", "bold")
              + (c(f"  {_truncate(reason,90)}", "dim") if reason else ""))


def note(msg, color="blue"):
    _print(c("  • " + msg, color))


def done(path):
    print()
    _print(c("  ✓ briefing ready  ", "green", "bold") + c(path, "dim"))


# Back-compat: run_core may still pass on_step; keep a simple summary available.
def step_summary(entry):
    if entry.get("terminal"):
        synthesis(entry.get("synth_verdict"), entry.get("g1_satisfied"))
    else:
        executed(entry)


class Spinner:
    """Background spinner for a blocking call. No-op when styling is disabled."""
    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, label="working"):
        self.label = label
        self._stop = None
        self._thread = None

    def __enter__(self):
        if not ENABLED:
            return self
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def _spin(self):
        t0 = time.time()
        for frame in itertools.cycle(self.FRAMES):
            if self._stop.is_set():
                break
            line = (c(f"  {frame} {self.label}…", "cyan")
                    + c(f" {int(time.time() - t0)}s", "dim"))
            sys.stdout.write("\r" + line + "  ")
            sys.stdout.flush()
            self._stop.wait(0.1)

    def __exit__(self, *exc):
        if self._thread:
            self._stop.set()
            self._thread.join(timeout=0.5)
            sys.stdout.write("\r" + " " * 64 + "\r")
            sys.stdout.flush()
        return False