"""
Persistent execution kernel for delv-e's inverted-core loop.

A single long-lived worker process holds ONE namespace. `df` is loaded once;
derived columns and intermediate objects created in step k survive into step
k+1. This is what makes analytical (not line-count) decomposition possible:
each Executor step can stay junior-simple because its prerequisites already
live in the namespace.

Crash isolation: the worker is itself a killable subprocess. If a step hangs
(timeout) or the worker dies (OOM/segfault), the parent kills/restarts it and
replays the history of previously-successful steps to reconstruct the
namespace. Determinism of the replayed steps is the caller's responsibility
(seed your RNG).

Contract (kept deliberately close to executor.CodeExecutor):
    kernel = PersistentKernel(df=df, analysis_root="output/exploration")
    stdout, error, plots = kernel.execute(code, analysis_dir=".../01")
    print(kernel.describe_namespace())   # registry for prompts
    kernel.cleanup()

`execute` returns the familiar (stdout, error, plots) triple so it can slot in
beside the existing stateless executor. The namespace registry is exposed
separately via `registry` / `describe_namespace()`.
"""

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading

from logger_config import get_logger

# Reuse the security blacklist and temp-file helpers already proven in executor.
from executor import BLACKLIST, _write_temp_text, _cleanup_files, _serialize_dataframe

logger = get_logger(__name__)

# Wall-clock seconds for a single step. The persistent worker means a hang on
# one step must not wedge the whole run, so on timeout we kill and restart.
STEP_TIMEOUT = 300

# Token the worker prints on its real stdout after each step completes. User
# code output is captured in-worker and written to a result file, so the only
# thing on the worker's real stdout is this control token (and the startup
# ready token). That keeps the control channel clean.
_DONE = "__DELVE_KERNEL_DONE__"
_READY = "__DELVE_KERNEL_READY__"

# Names that are part of the kernel's own plumbing and must never be surfaced
# in the namespace registry shown to the Investigator/Executor.
_INTERNAL_NAMES = {
    "df",  # handled explicitly (we show its columns, not as a generic object)
    "os", "io", "json", "sys", "traceback", "warnings",
    "matplotlib", "plt", "_mpl_figure", "pd",
    "_real_savefig", "_saved_plots", "_plot_counter", "_analysis_dir",
    "_patched_show", "_patched_savefig",
    "redirect_stdout",
}


# The worker script. Runs in a fresh interpreter; holds the persistent
# namespace in the module globals `G`. Reads one control-JSON line per step
# from stdin, execs the step's code into G (so state persists), writes a result
# JSON, then prints the done token.
_WORKER_SCRIPT = r'''
import io, json, os, sys, traceback, warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.figure as _mpl_figure
import pandas as pd

# Prevent pandas from SILENTLY truncating printed results. pandas defaults
# (max_rows=60, limited columns/width) insert "..." into the MIDDLE of a table,
# which would hand the Investigator/Synthesizer numbers with gaps they cannot see.
# Show all columns and full width; cap rows/cell-width only at levels no normal
# summarized result reaches (so a pathological raw dump is still bounded).
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", 200)
pd.set_option("display.max_rows", 2000)

_DONE = "__DELVE_KERNEL_DONE__"
_READY = "__DELVE_KERNEL_READY__"

_df_path = sys.argv[1] if len(sys.argv) > 1 else ""

# ---- Persistent namespace -------------------------------------------------
# Everything user code defines lands in G and survives across steps.
G = {}
G["os"] = os

if _df_path:
    if _df_path.endswith(".pkl"):
        G["df"] = pd.read_pickle(_df_path)
    elif _df_path.endswith(".parquet"):
        G["df"] = pd.read_parquet(_df_path)
    else:
        G["df"] = pd.read_csv(_df_path, low_memory=False)

# ---- Plot capture (patched once; reads _analysis_dir from G per step) ------
_real_savefig = _mpl_figure.Figure.savefig
G["_real_savefig"] = _real_savefig
G["_saved_plots"] = []
G["_plot_counter"] = [0]
G["_analysis_dir"] = "/tmp"

def _patched_show(*args, **kwargs):
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        G["_plot_counter"][0] += 1
        os.makedirs(G["_analysis_dir"], exist_ok=True)  # lazy: only when a plot exists
        path = os.path.join(G["_analysis_dir"], "plot_%03d.png" % G["_plot_counter"][0])
        _real_savefig(fig, path, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
        G["_saved_plots"].append(path)
    plt.close("all")

def _patched_savefig(self, fname, *args, **kwargs):
    G["_plot_counter"][0] += 1
    if isinstance(fname, str):
        basename = os.path.basename(fname)
    else:
        basename = "plot_%03d.png" % G["_plot_counter"][0]
    os.makedirs(G["_analysis_dir"], exist_ok=True)  # lazy: only when a plot exists
    path = os.path.join(G["_analysis_dir"], basename)
    kwargs.setdefault("dpi", 150)
    kwargs.setdefault("bbox_inches", "tight")
    kwargs.setdefault("facecolor", "white")
    _real_savefig(self, path, *args, **kwargs)
    G["_saved_plots"].append(path)

plt.show = _patched_show
_mpl_figure.Figure.savefig = _patched_savefig


def _namespace_summary():
    """Compact description of user-defined names currently in G."""
    internal = {
        "os", "io", "json", "sys", "traceback", "warnings",
        "matplotlib", "plt", "_mpl_figure", "pd", "df",
        "_real_savefig", "_saved_plots", "_plot_counter", "_analysis_dir",
        "_patched_show", "_patched_savefig", "redirect_stdout",
    }
    out = []
    for name, val in list(G.items()):
        if name in internal or name.startswith("__"):
            continue
        t = type(val).__name__
        try:
            if t == "DataFrame":
                cols = list(map(str, val.columns))
                shown = ",".join(cols[:60])
                more = "" if len(cols) <= 60 else (" ...(+%d more cols)" % (len(cols) - 60))
                desc = "DataFrame %dx%d cols=%s%s" % (val.shape[0], val.shape[1], shown, more)
            elif t == "Series":
                desc = "Series len=%d name=%s dtype=%s" % (len(val), val.name, val.dtype)
            elif t in ("int", "float", "bool", "str"):
                r = repr(val)
                desc = r if len(r) <= 80 else r[:77] + "..."
            elif t in ("list", "tuple", "set", "dict"):
                desc = "%s len=%d" % (t, len(val))
            elif t in ("ndarray",):
                desc = "ndarray shape=%s dtype=%s" % (getattr(val, "shape", "?"), getattr(val, "dtype", "?"))
            elif t in ("function", "type", "module"):
                continue  # imports/defs are not "derived state" worth listing
            else:
                desc = t
        except Exception:
            desc = t
        out.append({"name": name, "type": t, "desc": desc})
    return out


def _df_columns():
    df = G.get("df")
    if df is None:
        return []
    try:
        return list(map(str, df.columns))
    except Exception:
        return []


# ---- Signal ready, then serve steps ---------------------------------------
sys.stdout.write(_READY + "\n")
sys.stdout.flush()

from contextlib import redirect_stdout

for _line in sys.stdin:
    _line = _line.strip()
    if not _line:
        continue
    try:
        _req = json.loads(_line)
    except Exception:
        continue
    if _req.get("cmd") == "shutdown":
        break

    _code_path = _req.get("code_path")
    _result_path = _req.get("result_path")
    G["_analysis_dir"] = _req.get("analysis_dir") or "/tmp"
    G["_saved_plots"] = []

    _stdout = None
    _error = None
    try:
        with open(_code_path, "r", encoding="utf-8") as _f:
            _user_code = _f.read()
        _buf = io.StringIO()
        plt.close("all")
        with redirect_stdout(_buf):
            exec(_user_code, G)   # exec into G so derived state persists
        _stdout = _buf.getvalue()
    except Exception:
        _error = traceback.format_exc()
    finally:
        plt.close("all")

    _payload = {
        "stdout": _stdout,
        "error": _error,
        "plots": list(G.get("_saved_plots", [])),
        "namespace": _namespace_summary(),
        "columns": _df_columns(),
    }
    try:
        with open(_result_path, "w", encoding="utf-8") as _f:
            json.dump(_payload, _f)
    except Exception:
        pass

    sys.stdout.write(_DONE + "\n")
    sys.stdout.flush()
'''


def _truncate_traceback(tb, limit=1500):
    if not tb:
        return tb
    if len(tb) <= limit:
        return tb
    last = tb.rstrip().rsplit("\n", 1)[-1]
    return tb[: limit - 100] + "\n[...truncated]\n" + last


class KernelDead(Exception):
    """Raised internally when the worker process is gone and must be restarted."""


class PersistentKernel:
    """A long-lived worker process holding one persistent analysis namespace."""

    def __init__(self, df=None, analysis_root=None, step_timeout=STEP_TIMEOUT):
        self.step_timeout = step_timeout
        self._df_path = _serialize_dataframe(df) if df is not None else None
        self._worker_script_path = _write_temp_text(
            _WORKER_SCRIPT, suffix=".py", prefix="delve_kernel_"
        )
        # Ordered list of code blocks that previously executed without error.
        # Replayed (in order) to reconstruct the namespace after a restart.
        self._history = []
        # Latest namespace registry from the worker.
        self.registry = {"namespace": [], "columns": []}
        self._proc = None
        self._reader = None
        self._q = None
        self._start_worker()

    # ----- worker lifecycle ------------------------------------------------

    def _start_worker(self):
        self._proc = subprocess.Popen(
            [sys.executable, "-u", self._worker_script_path, self._df_path or ""],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._q = queue.Queue()
        # Bind this worker's proc and queue into the thread so a dying OLD worker
        # can never drop a stale sentinel into a NEW worker's queue after restart.
        self._reader = threading.Thread(
            target=self._read_stdout, args=(self._proc, self._q), daemon=True
        )
        self._reader.start()
        # Wait for the ready token.
        if not self._await_token(_READY, timeout=60):
            raise RuntimeError("Persistent kernel worker failed to start.")

    @staticmethod
    def _read_stdout(proc, q):
        try:
            for line in proc.stdout:
                q.put(line.strip())
        except Exception:
            pass
        finally:
            q.put(None)  # sentinel: this worker's stdout closed (worker died)

    def _await_token(self, token, timeout):
        """Block until `token` appears on the worker's stdout, or timeout/death."""
        import time
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            try:
                item = self._q.get(timeout=remaining)
            except queue.Empty:
                return False
            if item is None:
                raise KernelDead()
            if item == token:
                return True
            # Any other stray line is ignored (shouldn't happen; user output is
            # captured in-worker).

    def _kill_worker(self):
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.kill()
            except Exception:
                pass
        try:
            if self._proc:
                self._proc.wait(timeout=5)
        except Exception:
            pass
        self._proc = None

    def _restart_and_replay(self):
        """Restart the worker and replay successful history to rebuild state."""
        logger.warning("Persistent kernel restarting; replaying %d step(s).", len(self._history))
        self._kill_worker()
        self._start_worker()
        replayed = 0
        for code in self._history:
            ok = self._run_once(code, analysis_dir=None)
            if ok is None:  # death during replay → give up cleanly
                logger.error("Kernel died during replay at step %d; namespace partial.", replayed)
                break
            replayed += 1

    # ----- execution -------------------------------------------------------

    def _security_refuse(self, code):
        import re
        for banned in BLACKLIST:
            pattern = r"\b" + re.escape(banned) + r"\b"
            if re.search(pattern, code):
                if banned == "sys" and "sys." not in code:
                    continue
                return (
                    "Security notice: code contains restricted module "
                    f"({banned}) and was not executed."
                )
        return None

    def _run_once(self, code, analysis_dir):
        """One round-trip to the worker. Returns the result dict, or None if the
        worker died (caller decides whether to restart). Does NOT itself restart."""
        # NOTE: analysis_dir is NOT created here. The folder is made lazily, only
        # when a plot is actually written (see the worker plot patch), so steps
        # that produce no plots — which is all of them, since the executor is told
        # not to plot — never leave behind an empty NN/ folder.
        code_path = _write_temp_text(code, suffix=".py", prefix="delve_step_")
        fd, result_path = tempfile.mkstemp(suffix=".json", prefix="delve_res_")
        os.close(fd)
        try:
            req = json.dumps({
                "code_path": code_path,
                "result_path": result_path,
                "analysis_dir": analysis_dir or "",
            })
            try:
                self._proc.stdin.write(req + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                return None  # worker already gone

            try:
                alive = self._await_token(_DONE, timeout=self.step_timeout)
            except KernelDead:
                return None
            if not alive:
                # Hung: kill so the caller can restart+replay.
                self._kill_worker()
                return {"_timeout": True}

            try:
                with open(result_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                return {"stdout": None, "error": f"Failed to read result: {exc}",
                        "plots": [], "namespace": [], "columns": []}
        finally:
            _cleanup_files(code_path, result_path)

    def execute(self, code, analysis_dir=None):
        """Execute one step in the persistent namespace.

        Returns (stdout, error, plots), matching executor.CodeExecutor.execute
        (minus the df argument — df lives in the kernel).
        """
        refusal = self._security_refuse(code)
        if refusal:
            return None, refusal, []

        result = self._run_once(code, analysis_dir)

        # Timeout → restart + replay, then report.
        if result is not None and result.get("_timeout"):
            self._restart_and_replay()
            return None, (
                f"Execution killed: exceeded {self.step_timeout}s time limit. "
                "Simplify the step (fewer iterations, aggregate before fitting, "
                "cap pairwise operations). Namespace was preserved."
            ), []

        # Worker died (OOM/segfault) → restart + replay, then report.
        if result is None:
            self._restart_and_replay()
            return None, (
                "Execution killed by the OS (likely out of memory) or the worker "
                "crashed. Reduce memory use (sample/aggregate first, avoid large "
                "intermediate frames). Namespace was restored to the last good state."
            ), []

        stdout = result.get("stdout")
        error = _truncate_traceback(result.get("error"))
        plots = result.get("plots", []) or []
        self.registry = {
            "namespace": result.get("namespace", []),
            "columns": result.get("columns", []),
        }
        if error is None:
            self._history.append(code)  # only successful steps are replayable
        return stdout, error, plots

    # ----- introspection / lifecycle --------------------------------------

    def describe_namespace(self, max_items=120, names=None):
        """Human/LLM-readable registry of current derived state and df columns.
        ALL df column names are shown — the Executor writes code against these, so
        a truncated list could make it reference a column it cannot see.

        When `names` is given, only those derived objects are listed (plus a count
        of the rest); callers use this to show the Executor just the objects its
        spec references, and the Investigator just the live ones. When `names` is
        None, the MOST RECENT `max_items` objects are shown (newest kept, oldest
        dropped) — never the reverse, so freshly created objects are never hidden."""
        cols = self.registry.get("columns", [])
        ns = self.registry.get("namespace", [])
        lines = []
        if cols:
            lines.append(f"df columns ({len(cols)}): {', '.join(cols)}")
        if not ns:
            lines.append("Derived objects in namespace: (none yet)")
            return "\n".join(lines)
        if names is not None:
            shown = [it for it in ns if it["name"] in names]
            lines.append("Derived objects in scope:" if shown
                         else "Derived objects in scope: (none referenced)")
            for item in shown:
                lines.append(f"  - {item['name']}: {item['desc']}")
            hidden = len(ns) - len(shown)
            if hidden > 0:
                lines.append(f"  ... (+{hidden} other derived objects exist; "
                             f"reference one by its exact name to use it)")
        else:
            shown = ns[-max_items:]            # keep the NEWEST, not the oldest
            lines.append("Derived objects in namespace (most recent):")
            for item in shown:
                lines.append(f"  - {item['name']}: {item['desc']}")
            hidden = len(ns) - len(shown)
            if hidden > 0:
                lines.append(f"  ... (+{hidden} older derived objects hidden)")
        return "\n".join(lines)

    def reset(self):
        """Clear the namespace and history; start a fresh worker (df reloaded)."""
        self._kill_worker()
        self._history = []
        self.registry = {"namespace": [], "columns": []}
        self._start_worker()

    @property
    def history(self):
        """The ordered code blocks that executed successfully — persist this to
        resume a run later (see restore_history)."""
        return list(self._history)

    def restore_history(self, history):
        """Replay an external history (from a prior run) to rebuild the namespace
        on a fresh worker. Used by --continue. Best-effort: a step that fails to
        replay is logged and skipped."""
        for code in history or []:
            res = self._run_once(code, analysis_dir=None)
            if res is None:                      # worker died mid-replay
                self._restart_and_replay()
                res = self._run_once(code, analysis_dir=None)
            if res and not res.get("error") and not res.get("_timeout"):
                self._history.append(code)
                self.registry = {"namespace": res.get("namespace", []),
                                 "columns": res.get("columns", [])}
            else:
                logger.warning("restore_history: a step failed to replay; namespace may be partial.")

    def cleanup(self):
        try:
            if self._proc and self._proc.poll() is None:
                try:
                    self._proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                    self._proc.stdin.flush()
                    self._proc.wait(timeout=5)
                except Exception:
                    self._kill_worker()
        finally:
            self._kill_worker()
            _cleanup_files(self._worker_script_path, self._df_path)
            self._worker_script_path = None
            self._df_path = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.cleanup()
        return False

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass