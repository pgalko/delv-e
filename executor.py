"""
Local code executor with subprocess isolation, timeout, and OOM protection.

Runs generated code in a separate Python process via subprocess.Popen.
If the code exhausts memory or hangs, the subprocess is killed without
crashing the parent process. Works on Linux, macOS, and Windows.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap

from logger_config import get_logger

logger = get_logger(__name__)

# Matplotlib monkey-patch injected before user code.
# In the subprocess model each execution runs in a fresh interpreter, so we do
# not need the old parent-side _REAL_SAVEFIG capture to prevent cross-run patch
# leakage. The real savefig is resolved inside the runner immediately before
# user code executes.
PLOT_PATCH = '''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.figure as _mpl_figure
_plot_counter = [0]
_saved_plots = []

def _patched_show(*args, **kwargs):
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        _plot_counter[0] += 1
        path = os.path.join(_analysis_dir, f"plot_{_plot_counter[0]:03d}.png")
        _real_savefig(fig, path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        _saved_plots.append(path)
    plt.close('all')

def _patched_savefig(self, fname, *args, **kwargs):
    _plot_counter[0] += 1
    if isinstance(fname, str):
        basename = os.path.basename(fname)
    else:
        basename = f"plot_{_plot_counter[0]:03d}.png"
    path = os.path.join(_analysis_dir, basename)
    kwargs.setdefault('dpi', 150)
    kwargs.setdefault('bbox_inches', 'tight')
    kwargs.setdefault('facecolor', 'white')
    _real_savefig(self, path, *args, **kwargs)
    _saved_plots.append(path)

plt.show = _patched_show
_mpl_figure.Figure.savefig = _patched_savefig
'''

# Modules banned from generated code (security)
BLACKLIST = [
    'subprocess', 'sys', 'exec', 'eval', 'socket', 'urllib',
    'shutil', 'pickle', 'ctypes', 'multiprocessing', 'tempfile',
    'pty', 'commands', 'cgi', 'builtins', 'importlib',
    'webbrowser', 'http', 'ftplib', 'smtplib', 'xmlrpc',
    'os.system', 'os.popen', 'os.exec', 'os.spawn',
    'os.remove', 'os.rmdir', 'os.unlink', '__import__',
]

# Maximum wall-clock seconds for a single code execution.
EXECUTION_TIMEOUT = 300

_TIMEOUT_ERROR = (
    "Execution killed: exceeded {timeout}s time limit.\n\n"
    "The code is too computationally expensive. Common causes and fixes:\n"
    "- Bootstrap/permutation/cross-validation: use ≤200 iterations, ≤5 folds\n"
    "- Iterative model fits (MixedLM, Bradley-Terry, EM, MCMC): reduce groups "
    "to ≤300, filter to groups with ≥10 observations BEFORE fitting\n"
    "- Pairwise operations (distance matrices, all-pairs tests): cap at ≤500 "
    "entities per dimension, or sample\n"
    "- Large regression: aggregate to per-group summaries before fitting, "
    "don't pass row-level data with >10,000 rows to iterative solvers\n"
    "- Grid search: limit parameter combinations to ≤50 total\n"
    "- Nested stratification: N_iterations × N_strata × N_groups must stay "
    "under 500,000 total operations\n"
    "Simplify the approach. The result must compute in under {timeout} seconds."
)

_OOM_ERROR = (
    "Execution killed by operating system (likely out of memory).\n\n"
    "The code consumed too much memory. Common causes and fixes:\n"
    "- Bootstrap/permutation combined with large models: reduce to ≤50 iterations "
    "or eliminate bootstrap entirely and use analytical standard errors\n"
    "- Pairwise comparison matrices on >200 entities: sample or aggregate first\n"
    "- Multiple large intermediate DataFrames: process sequentially, use del\n"
    "- Large pivot/cross-tabulation with high-cardinality dimensions: aggregate first\n"
    "Simplify the approach. The code must run within available memory."
)


_RUNNER_SCRIPT = r'''import io, json, os, sys, traceback, warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.figure as _mpl_figure
import pandas as pd

_df_path, _code_path, _result_path = sys.argv[1], sys.argv[2], sys.argv[3]
_analysis_dir = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else None

_df = None
if _df_path:
    if _df_path.endswith(".pkl"):
        _df = pd.read_pickle(_df_path)
    elif _df_path.endswith(".parquet"):
        _df = pd.read_parquet(_df_path)
    else:
        _df = pd.read_csv(_df_path, low_memory=False)

_real_savefig = _mpl_figure.Figure.savefig

with open(_code_path, "r", encoding="utf-8") as _f:
    _user_code = _f.read()

_buf = io.StringIO()
try:
    plt.close("all")
    from contextlib import redirect_stdout
    with redirect_stdout(_buf):
        _vars = {
            "_analysis_dir": _analysis_dir or "/tmp",
            "_real_savefig": _real_savefig,
            "os": os,
        }
        if _df is not None:
            _vars["df"] = _df
        exec(_user_code, _vars)
    _plots = _vars.get("_saved_plots", [])
    with open(_result_path, "w", encoding="utf-8") as _f:
        json.dump({"stdout": _buf.getvalue(), "error": None, "plots": _plots}, _f)
except Exception:
    with open(_result_path, "w", encoding="utf-8") as _f:
        json.dump({"stdout": None, "error": traceback.format_exc(), "plots": []}, _f)
finally:
    plt.close("all")
    _buf.close()
'''


def _filter_traceback(full_traceback, patch_lines):
    """Produce a traceback that references user code line numbers."""
    filtered_parts = []
    for line in full_traceback.splitlines():
        if '<string>' in line and 'line' in line:
            try:
                num = int(line.split(', line ')[1].split(',')[0])
                adjusted = num - patch_lines
                line = line.replace(f'line {num}', f'line {adjusted}')
            except (IndexError, ValueError):
                pass
        filtered_parts.append(line)

    result = '\n'.join(filtered_parts)
    if len(result) > 1500:
        last_line = result.rstrip().rsplit('\n', 1)[-1]
        result = result[:1400] + "\n[...truncated]\n" + last_line
    return result


def _cleanup_files(*paths):
    """Remove temp files, ignoring missing files and empty paths."""
    for path in paths:
        if not path:
            continue
        try:
            os.unlink(path)
        except OSError:
            pass


def _write_temp_text(content, *, suffix, prefix):
    """Write a UTF-8 text file and return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    os.close(fd)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return path


def _serialize_dataframe(df):
    """Serialize the current DataFrame state to a temp file.

    Prefer pickle for fidelity and broad compatibility between the trusted
    parent and child processes. Fall back to parquet, then CSV only if needed.
    """
    path = None
    try:
        fd, path = tempfile.mkstemp(suffix='.pkl', prefix='delve_')
        os.close(fd)
        df.to_pickle(path)
        return path
    except Exception:
        _cleanup_files(path)

    try:
        fd, path = tempfile.mkstemp(suffix='.parquet', prefix='delve_')
        os.close(fd)
        df.to_parquet(path)
        return path
    except Exception:
        _cleanup_files(path)

    fd, path = tempfile.mkstemp(suffix='.csv', prefix='delve_')
    os.close(fd)
    df.to_csv(path, index=False)
    return path


class CodeExecutor:
    """Execute Python code in an isolated subprocess with timeout and OOM protection."""

    def __init__(self):
        self._runner_path = None

    def _ensure_runner_cached(self):
        """Write the runner script to disk on first use."""
        if self._runner_path:
            return
        self._runner_path = _write_temp_text(
            _RUNNER_SCRIPT,
            suffix='.py',
            prefix='delve_runner_',
        )

    def cleanup(self):
        """Remove persistent temp files owned by this executor instance."""
        _cleanup_files(self._runner_path)
        self._runner_path = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.cleanup()
        return False

    def __del__(self):
        # Best-effort cleanup only. Call cleanup() explicitly when possible.
        try:
            self.cleanup()
        except Exception:
            pass

    def execute(self, code, df=None, analysis_dir=None):
        """
        Execute code with df in scope.

        Args:
            code: Python code string
            df: pandas DataFrame (None for computation-only mode)
            analysis_dir: directory for saving plots (None = no plot saving)

        Returns:
            (stdout_text, error_text, saved_plot_paths)
        """
        if analysis_dir:
            os.makedirs(analysis_dir, exist_ok=True)

        self._ensure_runner_cached()

        df_path = _serialize_dataframe(df) if df is not None else None
        code_path = None
        result_path = None

        try:
            combined_code = f"{PLOT_PATCH}\n{code}" if analysis_dir else code
            code_path = _write_temp_text(combined_code, suffix='.py', prefix='delve_')

            fd, result_path = tempfile.mkstemp(suffix='.json', prefix='delve_')
            os.close(fd)

            process = subprocess.Popen(
                [
                    sys.executable,
                    self._runner_path,
                    df_path or '',
                    code_path,
                    result_path,
                    analysis_dir or '',
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            try:
                _, stderr_text = process.communicate(timeout=EXECUTION_TIMEOUT)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                return None, _TIMEOUT_ERROR.format(timeout=EXECUTION_TIMEOUT), []

            error = None
            stdout = None
            plots = []

            if os.path.exists(result_path):
                try:
                    with open(result_path, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                    stdout = result.get('stdout')
                    error = result.get('error')
                    plots = result.get('plots', []) or []
                except (json.JSONDecodeError, OSError) as exc:
                    error = f"Failed to read execution results: {exc}"

            if process.returncode != 0:
                if process.returncode < 0:
                    sig = -process.returncode
                    if sig in (9, 11):
                        error = _OOM_ERROR
                    else:
                        error = error or (
                            f"Execution killed by signal {sig}.\n"
                            "The code may have consumed too much memory or CPU. "
                            "Simplify the approach."
                        )
                elif not error:
                    if stderr_text and stderr_text.strip():
                        error = stderr_text.strip()
                    else:
                        error = f"Execution failed with exit code {process.returncode}"
                return None, error, []

            if error:
                patch_lines = len(PLOT_PATCH.splitlines()) if analysis_dir else 0
                error = _filter_traceback(error, patch_lines)
                return None, error, []

            return stdout, None, plots

        finally:
            _cleanup_files(df_path, code_path, result_path)

    @staticmethod
    def _read_error_from_result(result_path):
        """Backward-compatible helper retained for external callers/tests."""
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            return result.get('error')
        except (json.JSONDecodeError, FileNotFoundError, KeyError, OSError):
            return None


def extract_code(response):
    """Extract Python code from an LLM response."""
    if not response:
        return ""

    patterns = [
        r'```[Pp]ython\s*\n(.*?)```',
        r'```\s*\n(.*?)```',
    ]

    segments = []
    for pattern in patterns:
        segments = re.findall(pattern, response, re.DOTALL)
        if segments:
            break

    if not segments:
        lines = response.split('\n')
        code_lines = []
        in_code = False
        for line in lines:
            stripped = line.strip()
            if not in_code and (
                stripped.startswith('import ')
                or stripped.startswith('from ')
                or stripped.startswith('import\t')
            ):
                in_code = True
            if in_code:
                if (
                    stripped
                    and not any(c in stripped for c in '=()[]{}:.,+-*/#%')
                    and len(stripped) > 60
                    and not stripped.startswith('#')
                ):
                    break
                code_lines.append(line)
        if len(code_lines) >= 3:
            segments = ['\n'.join(code_lines)]

    if not segments:
        return ""

    code = '\n\n'.join(textwrap.dedent(segment) for segment in segments).strip()

    for banned in BLACKLIST:
        pattern = r'\b' + re.escape(banned) + r'\b'
        if re.search(pattern, code):
            if banned == 'sys' and 'sys.' not in code:
                continue
            return (
                'print("Security notice: generated code contains restricted '
                f'module ({banned}) and cannot be executed.")'
            )

    lines = code.split('\n')
    processed = []
    in_main = False
    main_indent = 0
    for line in lines:
        if '__name__' in line and '__main__' in line:
            in_main = True
            base_indent = len(line) - len(line.lstrip())
            main_indent = base_indent + 4
            continue
        if in_main:
            current_indent = len(line) - len(line.lstrip())
            if line.strip() and current_indent <= (main_indent - 4):
                in_main = False
                processed.append(line)
            else:
                if line.strip():
                    stripped = line[main_indent:] if len(line) > main_indent else line.lstrip()
                    processed.append(stripped)
                else:
                    processed.append('')
        else:
            processed.append(line)

    result = '\n'.join(processed)
    result = re.sub(r'n_jobs\s*=\s*-1', 'n_jobs=4', result)
    result = re.sub(r'n_jobs\s*=\s*None', 'n_jobs=4', result)
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()