"""
Local code executor using exec() with stdout capture and matplotlib plot saving.
"""

import io
import os
import re
import sys
import textwrap
import traceback
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt

from logger_config import get_logger
logger = get_logger(__name__)

# Matplotlib monkey-patch injected before user code.
# CRITICAL: _real_savefig is passed in via local_vars to avoid closure chain bug.
# If we captured Figure.savefig inside the exec, the second exec would capture
# the FIRST exec's patched version, causing all plots to go to exec1's directory.
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

# Save the REAL Figure.savefig once at import time, before any exec can patch it
import matplotlib.figure as _mpl_figure_ref
_REAL_SAVEFIG = _mpl_figure_ref.Figure.savefig

# Modules banned from generated code (security)
BLACKLIST = [
    'subprocess', 'sys', 'exec', 'eval', 'socket', 'urllib',
    'shutil', 'pickle', 'ctypes', 'multiprocessing', 'tempfile',
    'pty', 'commands', 'cgi', 'builtins', 'importlib',
    'webbrowser', 'http', 'ftplib', 'smtplib', 'xmlrpc',
    'os.system', 'os.popen', 'os.exec', 'os.spawn',
    'os.remove', 'os.rmdir', 'os.unlink', '__import__',
]


class CodeExecutor:
    """Execute Python code locally with stdout capture and plot saving."""

    def execute(self, code, df, analysis_dir=None):
        """
        Execute code with df in scope.
        
        Args:
            code: Python code string
            df: pandas DataFrame  
            analysis_dir: directory for saving plots (None = no plot saving)
        
        Returns:
            (stdout_text, error_text, saved_plot_paths)
        """
        if analysis_dir:
            os.makedirs(analysis_dir, exist_ok=True)

        output_buffer = io.StringIO()
        saved_plots = []

        try:
            plt.close('all')
            
            # Suppress warnings (FutureWarning etc.) from generated code
            import warnings
            warnings.filterwarnings('ignore')
            
            from contextlib import redirect_stdout
            with redirect_stdout(output_buffer):
                local_vars = {
                    'df': df.copy(),
                    '_analysis_dir': analysis_dir or '/tmp',
                    '_real_savefig': _REAL_SAVEFIG,
                    'os': __import__('os'),
                }
                
                if analysis_dir:
                    exec(PLOT_PATCH + "\n" + code, local_vars)
                    saved_plots = local_vars.get('_saved_plots', [])
                else:
                    exec(code, local_vars)

            results = output_buffer.getvalue()
            return results, None, saved_plots

        except Exception:
            # Build a clean traceback pointing to user code lines
            full_tb = traceback.format_exc()
            clean_tb = self._filter_traceback(code, full_tb, analysis_dir)
            return None, clean_tb, []

        finally:
            plt.close('all')
            output_buffer.close()

    def _filter_traceback(self, code, full_traceback, analysis_dir):
        """Produce a traceback that references user code line numbers."""
        # Offset from the injected patch code
        patch_lines = len(PLOT_PATCH.split('\n')) if analysis_dir else 0
        
        tb_lines = full_traceback.split('\n')
        code_lines = code.split('\n')
        
        # Find error lines referencing <string> and adjust line numbers
        filtered_parts = []
        for line in tb_lines:
            if '<string>' in line and 'line' in line:
                try:
                    num = int(line.split(', line ')[1].split(',')[0])
                    adjusted = num - patch_lines
                    line = line.replace(f'line {num}', f'line {adjusted}')
                except (IndexError, ValueError):
                    pass
            filtered_parts.append(line)
        
        result = '\n'.join(filtered_parts)
        # Truncate very long tracebacks, but always preserve the final exception line
        if len(result) > 1500:
            last_line = result.rstrip().rsplit('\n', 1)[-1]
            result = result[:1400] + "\n[...truncated]\n" + last_line
        return result


def extract_code(response):
    """Extract Python code from LLM response (```python ... ``` blocks)."""
    if not response:
        return ""

    # Try multiple patterns in order of specificity
    # Pattern 1: ```python ... ``` (with flexible whitespace)
    segments = re.findall(r'```python\s*\n(.*?)```', response, re.DOTALL)
    
    # Pattern 2: ``` ... ``` (any fenced code block)
    if not segments:
        segments = re.findall(r'```\s*\n(.*?)```', response, re.DOTALL)
    
    # Pattern 3: Haiku sometimes uses ```Python (capital P)
    if not segments:
        segments = re.findall(r'```[Pp]ython\s*\n(.*?)```', response, re.DOTALL)

    # Pattern 4: Look for obvious Python code even without backticks
    # (starts with import or common patterns, multiple lines)
    if not segments:
        lines = response.split('\n')
        code_lines = []
        in_code = False
        for line in lines:
            stripped = line.strip()
            if not in_code and (stripped.startswith('import ') or stripped.startswith('from ') 
                                or stripped.startswith('import\t')):
                in_code = True
            if in_code:
                # Stop if we hit obvious prose (long line without code chars)
                if (stripped and not any(c in stripped for c in '=()[]{}:.,+-*/#%') 
                    and len(stripped) > 60 and not stripped.startswith('#')):
                    break
                code_lines.append(line)
        if len(code_lines) >= 3:
            segments = ['\n'.join(code_lines)]

    if not segments:
        return ""

    code = '\n\n'.join(textwrap.dedent(s) for s in segments).strip()

    # Security check
    for banned in BLACKLIST:
        pattern = r'\b' + re.escape(banned) + r'\b'
        if re.search(pattern, code):
            if banned in ('sys',) and 'sys.' not in code:
                continue  # Allow 'sys' in comments but not sys.xxx
            return (
                'print("Security notice: generated code contains restricted '
                f'module ({banned}) and cannot be executed.")'
            )

    # Remove if __name__ == '__main__' wrapper — extract its body
    lines = code.split('\n')
    processed = []
    in_main = False
    main_indent = 0
    for line in lines:
        if '__name__' in line and '__main__' in line:
            in_main = True
            base = len(line) - len(line.lstrip())
            main_indent = base + 4  # typical indent
            continue
        if in_main:
            if line.strip() and (len(line) - len(line.lstrip())) <= (main_indent - 4):
                in_main = False
                processed.append(line)
            else:
                # Dedent main block content
                if line.strip():
                    stripped = line[main_indent:] if len(line) > main_indent else line.lstrip()
                    processed.append(stripped)
                else:
                    processed.append('')
        else:
            processed.append(line)

    # Cap n_jobs for ML operations
    result = '\n'.join(processed)
    result = re.sub(r'n_jobs\s*=\s*-1', 'n_jobs=4', result)
    result = re.sub(r'n_jobs\s*=\s*None', 'n_jobs=4', result)

    # Clean up multiple blank lines
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()