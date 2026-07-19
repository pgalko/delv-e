# --- test bootstrap: runnable from the repo root via `python3 tests/<n>.py` ---
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:0] = [os.path.join(_HERE, "stubs"), _ROOT]  # bundled httpx stub + delv-e modules
# --- end bootstrap ---

# Source-side noise reduction, added after auditing a real 17-turn heavy run:
# 83% of resident raw mass was numeric table rows and ~6% of the prompt was
# float digits past the 4th significant figure. Two levers: (1) a PRINT BUDGET
# clause in the Investigator's spec-writing contract (decision-sufficient
# prints, not listings), present in BOTH prompt modes and itself clean under
# the spec leakage audit; (2) the kernel worker prints floats at 4 significant
# digits, display-only.

import inspect

import prompts as P
import kernel as K
from investigation import scan_spec_for_leakage

# ── 1) The clause exists in both prompt modes and carries the key guidance ──
src = inspect.getsource(P)
assert src.count("PRINT BUDGET (the context cost of a print)") == 2, \
    "the clause must appear in both spec-rule blocks"
i = src.index("PRINT BUDGET (the context cost of a print)")
clause = src[i:src.index("ONE MOVE PER SPEC", i)]
for phrase in ("decision-sufficient", "top and bottom rows", "row counts",
               "summary statistics", "persist in the namespace",
               "seeing every row"):
    assert phrase in clause, f"clause lost its guidance: {phrase!r}"
print("prompt clause: present in both modes, guidance intact: OK")

# ── 2) The clause itself passes the leakage audit it sits beside ──
assert scan_spec_for_leakage(clause) == [], "the clause must model closed-spec language"
print("prompt clause: leakage-clean: OK")

# ── 3) The worker script sets the float format (that is where steps print) ──
assert 'pd.set_option("display.float_format", lambda v: f"{v:.4g}")' in K._WORKER_SCRIPT, \
    "the worker display block must set the 4-significant-digit float format"
# The anti-truncation options the format joins must remain untouched.
for opt in ('("display.max_columns", None)', '("display.max_rows", 2000)'):
    assert opt in K._WORKER_SCRIPT
print("worker wiring: float format beside the anti-truncation block: OK")

# ── 4) Behavior: the exact option the worker sets produces the intended text ──
import pandas as pd
prev = pd.get_option("display.float_format")
try:
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    s = str(pd.DataFrame({"ability": [0.739847123, -1.077487, 123456.789]}))
    assert "0.7398" in s and "-1.077" in s and "1.235e+05" in s, s
    assert "0.739847" not in s, "digits past the 4th must not print"
finally:
    pd.set_option("display.float_format", prev)
print("behavior: 4 significant digits, display-only: OK")

print("test_print_discipline: OK")
