# --- test bootstrap: runnable from the repo root via `python3 tests/<name>.py` ---
import os, sys, tempfile
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:0] = [os.path.join(_HERE, "stubs"), _ROOT]  # bundled httpx stub + delv-e modules
def _tmpdir(tag):
    return tempfile.mkdtemp(prefix=f"delve_{tag}_")
def _require_dataset():
    p = os.environ.get("EEDR_DATASET", os.path.join(_ROOT, "datasets", "EEDR_sessions_laps_enriched.csv"))
    if not os.path.exists(p):
        print(f"SKIP: EEDR dataset not found at {p} (set EEDR_DATASET or place the CSV in datasets/)")
        sys.exit(0)
    return p
# --- end bootstrap ---

# Covers the DETERMINISTIC half of the estimand discipline (name-the-estimand-up-front).
# The G3 synthesis gate itself is a model-facing reasoning discipline (the premium
# Synthesizer self-gates to NEEDS_MORE_WORK), so it is validated on real runs, not here.

import json, shutil, pandas as pd
from investigation import _parse_investigator, run_investigation
from nav_state import NavState
from kernel import PersistentKernel

# 1) PARSER: the ###ESTIMAND### block is captured; none/n-a/absent collapse to "".
d = _parse_investigator(
    "###ESTIMAND###\nThe sea-to-altitude pace correction at matched effort, in s/km.\n"
    "###THINKING###\nx\n###STATUS###\nCONTINUE\n###SPEC###\ncompute X; print.\n###LEDGER###\n")
assert d["estimand"].startswith("The sea-to-altitude"), d["estimand"]
assert _parse_investigator(
    "###ESTIMAND###\nnone\n###THINKING###\nx\n###STATUS###\nCONTINUE\n###SPEC###\ns\n###LEDGER###\n"
)["estimand"] == ""
assert _parse_investigator(  # absent block -> ""
    "###THINKING###\nx\n###STATUS###\nCONTINUE\n###SPEC###\ns\n###LEDGER###\n"
)["estimand"] == ""
print("parser captures ESTIMAND; none/absent -> empty: OK")

# 2) NAVSTATE: target_estimand round-trips, and old dicts (no key) load as "".
ns = NavState()
ns.target_estimand = "Correction C: speed ratio sea-level vs home at equal %HRmax, in percent."
back = NavState.from_dict(ns.to_dict())
assert back.target_estimand == ns.target_estimand
assert NavState.from_dict({"frontier": [], "regimes": [], "risks": [], "breakdown": []}).target_estimand == ""
print("nav target_estimand round-trips; backward-compatible default: OK")

# 3) RENDER: shown at the top when set, absent when empty.
r_set = ns.render_for_investigator([])
assert "TARGET ESTIMAND" in r_set and "speed ratio sea-level vs home" in r_set
assert r_set.index("TARGET ESTIMAND") < r_set.index("FRONTIER")  # pinned at the top
assert "TARGET ESTIMAND" not in NavState().render_for_investigator([])
print("estimand rendered at top when set, absent when empty: OK")

# 4) PIN-ONCE through the real loop: named on step 1, NOT overwritten by a step-2 drift.
df = pd.DataFrame({"a": [1, 2, 3, 4],
                   "location_status": ["abroad", "ethiopia", "abroad", "ethiopia"],
                   "lap_avg_speed_ms": [5, 4, 5.2, 4.1]})
out = _tmpdir("estimand_out")
if os.path.exists(out): shutil.rmtree(out)

def inv(est, t, s, sp, l):
    head = f"###ESTIMAND###\n{est}\n" if est else ""
    return head + f"###THINKING###\n{t}\n###STATUS###\n{s}\n###SPEC###\n{sp}\n###LEDGER###\n{l}\n"
def code(b): return "```python\n" + b + "\n```"

L_OK = "FRONTIER | f1 | tested | steps: 1\nREGIME | intensity | examined | steps: 1"
PINNED = "Correction C: median speed ratio, sea-level vs home, matched on %HRmax, in percent."
DRIFT = "A within-altitude gradient in s/km per 1000m."  # later restatement; must be ignored
INV = [inv(PINNED, "step1", "CONTINUE", "median speed by location_status; print.", L_OK),
       inv(DRIFT,  "step2", "SYNTHESIZE", "none", L_OK)]
EXEC = {1: code("print('###RESULTS_START###');"
                "print(df.groupby('location_status')['lap_avg_speed_ms'].median().to_string());"
                "print('###RESULTS_END###')")}

class Mock:
    def __init__(s): s.i = 0; s.e = 0
    def call(s, m, model, max_tokens=10000, temperature=0, agent=None):
        if agent == "Investigator": r = INV[s.i]; s.i += 1; return r
        if agent == "Executor":
            if "raised an error" not in m[-1]["content"]: s.e += 1
            return EXEC[s.e]
        if agent == "Synthesizer":
            return "###VERDICT###\nFINAL\n###BRIEFING###\n## Summary\nok.\n"
        return ""

k = PersistentKernel(df=df)
log, _, nav, briefing = run_investigation(
    seed="t", df=df, client=Mock(), investigator_model="m:p", executor_model="m:c",
    schema_text="(s)", max_steps=5, output_dir=out, kernel=k, nav=NavState())
k.cleanup()

assert nav.target_estimand == PINNED, nav.target_estimand  # pinned step 1, drift ignored
saved = json.load(open(os.path.join(out, "nav_state.json")))
assert saved.get("target_estimand") == PINNED              # and persisted to disk
print("estimand pinned on first emission, later drift ignored, persisted: OK")

print("ESTIMAND PLUMBING (name-up-front + pin + render + round-trip) WORKS")
