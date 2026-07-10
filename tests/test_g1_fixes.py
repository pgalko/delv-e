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

# Q2 (code-grounded G1 backstop) + Q3 (provisional briefing when gated with budget spent).
import os, shutil, pandas as pd
from kernel import PersistentKernel
from nav_state import NavState, code_shows_stratification

# ---------- Q2: generic stratification detector + code-aware g1_satisfied ----------
def L(code): return [{"step": 1, "code": code, "terminal": False}]
assert code_shows_stratification(L("g = df.groupby('x')['y'].mean()"))
assert code_shows_stratification(L("t = df.pivot_table(index='a', values='v')"))
assert code_shows_stratification(L("df['band'] = pd.qcut(df['z'], 4)"))
assert code_shows_stratification(L("for lvl in df['athlete'].unique():\n    sub = df[df.athlete==lvl]"))
assert code_shows_stratification(L("m = smf.ols('y ~ alt * C(zone)', df).fit()"))
assert not code_shows_stratification(L("m = df['y'].mean(); print(m)"))      # pooled only
assert not code_shows_stratification(L("df2 = df[df.y>0]; print(df2.shape)")) # filter, not strat
assert not code_shows_stratification([])                                     # empty

nav = NavState()
# no regime marked, no log -> not satisfied (back-compat)
assert nav.g1_satisfied() is False
# no regime marked, but code stratified -> satisfied via backstop
assert nav.g1_satisfied(L("df.groupby('g').size()")) is True
# no regime marked, pooled-only code -> still not satisfied
assert nav.g1_satisfied(L("df['y'].median()")) is False
print("Q2 code-grounded G1 backstop: OK")

# ---------- Q3: gated synth with budget spent must still yield a briefing ----------
df = pd.DataFrame({"a":[1,2,3,4], "g":["x","y","x","y"], "v":[5,4,5,4]})
out = _tmpdir("g1fix_out")
if os.path.exists(out): shutil.rmtree(out)

def inv(t,s,sp,l): return f"###THINKING###\n{t}\n###STATUS###\n{s}\n###SPEC###\n{sp}\n###LEDGER###\n{l}\n"
# Investigator immediately wants to SYNTHESIZE, ledger leaves a regime not_examined,
# and NO executor step runs (so no stratification in code -> g1 genuinely unmet).
LED = "FRONTIER | f1 | in_progress | steps: -\nREGIME | intensity | not_examined | steps: -"
INV = [inv("done thinking","SYNTHESIZE","(none)",LED)]
# Synthesizer (non-final) returns FINAL -> override flips to NEEDS_MORE_WORK; budget=0
# means no pushback, so the Q3 fallback must force a provisional FINAL briefing.
SYNTH = ("Pre-verdict reasoning here.\n###VERDICT###\nFINAL\n###BRIEFING###\n"
         "## Summary\nThe effect appears positive.\n")
class Mock:
    def __init__(s): s.i=0
    def call(s, m, model, max_tokens=10000, temperature=0, agent=None):
        if agent=="Investigator":
            r=INV[min(s.i,len(INV)-1)]; s.i+=1; return r
        if agent=="Synthesizer": return SYNTH
        return ""

k = PersistentKernel(df=df)
log,_,nav,briefing = run_investigation = __import__("investigation").run_investigation(
    seed="test", df=df, client=Mock(),
    investigator_model="m:p", executor_model="m:c", schema_text="(s)",
    max_steps=3, output_dir=out, kernel=k, nav=NavState(), g1_pushback_budget=0)
k.cleanup()

assert briefing, "Q3 FAILED: gated-with-budget-spent produced NO briefing (the bug)"
assert os.path.exists(os.path.join(out,"briefing.md")), "briefing.md not written"
term = [e for e in log if e.get("terminal")]
assert term and term[-1].get("synth_verdict")=="FINAL", "terminal verdict should be FINAL after forced synthesis"
assert "PROVISIONAL" in briefing or "effect appears positive" in briefing, "should salvage/forced briefing"
print("Q3 provisional-on-gate-with-budget-spent: OK")
print("\nALL Q2 + Q3 ASSERTIONS PASSED")
