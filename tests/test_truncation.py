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

# Investigator truncation handling: an empty/token-capped turn must be RETRIED,
# not silently finalized; persistent truncation falls back to a provisional briefing.
import os, shutil, pandas as pd
from kernel import PersistentKernel
from nav_state import NavState
from investigation import run_investigation, INV_TRUNCATION_RETRIES

df = pd.DataFrame({"a":[1,2,3,4], "g":["x","y","x","y"], "v":[5.0,4.0,5.2,4.1]})

def inv(t,s,sp,l): return f"###THINKING###\n{t}\n###STATUS###\n{s}\n###SPEC###\n{sp}\n###LEDGER###\n{l}\n"
def code(b): return "```python\n"+b+"\n```"
LED = "FRONTIER | f1 | in_progress | steps: 1\nREGIME | g | not_examined | steps: -"
EXEC = {1: code("print('###RESULTS_START###')\nprint(df.groupby('g')['v'].median().to_string())\nprint('###RESULTS_END###')")}
SYNTH = "Reasoning.\n###VERDICT###\nFINAL\n###BRIEFING###\n## Summary\nEffect is positive.\n"

class Mock:
    def __init__(s, inv_seq): s.inv_seq=inv_seq; s.inv_calls=0; s.e=0
    def call(s, m, model, max_tokens=10000, temperature=0, agent=None, return_meta=False):
        if agent=="Investigator":
            content, trunc = s.inv_seq[min(s.inv_calls, len(s.inv_seq)-1)]
            s.inv_calls += 1
            if return_meta:
                return content, {"output_tokens": (max_tokens if trunc else 50),
                                 "max_tokens": max_tokens, "truncated": trunc}
            return content
        if agent=="Executor":
            if "raised an error" not in m[-1]["content"]: s.e += 1
            return EXEC[s.e]
        if agent=="Synthesizer":
            return SYNTH
        return ""

# ---------- Case A: one truncated turn is retried, then the run proceeds normally ----------
outA=_tmpdir("trunc_A");  shutil.rmtree(outA, ignore_errors=True)
seqA=[("", True),                                   # empty + truncated -> must retry
      (inv("look","CONTINUE","median v by g; print.",LED), False),  # real step 1
      (inv("done","SYNTHESIZE","(none)",LED), False)]               # then synthesize
mA=Mock(seqA); kA=PersistentKernel(df=df)
log,_,nav,briefing = run_investigation(seed="t", df=df, client=mA,
    investigator_model="m:p", executor_model="m:c", schema_text="(s)",
    max_steps=6, output_dir=outA, kernel=kA, nav=NavState())
kA.cleanup()
assert briefing, "A: no briefing"
assert mA.inv_calls == 3, f"A: expected 3 investigator calls (1 retried + 2 real), got {mA.inv_calls}"
assert any(e.get("code") for e in log), "A: should have a real executed step (did not finalize on the empty turn)"
assert "positive" in briefing
print(f"Case A (retry-then-proceed): OK — investigator called {mA.inv_calls}x, real step executed, briefing produced")

# ---------- Case B: persistent truncation -> provisional briefing, no infinite loop ----------
outB=_tmpdir("trunc_B");  shutil.rmtree(outB, ignore_errors=True)
mB=Mock([("", True)]); kB=PersistentKernel(df=df)   # ALWAYS empty+truncated
log,_,nav,briefing = run_investigation(seed="t", df=df, client=mB,
    investigator_model="m:p", executor_model="m:c", schema_text="(s)",
    max_steps=6, output_dir=outB, kernel=kB, nav=NavState())
kB.cleanup()
assert briefing, "B: persistent truncation produced no briefing"
assert mB.inv_calls == INV_TRUNCATION_RETRIES + 1, \
    f"B: expected {INV_TRUNCATION_RETRIES+1} investigator calls before fallback, got {mB.inv_calls}"
assert os.path.exists(os.path.join(outB,"briefing.md")), "B: briefing.md not written"
term=[e for e in log if e.get("terminal")]
assert term and term[-1].get("synth_verdict")=="FINAL", "B: terminal verdict should be FINAL (forced)"
print(f"Case B (persistent truncation -> provisional): OK — {mB.inv_calls} calls then provisional briefing")
print("\nALL TRUNCATION-RETRY ASSERTIONS PASSED")
