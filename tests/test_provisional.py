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

import os, shutil, pandas as pd
from kernel import PersistentKernel
from nav_state import NavState
from investigation import run_investigation

df = pd.DataFrame({"a":[1,2,3,4],"location_status":["abroad","ethiopia","abroad","ethiopia"],"lap_avg_speed_ms":[5,4,5.2,4.1]})
out=_tmpdir("prov_out")
if os.path.exists(out): shutil.rmtree(out)

def inv(t,s,sp,l): return f"###THINKING###\n{t}\n###STATUS###\n{s}\n###SPEC###\n{sp}\n###LEDGER###\n{l}\n"
def code(b): return "```python\n"+b+"\n```"
# Investigator keeps working (never says SYNTHESIZE) -> hits max_steps=2 -> FINAL synthesis at ceiling.
# Ledger leaves a regime not_examined so a NON-final synth would gate; FINAL must NOT gate.
L="FRONTIER | f1 | in_progress | steps: 1\nREGIME | intensity | not_examined | steps: -"
INV=[inv("look at speeds","CONTINUE","median lap_avg_speed_ms by location_status; print.",L),
     inv("noisy, keep going","CONTINUE","count rows by location_status; print.",L)]
EXEC={1:code("print('###RESULTS_START###');print(df.groupby('location_status')['lap_avg_speed_ms'].median().to_string());print('###RESULTS_END###')"),
      2:code("print('###RESULTS_START###');print(df['location_status'].value_counts().to_string());print('###RESULTS_END###')")}
# Synthesizer (final mode) writes reasoning then would gate — but final=True must coerce to a briefing.
# Simulate the model STILL trying to gate (worst case) to prove salvage works:
SYNTH_GATES=("Here is my analysis of the speeds: abroad faster but intensity not yet examined, "
             "so I cannot confirm.\n###VERDICT###\nNEEDS_MORE_WORK: examine intensity\n###BRIEFING###\nnone\n")
class Mock:
    def __init__(s):s.i=0;s.e=0
    def call(s,m,model,max_tokens=10000,temperature=0,agent=None):
        if agent=="Investigator":r=INV[s.i];s.i+=1;return r
        if agent=="Executor":
            if "raised an error" not in m[-1]["content"]:s.e+=1
            return EXEC[s.e]
        if agent=="Synthesizer":return SYNTH_GATES
        return ""

k=PersistentKernel(df=df)
log,_,nav,briefing=run_investigation(seed="test",df=df,client=Mock(),
    investigator_model="m:p",executor_model="m:c",schema_text="(s)",
    max_steps=2,output_dir=out,kernel=k,nav=NavState())
k.cleanup()

print("briefing.md exists:", os.path.exists(os.path.join(out,"briefing.md")))
print("briefing non-empty:", bool(briefing))
print("\n=== PROVISIONAL BRIEFING ===")
print(briefing)
assert briefing, "ceiling produced NO briefing — fix failed!"
assert "PROVISIONAL" in briefing, "should be flagged provisional"
assert "abroad faster" in briefing or "analysis of the speeds" in briefing, "synthesizer reasoning not salvaged"
assert "intensity" in briefing.lower(), "open questions should list unexamined axis"
assert os.path.exists(os.path.join(out,"briefing.md"))
print("\nPROVISIONAL-BRIEFING FIX VERIFIED: ceiling + gating synth still yields a usable briefing")
