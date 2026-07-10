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

import sys, os, tempfile, shutil
import pandas as pd
from kernel import PersistentKernel
from nav_state import NavState
from investigation import run_investigation

df=pd.DataFrame({"a":[1,2,3,4],"location_status":["abroad","ethiopia","abroad","ethiopia"],"lap_avg_speed_ms":[5,4,5.2,4.1]})
out=tempfile.mkdtemp()
def inv(t,s,sp,l): return f"###THINKING###\n{t}\n###STATUS###\n{s}\n###SPEC###\n{sp}\n###LEDGER###\n{l}\n"
def code(b): return "```python\n"+b+"\n```"
L="FRONTIER | f1 | in_progress | steps: 1\nREGIME | intensity | examined | steps: 1"
INV=[inv("I want to compare speed by location to start orienting.","CONTINUE","median lap_avg_speed_ms by location_status; print.",L),
     inv("Integrated; intensity examined; done.","SYNTHESIZE","none",L)]
EXEC={1:code("print('###RESULTS_START###');print(df.groupby('location_status')['lap_avg_speed_ms'].median().to_string());print('###RESULTS_END###')")}
class Mock:
    def __init__(s):s.i=0;s.e=0
    def call(s,m,model,max_tokens=10000,temperature=0,agent=None):
        if agent=="Investigator":r=INV[s.i];s.i+=1;return r
        if agent=="Executor":
            if "cut off" not in m[-1]["content"] and "raised an error" not in m[-1]["content"]:s.e+=1
            return EXEC[s.e]
        if agent=="Synthesizer":return "###VERDICT###\nFINAL\n###BRIEFING###\n## Summary\nok\n"
        return ""
k=PersistentKernel(df=df)
run_investigation(seed="t",df=df,client=Mock(),investigator_model="m:p",executor_model="m:c",
                  schema_text="(s)",max_steps=5,output_dir=out,kernel=k,nav=NavState())
k.cleanup()

ap=os.path.join(out,"exploration","01","analysis.md")
print("analysis.md exists:", os.path.exists(ap))
txt=open(ap).read()
print("---- analysis.md ----")
print(txt)
assert os.path.exists(ap)
for must in ["# Step 1","Iteration | 1 of 5","## Spec","## Rationale","orienting","## Code","groupby","## Output","ethiopia"]:
    assert must in txt, f"missing: {must}"
print("ALL CHECKS PASS: analysis.md contains move, rationale, code, and output")
shutil.rmtree(out, ignore_errors=True)
