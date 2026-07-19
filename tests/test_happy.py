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
out=_tmpdir("happy_out")
if os.path.exists(out): shutil.rmtree(out)
def inv(t,s,sp,l): return f"###THINKING###\n{t}\n###STATUS###\n{s}\n###SPEC###\n{sp}\n###LEDGER###\n{l}\n"
def code(b): return "```python\n"+b+"\n```"
L_OK="FRONTIER | f1 | tested | steps: 1\nREGIME | intensity | examined | steps: 1\nBREAKDOWN | high effort | holds | why: faster | steps: 1"
INV=[inv("examine within intensity","CONTINUE","median speed by location_status; print.",L_OK),
     inv("intensity examined, done","SYNTHESIZE","none",L_OK)]
EXEC={1:code("print('###RESULTS_START###');print(df.groupby('location_status')['lap_avg_speed_ms'].median().to_string());print('###RESULTS_END###')")}
class Mock:
    def __init__(s):s.i=0;s.e=0
    def call(s,m,model,max_tokens=10000,temperature=0,agent=None):
        if agent=="Investigator":r=INV[s.i];s.i+=1;return r
        if agent=="Executor":
            if "raised an error" not in m[-1]["content"]:s.e+=1
            return EXEC[s.e]
        if agent=="Synthesizer":return "###VERDICT###\nFINAL\n###FINDINGS###\nF1 | decisive\nCLAIM: The effect is intensity-dependent; it holds at high effort.\nNUMBERS: ratio 1.0622 (95% CI 0.9994-1.1083), n=43 [step 1]\nCAVEATS: none\n"
        if agent=="Editor":return "## Summary\n\nThe effect is intensity-dependent; it holds at high effort [F1].\n"
        return ""
k=PersistentKernel(df=df)
log,_,nav,briefing=run_investigation(seed="t",df=df,client=Mock(),
    investigator_model="m:p",executor_model="m:c",schema_text="(s)",max_steps=5,output_dir=out,kernel=k,nav=NavState())
k.cleanup()
term=[e for e in log if e.get("terminal")]
print("terminal verdict:", term[-1].get("synth_verdict") if term else None)
print("briefing:", repr(briefing.strip()))
assert briefing and "intensity-dependent" in briefing
assert term and term[-1]["synth_verdict"]=="FINAL"
assert os.path.exists(os.path.join(out,"briefing.md"))
print("HAPPY-PATH (natural SYNTHESIZE -> FINAL) STILL WORKS")
