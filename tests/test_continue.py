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

import json, os, shutil, pandas as pd
from kernel import PersistentKernel
from nav_state import NavState
from investigation import run_investigation
from dataio import build_schema

df = pd.read_csv(_require_dataset(), low_memory=False)
print("schema sample:\n" + "\n".join(build_schema(df).splitlines()[:4]))

out=_tmpdir("cont_out")
if os.path.exists(out): shutil.rmtree(out)

def inv(t,s,sp,l): return f"###THINKING###\n{t}\n###STATUS###\n{s}\n###SPEC###\n{sp}\n###LEDGER###\n{l}\n"
def code(b): return "```python\n"+b+"\n```"
L="FRONTIER | f1 | in_progress | steps: -\nREGIME | intensity | not_examined | steps: -"

# Phase 1: 2 CONTINUE steps then hit max_steps=2 -> final synthesis (no terminal entry)
INV1=[inv("build core","CONTINUE","Create core = df[df['sport_name']=='running'].copy(); print len(core).",L),
      inv("add a col","CONTINUE","On core add col flag1 = core['lap_avg_speed_ms']>4; print int(core['flag1'].sum()).",L)]
EXEC1={1:code("core=df[df['sport_name']=='running'].copy()\nprint('###RESULTS_START###');print('n',len(core));print('###RESULTS_END###')"),
       2:code("core['flag1']=core['lap_avg_speed_ms']>4\nprint('###RESULTS_START###');print('flag1',int(core['flag1'].sum()));print('###RESULTS_END###')")}
SYNTH=("###VERDICT###\nFINAL\n###BRIEFING###\n## Summary\nphase1 interim briefing\n")
class Mock1:
    def __init__(s):s.i=0;s.e=0
    def call(s,m,model,max_tokens=10000,temperature=0,agent=None):
        if agent=="Investigator":r=INV1[s.i];s.i+=1;return r
        if agent=="Executor":
            if "raised an error" not in m[-1]["content"]:s.e+=1
            return EXEC1[s.e]
        if agent=="Synthesizer":return SYNTH
        return ""

k1=PersistentKernel(df=df)
log1,_,nav1,brief1=run_investigation(seed="seedQ",df=df,client=Mock1(),
    investigator_model="m:p",executor_model="m:c",schema_text="(s)",
    max_steps=2,output_dir=out,kernel=k1,nav=NavState())
# persist seed manually (run_core does this; here we just check loop persistence)
k1.cleanup()
print("\nphase1 steps:",[e['step'] for e in log1 if not e.get('terminal')])
print("phase1 persisted files:",sorted(os.listdir(out)))
hist1=json.load(open(os.path.join(out,"kernel_history.json")))
print("phase1 kernel_history len:",len(hist1))
assert os.path.exists(os.path.join(out,"log.json"))
assert os.path.exists(os.path.join(out,"nav_state.json"))
assert len(hist1)==2

# Phase 2: RESUME from disk into a FRESH kernel
nav2=NavState.from_dict(json.load(open(os.path.join(out,"nav_state.json"))))
prior_log=json.load(open(os.path.join(out,"log.json")))
history=json.load(open(os.path.join(out,"kernel_history.json")))
k2=PersistentKernel(df=df)
k2.restore_history(history)
print("\nphase2 fresh kernel registry after restore:")
print("  "+k2.describe_namespace().replace("\n","\n  "))
assert "core" in k2.describe_namespace(), "restored 'core' missing!"
assert "x50" in k2.describe_namespace(), "core should have 50 cols incl flag1"

# Continue: 1 more step that USES restored state, then synthesize
INV2=[inv("use restored core+flag1","CONTINUE","On core compute core[core['flag1']]['lap_avg_speed_ms'].mean(); print it.",L),
      inv("intensity examined; done","SYNTHESIZE","none","FRONTIER | f1 | tested | steps: 3\nREGIME | intensity | examined | steps: 3")]
EXEC2={1:code("v=core[core['flag1']]['lap_avg_speed_ms'].mean()\nprint('###RESULTS_START###');print('mean_fast',round(float(v),3));print('###RESULTS_END###')")}
class Mock2:
    def __init__(s):s.i=0;s.e=0
    def call(s,m,model,max_tokens=10000,temperature=0,agent=None):
        if agent=="Investigator":r=INV2[s.i];s.i+=1;return r
        if agent=="Executor":
            if "raised an error" not in m[-1]["content"]:s.e+=1
            return EXEC2[s.e]
        if agent=="Synthesizer":return "###VERDICT###\nFINAL\n###BRIEFING###\n## Summary\nfinal after continue\n"
        return ""

log2,_,nav2,brief2=run_investigation(seed="seedQ",df=df,client=Mock2(),
    investigator_model="m:p",executor_model="m:c",schema_text="(s)",
    max_steps=3,output_dir=out,kernel=k2,nav=nav2,log=prior_log)
k2.cleanup()

steps2=[e['step'] for e in log2 if not e.get('terminal')]
print("\nphase2 all steps (continued numbering):",steps2)
print("phase2 briefing:",repr(brief2.strip()[:40]))
assert 3 in steps2, "continued step numbering broken (expected step 3)"
assert any(e['step']==3 and e.get('stdout') and 'mean_fast' in e['stdout'] for e in log2), "restored-state step didn't run"
assert brief2 and "final after continue" in brief2
print("\nALL --CONTINUE / STEP-5 ASSERTIONS PASSED")
