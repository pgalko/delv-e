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
from prompts import DIRECTIVE_EXTEND_LEDGER

df = pd.read_csv(_require_dataset(), low_memory=False)
out=_tmpdir("ext_out")
if os.path.exists(out): shutil.rmtree(out)

def inv(t,s,sp,l): return f"###THINKING###\n{t}\n###STATUS###\n{s}\n###SPEC###\n{sp}\n###LEDGER###\n{l}\n"
def code(b): return "```python\n"+b+"\n```"
# Inherited ledger: a RISK flagged but never resolved (the reconcile scenario)
L=("FRONTIER | f1 | tested | steps: 1\n"
   "REGIME | intensity | examined | steps: 1\n"
   "RISK | race_confound | flagged | steps: -")

# ---- Phase 1: a FINISHED run (2 steps then FINAL) ----
INV1=[inv("core","CONTINUE","Create core = df[df['sport_name']=='running'].copy(); print len(core).",L),
      inv("done","SYNTHESIZE","none",L)]
EXEC1={1:code("core=df[df['sport_name']=='running'].copy()\nprint('###RESULTS_START###');print('n',len(core));print('###RESULTS_END###')")}
class Mock1:
    def __init__(s):s.i=0;s.e=0
    def call(s,m,model,max_tokens=10000,temperature=0,agent=None):
        if agent=="Investigator":r=INV1[s.i];s.i+=1;return r
        if agent=="Executor":
            if "raised an error" not in m[-1]["content"]:s.e+=1
            return EXEC1[s.e]
        if agent=="Synthesizer":return "###VERDICT###\nFINAL\n###BRIEFING###\n## Summary\noriginal: effect ~8% (race confound UNRESOLVED)\n"
        return ""
k1=PersistentKernel(df=df)
log1,_,nav1,brief1=run_investigation(seed="What is the altitude correction?",df=df,client=Mock1(),
    investigator_model="m:p",executor_model="m:c",schema_text="(s)",
    max_steps=4,output_dir=out,kernel=k1,nav=NavState())
k1.cleanup()
has_terminal = any(e.get("terminal") for e in log1)
print("phase1 finished. has terminal entry:", has_terminal, "| steps:", [e['step'] for e in log1 if not e.get('terminal')])

# ---- Phase 2: EXTEND with a new seed; rehydrate from disk ----
nav2=NavState.from_dict(json.load(open(os.path.join(out,"nav_state.json"))))
prior_log=json.load(open(os.path.join(out,"log.json")))
history=json.load(open(os.path.join(out,"kernel_history.json")))
assert any(e.get("terminal") for e in prior_log), "phase1 should have persisted a terminal entry"
k2=PersistentKernel(df=df); k2.restore_history(history)

captured={"inv_first_msg":None,"synth_msg":None}
INV2=[inv("pursue the flagged race confound","CONTINUE",
          "On core compute median lap_avg_pace_min_per_km; print it.",
          L.replace("flagged","examined")),
      inv("confound real; revise","SYNTHESIZE","none",L.replace("flagged","examined"))]
EXEC2={1:code("v=core['lap_avg_pace_min_per_km'].median()\nprint('###RESULTS_START###');print('median',round(float(v),3));print('###RESULTS_END###')")}
class Mock2:
    def __init__(s):s.i=0;s.e=0
    def call(s,m,model,max_tokens=10000,temperature=0,agent=None):
        if agent=="Investigator":
            if captured["inv_first_msg"] is None: captured["inv_first_msg"]=m[-1]["content"]
            r=INV2[s.i];s.i+=1;return r
        if agent=="Executor":
            if "raised an error" not in m[-1]["content"]:s.e+=1
            return EXEC2[s.e]
        if agent=="Synthesizer":
            captured["synth_msg"]=" ".join(b["content"] for b in m if isinstance(b.get("content"),str))
            return "###VERDICT###\nFINAL\n###BRIEFING###\n## Summary\nrevised: race confound real; correction now ~3% (was ~8%)\n"
        return ""

log2,_,nav2,brief2=run_investigation(seed="Is the ~8% inflated by race-day pacing?",df=df,client=Mock2(),
    investigator_model="m:p",executor_model="m:c",schema_text="(s)",
    max_steps=3,output_dir=out,kernel=k2,nav=nav2,log=prior_log,
    prior_seeds=["What is the altitude correction?"])
k2.cleanup()

steps2=[e['step'] for e in log2 if not e.get('terminal')]
n_terminal=sum(1 for e in log2 if e.get('terminal'))
print("extend steps (continued numbering):", steps2)
print("terminal entries in final log:", n_terminal)

# 1) terminal entry from phase 1 was dropped (not carried as a stale step), exactly one at end
assert steps2==[1,2], f"expected continued numbering [1,2] over the dropped-terminal log, got {steps2}"
# 2) step numbering continued past prior (prior had step 1; extend added step 2), no overwrite
assert 2 in steps2 and 1 in steps2
# 3) first Investigator turn received the inherited-ledger directive
assert "EXTENSION: you are continuing" in (captured["inv_first_msg"] or ""), "extend directive not delivered to Investigator turn 1"
# 4) Synthesizer received the combined-briefing notice and BOTH questions co-equally
assert "EXTENSION CONTEXT" in (captured["synth_msg"] or ""), "synth extension notice missing"
assert "What is the altitude correction?" in (captured["synth_msg"] or ""), "prior seed not in synth"
assert "Is the ~8% inflated by race-day pacing?" in (captured["synth_msg"] or ""), "extension seed not in synth"
assert "crowd out" in (captured["synth_msg"] or ""), "anti-crowding instruction missing"
# 5) combined briefing produced
assert brief2 and "revised" in brief2.lower()
print("\nALL --EXTEND ASSERTIONS PASSED")
