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
def _text(m):
    parts=[]
    for b in m:
        c=b.get("content")
        if isinstance(c,str): parts.append(c)
        elif isinstance(c,list):
            for blk in c:
                if isinstance(blk,dict) and isinstance(blk.get("text"),str):
                    parts.append(blk["text"])
    return " ".join(parts)
from kernel import PersistentKernel
from nav_state import NavState
from investigation import run_investigation

df = pd.read_csv(_require_dataset(), low_memory=False)
def inv(t,s,sp,l,se=""): 
    blocks=f"###THINKING###\n{t}\n###STATUS###\n{s}\n###SPEC###\n{sp}\n###LEDGER###\n{l}\n"
    if se: blocks+=f"###QUERY###\n{se}\n"
    return blocks
def code(b): return "```python\n"+b+"\n```"
L="FRONTIER | f1 | in_progress | steps: -\nREGIME | intensity | examined | steps: 1"

# ---------- Scenario A: search ENABLED, budget 2 ----------
out=_tmpdir("search_out")
if os.path.exists(out): shutil.rmtree(out)
cap={"inv_msgs":[], "search_prompts":[], "synth_msg":None}
INV=[ inv("I should calibrate against published altitude effects before reading this.",
          "SEARCH","", L, se="plausible sea-level vs altitude pace correction elite runners"),
      inv("Now compute the core frame.","CONTINUE",
          "Create core = df[df['sport_name']=='running'].copy(); print len(core).", L),
      inv("done","SYNTHESIZE","none", L) ]
EXEC={1:code("core=df[df['sport_name']=='running'].copy()\nprint('###RESULTS_START###');print('n',len(core));print('###RESULTS_END###')")}
class MockA:
    def __init__(s): s.i=0; s.e=0
    def call(s,m,model,max_tokens=10000,temperature=0,agent=None):
        txt=_text(m)
        if agent=="Investigator":
            cap["inv_msgs"].append(txt); r=INV[s.i]; s.i+=1; return r
        if agent=="Executor":
            if "raised an error" not in m[-1]["content"]: s.e+=1
            return EXEC[s.e]
        if agent=="Synthesizer":
            cap["synth_msg"]=txt
            return "###VERDICT###\nFINAL\n###BRIEFING###\n## Summary\ncalibrated briefing\n"
        return ""
    def search_call(s, messages, model, max_tokens=8000, temperature=0, agent=None, max_uses=5):
        cap["search_prompts"].append(messages[-1]["content"])
        return "[PUBLISHED] Altitude pace penalty ~5-8% at threshold (Jones 2019)."

k=PersistentKernel(df=df)
log,_,nav,brief=run_investigation(seed="altitude correction?",df=df,client=MockA(),
    investigator_model="anthropic:opus",executor_model="m:c",synth_model="anthropic:opus",
    schema_text="(s)",max_steps=4,output_dir=out,kernel=k,nav=NavState(),
    search_model="anthropic:claude-x",search_budget=2)
k.cleanup()

search_entries=[e for e in log if e.get("kind")=="search"]
assert len(cap["search_prompts"])==1, f"expected 1 search_call, got {len(cap['search_prompts'])}"
assert len(search_entries)==1, "search evidence entry missing"
assert search_entries[0]["query"].startswith("plausible sea-level"), "query not stored"
assert "Jones 2019" in search_entries[0]["result"], "synthesized result not stored"
assert os.path.exists(os.path.join(out,"exploration","01","search.md")), "search.md not written"
# search instruction present in the Investigator prompt (enabled)
assert "EXTERNAL SEARCH IS AVAILABLE" in cap["inv_msgs"][0], "search instruction missing when enabled"
# the search call prompt carried the brief (the Investigator's thinking) and the calibration framing
assert "calibrate against published" in cap["search_prompts"][0], "brief (thinking) not used as context"
assert "not to answer the investigation" in cap["search_prompts"][0].lower(), "calibration discipline missing"
# search rendered FULL to the next Investigator turn and to the Synthesizer
assert "WEB SEARCH" in cap["inv_msgs"][1] and "Jones 2019" in cap["inv_msgs"][1], "search not shown full to Investigator"
assert "WEB SEARCH" in (cap["synth_msg"] or "") and "Jones 2019" in (cap["synth_msg"] or ""), "search not in synthesis evidence"
print("Scenario A (enabled, used, rendered): OK")

# ---------- Scenario B: budget exhausted refuses a 2nd search ----------
out2=_tmpdir("search_out2")
if os.path.exists(out2): shutil.rmtree(out2)
capB={"searches":0}
INVB=[ inv("calibrate","SEARCH","",L,se="first query"),
       inv("calibrate again","SEARCH","",L,se="second query"),
       inv("done","SYNTHESIZE","none",L) ]
class MockB:
    def __init__(s): s.i=0
    def call(s,m,model,max_tokens=10000,temperature=0,agent=None):
        if agent=="Investigator": r=INVB[s.i]; s.i+=1; return r
        if agent=="Synthesizer": return "###VERDICT###\nFINAL\n###BRIEFING###\n## Summary\nb\n"
        return ""
    def search_call(s,messages,model,max_tokens=8000,temperature=0,agent=None,max_uses=5):
        capB["searches"]+=1; return "[PUBLISHED] x."
k=PersistentKernel(df=df)
logB,_,_,_=run_investigation(seed="q",df=df,client=MockB(),
    investigator_model="anthropic:opus",executor_model="m:c",synth_model="anthropic:opus",
    schema_text="(s)",max_steps=5,output_dir=out2,kernel=k,nav=NavState(),
    search_model="anthropic:claude-x",search_budget=1)
k.cleanup()
assert capB["searches"]==1, f"budget cap breached: {capB['searches']} searches (cap was 1)"
assert sum(1 for e in logB if e.get('kind')=='search')==1, "should have exactly one search entry"
print("Scenario B (budget cap enforced): OK")

# ---------- Scenario C: search DISABLED -> no instruction, SEARCH refused ----------
out3=_tmpdir("search_out3")
if os.path.exists(out3): shutil.rmtree(out3)
capC={"inv0":None,"searches":0}
INVC=[ inv("try search","SEARCH","",L,se="should be refused"),
       inv("done","SYNTHESIZE","none",L) ]
class MockC:
    def __init__(s): s.i=0
    def call(s,m,model,max_tokens=10000,temperature=0,agent=None):
        txt=_text(m)
        if agent=="Investigator":
            if capC["inv0"] is None: capC["inv0"]=txt
            r=INVC[s.i]; s.i+=1; return r
        if agent=="Synthesizer": return "###VERDICT###\nFINAL\n###BRIEFING###\n## Summary\nc\n"
        return ""
    def search_call(s,*a,**k): capC["searches"]+=1; return "x"
k=PersistentKernel(df=df)
logC,_,_,_=run_investigation(seed="q",df=df,client=MockC(),
    investigator_model="m:p",executor_model="m:c",
    schema_text="(s)",max_steps=4,output_dir=out3,kernel=k,nav=NavState(),
    search_model=None)
k.cleanup()
assert "EXTERNAL SEARCH IS AVAILABLE" not in capC["inv0"], "instruction leaked when search disabled"
assert capC["searches"]==0, "search fired while disabled"
assert not any(e.get('kind')=='search' for e in logC), "search entry created while disabled"
print("Scenario C (disabled: no instruction, no search): OK")

print("\nALL SEARCH ASSERTIONS PASSED")
