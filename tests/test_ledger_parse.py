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

# Ledger render/parse alignment (Layer 1) + tolerant parser (Layer 2), uniform for all models.
from nav_state import NavState, Entry

def sig(entries): return [(e.label, e.status, e.steps, e.why) for e in entries]

# ---------- Layer 1: render -> parse round-trips exactly (what the model is shown, it can echo) ----------
nav = NavState()
nav.frontier  = [Entry("frontier","hr_matched_ratio","in_progress",[1,3]), Entry("frontier","mixed_model","untested",[])]
nav.regimes   = [Entry("regime","effort_hr_zone","examined",[3,4,5]), Entry("regime","altitude_band","partial",[2]),
                 Entry("regime","gender","not_examined",[])]
nav.risks     = [Entry("risk","effort_confound","resolved",[6]), Entry("risk","sea_level_race","open",[])]
nav.breakdown = [Entry("breakdown","above_2600m","unrecoverable",[8],"single venue/surface, no variation")]

for with_log in (None, [{"step":1,"spec":"orient","stdout":"x"},{"step":2,"spec":"hr bins","stdout":"y"}]):
    rendered = nav.render_for_investigator(with_log)
    nv = NavState(); nv.apply_ledger_block(rendered)
    assert sig(nv.frontier)==sig(nav.frontier), ("frontier", sig(nv.frontier))
    assert sig(nv.regimes)==sig(nav.regimes),   ("regime",   sig(nv.regimes))
    assert sig(nv.risks)==sig(nav.risks),       ("risk",     sig(nv.risks))
    assert sig(nv.breakdown)==sig(nav.breakdown),("breakdown",sig(nv.breakdown))
print("Layer 1 round-trip (render -> parse identical, with & without evidence index): OK")

# ---------- Layer 2: all three observed model shapes parse to the same result ----------
canonical = ("FRONTIER:\n  f1 [tested] steps:7\nREGIME:\n  effort [examined] steps:3,4\n"
             "RISK:\n  conf [resolved] steps:6\nBREAKDOWN:\n  hi_alt [thin] steps:8 — why: one venue")
header_pipe = ("FRONTIER\n  f1 | tested | steps: 7\nREGIME\n  effort | examined | steps: 3,4\n"
               "RISK\n  conf | resolved | steps: 6\nBREAKDOWN\n  hi_alt | thin | why: one venue | steps: 8")
legacy_pipe = ("FRONTIER | f1 | tested | steps: 7\nREGIME | effort | examined | steps: 3,4\n"
               "RISK | conf | resolved | steps: 6\nBREAKDOWN | hi_alt | thin | why: one venue | steps: 8")
results=[]
for name, blk in [("canonical_bracket",canonical),("header+pipe",header_pipe),("legacy_pipe",legacy_pipe)]:
    nv=NavState(); nv.apply_ledger_block(blk)
    r=(sig(nv.frontier),sig(nv.regimes),sig(nv.risks),sig(nv.breakdown))
    results.append(r)
    assert nv.regimes and nv.regimes[0].status=="examined" and nv.regimes[0].steps==[3,4], (name, sig(nv.regimes))
    assert nv.breakdown and nv.breakdown[0].why=="one venue", (name, sig(nv.breakdown))
    print(f"  {name}: parsed frontier/regime/risk/breakdown = {[len(x) for x in r]}")
assert results[0]==results[1]==results[2], "the three shapes should yield identical ledgers"
print("Layer 2 three shapes parse identically: OK")

# ---------- Tolerance: malformed lines skipped, valid ones kept; evidence index ignored; empty leaves intact ----------
nv=NavState()
nv.apply_ledger_block("REGIME:\n  good [examined] steps:3\n  garbage line no status\n  bad [nonsense_status] steps:1\n"
                      "EVIDENCE INDEX:\n  step 1: [ok] did a thing\n  step 2: [ok] did another")
assert [r.label for r in nv.regimes]==["good"], sig(nv.regimes)   # garbage + bad-status + evidence lines all skipped
print("tolerance (skip malformed/bad-status/evidence lines): OK")

prior = NavState(); prior.regimes=[Entry("regime","keep","examined",[1])]
prior.apply_ledger_block("total garbage with no parseable ledger at all\njust prose")
assert [r.label for r in prior.regimes]==["keep"], "a garbled block must leave the prior map intact"
print("garbled block leaves prior map intact: OK")

print("\nALL LEDGER PARSE/RENDER ASSERTIONS PASSED")
