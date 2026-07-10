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

import sys
from investigation import _history_blocks, _parse_investigator
from nav_state import NavState

# Build a 7-step log; each step has raw + thinking
log=[{"step":i,"spec":f"spec{i}","stdout":f"<<TABLE {i} with numbers>>","thinking":f"note{i}"} for i in range(1,8)]

print("=== TEST 1: recent stay full, old closed-thread steps collapse ===")
nav=NavState()
# nav protects nothing (all closed) -> only recent_full=3 stay full
blocks=_history_blocks(log, recent_full=3, protected=set(), forced_full=set())
def is_full(b): return "RAW RESULT" in b or "RESULT:" in b and "collapsed" not in b
for i,b in enumerate(blocks,1):
    coll="(collapsed)" in b
    print(f"  step{i}: {'COLLAPSED' if coll else 'FULL'}")
# steps 5,6,7 full (recent 3); 1-4 collapsed
assert all("(collapsed)" in blocks[i] for i in range(0,4)), "old steps should collapse"
assert all("(collapsed)" not in blocks[i] for i in range(4,7)), "recent 3 should be full"
print("  OK: steps 1-4 collapsed, 5-7 full")

print("\n=== TEST 2: protected (live-thread) step stays full even if old ===")
nav2=NavState()
nav2.apply_ledger_block("FRONTIER | f-open | in_progress | steps: 2\nFRONTIER | f-done | tested | steps: 1")
prot=nav2.protected_steps()
print("  protected steps (open frontier):", sorted(prot))
assert 2 in prot and 1 not in prot
blocks2=_history_blocks(log, recent_full=3, protected=prot, forced_full=set())
print(f"  step2 (feeds OPEN frontier): {'FULL' if '(collapsed)' not in blocks2[1] else 'COLLAPSED'}")
assert "(collapsed)" not in blocks2[1], "step feeding open frontier must stay full"
print(f"  step1 (feeds tested frontier, old): {'COLLAPSED' if '(collapsed)' in blocks2[0] else 'FULL'}")
assert "(collapsed)" in blocks2[0]
print("  OK: open-thread step protected, closed-thread old step collapsed")

print("\n=== TEST 3: rehydrate forces a collapsed step full ===")
blocks3=_history_blocks(log, recent_full=3, protected=set(), forced_full={2})
print(f"  step2 with rehydrate: {'FULL' if '(collapsed)' not in blocks3[1] else 'COLLAPSED'}")
assert "(collapsed)" not in blocks3[1]
print("  OK: rehydrated step shown full")

print("\n=== TEST 4: parser extracts REHYDRATE block ===")
out="###THINKING###\nneed old numbers\n###STATUS###\nCONTINUE\n###SPEC###\ndo x\n###LEDGER###\nFRONTIER | f | in_progress | steps: 5\n###REHYDRATE###\n6, 9\n"
d=_parse_investigator(out)
print("  parsed rehydrate:", d["rehydrate"])
assert d["rehydrate"]==[6,9]
# no rehydrate block -> empty
d2=_parse_investigator("###THINKING###\nx\n###STATUS###\nCONTINUE\n###SPEC###\ny\n###LEDGER###\nz\n")
assert d2["rehydrate"]==[]
print("  OK: REHYDRATE parsed; absent -> empty")

print("\n=== TEST 5: collapsed blocks still byte-stable across turns (cache-safe) ===")
# step 1 collapsed at turn A and turn B must be identical bytes
a=_history_blocks(log[:6], recent_full=3, protected=set())[0]   # 6-step run, step1 collapsed
b=_history_blocks(log[:7], recent_full=3, protected=set())[0]   # 7-step run, step1 collapsed
print("  step1 collapsed block identical across turns:", a==b)
assert a==b
print("  OK: collapsed rendering deterministic -> cache prefix holds")

print("\nALL TIERING + REHYDRATE TESTS PASSED")
