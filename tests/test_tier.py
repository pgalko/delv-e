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
from investigation import _render_context, _parse_investigator
from nav_state import NavState

# Build a 7-step log; each step has raw + thinking
log=[{"step":i,"spec":f"spec{i}","stdout":f"<<TABLE {i} with numbers>>","thinking":f"note{i}"} for i in range(1,8)]

print("=== TEST 1: recents ride the working set; older steps archive to the prefix ===")
nav=NavState()
# nav protects nothing (all closed) -> only recent_full=3 in the working set
prefix, work = _render_context(log, recent_full=3, protected=set(), pinned=set())
assert len(prefix)==4 and all("(archived)" in b for b in prefix), "steps 1-4 archived"
assert all("RESULT EXCERPT" in b for b in prefix), "archived form carries the excerpt card"
assert len(work)==3 and all("RAW RESULT" in b for b in work), "steps 5-7 full in the working set"
assert "<<TABLE 1 with numbers>>" in prefix[0], "short raw kept WHOLE in the archived form"
print("  OK: 1-4 archived (with excerpts), 5-7 full")

print("\n=== TEST 2: protected (live-thread) step rides the working set even if old ===")
nav2=NavState()
nav2.apply_ledger_block("FRONTIER | f-open | in_progress | steps: 2\nFRONTIER | f-done | tested | steps: 1")
prot=nav2.protected_steps()
print("  protected steps (open frontier):", sorted(prot))
assert 2 in prot and 1 not in prot
prefix2, work2 = _render_context(log, recent_full=3, protected=prot, pinned=set())
joined2 = "\n\n".join(work2)
assert "[LIVE-THREAD step 2" in joined2 and "<<TABLE 2 with numbers>>" in joined2, \
    "step feeding open frontier keeps its full raw resident"
assert prefix2 == prefix, "protection must not reshape the cached prefix (append-only)"
print("  OK: open-thread step resident in the working set; prefix untouched")

print("\n=== TEST 3: rehydrate pins a step's full raw into the working set ===")
prefix3, work3 = _render_context(log, recent_full=3, protected=set(), pinned={2})
joined3 = "\n\n".join(work3)
assert "[REHYDRATED step 2" in joined3 and "<<TABLE 2 with numbers>>" in joined3
assert prefix3 == prefix, "pins must not reshape the cached prefix"
print("  OK: rehydrated step full in the working set")

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
a=_render_context(log[:6], recent_full=3)[0][0]   # 6-step run, step1 archived
b=_render_context(log[:7], recent_full=3)[0][0]   # 7-step run, step1 archived
print("  step1 permanent block identical across turns:", a==b)
assert a==b
print("  OK: permanent rendering deterministic -> cache prefix holds")

print("\nALL TIERING + REHYDRATE TESTS PASSED")
