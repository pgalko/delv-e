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

# Regression test for the context-growth / compaction fixes.
from kernel import PersistentKernel
import investigation as I
from nav_state import NavState, Entry

k = PersistentKernel.__new__(PersistentKernel)
k.registry = {"columns": ["a", "b"],
              "namespace": [{"name": f"obj{i}", "type": "DataFrame", "desc": f"DataFrame {i}x2"}
                            for i in range(130)]}

# #1 Investigator default view: NEWEST objects carry descriptions; every older
# object is listed by NAME only (names are the reuse contract and never drop —
# an improvement over the old 120-cap, which hid the oldest names entirely).
out = k.describe_namespace(max_items=120)
assert "- obj129:" in out and "- obj105:" in out, "newest 25 keep descriptions"
assert "- obj104:" not in out and "- obj10:" not in out, "older objects drop descriptions"
assert "Earlier derived objects, by NAME only" in out
assert "obj10" in out and "obj0" in out, "every name stays present"

# #2/#3 names subset shows only named objects + count of the rest
out_focus = k.describe_namespace(names={"obj5", "obj129"})
assert "- obj5:" in out_focus and "- obj129:" in out_focus and "- obj7:" not in out_focus
assert "+128 other derived objects exist" in out_focus

# #2 Executor focus = objects the spec references
k.registry["namespace"] = [{"name": n, "type": "X", "desc": "d"} for n in ["alt", "sea", "tmp_old", "ratio"]]
assert I._referenced_names("median of alt and sea; store as ratio.", k) == {"alt", "sea", "ratio"}

# #3 Investigator live = used-recently OR newest; dormant excluded
log = [{"step": 1, "code": "alt=1", "terminal": False}, {"step": 2, "code": "sea=1", "terminal": False},
       {"step": 3, "code": "x=1", "terminal": False}, {"step": 4, "code": "ratio=alt/sea", "terminal": False},
       {"step": 5, "code": "y=2", "terminal": False}]
live = I._live_names(k, log, window=2, newest=1)
assert {"ratio", "alt", "sea"} <= live and "tmp_old" not in live

# #5 protected = latest step per LIVE thread, closed excluded, capped
nav = NavState()
nav.frontier = [Entry("frontier", "f1", "in_progress", [1, 2, 3]), Entry("frontier", "f2", "tested", [4])]
nav.risks = [Entry("risk", "r1", "open", [2, 5]), Entry("risk", "r2", "resolved", [6])]
assert nav.protected_steps() == {3, 5}
nav.frontier = [Entry("frontier", f"f{i}", "in_progress", [i]) for i in range(1, 12)]
nav.risks = []
assert nav.protected_steps() == set(range(6, 12))

print("ALL CONTEXT-GROWTH / COMPACTION ASSERTIONS PASSED")
