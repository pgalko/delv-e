# --- test bootstrap: runnable from the repo root via `python3 tests/<name>.py` ---
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:0] = [os.path.join(_HERE, "stubs"), _ROOT]
# --- end bootstrap ---

# The append-only context layout (audit 5.2) and its two budgets.
#
# The cached PREFIX holds one PERMANENT block per step that has exited the
# recent window — a pure function of the frozen log entry, independent of
# protection and pins, so it never rewrites (the property that makes the
# prompt cache read back every turn; the old rendering survived 0-35% of
# turns under protection churn). The volatile WORKING SET carries the full
# raws of recent / protected / pinned steps and absorbs ALL budget dynamics,
# where rewriting is cache-free. The one remaining prefix rewrite is the
# over-budget archive fold — deliberate and rare.

import random
from investigation import (_render_context, _permanent_block, _result_excerpt,
                           _step_block, HISTORY_CHAR_BUDGET,
                           PROTECTED_SLIM_CEILING, WORKING_SET_CHAR_BUDGET)

# Sized from the first live run on this layout (working set 21-32k chars,
# zero trims at 60k): the budget must sit where the observed excess trims.
assert WORKING_SET_CHAR_BUDGET == 30_000

SPEC = ("For each of the {n} groups in the qualifying table compute the median "
        "gap to the session best, printing counts per group and the medians "
        "sorted ascending. Then join the qualifying table and report per-driver contrasts.")


def mklog(n, raw=4000):
    return [{"step": i, "spec": SPEC.format(n=i), "stdout": "T" * raw,
             "thinking": f"note{i}. more detail here."} for i in range(1, n + 1)]


# ── 1) THE property: the prefix is append-only under arbitrary churn ──
log = mklog(14)
rng = random.Random(0)
prev = None
for k in range(1, len(log) + 1):
    protected = set(rng.sample(range(1, k + 1), min(k, rng.randint(0, 4))))
    pinned = set(rng.sample(range(1, k + 1), min(k, rng.randint(0, 2))))
    prefix, _ = _render_context(log[:k], protected=protected, pinned=pinned)
    if prev is not None:
        assert prefix[:len(prev)] == prev, \
            f"prefix rewrote at k={k}: append-only violated"
    prev = prefix
assert len(prev) == len(log) - 3, "one permanent block per step beyond the recent window"
print("1 (prefix append-only under random protection/pin churn, byte-exact): OK")

# ── 2) The prefix ignores protection and pins entirely ──
p1, _ = _render_context(log, protected=set(), pinned=set())
p2, _ = _render_context(log, protected={1, 2, 5}, pinned={3, 7})
assert p1 == p2, "protection/pins must never reshape the cached prefix"
print("2 (prefix independent of protection and pins): OK")

# ── 3) Working set: recents always; protected labeled; pinned labeled ──
prefix, work = _render_context(log, protected={2}, pinned={5})
joined = "\n\n".join(work)
assert "--- STEP 12 ---" in joined and "--- STEP 14 ---" in joined, "recents full"
assert "[LIVE-THREAD step 2" in joined and "--- STEP 2 ---" in joined
assert "[REHYDRATED step 5" in joined
assert "--- STEP 2 (archived) ---" in "\n".join(prefix), \
    "a protected step's permanent block still sits in the prefix (append-only)"
prefix2, work2 = _render_context(log, protected=set(), pinned=set())
assert len(work2) == 3 and all("[" not in b.split("\n")[0] for b in work2), \
    "plain recents carry no residency label"
print("3 (working-set membership and labels): OK")

# ── 4) Permanent blocks: verbatim excerpt; short raws kept WHOLE ──
short = {"step": 6, "spec": "Count rows per era; print counts.",
         "stdout": "era_a 512\nera_b 480\nera_c 233", "thinking": "Counts are balanced. Next stratify."}
blk = _permanent_block(short)
assert "--- STEP 6 (archived) ---" in blk and "REHYDRATE 6" in blk
assert "era_a 512" in blk and "era_c 233" in blk, \
    "a print-budget-disciplined step loses NOTHING when archived"
assert "NOTE AT THE TIME: Counts are balanced." in blk
long_raw = "N" * 9000
lb = _permanent_block({"step": 7, "spec": "s", "stdout": long_raw, "thinking": ""})
assert "chars omitted; full raw on disk" in lb and len(lb) < 1100
assert _result_excerpt("x" * 100) == "x" * 100
# keep-whole is decoupled from head+tail (slimming the excerpt must never
# truncate a raw that was previously archived whole): 680 stays the line.
assert _result_excerpt("y" * 680) == "y" * 680
assert "chars omitted" in _result_excerpt("y" * 681)
print("4 (verbatim excerpt card; short raws whole; long raws bounded): OK")

# ── 5) Over the prefix budget: oldest fold into ONE archive block ──
log2 = mklog(14)
perm_total = sum(len(_permanent_block(e)) for e in log2[:-3])
budget = perm_total - 600            # force a fold of at least one block
prefix, _ = _render_context(log2, prefix_budget=budget)
arch = [b for b in prefix if b.startswith("--- ARCHIVED STEPS")]
assert len(arch) == 1, "all archived steps share a single block"
assert sum(map(len, prefix)) <= budget
lines = arch[0].splitlines()
assert "###REHYDRATE###" in lines[0]
got = [ln.split(":")[0] for ln in lines[1:]]
assert got == sorted(got, key=lambda s: int(s.split()[1])), "archive lines chronological"
assert "--- STEP 1 (archived) ---" not in "\n".join(prefix), "folded steps leave the block list"
# below the budget the rendering is byte-identical to the unbudgeted one
pb, _ = _render_context(log2, prefix_budget=HISTORY_CHAR_BUDGET)
pu, _ = _render_context(log2)
assert pb == pu
print("5 (prefix fold: single chronological archive block; inert below budget): OK")

# ── 6) Working-set budget: protected trim oldest-first; recents/pins never ──
log3 = mklog(12, raw=9000)
protected = {2, 5}
_, work = _render_context(log3, protected=protected, pinned={7},
                          working_budget=30_000)
joined = "\n\n".join(work)
assert "trimmed here under the working-set budget" in joined, "protected residents trim"
i2 = joined.index("[LIVE-THREAD step 2")
i5 = joined.index("[LIVE-THREAD step 5")
seg2 = joined[i2:i5]
assert "trimmed here" in seg2, "oldest protected trims first"
for sid in (7, 10, 11, 12):        # the pin and the recents keep full raw
    start = joined.index(f"--- STEP {sid} ---")
    seg = joined[start:start + 12000]
    assert "trimmed here" not in seg.split("--- STEP")[0], f"step {sid} must not trim"
_, wu = _render_context(log3, protected=protected, pinned={7})
assert "trimmed here" not in "\n\n".join(wu), "no trim without a budget"
print("6 (working-set trim: protected oldest-first; recents and pins untouchable): OK")

print("\ntest_compaction_budget (append-only layout): OK")
