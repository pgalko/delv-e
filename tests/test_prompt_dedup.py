# --- test bootstrap: runnable from the repo root via `python3 tests/<name>.py` ---
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:0] = [os.path.join(_HERE, "stubs"), _ROOT]  # bundled httpx stub + delv-e modules
# --- end bootstrap ---

# Audit 4.2, the de-duplication set (six cuts, one change), plus the honest
# collapse wording from 5.3. What is pinned here:
#   1. ESTIMAND rules ride ONLY the unpinned-estimand turn (tail), never the
#      permanent system prompt; the HEAD stays byte-stable across turns.
#   2. The nav render is bare headers — legends are taught once, in the cached
#      system prompt; the EVIDENCE INDEX is Synthesizer payload only.
#   3. Collapse notices are one honest line: the raw is on disk, NOT "recorded
#      in the NAV MAP" (the map holds no findings).
#   4. The ablation-validated clauses survive the cuts verbatim where measured.
#   5. Size ceilings so the prompt cannot silently re-bloat.

import prompts as P
import investigation as I
import nav_state as N
from nav_state import Entry

# ====================================================================
# 1. ESTIMAND relocation
# ====================================================================
for sysp in (P.INVESTIGATOR_SYSTEM, P.COMPUTE_INVESTIGATOR_SYSTEM):
    assert "###ESTIMAND###" in sysp, "the marker must still be taught in the emission list"
    assert "do not narrow" not in sysp, "the long estimand rules must not ride every turn"
assert "do not narrow, restructure, or pre-commit" in P.ESTIMAND_NOTE_DATA
assert "do not narrow or restructure" in P.ESTIMAND_NOTE_COMPUTE
assert "{estimand_note}" in P.INVESTIGATOR_TASK_FIRST
assert P.DATA_MODE.estimand_note == P.ESTIMAND_NOTE_DATA
assert P.COMPUTE_MODE.estimand_note == P.ESTIMAND_NOTE_COMPUTE
print("1a (estimand rules moved out of the permanent system prompt): OK")

class CaptureClient:
    def __init__(s): s.captured = []
    def call(s, m, model, max_tokens=10000, temperature=0, agent=None,
             return_meta=False, reasoning_effort=None):
        s.captured.append(m)
        r = ("###THINKING###\nt\n###STATUS###\nCONTINUE\n###SPEC###\ncount rows; print it.\n"
             "###LEDGER###\nFRONTIER:\n  f1 [untested] steps: -\n")
        return (r, {"output_tokens": 5, "max_tokens": max_tokens,
                    "truncated": False}) if return_meta else r

cc = CaptureClient()
inv = I.Investigator(cc, "m:p")
nav = N.NavState()
LOG1 = [{"step": 1, "spec": "count rows; print it.", "stdout": "26234", "thinking": "t", "attempts": 1}]

inv.decide("seed?", "(schema)", "(registry)", [], nav)              # turn 1: unpinned
nav.target_estimand = "the pinned question"
inv.decide("seed?", "(schema)", "(registry)", LOG1, nav)            # turn 2: pinned
def _texts(msgs):
    return [p.get("content", "") if isinstance(p, dict) else str(p) for p in msgs]
t1, t2 = ["\n".join(_texts(m)) for m in cc.captured]
assert "ESTIMAND INSTRUCTIONS" in t1 and "do not narrow" in t1
assert "ESTIMAND INSTRUCTIONS" not in t2 and "do not narrow" not in t2, \
    "the instructions must not ride turns after the estimand pins"
assert P.INVESTIGATOR_TASK_LATER in t2
# HEAD byte-stability: the cached block 0 must be byte-identical across turns
# (asserted structure-independently: the exact head string appears verbatim in
# both turns; only blocks append after it and only the tail varies).
expected_head = P.INVESTIGATOR_HEAD_TEMPLATE.format(seed="seed?", schema="(schema)")
assert expected_head in t1 and expected_head in t2, \
    "per-turn variation must live in the volatile tail, never the cached head"
# Self-healing: if turn 1 failed to pin an estimand, the instructions persist.
cc2 = CaptureClient()
inv2 = I.Investigator(cc2, "m:p")
inv2.decide("seed?", "(schema)", "(registry)", LOG1, N.NavState())  # later turn, still unpinned
assert "ESTIMAND INSTRUCTIONS" in "\n".join(_texts(cc2.captured[0]))
print("1b (instructions ride only unpinned turns; head byte-stable; self-healing): OK")

# ====================================================================
# 2. Bare-legend render; evidence index is Synthesizer payload
# ====================================================================
nav = N.NavState()
nav.target_estimand = "whether v is higher for x"
nav.frontier = [Entry("f1", "untested", [])]
nav.regimes = [Entry("era", "not_examined", [])]
nav.risks = [Entry("r1", "open", [1])]
nav.breakdown = [Entry("b1", "holds", [1], why="thin data")]
LOG = [{"step": 1, "spec": "count rows per group; print counts.", "stdout": "x 2\ny 2"}]

r = nav.render_for_investigator()
for header in ("TARGET ESTIMAND (pinned", "FRONTIER:", "REGIME LEDGER:",
               "OPEN RISKS:", "BREAKDOWN MAP:"):
    assert header in r, f"header keyword must survive (parser echo contract): {header}"
for legend in ("pursue untested ones", "estimated WITHIN each level",
               "at least one axis must be examined", "holds / thin / blocked"):
    assert legend not in r, f"legend re-teaching must not ride the render: {legend!r}"
assert "estimated WITHIN each level" not in r and \
       "ESTIMATE THE EFFECT WITHIN EACH LEVEL" in P.INVESTIGATOR_SYSTEM, \
    "the regime/G1 teaching lives in the cached system prompt, once"
assert "EVIDENCE INDEX" not in r, "no index without a log"
assert "EVIDENCE INDEX" in nav.render_for_investigator(LOG), "Synthesizer view keeps the index"
# The investigator's assembled turn carries no index and no legend text:
assert "EVIDENCE INDEX" not in t2 and "pursue untested ones" not in t2
# The echo round-trip still parses: feed the bare-header shape back through the merge.
nav2 = N.NavState()
nav2.apply_ledger_block("FRONTIER:\n  f1 [untested] steps: -\n"
                        "REGIME LEDGER:\n  era [not_examined] steps: -\n"
                        "OPEN RISKS:\n  r1 [open] steps: 1\n"
                        "BREAKDOWN MAP:\n  b1 [holds] steps: 1 — why: thin data\n")
assert [e.label for e in nav2.frontier] == ["f1"] and [e.label for e in nav2.regimes] == ["era"]
assert [e.label for e in nav2.risks] == ["r1"] and [e.label for e in nav2.breakdown] == ["b1"]
print("2 (bare headers, teaching single-sourced, index synth-only, echo parses): OK")

# ====================================================================
# 3. Honest one-line collapse notices
# ====================================================================
e = {"step": 6, "spec": "Compute grouped medians of v by g. Print them sorted.",
     "stdout": "x 5.1\ny 4.05", "thinking": "x leads on medians. Next stratify by era.", "attempts": 1}
blk = I._permanent_block(e)
assert "REHYDRATE 6" in blk and "full raw on disk" in blk
assert "NAV MAP" not in blk, \
    "the archived form must not claim the map records findings — it records status pointers"
assert "x 5.1" in blk and "y 4.05" in blk, "the excerpt card keeps the numbers resident"
assert "NOTE AT THE TIME: x leads on medians." in blk
for sysp in (P.INVESTIGATOR_SYSTEM, P.COMPUTE_INVESTIGATOR_SYSTEM):
    assert "NOT resident" in sysp and "Their findings remain in this ledger" not in sysp, \
        "the system prompt's REHYDRATE section must tell the truth about residency"
print("3 (collapse notices honest, one line, system prompt corrected): OK")

# ====================================================================
# 4. Ablation-validated clauses survive the cuts
# ====================================================================
s = P.INVESTIGATOR_SYSTEM
for clause in (
    "METHOD ADEQUACY",                                     # Opus 76→82
    "cluster_bootstrap(df, cluster_col, stat_fn",          # toolkit (glm 73→82)
    "paired_ability(df, a_col, b_col",
    "rank_uncertainty(estimates=",
    "Their outputs use EXACT column names",
    "ONE MOVE PER SPEC",                                   # glm multi-part 5/7→0/14
    "standardize the metric within each group; regress",   # the validated example, verbatim
    "Right, three specs: (1) standardize within group",
    "Banned in a spec: appropriate, best, robust, handle, clean, reasonable, meaningful, optimal, sensible",
    "PRINT BUDGET",
    "G1 — SHAPE BEFORE NULL",
    "G2 — VARIABLE ROLES ARE FLUID",
):
    assert clause in s, f"validated clause lost: {clause!r}"
assert s.count("junior coder") == 1, "the merge must state the framing once"
assert s.count("never prior steps, prior specs, or prior code") == 1
# Print-budget sharpening (first live run on the append-only layout: a
# full-table print rode the working set 6 turns): the residency multiplier
# and the size anchor must survive future edits.
assert "rides your context for MULTIPLE turns" in s
assert "under ~2,000 characters" in s
print("4 (validated clauses intact; framing single-stated): OK")

# ====================================================================
# 5. Re-bloat ceilings (measured 2026-07: 12,789 / 11,546 / 656 chars)
# ====================================================================
assert len(P.INVESTIGATOR_SYSTEM) < 13_200, len(P.INVESTIGATOR_SYSTEM)
assert len(P.COMPUTE_INVESTIGATOR_SYSTEM) < 12_000, len(P.COMPUTE_INVESTIGATOR_SYSTEM)
assert len(r) < 900, f"bare-legend render regressed: {len(r)}"
fixed_tail = P.INVESTIGATOR_TAIL_TEMPLATE.format(registry="", nav="",
                                                 task=P.INVESTIGATOR_TASK_LATER)
assert len(fixed_tail) < 500, f"fixed tail text regressed: {len(fixed_tail)}"
print("5 (size ceilings pinned): OK")

print("\nALL PROMPT DE-DUP CONTRACTS PASSED")
