# --- test bootstrap: runnable from the repo root via `python3 tests/<name>.py` ---
import os, sys, tempfile
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:0] = [os.path.join(_HERE, "stubs"), _ROOT]  # bundled httpx stub + delv-e modules
def _tmpdir(tag):
    return tempfile.mkdtemp(prefix=f"delve_{tag}_")
# --- end bootstrap ---

# Three hardening fixes, tested together because they share one failure theme
# (a formatting lapse or a partial parse silently steering the run):
#   1. _parse_investigator terminates blocks ONLY at named markers, so markdown
#      "### Subsection" headers inside THINKING/SPEC no longer amputate them
#      (the _parse_synth fix, ported), with the same trailing-hash tolerance.
#   2. NavState.apply_ledger_block merges PER SECTION: a turn whose FRONTIER/
#      REGIME lines are garbled but whose one RISK line parses no longer wipes
#      the frontier and regimes (which disarmed the G1 gate); drift is diffed.
#   3. Loop: a marker-less turn is retried with DIRECTIVE_FORMAT_RETRY instead
#      of finalizing; the search-spent notice is preemptive (a standing tail
#      line, no wasted premium turn); gate pushbacks refund their iteration.

import shutil
import pandas as pd

from investigation import _parse_investigator, run_investigation
from kernel import PersistentKernel
from nav_state import NavState, Entry
from llm import RunStats
from prompts import (DIRECTIVE_FORMAT_RETRY, DIRECTIVE_SEARCH_SPENT,
                     DIRECTIVE_TRUNCATED_RETRY)


# ====================================================================
# 1. Parser: named-marker terminators + trailing-hash tolerance
# ====================================================================

# ---------- 1a: markdown subsections inside THINKING and SPEC survive ----------
out = _parse_investigator(
    "###THINKING###\nThe result shows X.\n"
    "### Detailed interpretation\nThis subsection continues the thinking.\n"
    "###STATUS###\nCONTINUE\n"
    "###SPEC###\nGroup df by season.\n"
    "### Output requirements\nPrint top 10 rows sorted by p_rank1 descending.\n"
    "###LEDGER###\nFRONTIER:\n  teammate_pairing [in_progress] steps:1\n")
assert "subsection continues" in out["thinking"], out["thinking"]
assert "p_rank1" in out["spec"], "### subsection amputated the spec"
assert out["status"] == "CONTINUE"
assert "teammate_pairing" in out["ledger"]
assert out["unparsed"] is False
print("1a (### subsections inside THINKING/SPEC survive): OK")

# ---------- 1b: named markers still terminate blocks ----------
assert "Group df" not in out["thinking"], "THINKING must stop at ###STATUS###"
assert "teammate_pairing" not in out["spec"], "SPEC must stop at ###LEDGER###"
print("1b (named markers still terminate): OK")

# ---------- 1c: trailing-hash malformations tolerated (the live glm shape) ----------
out = _parse_investigator(
    "###THINKING##\nshort note\n###STATUS\nCONTINUE\n###SPEC##\ncount rows by g; print.\n"
    "###LEDGER\nREGIME:\n  g [not_examined] steps:-\n")
assert out["thinking"] == "short note", out["thinking"]
assert out["spec"] == "count rows by g; print.", out["spec"]
assert out["status"] == "CONTINUE"
assert "not_examined" in out["ledger"]
print("1c (malformed trailing hashes tolerated): OK")

# ---------- 1d: REHYDRATE + QUERY + ESTIMAND regressions ----------
out = _parse_investigator(
    "###ESTIMAND###\nThe pinned target.\n###THINKING###\nx\n###STATUS###\nCONTINUE\n"
    "###SPEC###\ny\n###LEDGER###\nz\n###REHYDRATE###\n6, 9\n")
assert out["rehydrate"] == [6, 9] and out["estimand"] == "The pinned target."
out = _parse_investigator(
    "###THINKING###\nneed calibration\n###STATUS###\nSEARCH\n###SPEC###\nnone\n"
    "###QUERY###\naltitude adaptation effect sizes\n###LEDGER###\nz\n")
assert out["search"] == "altitude adaptation effect sizes" and out["status"] == "SEARCH"
print("1d (REHYDRATE/QUERY/ESTIMAND regressions): OK")

# ---------- 1e: marker-less fallback keeps SYNTHESIZE but is now flagged ----------
out = _parse_investigator("just some prose with no markers at all")
assert out["status"] == "SYNTHESIZE", "terminal fallback must remain SYNTHESIZE"
assert out["unparsed"] is True, "a marker-less turn must be flagged for the loop retry"
# a partially-markered turn (LEDGER only) keeps the old silent fallback, unflagged
out = _parse_investigator("###LEDGER###\nFRONTIER:\n  f [untested] steps:-\n")
assert out["status"] == "SYNTHESIZE" and out["unparsed"] is False
print("1e (marker-less flagged; partial-marker fallback unchanged): OK")


# ====================================================================
# 2. Ledger: per-section merge + drift diff
# ====================================================================

def sig(entries):
    return [(e.label, e.status, e.steps) for e in entries]

FULL = ("FRONTIER:\n  teammate_pairing [in_progress] steps:1,2\n  era_effects [untested] steps:-\n"
        "REGIME:\n  season [not_examined] steps:-\n  era [not_examined] steps:-\n"
        "RISK:\n  car_quality [open] steps:-\n")

# ---------- 2a: THE WIPE SCENARIO — garbled sections keep prior; G1 stays armed ----------
nav = NavState()
nav.apply_ledger_block(FULL)
assert len(nav.frontier) == 2 and len(nav.regimes) == 2 and len(nav.risks) == 1
# next turn: FRONTIER/REGIME lines malformed (brackets forgotten), one RISK line parses
diff = nav.apply_ledger_block(
    "FRONTIER:\n  teammate_pairing - in progress - steps 1,2\n  era_effects - untested\n"
    "REGIME:\n  season not examined yet\n  era still open\n"
    "RISK:\n  car_quality [open] steps:-\n")
assert len(nav.frontier) == 2, "frontier wiped by one parseable RISK line"
assert len(nav.regimes) == 2, "regimes wiped by one parseable RISK line"
assert nav.open_regimes() == ["season", "era"], nav.open_regimes()
log = [{"step": 1, "code": "df['x'].mean()", "stdout": "1.2"}]   # no stratification ran
assert (not nav.g1_satisfied(log) and nav.open_regimes()), \
    "G1 hard-gate condition must stay armed after a partial parse"
assert diff is not None and set(diff.keys()) == {"risk"}, diff
print("2a (partial parse keeps frontier/regimes; G1 gate stays armed): OK")

# ---------- 2b: a full echo still replaces all four sections (round trip) ----------
nav2 = NavState()
nav2.apply_ledger_block(FULL)
rendered = nav2.render_for_investigator(None)
nav3 = NavState()
nav3.frontier = [Entry("frontier", "stale_old_item", "untested", [])]
diff = nav3.apply_ledger_block(rendered)
assert sig(nav3.frontier) == sig(nav2.frontier)
assert sig(nav3.regimes) == sig(nav2.regimes) and sig(nav3.risks) == sig(nav2.risks)
assert "stale_old_item" in diff["frontier"]["removed"], diff["frontier"]
assert "stale_old_item" in diff["frontier"]["dropped_live"], \
    "an untested item that vanished must be flagged as a live drop"
print("2b (full echo replaces all sections; diff reports the live drop): OK")

# ---------- 2c: explicit '(none)' under a NON-empty section keeps prior ----------
nav4 = NavState()
nav4.apply_ledger_block(FULL)
nav4.apply_ledger_block("FRONTIER:\n  teammate_pairing [tested] steps:1,2\n"
                        "REGIME:\n  (none)\nRISK:\n  car_quality [open] steps:-\n")
assert len(nav4.regimes) == 2, "explicit (none) must not wipe a populated section"
assert nav4.frontier[0].status == "tested"      # the parsed sections did update
print("2c (explicit '(none)' keeps a populated prior section): OK")

# ---------- 2d: totally garbled block leaves the whole map intact (regression) ----------
nav5 = NavState()
nav5.regimes = [Entry("regime", "keep", "examined", [1])]
assert nav5.apply_ledger_block("total garbage with no parseable ledger at all\njust prose") is None
assert [r.label for r in nav5.regimes] == ["keep"]
print("2d (garbled block leaves prior map intact): OK")

# ---------- 2e: fresh-map single-section parse unchanged (tolerance regression) ----------
nav6 = NavState()
nav6.apply_ledger_block("REGIME:\n  good [examined] steps:3\n  garbage line no status\n"
                        "EVIDENCE INDEX:\n  step 1: [ok] did a thing\n")
assert [r.label for r in nav6.regimes] == ["good"] and not nav6.frontier
print("2e (single-section parse on a fresh map, evidence lines ignored): OK")


# ====================================================================
# 3. Loop: format retry, preemptive search-spent note, pushback refund
# ====================================================================

df = pd.DataFrame({"a": [1, 2, 3, 4], "g": ["x", "y", "x", "y"],
                   "v": [5.0, 4.0, 5.2, 4.1]})

def inv(t, s, sp, l, extra=""):
    return (f"###THINKING###\n{t}\n###STATUS###\n{s}\n###SPEC###\n{sp}\n"
            f"{extra}###LEDGER###\n{l}\n")

CODE = ("```python\nprint('###RESULTS_START###')\n"
        "print(df.groupby('g')['v'].median().to_string())\n"
        "print('###RESULTS_END###')\n```")
LED = "FRONTIER | f1 | in_progress | steps: 1\nREGIME | g | not_examined | steps: -"
SYNTH = "Reasoning.\n###VERDICT###\nFINAL\n###FINDINGS###\n## Summary\nEffect is positive.\n"


class Mock:
    """Scripted client recording every Investigator input; optional search seat."""
    def __init__(s, inv_seq):
        s.inv_seq = inv_seq
        s.inv_calls = 0
        s.inv_inputs = []
        s.search_calls = 0

    def call(s, m, model, max_tokens=10000, temperature=0, agent=None,
             return_meta=False):
        if agent == "Investigator":
            s.inv_inputs.append("\n".join(
                p.get("content", "") if isinstance(p, dict) else str(p) for p in m))
            content = s.inv_seq[min(s.inv_calls, len(s.inv_seq) - 1)]
            s.inv_calls += 1
            if return_meta:
                return content, {"output_tokens": 50, "max_tokens": max_tokens,
                                 "truncated": False}
            return content
        if agent == "Executor":
            return CODE
        if agent == "Synthesizer":
            return SYNTH
        return ""

    def search_call(s, m, model, **kw):
        s.search_calls += 1
        return "[PUBLISHED] a calibration finding (Some Source)"


# ---------- 3a: marker-less turn -> ONE format retry, run completes ----------
outA = _tmpdir("fmt_retry")
seqA = ["I have been thinking about the data in plain prose, with no blocks.",
        inv("t1", "CONTINUE", "median v by g; print.", LED),
        inv("done", "SYNTHESIZE", "(none)", LED)]
mA = Mock(seqA)
kA = PersistentKernel(df=df)
statsA = RunStats()
log, _, nav, briefing = run_investigation(
    seed="t", df=df, client=mA, investigator_model="m:p", executor_model="m:c",
    schema_text="(s)", max_steps=6, output_dir=outA, kernel=kA, nav=NavState(),
    stats=statsA)
kA.cleanup()
assert briefing and "positive" in briefing, "run did not complete after the format retry"
assert mA.inv_calls == 3, f"expected 3 investigator calls (1 retried + 2 real), got {mA.inv_calls}"
assert any(e.get("code") for e in log), "should have executed a real step, not finalized on prose"
assert statsA.get("investigator_format_retries", 0) == 1
distinct = "contained none of the required ### blocks"
assert distinct in DIRECTIVE_FORMAT_RETRY
assert distinct not in mA.inv_inputs[0], "first turn must not carry the format directive"
assert distinct in mA.inv_inputs[1], "the retry must carry DIRECTIVE_FORMAT_RETRY"
assert DIRECTIVE_TRUNCATED_RETRY[:20] not in mA.inv_inputs[1], \
    "a format lapse must not be misdiagnosed as truncation"
print("3a (marker-less turn retried with the format directive, run completes): OK")

# ---------- 3b: search budget 0 -> spent note is preemptive from turn 1 ----------
outB = _tmpdir("search_note0")
seqB = [inv("t1", "CONTINUE", "median v by g; print.", LED),
        inv("done", "SYNTHESIZE", "(none)", LED)]
mB = Mock(seqB)
kB = PersistentKernel(df=df)
log, _, nav, briefing = run_investigation(
    seed="t", df=df, client=mB, investigator_model="m:p", executor_model="m:c",
    schema_text="(s)", max_steps=6, output_dir=outB, kernel=kB, nav=NavState(),
    search_model="m:s", search_budget=0)
kB.cleanup()
spent = "external-search budget for this run is spent"
assert spent in DIRECTIVE_SEARCH_SPENT
assert all(spent in t for t in mB.inv_inputs), \
    "with budget 0, every turn must carry the standing spent note"
assert mB.search_calls == 0
print("3b (budget 0: spent note standing from turn 1, no search attempted): OK")

# ---------- 3c: after the last search, the note appears on the NEXT turn ----------
outC = _tmpdir("search_note1")
seqC = [inv("want context", "SEARCH", "none", LED,
            extra="###QUERY###\naltitude adaptation effect sizes\n"),
        inv("t2", "CONTINUE", "median v by g; print.", LED),
        inv("done", "SYNTHESIZE", "(none)", LED)]
mC = Mock(seqC)
kC = PersistentKernel(df=df)
log, _, nav, briefing = run_investigation(
    seed="t", df=df, client=mC, investigator_model="m:p", executor_model="m:c",
    schema_text="(s)", max_steps=6, output_dir=outC, kernel=kC, nav=NavState(),
    search_model="m:s", search_budget=1)
kC.cleanup()
assert mC.search_calls == 1, "the budgeted search itself must still run"
assert spent not in mC.inv_inputs[0], "note must be absent while budget remains"
assert all(spent in t for t in mC.inv_inputs[1:]), \
    "note must stand on every turn after the budget is spent"
assert any(e.get("kind") == "search" for e in log)
print("3c (note absent before, standing after the last search): OK")

# ---------- 3d: a gate pushback refunds its iteration ----------
# max_steps=2: turn 1 tries to SYNTHESIZE with a not_examined regime and no
# executed code -> G1 gate pushes back. WITHOUT the refund the run would have
# one turn left (the analysis step) and end at the ceiling on the ungated
# provisional path. WITH the refund it gets its two planned analysis-capable
# turns after the pushback: the step runs (its groupby satisfies G1's code
# backstop), the re-requested synthesis proceeds, and the verdict is a natural
# gated FINAL, not a provisional one.
outD = _tmpdir("refund")
seqD = [inv("done already?", "SYNTHESIZE", "(none)", LED),          # gate fires
        inv("ok, stratify", "CONTINUE", "median v by g; print.", LED),
        inv("now done", "SYNTHESIZE", "(none)", LED)]
mD = Mock(seqD)
kD = PersistentKernel(df=df)
statsD = RunStats()
log, _, nav, briefing = run_investigation(
    seed="t", df=df, client=mD, investigator_model="m:p", executor_model="m:c",
    schema_text="(s)", max_steps=2, output_dir=outD, kernel=kD, nav=NavState(),
    g1_pushback_budget=1, stats=statsD)
kD.cleanup()
assert statsD.get("g1_gate_overrides", 0) == 1, "the G1 gate should have fired once"
assert mD.inv_calls == 3, \
    f"expected 3 investigator turns (pushback refunded), got {mD.inv_calls}"
assert any(e.get("code") for e in log), "the analysis step must still have run"
term = [e for e in log if e.get("terminal")]
assert term and term[-1].get("synth_verdict") == "FINAL"
assert briefing and "PROVISIONAL" not in briefing, \
    "with the refund, the run must finish through the gated path, not the ceiling"
print("3d (gate pushback refunds its iteration; run finishes gated, not provisional): OK")

print("\nALL HARDENING-FIX ASSERTIONS PASSED")
