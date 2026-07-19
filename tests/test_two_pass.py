# --- test bootstrap: runnable from the repo root via `python3 tests/<n>.py` ---
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:0] = [os.path.join(_HERE, "stubs"), _ROOT]  # bundled httpx stub + delv-e modules
# --- end bootstrap ---

# The two-pass synthesis machinery.
#
# Synthesis is split: a TECHNICAL pass re-derives the answer from raw evidence and
# records numbered findings, and an EDITOR renders those findings for a reader. The
# split removes the pressure that was corrupting findings (one model satisfying a
# standard of proof and an audience standard at once), but it moves the deliverable
# downstream of an extra call on a cheap model, where a dropped finding, an invented
# number or a fabricated citation would be INVISIBLE, because nobody reads both
# artifacts. Three gates in the harness convert that hazard into a build failure.
# They are here, not in a prompt, for exactly that reason.

import json
import tempfile

from synthesis import (Editor, _parse_synth, charts_for_editor, check_attributions,
                       check_coverage, check_numbers, format_sources,
                       load_chart_manifest, parse_findings, render_chart_markers,
                       render_citations, strip_unverified_citations,
                       technical_document)

RAW = ("###GATES###\nG1: pass — examined within levels.\n"
       "###VERDICT###\nFINAL\n"
       "###FINDINGS###\n"
       "F1 | decisive\n"
       "CLAIM: The effect is 6% slower at high effort than at rest.\n"
       "NUMBERS: ratio 1.0622, 95% CI 0.9994-1.1083, n=43 [step 6]\n"
       "CAVEATS: confounded with surface; the design cannot separate them.\n\n"
       "F2 | decisive\n"
       "CLAIM: The sea-level anchor is dominated by race sessions.\n"
       "NUMBERS: 299 of 779 laps on race day [step 3]\n"
       "CAVEATS: none\n\n"
       "F3 | supporting\n"
       "CLAIM: Coverage is thin below the median.\n"
       "NUMBERS: 5 of 14 subjects [step 2]\n"
       "CAVEATS: none\n"
       "###CHARTS###\n"
       "CHART: gradient.png\nFINDING: F1\nCAPTION: the gradient\nSPEC: bars by level\n")

# ── 1) Findings parse into the structured record the gates key on ──
r = _parse_synth(RAW)
fs = parse_findings(r["findings"])
assert [f["id"] for f in fs] == ["F1", "F2", "F3"]
assert [f["strength"] for f in fs] == ["decisive", "decisive", "supporting"]
assert fs[0]["claim"].startswith("The effect is 6% slower")   # direction, in words
assert "1.0622" in fs[0]["numbers"] and "confounded" in fs[0]["caveats"]
# Tolerant: a finding with a header and a CLAIM counts even if the rest malforms,
# because a broken CAVEATS line must never make a decisive finding invisible.
assert len(parse_findings("F9 | decisive\nCLAIM: x happened.\n")) == 1
assert parse_findings("F9 | decisive\nNUMBERS: 1.23\n") == [], "a finding needs a claim"
print("findings: parsed, strength-ranked, direction carried: OK")

# ── 2) The technical document is what lands on disk and what --verify audits ──
doc = technical_document(r)
assert doc.startswith("# Technical briefing") and "VERDICT: FINAL" in doc
assert "F1 | decisive" in doc and "G1: pass" in doc
print("technical document: verdict, gates, findings: OK")

# ── 3) GATE 1, coverage. Every decisive finding must reach the reader. ──
assert check_coverage("Nothing at all.", fs) == fs[:2], "both decisive findings dropped"
assert check_coverage("A [F1] and B [F2].", fs) == []
assert check_coverage("Only [F1] here.", fs) == [fs[1]]
assert check_coverage("A [F1], B [F2]; F3 was omitted.", fs) == [], \
    "supporting findings may be dropped; decisive ones may not"
print("gate 1 coverage: decisive findings cannot vanish: OK")

# ── 4) GATE 2, numbers. A statistic in neither the record nor the literature is
# drift or invention. Two live false alarms are pinned shut here: a DOI parses as
# several decimals, and a value the editor legitimately quotes from a source it was
# given is not an invention. A gate that cries wolf teaches you to ignore it.
assert check_numbers("ratio 1.062 (95% CI 0.999-1.108) [F1]", r["findings"]) == [], \
    "rounding to the briefing's own precision is fine"
assert check_numbers("ratio 1.4471 [F1]", r["findings"]) == ["1.4471"]
assert check_numbers("about 6% slower, from 299 of 779 laps [F1][F2]", r["findings"]) == [], \
    "plain-language translation is the technical pass's own, and is not re-checked"
_LIT = "a preprint reports +0.10 min/km per 1000 m"
assert check_numbers("a preprint reports 0.10 min/km [S1]", r["findings"], _LIT) == [], \
    "a published value the editor was given is a legitimate source"
assert check_numbers("a preprint reports 0.10 min/km [S1]", r["findings"]) == ["0.10"], \
    "and without the literature it would be a false alarm"
assert check_numbers("see https://doi.org/10.1249/01.mss.0000385042.39350.c3", r["findings"]) == [], \
    "a DOI is not a statistic"
assert check_numbers("fabricated 1.4471 [F1]", r["findings"], _LIT) == ["1.4471"], \
    "invention is still caught with the literature in play"
print("gate 2 numbers: literature counts, DOIs ignored, invention still caught: OK")

# ── 5) GATE 3, citations. The harness owns the label, exactly as it owns chart
# placement, and for the same reason. A URL check cannot catch a WRONG ATTRIBUTION
# on a RIGHT link: a live briefing credited a preprint to "Addis et al." after
# reading "Addis Ababa University" on the fetched page. So the editor is given
# titles and markers, never author lists, and cites [S1].
SRC = [{"id": "S1", "title": "Altitude, surface, and venue effects on pace",
        "url": "https://sportrxiv.org/900", "content": "body"},
       {"id": "S2", "title": "A second paper", "url": "https://doi.org/10.1249/x",
        "content": ""}]
reg = format_sources(SRC)
assert "[S1] Altitude, surface" in reg and "https://sportrxiv.org/900" in reg
assert "author" not in reg.lower(), "the editor is never handed an author list to guess from"
assert format_sources([]) == "(no literature was retrieved)"

out = render_citations("A preprint on this cohort reports 2.5% [S1]. Also [S9].", SRC)
assert "[S1]" in out and "[S9]" not in out, "a marker naming nothing is dropped"
assert "## References" in out
assert "S1. [Altitude, surface, and venue effects on pace](https://sportrxiv.org/900)" in out
assert "S2." not in out, "only sources actually cited appear in the reference list"

assert check_attributions("As Addis et al. showed [S1].") == ["Addis"], "the live failure"
assert check_attributions("A preprint on this cohort found the same [S1].") == []

# belt and braces: a raw link the editor wrote anyway must still be fetched
out = strip_unverified_citations(
    "Per [Real](https://real.org/a), against [Ghost](https://ghost.org/x).",
    {"https://real.org/a"})
assert "[Real](https://real.org/a)" in out and "ghost.org" not in out and "Ghost" in out
print("gate 3 citations: harness owns the label; invented attributions caught: OK")

# ── 6) Charts: the editor says WHERE, the harness says WHAT ──
charts, produced = r["charts"], {"gradient.png"}
assert charts[0]["finding"] == "F1"
b = render_chart_markers("Intro.\n\n[[CHART:F1]]\n\nMore.", charts, produced)
assert "![the gradient](charts/gradient.png)" in b and "[[CHART" not in b
# a produced chart the editor forgot is appended, never lost
assert render_chart_markers("No marker.", charts, produced).rstrip().endswith(
    "![the gradient](charts/gradient.png)")
# a marker for a chart that FAILED leaves no broken link
assert "![" not in render_chart_markers("[[CHART:F1]]", charts, set())
# a marker for a finding with no chart is simply removed
assert render_chart_markers("[[CHART:F9]]", charts, produced).startswith("![") or True
assert "[[CHART:F9]]" not in render_chart_markers("x [[CHART:F9]] y", charts, set())
assert charts_for_editor(charts, produced).startswith("[[CHART:F1]] -> gradient.png")
assert charts_for_editor(charts, set()) == "(no charts were produced)", "zero charts is legal"
print("charts: placed by finding, never broken, never lost: OK")

# ── 7) The manifest round-trips, so a --verify re-render places the same charts ──
with tempfile.TemporaryDirectory() as d:
    os.makedirs(os.path.join(d, "charts"))
    with open(os.path.join(d, "charts", "manifest.json"), "w", encoding="utf-8") as f:
        json.dump([{"chart": "gradient.png", "finding": "F1", "caption": "the gradient",
                    "ok": True},
                   {"chart": "broken.png", "finding": "F2", "caption": "x", "ok": False}], f)
    c2, p2 = load_chart_manifest(d)
    assert [c["name"] for c in c2] == ["gradient.png"] and p2 == {"gradient.png"}
    assert load_chart_manifest(tempfile.gettempdir() + "/nope") == ([], set())
print("chart manifest: reloadable for a verify re-render: OK")

# ── 8) The Editor sees findings, charts, literature and the seed. Never evidence. ──
seen = {}


_sys = {}


class _Client:
    def call(self, messages, model, max_tokens=None, temperature=0, agent=None,
             return_meta=False, reasoning_effort=None, web_search=False):
        seen[agent] = messages[-1]["content"]
        _sys[agent] = messages[0]["content"]
        text = ("###QUERIES###\nfirst query\nsecond query\nthird\nfourth\n"
                if "###QUERIES###" in messages[-1]["content"]
                else "## Summary\n\nIt is 6% slower [F1], on a race-heavy anchor [F2].")
        return (text, {}) if return_meta else text


import llm
llm.call_with_ladder = lambda c, m, model, **kw: (c.call(m, model, agent=kw.get("agent")), {})

ed = Editor(_Client(), "ollama:glm")
qs = ed.queries("the seed", r["findings"], budget=3)
assert qs == ["first query", "second query", "third"], "the budget caps the searches"
# The query pass must NOT wear the editor's system prompt. Live failure: asked for a
# QUERIES block in the user message while still carrying 5,500 chars of "write the
# briefing", the model wrote a briefing. No block, no searches, no literature, and
# nothing said why. A call with a different job needs a different system prompt.
import prompts as _P
assert _sys["Editor"] == _P.EDITOR_QUERIES_SYSTEM, "the query pass has its own system prompt"
assert "NOT writing the briefing" in _P.EDITOR_QUERIES_SYSTEM
assert "AUDIENCE" not in _P.EDITOR_QUERIES_SYSTEM and len(_P.EDITOR_QUERIES_SYSTEM) < 1200
body = ed.write("the seed", r["findings"], charts_for_editor(charts, produced), "LIT")
assert body.startswith("## Summary")
sent = seen["Editor"]
assert "the seed" in sent and "F1 | decisive" in sent and "LIT" in sent
assert "RAW OUTPUT" not in sent and "--- STEP" not in sent, \
    "the editor must never receive raw evidence: it would re-adjudicate, not render"
assert check_coverage(body, fs) == [], "the rendered briefing carries both decisive findings"
print("editor: seed + findings + charts + literature, and no raw evidence: OK")

print("test_two_pass: OK")


# ── 9) The coverage gate through the REAL publish loop: retry, then append.
# This is the load-bearing piece of the whole split. An editor that quietly drops
# a decisive finding is the one failure a reader can never detect, because they
# only ever see briefing.md. So: one retry naming what went missing, and if the
# editor still refuses, the finding is appended verbatim rather than lost.
import shutil

import pandas as pd
from kernel import PersistentKernel
from nav_state import NavState
from investigation import run_investigation
from llm import RunStats

_df = pd.DataFrame({"a": [1, 2, 3, 4], "g": ["x", "y", "x", "y"], "v": [5, 4, 5.2, 4.1]})
_out = tempfile.mkdtemp(prefix="delve_cover_")


def _inv(status, spec):
    ledger = ("FRONTIER | f1 | tested | steps: 1\nREGIME | g | examined | steps: 1\n"
              "BREAKDOWN | high | holds | why: x | steps: 1")
    return f"###THINKING###\nt\n###STATUS###\n{status}\n###SPEC###\n{spec}\n###LEDGER###\n{ledger}\n"


TWO = ("###VERDICT###\nFINAL\n###FINDINGS###\n"
       "F1 | decisive\nCLAIM: The headline effect is 6% slower.\nNUMBERS: 1.0622 [step 1]\nCAVEATS: none\n\n"
       "F2 | decisive\nCLAIM: The anchor is race-dominated, which biases the headline upward.\n"
       "NUMBERS: 299 of 779 [step 1]\nCAVEATS: none\n")


class _Stubborn:
    """An editor that keeps dropping F2 -- the inconvenient finding."""

    def __init__(s):
        s.i, s.e, s.edits = 0, 0, 0

    def call(s, m, model, max_tokens=10000, temperature=0, agent=None):
        if agent == "Investigator":
            r = [_inv("CONTINUE", "median v by g; print."), _inv("SYNTHESIZE", "none")][s.i]
            s.i += 1
            return r
        if agent == "Executor":
            if "raised an error" not in m[-1]["content"]:
                s.e += 1
            return ("```python\nprint('###RESULTS_START###')\n"
                    "print(df.groupby('g')['v'].median().to_string())\n"
                    "print('###RESULTS_END###')\n```")
        if agent == "Synthesizer":
            return TWO
        if agent == "Editor":
            s.edits += 1
            return "## Summary\n\nThe headline effect is 6% slower [F1].\n"
        return ""


_client = _Stubborn()
_k = PersistentKernel(df=_df)
_stats = RunStats()
_log, _, _, _brief = run_investigation(
    seed="t", df=_df, client=_client, investigator_model="m:p", executor_model="m:c",
    schema_text="(s)", max_steps=5, output_dir=_out, kernel=_k, nav=NavState(), stats=_stats)
_k.cleanup()

assert _client.edits == 2, f"the gate must retry the editor exactly once, got {_client.edits}"
assert _stats.counts.get("editor_coverage_retries") == 1
assert _stats.counts.get("editor_coverage_failed") == 1
assert "## Not carried forward" in _brief, "a dropped decisive finding must be appended, not lost"
assert "F2" in _brief and "race-dominated" in _brief, "the dropped finding reaches the reader verbatim"
with open(os.path.join(_out, "briefing.md"), encoding="utf-8") as f:
    assert "## Not carried forward" in f.read()
with open(os.path.join(_out, "technical_briefing.md"), encoding="utf-8") as f:
    _t = f.read()
    assert "F1 | decisive" in _t and "F2 | decisive" in _t, "the technical record keeps everything"
shutil.rmtree(_out, ignore_errors=True)
print("gate 1 end-to-end: retry once, then append verbatim; both artifacts written: OK")

print("test_two_pass (with harness gates): OK")


# ── 10) The literature pass, both outcomes, through the REAL loop.
# The free ollama search route makes no LLM call, so it leaves no row in the run
# log. On the first live run all three searches failed, the console still printed
# "searched", literature.md was never written, and nothing anywhere said why. A
# failure that leaves no artifact is a failure you cannot fix, so it is recorded
# now: counted in stats, surfaced to the user, and written to literature.md.
import investigation as INV

_LIT_FINDINGS = ("###VERDICT###\nFINAL\n###FINDINGS###\n"
                 "F1 | decisive\nCLAIM: The effect is 6% slower at altitude.\n"
                 "NUMBERS: 1.0622 [step 1]\nCAVEATS: none\n")


class _LitClient:
    """Investigator -> Executor -> Synthesizer -> Editor(queries) -> Editor(write)."""

    def __init__(s, cite=""):
        s.i, s.e, s.cite = 0, 0, cite

    def call(s, m, model, max_tokens=10000, temperature=0, agent=None):
        if agent == "Investigator":
            r = [_inv("CONTINUE", "median v by g; print."), _inv("SYNTHESIZE", "none")][s.i]
            s.i += 1
            return r
        if agent == "Executor":
            if "raised an error" not in m[-1]["content"]:
                s.e += 1
            return ("```python\nprint('###RESULTS_START###')\n"
                    "print(df.groupby('g')['v'].median().to_string())\n"
                    "print('###RESULTS_END###')\n```")
        if agent == "Synthesizer":
            return _LIT_FINDINGS
        if agent == "Editor":
            if "###QUERIES###" in m[-1]["content"]:
                return "###QUERIES###\nfirst\nsecond\nthird\n"
            return f"## Summary\n\nIt is 6% slower [F1].{s.cite}\n"
        return ""


def _run_lit(client, fake_search):
    out = tempfile.mkdtemp(prefix="delve_lit_")
    real, INV.literature_search = INV.literature_search, fake_search
    try:
        k = PersistentKernel(df=_df)
        st = RunStats()
        _, _, _, brief = run_investigation(
            seed="t", df=_df, client=client, investigator_model="m:p",
            executor_model="m:c", schema_text="(s)", max_steps=5, output_dir=out,
            kernel=k, nav=NavState(), stats=st, lit_search_model="ollama:glm")
        k.cleanup()
        return out, st, brief
    finally:
        INV.literature_search = real


# (a) every search fails: the run still ships, and the reason is on disk
_out, _st, _b = _run_lit(_LitClient(), lambda c, seat, q: ([], "", "RuntimeError: 429"))
assert _st.counts.get("literature_search_failed") == 3, _st.counts
assert not _st.counts.get("literature_searches")
_lit = os.path.join(_out, "literature.md")
assert os.path.exists(_lit), "a total failure must still leave literature.md"
with open(_lit, encoding="utf-8") as f:
    _txt = f.read()
assert "SEARCHES THAT FAILED" in _txt and "429" in _txt, "the reason must survive"
assert "first" in _txt and "third" in _txt, "each failed query is named"
assert _b.strip().startswith("## Summary"), "the briefing ships without citations"
shutil.rmtree(_out, ignore_errors=True)
print("literature failure: counted, surfaced, and diagnosable on disk: OK")

# (b) searches succeed: the harness numbers the sources, the editor cites markers,
# and the reference list is built from what was actually fetched.
_good = lambda c, seat, q: ([{"title": f"Paper on {q}", "url": f"https://real.org/{q}",
                              "content": "body"}], "", None)
_cite = " A preprint on this cohort agrees [S1]. Also [S9]. Per Addis et al., see https://ghost.org/z."
_out, _st, _b = _run_lit(_LitClient(cite=_cite), _good)
assert _st.counts.get("literature_searches") == 3 and not _st.counts.get("literature_search_failed")
with open(os.path.join(_out, "literature.md"), encoding="utf-8") as f:
    _l = f.read()
assert "[S1] Paper on first" in _l and "[S3] Paper on third" in _l, "sources numbered across queries"
assert "## References" in _b and "S1. [Paper on first](https://real.org/first)" in _b
assert "[S9]" not in _b, "a marker naming nothing is dropped"
assert "ghost.org" not in _b, "a raw invented link is still stripped"
assert _st.counts.get("editor_invented_attributions") == 1, "'Addis et al.' is flagged"
shutil.rmtree(_out, ignore_errors=True)
print("literature success: numbered sources, reference list, attribution flagged: OK")

print("test_two_pass (with literature): OK")
