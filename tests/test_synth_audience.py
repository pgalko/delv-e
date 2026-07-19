# --- test bootstrap: runnable from the repo root via `python3 tests/<n>.py` ---
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:0] = [os.path.join(_HERE, "stubs"), _ROOT]  # bundled httpx stub + delv-e modules
# --- end bootstrap ---

# The synthesis prompt contract, after synthesis was split in two.
#
# One model asked to satisfy a standard of proof AND an audience standard at once
# is what smoothed findings away: a sign flip got reported as stability because
# the messy version was harder to write plainly. The two standards now live in two
# passes. The TECHNICAL pass answers to the standard of proof and records numbered
# findings; it never sees an audience rule. The EDITOR answers to the audience
# standard and renders those findings; it never sees the raw evidence, so it
# cannot re-adjudicate, only render.
#
# Also pins DOMAIN NEUTRALITY. delv-e runs on any dataset and on none. An example
# lifted from a live use case flatters the system on that case and teaches it
# nothing transferable, and it is meaningless for a simulation with no data at all.

import re

import prompts as P

TECH = (P.SYNTHESIZER_SYSTEM, P.COMPUTE_SYNTHESIZER_SYSTEM)


def require(text, *phrases):
    for phrase in phrases:
        assert phrase in text, f"missing prompt contract phrase: {phrase!r}"


# ── 1) The technical pass: proof, findings, and NO audience pressure ──
for t in TECH:
    require(t, "THE STANDARD OF PROOF",
            "the evidence would be unlikely to look as it does if the claim were false",
            "Name the number the claim rests on",
            "Name what else could have produced that number",
            "Carry the weakest link",
            # the four consequences that make the test actionable
            "A number is evidence only for the population, filter, or setting it was computed on",
            "A summary across strata is evidence for the summary only when the strata agree",
            "does not exclude that value, however narrowly the bound falls",
            "must predict that effect's direction and rough size",
            # and the symmetric half: under-reporting is a failure too
            "The test cuts both ways")
    assert "AUDIENCE AND WRITING STANDARD" not in t, \
        "the technical pass must carry no audience rule: that pressure is what smoothed findings"
    assert "scientific generalists" not in t
print("technical pass: standard of proof, no audience pressure: OK")

# ── 2) The findings format, and the direction rule that closes a live bug ──
for t in TECH:
    require(t, "###FINDINGS###", "FINDINGS FORMAT",
            "F<n> | decisive",
            "WITH ITS DIRECTION IN WORDS",
            # a live briefing said "expect 2-8% faster" from a ratio of 0.92, which
            # means slower: every number was right and the advice was inverted.
            "a sign inverted here is a sign inverted in the deliverable",
            "every decisive finding must survive into the briefing")
    assert "BRIEFING STRUCTURE" not in t, "the editor owns the structure now"
print("findings format: direction owned by the technical pass: OK")

# ── 3) Charts key on a finding, not a section: the editor picks its own headings ──
for t in TECH:
    require(t, "###CHARTS###", "FINDING: <the id of the finding",
            "Never ask for text annotations")
    assert "SECTION:" not in t, "a header is not a stable anchor once the editor chooses them"
print("charts: keyed on findings: OK")

# ── 4) The editor: register, and the four things it may not do ──
E = P.EDITOR_SYSTEM
# The harness owns the citation label. A live briefing credited a preprint to
# "Addis et al." after reading the author's UNIVERSITY on the fetched page, and a
# wrong attribution riding a real, fetched URL passes any URL check. So the editor
# is handed titles and markers, never author lists.
require(E, "Cite it by that marker alone",
        "Do NOT write a link, a URL, an author name, or a year",
        "any name you write is a guess",
        "The harness builds the reference list")
require(E, "AUDIENCE AND WRITING STANDARD", "scientific generalists",
        "Use progressive disclosure", "Do not patronize",
        "This is your ONLY source of fact",
        "Introduce a number that is not in the technical briefing",
        "Change the direction of a finding",
        "Drop a decisive finding",
        "Cite a source you were not given",
        "Divergence from the published record is a FINDING",
        "[[CHART:F3]]", "Never write an image link yourself")
assert "THE STANDARD OF PROOF" not in E, "the editor renders; it does not adjudicate"
assert "RAW EVIDENCE" not in E, "the editor must never see the evidence, or it re-litigates"
for tmpl, fields in ((P.EDITOR_QUERIES_TEMPLATE, ("{seed}", "{technical}", "{budget}")),
                     (P.EDITOR_BRIEFING_TEMPLATE,
                      ("{seed}", "{technical}", "{charts}", "{literature}"))):
    for f in fields:
        assert f in tmpl, f"editor template missing {f}"
assert P.DATA_MODE.editor_system is P.EDITOR_SYSTEM
assert P.COMPUTE_MODE.editor_system is P.EDITOR_SYSTEM
print("editor: renders findings, cannot re-derive: OK")

# ── 5) The audit trail is the finding id, which is what makes coverage checkable ──
require(P.AUDIENCE_STANDARD, "Reference findings by id",
        "every decisive finding must carry one")
print("audit trail: finding ids: OK")

# ── 6) Reconciliation adjudicates findings, and sheds every audience rule ──
for r in (P.RECONCILIATION_PROMPT, P.RECONCILIATION_PROMPT_COMPUTE):
    require(r, "Emit the corrected findings in the SAME numbered format",
            "## Verification record", "Output only the corrected findings.")
    assert "scientific generalists" not in r, "the editor re-renders; the reconciler adjudicates"
# the two modes dispute different things
require(P.RECONCILIATION_PROMPT, "the sample or filter")
require(P.RECONCILIATION_PROMPT_COMPUTE, "the parameters or resolution")
print("reconciliation: findings, per-mode adjudication vocabulary: OK")

# ── 7) Gates and grounding survive the split ──
for name, t in (("data", P.SYNTHESIZER_SYSTEM), ("compute", P.COMPUTE_SYNTHESIZER_SYSTEM)):
    require(t, "###GATES###", "###VERDICT###")
    fmt = re.search(r"OUTPUT FORMAT\. Emit exactly these (\w+)", t).group(1)
    assert fmt == "blocks", f"{name}: a hardcoded block count drifts when a block is added"
require(P.SYNTHESIZER_SYSTEM, "G1 —", "Do not invent numbers that are not in the evidence")
print("gates and grounding intact: OK")

# ── 8) The Investigator still owes an uncertainty on anything it hands over ──
require(P.INVESTIGATOR_SYSTEM, "ANY quantity you will hand over as the answer",
        "a bound whose width you never measured is not a bound")
require(P.COMPUTE_INVESTIGATOR_SYSTEM, "C2, QUANTIFY UNCERTAINTY")
from investigation import scan_spec_for_leakage
_i = P.INVESTIGATOR_SYSTEM.index("ANY quantity you will hand over")
assert scan_spec_for_leakage(P.INVESTIGATOR_SYSTEM[_i:_i + 520]) == []
print("investigator: uncertainty on any headline quantity: OK")

# ── 9) DOMAIN NEUTRALITY, permanent. Runs on any dataset, and on none. ──
DOMAIN = (r"\brunner|\bpace\b|pace ratio|altitude|athlete\b|heart[- ]rate|\bbpm\b|"
          r"hypoxia|asphalt|terrain|\bdriver\b|constructor|teammate|\blap\b")
offenders = []
for name in dir(P):
    if name.startswith("_"):
        continue
    val = getattr(P, name)
    if not isinstance(val, str):
        continue
    for m in re.finditer(DOMAIN, val, re.I):
        offenders.append(f"{name}: ...{val[max(0, m.start()-40):m.start()+40]}...")
assert not offenders, ("no prompt may name a use case; found:\n  " + "\n  ".join(offenders[:6]))
print("domain neutrality: no use-case vocabulary in any prompt: OK")

# ── 10) f-string guard: the prompts interpolate their standards ──
src = open(os.path.join(_ROOT, "prompts.py")).read()
for name, ph in (("SYNTHESIZER_SYSTEM", "STANDARD_OF_PROOF"),
                 ("COMPUTE_SYNTHESIZER_SYSTEM", "STANDARD_OF_PROOF"),
                 ("EDITOR_SYSTEM", "AUDIENCE_STANDARD")):
    m = re.search(rf'^{name} = (f?)"""(.*?)"""', src, re.S | re.M)
    assert m.group(1) == "f", f"{name} must stay an f-string"
    assert re.findall(r"\{([^}]*)\}", m.group(2)) == [ph], \
        f"{name}: an f-string prompt cannot hold literal braces"
for t in TECH + (E,):
    assert "{" not in t and "}" not in t
print("f-string guard: OK")

print("test_synth_audience: OK")
