# --- test bootstrap: runnable from the repo root via `python3 tests/<n>.py` ---
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:0] = [os.path.join(_HERE, "stubs"), _ROOT]  # bundled httpx stub + delv-e modules
# --- end bootstrap ---

# Scientific-generalist briefing contract. The Synthesizer (and the Reconciler,
# so --verify cannot undo it) writes for readers who understand evidence and the
# scientific method but are not statistics specialists: progressive disclosure,
# purpose-first method explanation, every decisive number preserved. Analytical
# adjudication is untouched -- the gates, the estimand, the evidence assembly and
# the Investigator all still work exactly as before; only the rendering changed.
#
# Pins the contract in BOTH modes plus four hardening rules added on review:
#   1. a translated quantity carries its exact form alongside it, so a bad
#      conversion is catchable by the very reader the briefing is written for
#      (banishing the exact value to Method notes hides errors from them);
#   2. a practical conclusion inherits the uncertainty of the estimate under it;
#   3. sections are omitted rather than padded (readability was the goal, and a
#      stub section on a five-step run reads worse, not better);
#   4. the audience standard sits adjacent to the BRIEFING STRUCTURE it governs,
#      leaving the analytical principles and the G1-G4 gates contiguous.
# Also pins the OUTPUT FORMAT block count (it said "three" while listing four
# once ###CHARTS### was added, which invites a model to drop the charts block)
# and guards the f-string prompts against stray braces.

import re

import prompts as P

DATA, COMPUTE = P.SYNTHESIZER_SYSTEM, P.COMPUTE_SYNTHESIZER_SYSTEM


def require(text, *phrases):
    for phrase in phrases:
        assert phrase in text, f"missing prompt contract phrase: {phrase!r}"


# ── 1) The audience contract, both modes ──
require(DATA, "scientific generalists", "Use progressive disclosure",
        "## Results at a glance", "## How the analysis reached this result",
        "## Where the result is stable", "## What the data can and cannot establish",
        "## Practical conclusion", "Do not patronize")
require(COMPUTE, "scientific generalists", "## Results at a glance",
        "## How the computation reached this result",
        "## What the computation can and cannot establish", "## Practical conclusion")
require(P.SYNTHESIZER_USER_TEMPLATE, "Write the BRIEFING for scientific generalists",
        "preserve all decisive numbers")
require(P.COMPUTE_SYNTHESIZER_USER_TEMPLATE, "Write the BRIEFING for scientific generalists",
        "preserve all decisive numbers")
# The verified briefing must not revert to the old shape.
require(P.RECONCILIATION_PROMPT, "Write for scientific generalists using progressive disclosure",
        "## Verification record", "## Practical conclusion")
require(P.RECONCILIATION_PROMPT_COMPUTE,
        "Write for scientific generalists using progressive disclosure",
        "## Verification record", "## Practical conclusion")
print("audience contract: both modes, both reconcilers: OK")

# ── 2) Analytical adjudication is untouched ──
for text in (DATA, COMPUTE):
    require(text, "###GATES###", "###VERDICT###", "###BRIEFING###", "###CHARTS###")
require(DATA, "G1 —", "Do not invent numbers that are not in the evidence")
print("analytical machinery intact (gates, blocks, grounding): OK")

# ── 3) Fix 1: exact quantity alongside its translation, not exiled to Method notes ──
for text in (DATA, COMPUTE):
    require(text, "Carry the exact quantity ALONGSIDE its translation at first use",
            "pace ratio 1.15", "worse than an untranslated one")
    assert "with the exact ratio, coefficient, or interval in parentheses at first use" in text
print("fix 1: translations stay auditable in place: OK")

# ── 4) Fix 2: the practical conclusion inherits its estimate's uncertainty ──
require(DATA, "inherits the uncertainty of the estimate it rests on",
        "never state a conclusion with more confidence than the evidence behind it carries")
require(COMPUTE, "inherits the uncertainty and the assumptions of the result it rests on")
print("fix 2: practical conclusion carries its uncertainty: OK")

# ── 5) Fix 3: omit, do not pad ──
for text in (DATA, COMPUTE):
    require(text, "a section with nothing evidenced to put in it is noise",
            "A short investigation earns a short briefing")
    # The four sections that must always survive an omission decision.
    for always in ("## Summary", "## Where it breaks down", "## Open questions",
                   "## Method notes"):
        assert always in text
print("fix 3: sections omitted rather than padded: OK")

# ── 6) Fix 4: the audience standard sits next to the structure it governs ──
for name, text in (("data", DATA), ("compute", COMPUTE)):
    i_chain = text.index("G1 —") if "G1 —" in text else text.index("OUTPUT FORMAT.")
    i_aud = text.index("AUDIENCE AND WRITING STANDARD")
    i_struct = text.index("BRIEFING STRUCTURE")
    assert i_chain < i_aud < i_struct, \
        f"{name}: the audience block must follow the analytical chain and precede the structure"
print("fix 4: audience block adjacent to the briefing structure: OK")

# ── 7) The OUTPUT FORMAT block count matches the blocks actually listed ──
for name, text in (("data", DATA), ("compute", COMPUTE)):
    fmt = re.search(r"OUTPUT FORMAT\. Emit exactly these (\w+)", text).group(1)
    assert fmt == "blocks", f"{name}: hardcoded block count ({fmt!r}) drifts when a block is added"
    assert "###CHARTS###, is optional" in text, f"{name}: charts must be declared optional"
print("output format: block count cannot drift: OK")

# ── 8) Evidence-fidelity rules, each added after a live briefing misreported its
#      own evidence. Both synthesizer modes carry them (shared standard).
for text in (DATA, COMPUTE):
    # Table provenance: the altitude briefing's results table paired step-7 pace
    # ratios with step-8 band lap counts, overstating the ratio's own sample by
    # 14-31%, while the right counts sat in the frame it charted from.
    require(text, "must describe the SAME population",
            "label the column with the step it came from")
    # CONSTANCY, not monotonicity. The next altitude run's HR bins ran
    # 0.99/0.94/0.98/0.95 at moderate effort and 1.03/1.02/1.06 at hard effort:
    # a clean sign change, which the briefing reported as "no monotonic trend"
    # and filed under STABLE. A bias that reverses across a stratum looks exactly
    # like that, and the pooled 0.996 was the place the two halves cancelled.
    require(text, "The test is whether the effect is CONSTANT",
            "A sign change across a stratum is a LARGER finding",
            "it is the place where they cancel and vanish",
            "Never file as unresolved a question the evidence already answers")
    # A pooled null has a fifth explanation: cancellation.
    require(text, "CANCELLATION, two influences running in opposite directions",
            "the null IS the cancellation")
    # A confounded quantity is a labelling problem, not a reason for silence: the
    # same run refused to report the high-altitude slowdown at all (surface is
    # collinear with altitude there), leaving the practitioner with nothing.
    require(text, "A confounded quantity is not thereby useless",
            "say plainly that it is not the estimand")
    # Stability must be earned, not inferred from a quiet table.
    require(text, "Stability is a claim about evidence, not the default reading of a quiet table")
    # The guard rail on the accessibility push itself: the technical briefings
    # surfaced the irregular pattern by dumping the table; the readable one
    # interpreted it, and interpretation is where a mess gets tidied away.
    require(text, "WHAT PLAIN LANGUAGE MAY NOT DO",
            "smooth an irregular pattern into a clean one",
            "call a result stable because the messy version is harder to explain",
            "state it plainly and completely rather than neatly and partially")
print("fidelity rules: provenance, constancy, cancellation, labelled confounds,"
      " earned stability, plain-language guard: OK")

# ── 9) The upstream half of the same contract: a briefing can only preserve
#      uncertainty the ANALYSIS computed. Data mode demanded uncertainty only for
#      "best/top/outlier" claims, so an estimate question (a correction, a bound)
#      never triggered it and a nine-step run shipped with no interval at all.
#      Compute mode already had this as C2, so it is deliberately not duplicated.
require(P.INVESTIGATOR_SYSTEM,
        "ANY quantity you will hand over as the answer",
        "a bound whose width you never measured is not a bound",
        "cluster_bootstrap over that unit is the honest default")
require(P.COMPUTE_INVESTIGATOR_SYSTEM, "C2, QUANTIFY UNCERTAINTY")
assert "ANY quantity you will hand over" not in P.COMPUTE_INVESTIGATOR_SYSTEM, \
    "compute mode already carries C2; do not duplicate the rule"
# The clause sits beside the leakage audit, so it must model closed-spec language.
from investigation import scan_spec_for_leakage
_i = P.INVESTIGATOR_SYSTEM.index("ANY quantity you will hand over")
assert scan_spec_for_leakage(P.INVESTIGATOR_SYSTEM[_i:_i + 520]) == [], \
    "the uncertainty clause must not use words the spec audit bans"
print("investigator: uncertainty required for any headline quantity: OK")

# ── 10) f-string guard: both synthesizer prompts are f-strings now, so a literal
#      brace added later would raise at import (or silently interpolate). The only
#      placeholder either may contain is the audience standard, already expanded.
src = open(os.path.join(_ROOT, "prompts.py")).read()
for name in ("SYNTHESIZER_SYSTEM", "COMPUTE_SYNTHESIZER_SYSTEM"):
    m = re.search(rf'^{name} = (f?)"""(.*?)"""', src, re.S | re.M)
    is_f, body = m.group(1), m.group(2)
    placeholders = re.findall(r"\{([^}]*)\}", body)
    assert is_f == "f", f"{name} must stay an f-string for the audience standard to expand"
    assert placeholders == ["SCIENTIFIC_GENERALIST_BRIEFING_STANDARD"], \
        f"{name}: unexpected brace content {placeholders!r}; an f-string prompt cannot hold literal braces"
# And the expanded prompts carry no leftover braces at all.
for text in (DATA, COMPUTE):
    assert "{" not in text and "}" not in text, "expanded prompt still contains braces"
print("f-string guard: no stray braces can slip into the synthesizer prompts: OK")

print("test_synth_audience: OK")