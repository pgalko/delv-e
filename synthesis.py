"""
Synthesizer for delv-e's inverted-core loop (step 4, spec §4.3 / §9 / §13.1).

The Synthesizer re-derives the whole judgment by reasoning over the ASSEMBLED
RAW EVIDENCE — the actual step outputs — and writes the handoff briefing. It is
the second premium role (same tier as the Investigator, separate prompt).

THE HARD RULE: the answer is derived here from the raw numbers, never copied
from the navigational ledger. The nav state is the MAP (what was tested, what's
open, where it breaks) — it tells the Synthesizer where to look; it is not the
source of the answer. So the Synthesizer is given the raw (spec, output) pairs
as PRIMARY input and the nav ledger as the coverage map. The Investigator's
per-step "thinking" notes are deliberately withheld: feeding them back would let
the Synthesizer parrot a digest instead of re-deriving from evidence — the exact
failure this architecture removes.

G1 AS A HARD GATE: a null or single-uniform answer is only admissible if the
effect was examined within at least one stratification axis (nav.g1_satisfied()).
If G1 is unmet and a candidate axis still exists, the Synthesizer returns
NEEDS_MORE_WORK naming the axis, rather than finalizing a (possibly false) null.
The loop turns that into another investigation step. This is what would have
caught the EEDR false-null.

§13.1 EVIDENCE VOLUME: the Synthesizer reads raw outputs, which grow with the
run. assemble_evidence() uses the nav ledger as a SELECTION INDEX — steps that
the ledger marks load-bearing (referenced by a tested/in-progress frontier item,
an examined regime, or a breakdown entry) are always included in full; other
steps are kept full only if recent, else trimmed to a headline. This keeps real
signal while bounding context, and is the first concrete cut at the open §13.1
question — smarter selection can layer on later.
"""

from logger_config import get_logger

logger = get_logger(__name__)


from prompts import (SYNTHESIZER_SYSTEM, SYNTHESIZER_USER_TEMPLATE,
                     SYNTHESIZER_EXTENSION_NOTICE)


def assemble_evidence(log, nav, max_chars=120000, headline_chars=1500,
                      recent_n=4, hard_ceiling=24000):
    """Build the raw-evidence payload, using the nav ledger as a selection index
    (spec §13.1). This is the step where the ANSWER is derived, so completeness
    is paramount: budgets are generous and trimming is a last resort.

    Policy: recent steps and load-bearing steps (referenced by the nav map) are
    always full, subject only to a high per-step safety ceiling that no normal
    result reaches. Other steps are full too unless the total exceeds max_chars,
    in which case only those are trimmed to a headline, signposted.
    """
    steps = [e for e in log if not e.get("terminal")]
    if not steps:
        return "(no steps run)"

    load_bearing = set()
    for coll in (nav.frontier, nav.regimes, nav.breakdown):
        for e in coll:
            for s in getattr(e, "steps", []) or []:
                load_bearing.add(s)
    recent = {e["step"] for e in steps[-recent_n:]}

    def raw_of(e):
        if e.get("kind") == "search":
            return e.get("result") or "(no result)"
        if e.get("error"):
            return "[ERROR after %s attempt(s)]\n%s" % (e.get("attempts", "?"), e["error"])
        return e.get("stdout") or "(no output)"

    full_blocks = [(e["step"], e.get("spec", ""), raw_of(e),
                    e["step"] in load_bearing, e.get("kind")) for e in steps]
    total = sum(len(b[2]) for b in full_blocks)
    trim = total > max_chars
    if trim:
        logger.info("assemble_evidence: %d chars > budget %d; trimming only older "
                    "non-load-bearing, non-recent steps. Recent + load-bearing kept full.",
                    total, max_chars)

    out = []
    for step, spec, raw, lb, kind in full_blocks:
        if kind == "search":
            # External calibration context: always full, never trimmed.
            out.append(f"--- STEP {step} (WEB SEARCH) ---\nQUERY: "
                       f"{spec.replace('(web search) ', '')}\n"
                       f"FINDINGS (external, for calibration):\n{raw}\n")
            continue
        essential = lb or (step in recent)
        if essential:
            if len(raw) > hard_ceiling:
                raw = (raw[:hard_ceiling].rstrip()
                       + "\n[...exceeded per-step safety ceiling; unusually large dump...]")
            body = raw
        elif trim and len(raw) > headline_chars:
            body = (raw[:headline_chars].rstrip()
                    + "\n[...older non-load-bearing step trimmed to bound context; "
                      "see nav map. Recompute if these numbers matter to the verdict...]")
        else:
            body = raw if len(raw) <= hard_ceiling else (
                raw[:hard_ceiling].rstrip() + "\n[...safety ceiling; very large dump...]")
        tag = " (load-bearing)" if lb else ""
        out.append(f"--- STEP {step}{tag} ---\nANALYSIS: {spec}\nRAW OUTPUT:\n{body}\n")
    return "\n".join(out)


def _parse_synth(text):
    import re
    def block(name):
        m = re.search(rf"###\s*{name}\s*###\s*(.*?)(?=###|\Z)", text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""
    verdict_raw = block("VERDICT")
    briefing = block("BRIEFING")
    # Reasoning the model wrote before the VERDICT block — often substantial even
    # when it gates. Salvaged as a provisional briefing at the iteration ceiling.
    preamble = ""
    vpos = re.search(r"###\s*VERDICT\s*###", text, re.IGNORECASE)
    if vpos:
        preamble = text[:vpos.start()].strip()
    needs = "NEEDS_MORE_WORK" in verdict_raw.upper()
    reason = ""
    if needs:
        parts = verdict_raw.split(":", 1)
        reason = parts[1].strip() if len(parts) > 1 else "unspecified additional analysis"
    if briefing.lower().strip() in ("none", "n/a", ""):
        briefing = ""
    if not verdict_raw and not briefing:
        return {"verdict": "FINAL", "reason": "", "briefing": text.strip(), "preamble": ""}
    return {"verdict": "NEEDS_MORE_WORK" if needs else "FINAL",
            "reason": reason, "briefing": briefing, "preamble": preamble}


class Synthesizer:
    """Premium holistic re-derivation over raw evidence → verdict + briefing."""

    def __init__(self, client, model):
        self.client = client
        self.model = model

    def synthesize(self, seed, schema, log, nav, max_tokens=16000, final=False,
                   prior_seeds=None):
        g1 = nav.g1_satisfied(log)
        open_regimes = nav.open_regimes()
        untested = nav.untested_frontier()
        evidence = assemble_evidence(log, nav)
        nav_text = nav.render_for_investigator(log)

        g1_status = ("SATISFIED — at least one effect-modifier regime was examined within levels."
                     if g1 else
                     "NOT SATISFIED — no stratification axis has been examined within levels. "
                     "A null/uniform answer is NOT admissible while a candidate axis remains.")

        if prior_seeds:
            # On an extension, the briefing must answer ALL questions, so feed them
            # to the template co-equally (original first) rather than making the
            # latest seed the sole primary anchor.
            qs = [s for s in (list(prior_seeds) + [seed]) if s]
            seed_for_template = (
                "This investigation now spans multiple questions; the briefing must "
                "address them together, each in full:\n"
                + "\n".join(f"  {i}. {q}" for i, q in enumerate(qs, 1)))
        else:
            seed_for_template = seed

        user = SYNTHESIZER_USER_TEMPLATE.format(
            seed=seed_for_template, schema=schema, g1_status=g1_status,
            open_regimes=", ".join(open_regimes) if open_regimes else "(none)",
            untested_frontier=", ".join(untested) if untested else "(none)",
            nav=nav_text, evidence=evidence)
        if prior_seeds:
            user += SYNTHESIZER_EXTENSION_NOTICE
        if final:
            user += ("\n\nFINALIZATION NOTICE: this is the LAST synthesis — the iteration "
                     "ceiling has been reached and no further analysis will run. You MUST "
                     "produce a briefing (verdict FINAL); do NOT return NEEDS_MORE_WORK. If "
                     "the answer is not fully settled, mark the Summary clearly as PROVISIONAL "
                     "and list, in Open questions, the specific unexamined axes/confounds "
                     f"(open regimes: {', '.join(open_regimes) or 'none'}; untested framings: "
                     f"{', '.join(untested) or 'none'}) that could still change it.")

        from llm import build_cached_messages
        # Cache the (stable) system prompt; the evidence/user content is volatile.
        messages = build_cached_messages(self.model, SYNTHESIZER_SYSTEM, "", user)
        resp = self.client.call(messages, self.model, agent="Synthesizer",
                                max_tokens=max_tokens)
        result = _parse_synth(resp or "")

        if final:
            # At the ceiling we never gate; always return a briefing. If the model
            # still gated (or gave none), salvage its own pre-verdict reasoning so
            # the run always ends with a usable artifact.
            if not result["briefing"]:
                salvage = result.get("preamble") or "(synthesizer produced no briefing text)"
                result["briefing"] = (
                    "## Summary (PROVISIONAL — stopped at the iteration ceiling)\n\n"
                    "This investigation was halted before it fully resolved; the reading below "
                    "is the best the evidence supports so far and may change with the "
                    "analyses listed under Open questions.\n\n"
                    + salvage
                    + "\n\n## Open questions\n"
                    + (f"- Examine the effect within: {', '.join(open_regimes)}\n" if open_regimes else "")
                    + (f"- Untested framings: {', '.join(untested)}\n" if untested else "")
                    + (f"- {result['reason']}\n" if result.get("reason") else ""))
            result["verdict"] = "FINAL"
            return result

        # Non-final: backstop G1 — if it tried to finalize while G1 unmet with an
        # open axis, override to NEEDS_MORE_WORK so the loop does the work.
        if result["verdict"] == "FINAL" and not g1 and open_regimes:
            logger.warning("Synthesizer returned FINAL but G1 unmet with open regimes %s; "
                           "overriding to NEEDS_MORE_WORK.", open_regimes)
            result = {"verdict": "NEEDS_MORE_WORK",
                      "reason": f"examine the effect within one of: {', '.join(open_regimes)}",
                      "briefing": "", "preamble": result.get("preamble", "")}
        return result