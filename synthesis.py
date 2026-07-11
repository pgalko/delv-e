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
                     SYNTHESIZER_EXTENSION_NOTICE, DATA_MODE)


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
    # Blocks terminate ONLY at the next NAMED marker, never at arbitrary "###":
    # a live run wrote "### Subsection" headers inside its briefing and the
    # generic ### terminator cut the deliverable at the first one. \b keeps
    # e.g. BRIEFINGS from matching.
    _END = r"(?=###\s*(?:GATES|VERDICT|BRIEFING|CHARTS)\b|\Z)"
    def block(name):
        # Marker tolerance: a model occasionally malforms the trailing hashes
        # (a live glm run emitted "###BRIEFING##" and the exact-match regex
        # silently discarded a complete briefing). Require the leading ### and
        # the name; accept any number of trailing hashes ON THE SAME LINE, so a
        # zero-hash marker cannot swallow the content's own markdown headers.
        m = re.search(rf"###\s*{name}[ \t]*#*\s*(.*?){_END}", text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""
    gates = block("GATES")
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
    if not briefing:
        # Briefing-marker salvage: if the model wrote a briefing marker in ANY
        # malformed shape (wrong leading hashes, stray text on the marker line),
        # recover the content after it rather than dropping the deliverable.
        m = re.search(r"#{2,}\s*BRIEFING[^\n]*\n?", text, re.IGNORECASE)
        if m:
            salvage = text[m.end():]
            cut = re.search(r"###\s*(GATES|VERDICT)\s*#*", salvage, re.IGNORECASE)
            if cut:
                salvage = salvage[:cut.start()]
            salvage = salvage.strip()
            if salvage.lower() not in ("none", "n/a", ""):
                briefing = salvage
    charts = _parse_charts(block("CHARTS"))
    if not verdict_raw and not briefing:
        return {"verdict": "FINAL", "reason": "", "briefing": text.strip(),
                "preamble": "", "gates_review": gates, "charts": charts}
    return {"verdict": "NEEDS_MORE_WORK" if needs else "FINAL",
            "reason": reason, "briefing": briefing, "preamble": preamble,
            "gates_review": gates, "charts": charts}


MAX_CHARTS = 3


def sanitize_chart_name(raw):
    """A chart filename the harness will trust: basename only, lowercase,
    [a-z0-9_] with runs of anything else collapsed to one underscore, forced
    .png. Returns None when nothing survives. Path mechanics never rest on
    model output: the kernel's patched savefig additionally basenames and
    routes every save into the step's analysis_dir."""
    import os as _os
    import re as _re
    stem = _os.path.basename((raw or "").strip()).lower()
    stem = _re.sub(r"\.png$", "", stem)
    stem = _re.sub(r"[^a-z0-9_]+", "_", stem).strip("_")
    return f"{stem}.png" if stem else None


def _parse_charts(block_text):
    """CHART entries from the ###CHARTS### block: tolerant of case and
    spacing, names sanitized, duplicates dropped (first wins), capped at
    MAX_CHARTS. Each entry is CHART/SECTION/CAPTION/SPEC (SECTION and CAPTION
    optional); returns a list of dicts with keys name, section, caption, spec."""
    import re as _re
    if not (block_text or "").strip():
        return []
    charts, seen = [], set()
    for chunk in _re.split(r"(?im)^\s*CHART:\s*", block_text)[1:]:
        lines = chunk.splitlines()
        name = sanitize_chart_name(lines[0] if lines else "")
        rest = "\n".join(lines[1:])
        m = _re.search(r"(?ims)^\s*SPEC:\s*(.*)\Z", rest)
        spec = m.group(1).strip() if m else ""
        def _field(f, _rest=rest):
            fm = _re.search(rf"(?im)^\s*{f}:\s*(.*)$", _rest)
            return fm.group(1).strip() if fm else ""
        if not name or not spec or name in seen:
            continue
        seen.add(name)
        charts.append({"name": name, "section": _field("SECTION"),
                       "caption": _field("CAPTION"), "spec": spec})
        if len(charts) >= MAX_CHARTS:
            break
    return charts


def apply_chart_results(briefing, charts, produced):
    """Place produced charts deterministically: placement is the HARNESS's job,
    not the model's (a live glm run dumped every image link after the last
    header despite instructions, so the contract moved to SECTION fields).
    Every standalone image line the model wrote is stripped; each produced
    chart is inserted at the end of the section its SECTION field names
    (fuzzy, case-insensitive header match), or appended at the end when the
    section is missing. A chart that failed simply does not appear, so the
    shipped briefing never carries a broken link."""
    import re as _re
    img_line = _re.compile(r"\s*!\[[^\]]*\]\([^)]+\)\s*$")
    lines = [ln for ln in briefing.split("\n") if not img_line.match(ln)]

    def _norm(h):
        return _re.sub(r"[^a-z0-9 ]+", "", (h or "").lower()).strip()

    headers = [(i, _norm(_re.sub(r"^#+", "", ln)))
               for i, ln in enumerate(lines) if _re.match(r"\s*##+\s", ln)]
    inserts = {}
    for c in charts:
        if c["name"] not in produced:
            continue
        cap = c.get("caption") or c["name"][:-4].replace("_", " ")
        img = f"![{cap}](charts/{c['name']})"
        target = None
        sec = _norm(c.get("section"))
        if sec:
            for k, (i, h) in enumerate(headers):
                if sec in h or h in sec:
                    target = headers[k + 1][0] if k + 1 < len(headers) else len(lines)
                    break
        if target is None:
            target = len(lines)
        inserts.setdefault(target, []).append(img)

    out = []
    for i, ln in enumerate(lines):
        for img in inserts.get(i, ()):
            if out and out[-1].strip():
                out.append("")
            out.append(img)
            out.append("")
        out.append(ln)
    for img in inserts.get(len(lines), ()):
        if out and out[-1].strip():
            out.append("")
        out.append(img)
    return "\n".join(out)


class Synthesizer:
    """Premium holistic re-derivation over raw evidence → verdict + briefing."""

    def __init__(self, client, model, prompts=None, reasoning_effort="medium"):
        self.client = client
        self.model = model
        self.p = prompts or DATA_MODE
        self.reasoning_effort = reasoning_effort

    def synthesize(self, seed, schema, log, nav, max_tokens=None, final=False,
                   prior_seeds=None, registry_text=None):
        # Synthesis uses the shared DEFAULT_MAX_TOKENS budget (via call_with_ladder
        # below). A reasoning synthesizer on Ollama can otherwise spend its whole
        # budget on hidden reasoning and emit no briefing (seen live); call_with_ladder
        # retries once with reasoning off on a capped turn instead of relying on the
        # cap alone, and Anthropic's non-streaming limit is clamped inside llm.call.
        compute = self.p.compute
        evidence = assemble_evidence(log, nav)
        if registry_text:
            # For CHART specs: lets the Synthesizer reference the run's live
            # derived objects by registry name instead of re-deriving from df.
            evidence += ("\n\n=== CURRENT NAMESPACE REGISTRY "
                         "(reference these objects by name in CHART specs) ===\n"
                         + registry_text)
        nav_text = nav.render_for_investigator(log)

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

        if compute:
            # Compute mode has no effect/regime/confound, so the statistical gates
            # do not apply. The compute synthesizer self-checks uncertainty,
            # convergence, and validity instead (see its system prompt).
            g1, open_regimes, untested = True, [], []
            user = self.p.synth_user.format(
                seed=seed_for_template, nav=nav_text, evidence=evidence)
        else:
            g1 = nav.g1_satisfied(log)
            open_regimes = nav.open_regimes()
            untested = nav.untested_frontier()
            g1_status = ("SATISFIED — at least one effect-modifier regime was examined within levels."
                         if g1 else
                         "NOT SATISFIED — no stratification axis has been examined within levels. "
                         "A null/uniform answer is NOT admissible while a candidate axis remains.")
            user = self.p.synth_user.format(
                seed=seed_for_template, schema=schema, g1_status=g1_status,
                open_regimes=", ".join(open_regimes) if open_regimes else "(none)",
                untested_frontier=", ".join(untested) if untested else "(none)",
                nav=nav_text, evidence=evidence)
        if prior_seeds:
            user += SYNTHESIZER_EXTENSION_NOTICE
        if final:
            notice = ("\n\nFINALIZATION NOTICE: this is the LAST synthesis — the iteration "
                      "ceiling has been reached and no further analysis will run. You MUST "
                      "produce a briefing (verdict FINAL); do NOT return NEEDS_MORE_WORK. If "
                      "the answer is not fully settled, mark the Summary clearly as PROVISIONAL")
            if compute:
                notice += " and list any remaining computations under Open questions."
            else:
                notice += (" and list, in Open questions, the specific unexamined axes/confounds "
                           f"(open regimes: {', '.join(open_regimes) or 'none'}; untested framings: "
                           f"{', '.join(untested) or 'none'}) that could still change it.")
            user += notice

        from llm import build_cached_messages, call_with_ladder
        # Cache the (stable) system prompt; the evidence/user content is volatile.
        messages = build_cached_messages(self.model, self.p.synth_system, "", user)
        resp, _meta = call_with_ladder(self.client, messages, self.model,
                                       agent="Synthesizer", max_tokens=max_tokens,
                                       reasoning_effort=self.reasoning_effort)
        result = _parse_synth(resp or "")

        if final:
            # At the ceiling we never gate; always return a briefing. If the model
            # still gated (or gave none), salvage its own pre-verdict reasoning so
            # the run always ends with a usable artifact.
            if not result["briefing"]:
                salvage = result.get("preamble") or "(synthesizer produced no briefing text)"
                if compute:
                    tail = (f"\n\n## Open questions\n- {result['reason']}\n"
                            if result.get("reason") else "")
                else:
                    tail = ("\n\n## Open questions\n"
                            + (f"- Examine the effect within: {', '.join(open_regimes)}\n" if open_regimes else "")
                            + (f"- Untested framings: {', '.join(untested)}\n" if untested else "")
                            + (f"- {result['reason']}\n" if result.get("reason") else ""))
                result["briefing"] = (
                    "## Summary (PROVISIONAL — stopped at the iteration ceiling)\n\n"
                    "This investigation was halted before it fully resolved; the reading below "
                    "is the best the evidence supports so far and may change with the "
                    "analyses listed under Open questions.\n\n"
                    + salvage + tail)
            result["verdict"] = "FINAL"
            return result

        # Non-final: backstop G1 (data mode only) — if it tried to finalize while
        # G1 unmet with an open axis, override to NEEDS_MORE_WORK so the loop does
        # the work. Compute mode has no such gate.
        if not compute and result["verdict"] == "FINAL" and not g1 and open_regimes:
            logger.warning("Synthesizer returned FINAL but G1 unmet with open regimes %s; "
                           "overriding to NEEDS_MORE_WORK.", open_regimes)
            result = {"verdict": "NEEDS_MORE_WORK",
                      "reason": f"examine the effect within one of: {', '.join(open_regimes)}",
                      "briefing": "", "preamble": result.get("preamble", "")}
        return result