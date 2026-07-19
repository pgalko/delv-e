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

import json
import os
import re

from logger_config import get_logger

logger = get_logger(__name__)


from prompts import (EDITOR_QUERIES_SYSTEM, EDITOR_QUERIES_TEMPLATE,
                     EDITOR_BRIEFING_TEMPLATE,
                     SYNTHESIZER_SYSTEM, SYNTHESIZER_USER_TEMPLATE,
                     SYNTHESIZER_EXTENSION_NOTICE, SYNTH_FORMAT_REPAIR, DATA_MODE)


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
    # a live run wrote "### Subsection" headers inside its output and the
    # generic ### terminator cut it at the first one. \b keeps e.g.
    # FINDINGS_SUMMARY from matching.
    _END = r"(?=###\s*(?:GATES|VERDICT|FINDINGS|CHARTS)\b|\Z)"
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
    findings = block("FINDINGS")
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
    if findings.lower().strip() in ("none", "n/a", ""):
        findings = ""
    if not findings:
        # Findings-marker salvage: if the model wrote the marker in ANY malformed
        # shape (wrong leading hashes, stray text on the marker line), recover the
        # content after it rather than dropping the technical record.
        m = re.search(r"#{2,}\s*FINDINGS[^\n]*\n?", text, re.IGNORECASE)
        if m:
            salvage = text[m.end():]
            cut = re.search(r"###\s*(GATES|VERDICT)\s*#*", salvage, re.IGNORECASE)
            if cut:
                salvage = salvage[:cut.start()]
            salvage = salvage.strip()
            if salvage.lower() not in ("none", "n/a", ""):
                findings = salvage
    charts = _parse_charts(block("CHARTS"))
    if not verdict_raw:
        # FAIL CLOSED. A missing ###VERDICT### block is a protocol failure, not
        # evidence that the work is complete. The old fallback promoted the raw
        # response text to `findings` under a clean FINAL verdict, so a
        # completely malformed synthesis (reasoning prose, a refusal, an error
        # message) could become the published technical record verbatim.
        # Instead: report MODEL_ERROR, keep whatever findings DID parse (a model
        # that emitted ###FINDINGS### but forgot the verdict keeps its work),
        # and stash the whole text as the preamble so the finalization salvage
        # can still use it — under its honest PROVISIONAL banner, never as
        # clean findings. Synthesizer.synthesize retries the format once and
        # then fails toward the existing salvage nets.
        return {"verdict": "MODEL_ERROR", "reason": "missing ###VERDICT### block",
                "findings": findings, "preamble": preamble or text.strip(),
                "gates_review": gates, "charts": charts}
    return {"verdict": "NEEDS_MORE_WORK" if needs else "FINAL",
            "reason": reason, "findings": findings, "preamble": preamble,
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
        charts.append({"name": name, "finding": _field("FINDING").upper(),
                       "caption": _field("CAPTION"), "spec": spec})
        if len(charts) >= MAX_CHARTS:
            break
    return charts


def parse_findings(text):
    """The technical record as structured findings. Tolerant by design: a finding
    counts once it has a header and a CLAIM, because a malformed CAVEATS line must
    never make a decisive finding invisible to the coverage gate."""
    heads = list(re.finditer(r"(?im)^\s*(F\d+)\s*\|\s*(decisive|supporting)\s*$", text or ""))
    out = []
    for k, m in enumerate(heads):
        end = heads[k + 1].start() if k + 1 < len(heads) else len(text)
        body = text[m.end():end]

        def field(name, _b=body):
            fm = re.search(rf"(?ims)^\s*{name}:\s*(.*?)(?=^\s*(?:CLAIM|NUMBERS|CAVEATS):|\Z)", _b)
            return fm.group(1).strip() if fm else ""

        claim = field("CLAIM")
        if not claim:
            continue
        out.append({"id": m.group(1).upper(), "strength": m.group(2).lower(),
                    "claim": claim, "numbers": field("NUMBERS"),
                    "caveats": field("CAVEATS")})
    return out


def technical_document(result):
    """The technical briefing as it lands on disk and as --verify reads it: the
    verdict, the gate reasoning, and the findings."""
    parts = [f"# Technical briefing\n\nVERDICT: {result.get('verdict', '?')}"]
    if result.get("gates_review"):
        parts.append("## Gates\n\n" + result["gates_review"].strip())
    parts.append("## Findings\n\n" + (result.get("findings") or "(none)").strip())
    return "\n\n".join(parts) + "\n"


# ── The three gates. They live in the harness, not in a prompt, because the
# deliverable now sits downstream of an extra call and its failures would
# otherwise be invisible: nobody reads both artifacts.

def check_coverage(briefing, findings):
    """Decisive findings the editor never carried. It cites findings by id, so a
    dropped one is detectable; a silently missing decisive finding is the whole
    hazard of splitting synthesis in two."""
    text = briefing or ""
    return [f for f in findings if f["strength"] == "decisive"
            and not re.search(rf"\b{f['id']}\b", text, re.I)]


_STAT_RE = re.compile(r"\d+\.\d{2,}")


def _strip_urls(text):
    """URLs are not numbers. A DOI (10.1249/01.mss.0000385042.39350.c3) parses as
    several decimals and made the numbers gate cry wolf on every cited briefing,
    which teaches you to ignore it."""
    return _URL_RE.sub(" ", text or "")


def _decimals(text):
    out = []
    for h in _STAT_RE.findall(_strip_urls(text)):
        try:
            out.append(float(h))
        except ValueError:
            pass
    return out


def check_numbers(briefing, technical, literature=""):
    """Statistics in the deliverable that appear in neither the technical record nor
    the literature the editor was given. Only decimals are checked: a plain-language
    "7% slower" is a translation the technical CLAIM already owns, while a fabricated
    estimate or interval always carries decimals. Rounding is tolerated to the
    briefing's own precision. The literature counts as a source, because quoting a
    published value is exactly what the editorial pass is for."""
    have = _decimals(technical) + _decimals(literature)
    bad = []
    for n in _STAT_RE.findall(_strip_urls(briefing)):
        try:
            v = float(n)
        except ValueError:
            continue
        tol = 0.5 * 10 ** -len(n.split(".")[1])
        if not any(abs(v - h) <= tol for h in have):
            bad.append(n)
    return sorted(set(bad))


_URL_RE = re.compile(r"https?://[^\s)>\]]+")


def _norm_url(u):
    return (u or "").rstrip("/.,;)").strip()


def strip_unverified_citations(briefing, fetched):
    """Replace any markdown link whose URL was never fetched with its own text.
    A model asked for references will invent plausible ones, and a reader cannot
    tell an invented citation from a real one, so this is not optional."""
    ok = {_norm_url(u) for u in (fetched or set())}

    def sub(m):
        return m.group(0) if _norm_url(m.group(2)) in ok else m.group(1)

    out = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", sub, briefing or "")
    # bare URLs that were never fetched go too
    return _URL_RE.sub(lambda m: m.group(0) if _norm_url(m.group(0)) in ok else "", out)


def render_chart_markers(briefing, charts, produced):
    """Turn [[CHART:F3]] into the image link for the chart tied to F3. The editor
    says WHERE, the harness says WHAT: markers for charts that failed or do not
    exist are removed rather than left as broken links, and a produced chart the
    editor never placed is appended so a rendered chart is never lost."""
    by_finding = {}
    for c in charts:
        if c["name"] in produced and c.get("finding"):
            by_finding.setdefault(c["finding"].upper(), c)
    placed = set()

    def sub(m):
        c = by_finding.get(m.group(1).upper())
        if not c:
            return ""
        placed.add(c["name"])
        cap = c.get("caption") or c["name"][:-4].replace("_", " ")
        return f"![{cap}](charts/{c['name']})"

    out = re.sub(r"\[\[\s*CHART:\s*(F\d+)\s*\]\]", sub, briefing or "", flags=re.I)
    tail = [c for c in charts if c["name"] in produced and c["name"] not in placed]
    for c in tail:
        cap = c.get("caption") or c["name"][:-4].replace("_", " ")
        out = out.rstrip() + f"\n\n![{cap}](charts/{c['name']})"
    return out


def format_sources(sources):
    """The literature as a numbered registry the editor cites by marker: [S1], [S2].

    Same contract as the charts: the model says WHICH source, the harness says what
    it is called. The editor never writes a link and never names an author, because
    it cannot do so reliably -- a live briefing credited a preprint to "Addis et al."
    after reading "Addis Ababa University" on the page -- and an invented attribution
    riding a real, fetched URL passes any URL check."""
    if not sources:
        return "(no literature was retrieved)"
    out = []
    for s in sources:
        block = f"[{s['id']}] {s['title']}\n{s['url']}"
        if s.get("content"):
            block += f"\n{s['content']}"
        out.append(block)
    return "\n\n".join(out)


def render_citations(briefing, sources):
    """Turn [S1] markers into a numbered reference list. Markers naming a source that
    does not exist are removed rather than left dangling."""
    by_id = {s["id"]: s for s in sources}
    used = []

    def sub(m):
        s = by_id.get(m.group(1).upper())
        if not s:
            return ""
        if s not in used:
            used.append(s)
        return f"[{s['id']}]"

    out = re.sub(r"\[\s*(S\d+)\s*\]", sub, briefing or "")
    if used:
        out = out.rstrip() + "\n\n## References\n\n" + "\n".join(
            f"{s['id']}. [{s['title']}]({s['url']})" for s in used)
    return out


_ATTRIB_RE = re.compile(r"\b([A-Z][a-z]+)\s+et\s+al\b")


def check_attributions(briefing):
    """Author attributions the editor invented. It is given titles and URLs, never
    author lists, so any "X et al." is a guess, and it cannot be verified against
    anything the harness fetched."""
    return sorted(set(_ATTRIB_RE.findall(briefing or "")))


def charts_for_editor(charts, produced):
    """What the editor is told about the charts: only the ones that exist."""
    live = [c for c in charts if c["name"] in produced]
    if not live:
        return "(no charts were produced)"
    return "\n".join(f"[[CHART:{c.get('finding') or '?'}]] -> {c['name']}: "
                      f"{c.get('caption') or ''}".rstrip() for c in live)


def load_chart_manifest(output_dir):
    """The charts a completed run produced, reloaded from disk. Lets a later pass
    (a --verify re-render) place the same charts without re-running them."""
    path = os.path.join(output_dir, "charts", "manifest.json")
    try:
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
    except (OSError, ValueError):
        return [], set()
    charts = [{"name": r["chart"], "finding": (r.get("finding") or "").upper(),
               "caption": r.get("caption") or "", "spec": ""}
              # The writer records success under "produced" (see _render_charts);
              # the old reader looked up "ok" — a key never written — whose
              # default of True marked every FAILED chart as available, so a
              # --verify re-render could place markers for images that do not
              # exist. Read the real key; accept legacy "ok" if some old manifest
              # has it; with neither, fail CLOSED (no phantom charts).
              for r in rows if r.get("chart") and r.get("produced", r.get("ok", False))]
    return charts, {c["name"] for c in charts}


class Editor:
    """Renders the technical findings into the deliverable.

    Sees the seed, the findings, the charts and the retrieved literature, and
    never the raw evidence: it cannot re-adjudicate, only render. That is the
    point of the split. The standard of proof governed the pass that produced the
    findings; this pass may not introduce a claim, a number, or a direction of
    its own, and the harness gates check that it did not."""

    def __init__(self, client, model, prompts=None, reasoning_effort="medium"):
        self.client = client
        self.model = model
        self.p = prompts or DATA_MODE
        self.reasoning_effort = reasoning_effort

    def queries(self, seed, technical, budget):
        """Up to `budget` literature searches, chosen from the findings.

        Runs on the shared token budget, not a tight cap: a reasoning model given
        1000 tokens spent all of them thinking and emitted nothing, costing a
        wasted call and a ladder retry on every run. The output is a handful of
        lines and anything past the block is ignored, so headroom is free."""
        from llm import call_with_ladder
        msg = EDITOR_QUERIES_TEMPLATE.format(seed=seed, technical=technical, budget=budget)
        resp, _ = call_with_ladder(
            self.client, [{"role": "system", "content": EDITOR_QUERIES_SYSTEM},
                          {"role": "user", "content": msg}],
            self.model, agent="Editor", max_tokens=None,
            reasoning_effort=self.reasoning_effort)
        m = re.search(r"###\s*QUERIES\s*###\s*(.*?)(?=\n###|\Z)", resp or "", re.S | re.I)
        if not m:
            logger.warning("Editor emitted no QUERIES block; no literature will be "
                           "retrieved. It answered with %d chars of something else.",
                           len(resp or ""))
            return []
        out = []
        for ln in m.group(1).splitlines():
            q = ln.strip().lstrip("-*0123456789. ").strip()
            if q and len(q) > 3:
                out.append(q)
        return out[:budget]

    def write(self, seed, technical, charts_text, literature, max_tokens=None,
              directive=None):
        from llm import call_with_ladder
        msg = EDITOR_BRIEFING_TEMPLATE.format(
            seed=seed, technical=technical, charts=charts_text,
            literature=literature or "(no literature was retrieved)")
        if directive:
            msg += "\n\n" + directive
        resp, _ = call_with_ladder(
            self.client, [{"role": "system", "content": self.p.editor_system},
                          {"role": "user", "content": msg}],
            self.model, agent="Editor", max_tokens=max_tokens,
            reasoning_effort=self.reasoning_effort)
        return (resp or "").strip()


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

        if result["verdict"] == "MODEL_ERROR":
            # The response had no ###VERDICT### block (see _parse_synth: this is
            # a protocol failure, never promoted to FINAL). Retry ONCE with the
            # format-repair notice appended, at the configured effort — the fix
            # is the format, but the model must re-derive, so thinking stays on.
            # Mirrors the Investigator's format retry in run_investigation.
            logger.warning("Synthesizer response had no ###VERDICT### block; "
                           "retrying once with a format-repair notice.")
            repair_messages = build_cached_messages(self.model, self.p.synth_system,
                                                    "", user + SYNTH_FORMAT_REPAIR)
            resp, _meta = call_with_ladder(self.client, repair_messages, self.model,
                                           agent="Synthesizer", max_tokens=max_tokens,
                                           reasoning_effort=self.reasoning_effort)
            repaired = _parse_synth(resp or "")
            repaired["format_repaired"] = True
            if repaired["verdict"] != "MODEL_ERROR":
                result = repaired
            else:
                # Still no verdict after the repair. Fail toward the existing
                # salvage nets rather than inventing a verdict path of our own:
                # keep whichever attempt yielded parseable FINDINGS (possibly
                # neither), and hand the loop a FINAL whose findings are ONLY
                # what actually parsed. Empty findings then trigger either the
                # final-mode coercion below (PROVISIONAL-bannered salvage from
                # the preamble) or the loop's finalization-mode re-run — the raw
                # prose can no longer stand in as clean findings.
                logger.warning("Synthesizer still returned no ###VERDICT### after "
                               "the format repair; failing toward the salvage nets.")
                best = repaired if repaired.get("findings") else result
                result = {"verdict": "FINAL",
                          "reason": "synthesizer protocol failure: no ###VERDICT### "
                                    "block after one format repair",
                          "findings": best.get("findings", ""),
                          "preamble": best.get("preamble", ""),
                          "gates_review": best.get("gates_review", ""),
                          "charts": best.get("charts", []),
                          "format_repaired": True}

        if final:
            # At the ceiling we never gate; always return a briefing. If the model
            # still gated (or gave none), salvage its own pre-verdict reasoning so
            # the run always ends with a usable artifact.
            if not result["findings"]:
                salvage = result.get("preamble") or "(synthesizer produced no briefing text)"
                if compute:
                    tail = (f"\n\n## Open questions\n- {result['reason']}\n"
                            if result.get("reason") else "")
                else:
                    tail = ("\n\n## Open questions\n"
                            + (f"- Examine the effect within: {', '.join(open_regimes)}\n" if open_regimes else "")
                            + (f"- Untested framings: {', '.join(untested)}\n" if untested else "")
                            + (f"- {result['reason']}\n" if result.get("reason") else ""))
                result["findings"] = (
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
                      "briefing": "", "preamble": result.get("preamble", ""),
                      "format_repaired": result.get("format_repaired", False)}
        return result