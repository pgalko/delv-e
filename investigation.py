"""
Inverted-core investigation loop for delv-e (step 2).

Three roles, two model tiers (see delve_rebuild_spec.md):

  Investigator (premium) — does ALL analytical thinking. Reads RAW results,
    integrates them holistically, decides the next move, and COMPILES that move
    into an analytically-closed spec for the executor. Maintains the running
    reasoning trail. Bound by guardrails G1 (test a regime stratification before
    declaring an effect null/uniform) and G2 (a variable's role is not fixed).

  Executor (cheap) — zero analytical latitude. Translates one closed spec into
    pandas against the PERSISTENT kernel namespace, runs it, returns raw output.
    Mechanical retry on error (traceback in, fixed code out) — no judgment, so
    it stays cheap.

  Synthesizer (premium) — periodic + final holistic re-derivation over assembled
    raw evidence. NOT in this module yet (step 4).

This module deliberately does NOT live in the old auto_explore.py. That file is
a rebuild target; this is the rebuilt core. The prompts are inline constants for
now so step 2 is self-contained and testable; they migrate into the rebuilt
prompts.py later.

The carried context in step 2 is the investigation log itself (each step's
spec + RAW output + the Investigator's own integration notes). The formal
navigational-state schema (spec §7) layers on top in step 3. The invariant held
here already: the Investigator re-reasons over the actual raw outputs every
turn — never over a digest.
"""

import json
import os
import re

from executor import extract_code
from kernel import PersistentKernel
from nav_state import NavState
from synthesis import Synthesizer
from logger_config import get_logger

logger = get_logger(__name__)
# Prompts now live in prompts.py (single source of truth; see the prompt audit).
from prompts import (
    INVESTIGATOR_SYSTEM, INVESTIGATOR_HEAD_TEMPLATE, INVESTIGATOR_TAIL_TEMPLATE,
    EXECUTOR_SYSTEM, EXECUTOR_USER_TEMPLATE, EXECUTOR_RETRY_TEMPLATE,
    EXECUTOR_TRUNCATION_RETRY, DIRECTIVE_G1_GATE, DIRECTIVE_SYNTH_GATE,
    DIRECTIVE_MIDPOINT, DIRECTIVE_EXTEND_LEDGER,
    SEARCH_INVESTIGATOR_INSTRUCTION, SEARCH_MIDSTREAM_TEMPLATE,
    DIRECTIVE_SEARCH_SPENT, DIRECTIVE_SEARCH_FAILED,
)


# Words that signal an unresolved analytical choice leaking into a spec.
# Used for auditing (warn, not block) per spec §6 / §13.4.
_LEAKAGE_WORDS = [
    "appropriate", "best", "robust", "handle", "clean", "reasonable",
    "meaningful", "optimal", "sensible", "if it looks like", "as needed",
    "etc.", "and so on",
]


def scan_spec_for_leakage(spec):
    """Return a list of banned words present in a spec (closure-rule audit)."""
    low = spec.lower()
    return [w for w in _LEAKAGE_WORDS if w in low]


def _decision_from_status(raw):
    """Map a ###STATUS### block to a decision: CONTINUE, SYNTHESIZE, or SEARCH.

    The Investigator is asked to emit one word, but a model sometimes wraps it in
    prose ("CONTINUE, not ready to synthesize yet" or "CONTINUE, no search needed").
    A naive substring test for SYNTH/SEARCH misfires on exactly those, silently
    turning a CONTINUE into a premature finalize or a spurious search. So the
    leading token decides; only if it is not one of the three verbs do we fall back
    to a whole-word scan, and CONTINUE wins any tie. The safe default is always to
    keep investigating rather than finalize or branch on an ambiguous signal.
    """
    s = (raw or "").strip().upper()
    if not s:
        return "CONTINUE"
    tokens = re.findall(r"[A-Z]+", s)
    lead = tokens[0] if tokens else ""
    if lead.startswith("CONTINU"):
        return "CONTINUE"
    if lead.startswith("SYNTH"):
        return "SYNTHESIZE"
    if lead == "SEARCH":
        return "SEARCH"
    # Leading token is not a decision verb (prose-wrapped). Prefer CONTINUE if it
    # appears at all; never finalize or search on an ambiguous, verb-less status.
    if re.search(r"\bCONTINU\w*", s):
        return "CONTINUE"
    if re.search(r"\bSYNTH\w*", s):
        return "SYNTHESIZE"
    if re.search(r"\bSEARCH\b", s):
        return "SEARCH"
    return "CONTINUE"


def _parse_investigator(text):
    """Parse the Investigator's three-block output. Tolerant of minor drift."""
    def block(name):
        m = re.search(rf"###\s*{name}\s*###\s*(.*?)(?=###|\Z)", text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""
    thinking = block("THINKING")
    status = _decision_from_status(block("STATUS"))
    spec = block("SPEC")
    ledger = block("LEDGER")
    search = block("QUERY")
    estimand = block("ESTIMAND")
    # Optional: steps the Investigator wants pulled back to full raw next turn.
    rehydrate = [int(n) for n in re.findall(r"\d+", block("REHYDRATE"))]
    if spec.lower().strip() in ("none", "n/a", ""):
        spec = ""
    if estimand.lower().strip() in ("none", "n/a", ""):
        estimand = ""
    # Fallback: if no markers at all, treat whole text as thinking and stop.
    if not thinking and not spec:
        thinking = text.strip()
        status = "SYNTHESIZE"
    return {"thinking": thinking, "status": status, "spec": spec,
            "ledger": ledger, "rehydrate": rehydrate, "search": search,
            "estimand": estimand}


def _write_step_artifact(analysis_dir, entry, iteration, max_steps):
    """Write a human-readable per-step record (the analytical move, the rationale,
    the code, and the raw output) to <exploration>/<NN>/analysis.md. This is a
    WRITE-ONLY audit artifact — it never enters any model's context, so it has no
    token cost; it just makes each step easy to review on disk. Plots (if any) land
    in the same folder."""
    out = entry.get("stdout")
    body = out if out else f"[error after {entry.get('attempts','?')} attempt(s)]\n{entry.get('error','')}"
    md = (
        f"# Step {entry['step']}\n\n"
        f"| Field | Value |\n|-------|-------|\n"
        f"| Step | {entry['step']} |\n"
        f"| Iteration | {iteration} of {max_steps} |\n"
        f"| Attempts | {entry.get('attempts','?')} |\n"
        f"| Status | {'error' if entry.get('error') else 'ok'} |\n\n"
        f"## Spec (closed instruction to the executor)\n\n{entry.get('spec','')}\n\n"
        f"## Rationale (the Investigator's reasoning)\n\n{entry.get('thinking','')}\n\n"
        f"## Code\n\n```python\n{entry.get('code','')}\n```\n\n"
        f"## Output\n\n```\n{body}\n```\n"
    )
    try:
        os.makedirs(analysis_dir, exist_ok=True)
        with open(os.path.join(analysis_dir, "analysis.md"), "w", encoding="utf-8") as f:
            f.write(md)
    except OSError as exc:
        logger.warning("Could not write step artifact to %s: %s", analysis_dir, exc)


class Executor:
    """Cheap coder: closed spec -> pandas -> run in kernel -> raw output.
    Mechanical retry on error (no judgment)."""

    def __init__(self, client, model, max_retries=2, max_tokens=20000):
        self.client = client
        self.model = model
        self.max_retries = max_retries
        self.max_tokens = max_tokens

    def run(self, spec, kernel, registry_text, analysis_dir=None):
        """Returns dict: {code, stdout, error, attempts}. Never raises on a bad
        executor response — if the model fails to produce runnable code, that is
        returned as an error dict so the Investigator can see it and adapt."""
        messages = [
            {"role": "system", "content": EXECUTOR_SYSTEM},
            {"role": "user", "content": EXECUTOR_USER_TEMPLATE.format(
                spec=spec, registry=registry_text)},
        ]
        code = ""
        stdout = None
        error = None
        for attempt in range(self.max_retries + 1):
            resp = self.client.call(messages, self.model, agent="Executor",
                                    max_tokens=self.max_tokens)
            code = extract_code(resp or "")
            if not code:
                # Most common cause: a reasoning model spent its whole token budget
                # on chain-of-thought and emitted no (or a truncated) code block.
                error = ("executor returned no runnable ```python``` code block "
                         f"(likely truncated after {self.max_tokens} output tokens of reasoning)")
                messages.append({"role": "assistant", "content": (resp or "")[-2000:]})
                messages.append({"role": "user", "content": EXECUTOR_TRUNCATION_RETRY})
                continue
            stdout, error, _plots = kernel.execute(code, analysis_dir=analysis_dir)
            if not error:
                return {"code": code, "stdout": stdout, "error": None, "attempts": attempt + 1}
            # Mechanical retry with the traceback.
            messages.append({"role": "assistant", "content": resp})
            messages.append({"role": "user", "content":
                             EXECUTOR_RETRY_TEMPLATE.format(traceback=error)})
        return {"code": code, "stdout": stdout,
                "error": error or "executor failed to produce runnable code",
                "attempts": self.max_retries + 1}


class Investigator:
    """Premium thinker: reads raw evidence, integrates, decides + compiles next step."""

    def __init__(self, client, model, search_enabled=False, search_budget=0):
        self.client = client
        self.model = model
        self.search_enabled = search_enabled
        self.search_budget = search_budget

    def decide(self, seed, schema, registry_text, log, nav, directive=None, rehydrate=None):
        nav_text = nav.render_for_investigator(log)
        head = INVESTIGATOR_HEAD_TEMPLATE.format(seed=seed, schema=schema)
        # Tiered history: recent + live-thread (protected) + rehydrated stay full;
        # older closed-thread steps collapse to a headline + pointer.
        blocks = _history_blocks(log, protected=nav.protected_steps(),
                                 forced_full=set(rehydrate or []))
        # Search instruction lives in the cached prefix (static for the run) so it
        # adds no per-turn cost; present only when search is enabled.
        if self.search_enabled:
            stable_blocks = [head,
                             SEARCH_INVESTIGATOR_INSTRUCTION.format(budget=self.search_budget)] + blocks
        else:
            stable_blocks = [head] + blocks
        tail = INVESTIGATOR_TAIL_TEMPLATE.format(registry=registry_text, nav=nav_text)
        if directive:
            tail += ("\n\nDIRECTIVE (act on this now): " + directive)
        from llm import build_cached_messages
        messages = build_cached_messages(self.model, INVESTIGATOR_SYSTEM, stable_blocks, tail)
        # 16k so a thinking model's reasoning AND its structured decision both fit;
        # at 8k a long chain-of-thought could exhaust the budget before emitting any
        # markers, yielding an empty turn. `truncated` lets the loop retry instead of
        # mis-reading an empty/cut-off turn as a decision.
        try:
            resp, meta = self.client.call(messages, self.model, agent="Investigator",
                                          max_tokens=16000, return_meta=True)
        except TypeError:
            # Client predates return_meta (e.g. a lightweight test stub): fall back to
            # text-only; truncation can then only be inferred from an empty response.
            resp = self.client.call(messages, self.model, agent="Investigator",
                                    max_tokens=16000)
            meta = {}
        decision = _parse_investigator(resp or "")
        decision["incomplete"] = (not (resp or "").strip()) or bool(meta.get("truncated"))
        return decision


def _step_block(e, full=True, hard_ceiling=20000):
    """Render ONE completed step as a deterministic, byte-stable block.

    full=True  → SPEC + complete RAW output + the Investigator's prior note.
    full=False → COLLAPSED: SPEC + a pointer. The raw is NOT destroyed (it stays
                 in the on-disk log, is always given to the Synthesizer, and can be
                 pulled back via REHYDRATE); it is merely non-resident this turn so
                 stale tables don't crowd the model's attention. The step's finding
                 lives in the nav map.
    """
    if e.get("kind") == "search":
        return (f"--- STEP {e['step']} (WEB SEARCH) ---\n"
                f"QUERY: {e.get('query', '')}\n"
                f"FINDINGS (external, for calibration):\n{e.get('result') or '(no result)'}")
    if not full:
        return (f"--- STEP {e['step']} (collapsed) ---\n"
                f"SPEC: {e.get('spec', '')}\n"
                f"[Raw output + integration note collapsed to keep context focused. "
                f"This step's finding is recorded in the NAV MAP. If you need its exact "
                f"numbers again, list step {e['step']} in a ###REHYDRATE### block and the "
                f"full raw will return next turn.]")
    raw = (f"[error after {e.get('attempts', '?')} attempt(s)]\n{e['error']}"
           if e.get("error") else (e.get("stdout") or "(no output)"))
    if len(raw) > hard_ceiling:
        raw = (raw[:hard_ceiling].rstrip()
               + "\n[...this step's raw output hit the per-step safety ceiling and was "
                 "cut here; unusual — the step likely dumped a very large table. Re-run "
                 "a summarized version if these numbers matter...]")
    parts = [f"--- STEP {e['step']} ---", f"SPEC: {e.get('spec', '')}",
             (f"RESULT: {raw}" if e.get("error") else f"RAW RESULT:\n{raw}")]
    if e.get("thinking"):
        parts.append(f"YOUR PRIOR NOTE: {e['thinking']}")
    return "\n".join(parts)


def _history_blocks(log, recent_full=3, protected=None, forced_full=None):
    """Per-step blocks, one per completed step, append-only and stable.

    A step is shown FULL when it is recent (last `recent_full`), protected (feeds a
    live thread — see NavState.protected_steps), or explicitly rehydrated this turn
    (`forced_full`). Otherwise it is COLLAPSED to a headline+pointer. This keeps the
    working set small so long runs don't bury the signal (needle-in-haystack), while
    nothing is lost: collapsed raw stays on disk, goes to the Synthesizer in full,
    and is fetchable via REHYDRATE.
    """
    protected = protected or set()
    forced_full = forced_full or set()
    steps = [e for e in log if not e.get("terminal")]
    recent_ids = {e["step"] for e in steps[-recent_full:]}
    blocks = []
    for e in steps:
        sid = e["step"]
        full = (e.get("kind") == "search") or (sid in recent_ids) \
            or (sid in protected) or (sid in forced_full)
        blocks.append(_step_block(e, full=full))
    return blocks


def _format_log(log):
    """Flat join of the per-step blocks (used for non-cached / debug rendering)."""
    return "\n\n".join(_history_blocks(log))


# How many times to re-call the Investigator when a turn comes back empty or
# token-capped (truncated) before giving up and emitting a provisional briefing.
INV_TRUNCATION_RETRIES = 2


def _referenced_names(text, kernel):
    """Names of existing derived objects that appear (as whole words) in `text`.
    Used to show the Executor only the objects its closed spec actually names."""
    ns = [it["name"] for it in kernel.registry.get("namespace", [])]
    text = text or ""
    return {n for n in ns if re.search(rf"\b{re.escape(n)}\b", text)}




def _live_names(kernel, log, window=4, newest=30):
    """Derived objects still 'live': referenced in the last `window` steps' code,
    or among the `newest` most recently created. Everything else is a dormant
    intermediate, summarized as a count, so the Investigator's registry reflects
    live state instead of every throwaway ever created."""
    ns = [it["name"] for it in kernel.registry.get("namespace", [])]
    if not ns:
        return set()
    steps = [e for e in log if not e.get("terminal")]
    recent_code = " ".join((e.get("code") or "") for e in steps[-window:])
    used = {n for n in ns if re.search(rf"\b{re.escape(n)}\b", recent_code)}
    return used | set(ns[-newest:])


def run_investigation(seed, df, client, investigator_model, executor_model,
                      synth_model=None, schema_text="", max_steps=12,
                      output_dir="output", kernel=None, nav=None, log=None,
                      periodic_every=0, g1_pushback_budget=2, on_step=None, ui=None,
                      prior_seeds=None, search_model=None, search_budget=3, stats=None):
    """The inverted-core loop with synthesis.

    Each iteration: one premium Investigator call (integrate last raw result,
    update the nav ledger, decide next move), then one cheap Executor cycle
    (with mechanical retry) against the persistent kernel.

    When the Investigator requests SYNTHESIZE, the premium Synthesizer re-derives
    the answer over RAW evidence and either returns FINAL (briefing) or
    NEEDS_MORE_WORK (G1 hard gate / insufficient evidence), in which case the loop
    resumes with a directive instead of finishing. Optional periodic re-derivation
    writes landscape snapshots and can course-correct.

    To RESUME a prior run (--continue): pass the restored `kernel` (with
    restore_history already applied), `nav`, and prior `log`; max_steps is then
    the number of ADDITIONAL steps. Step numbering continues from the prior log.

    Returns (log, kernel, nav, briefing).
    """
    own_kernel = kernel is None
    if own_kernel:
        kernel = PersistentKernel(df=df)
    if nav is None:
        nav = NavState()
    # Resume: extend a prior log, dropping any trailing terminal entry so we can
    # add more work, and continue the step numbering.
    log = list(log) if log else []
    if log and log[-1].get("terminal"):
        log.pop()
    step_offset = max((e["step"] for e in log if not e.get("terminal")), default=0)

    synth_model = synth_model or investigator_model
    search_enabled = bool(search_model)
    investigator = Investigator(client, investigator_model,
                                search_enabled=search_enabled, search_budget=search_budget)
    executor = Executor(client, executor_model)
    synthesizer = Synthesizer(client, synth_model)
    searches_used = 0
    briefing = ""
    # On an --extend run, the first Investigator turn is told to carry the
    # inherited (rehydrated) ledger forward rather than drop it.
    directive = DIRECTIVE_EXTEND_LEDGER if prior_seeds else None
    pushbacks = 0

    def _persist():
        try:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "nav_state.json"), "w", encoding="utf-8") as f:
                json.dump(nav.to_dict(), f, indent=2)
            with open(os.path.join(output_dir, "log.json"), "w", encoding="utf-8") as f:
                json.dump(log, f, indent=2, default=str)
            with open(os.path.join(output_dir, "kernel_history.json"), "w", encoding="utf-8") as f:
                json.dump(kernel.history, f, indent=2, default=str)
        except OSError as exc:
            logger.warning("Could not persist run state: %s", exc)

    def _final_synthesis(final=False):
        result = synthesizer.synthesize(seed, schema_text, log, nav, final=final,
                                        prior_seeds=prior_seeds)
        if result["verdict"] == "FINAL" and result["briefing"]:
            try:
                os.makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, "briefing.md"), "w", encoding="utf-8") as f:
                    f.write(result["briefing"])
            except OSError as exc:
                logger.warning("Could not write briefing.md: %s", exc)
        return result

    rehydrate = set()  # steps to show full next turn (requested via REHYDRATE)
    try:
        for i in range(1, max_steps + 1):
            step = step_offset + i
            registry_text = kernel.describe_namespace(names=_live_names(kernel, log))
            if ui:
                ui.iteration(step, max_steps, "ORIENTING" if i == 1 else "EXPLORING")
                ui.agent("Investigator", investigator_model)
            decision = investigator.decide(seed, schema_text, registry_text, log,
                                           nav, directive=directive, rehydrate=rehydrate)
            # An empty or token-capped (truncated) turn carries no real decision.
            # Retry it rather than letting the parser's markerless fallback finalize
            # the run prematurely. Only after repeated truncation do we give up and
            # emit a provisional briefing (never end empty-handed).
            inv_tries = 1
            while decision.get("incomplete") and inv_tries <= INV_TRUNCATION_RETRIES:
                logger.info("Step %d: Investigator turn was empty/truncated "
                            "(attempt %d/%d); retrying instead of finalizing.",
                            step, inv_tries, INV_TRUNCATION_RETRIES)
                if ui:
                    ui.note("Investigator turn was cut off; retrying.", "yellow")
                inv_tries += 1
                if stats:
                    stats.bump("investigator_truncation_retries")
                decision = investigator.decide(seed, schema_text, registry_text, log,
                                               nav, directive=directive, rehydrate=rehydrate)
            directive = None  # consumed (after any retries)
            if decision.get("incomplete"):
                logger.warning("Step %d: Investigator still truncated after %d attempts; "
                               "forcing a provisional briefing.", step, INV_TRUNCATION_RETRIES)
                if stats:
                    stats.flag("provisional_briefing", True)
                result = _final_synthesis(final=True)
                briefing = result.get("briefing", "")
                log.append({"step": step, "spec": "(synthesize)", "code": None,
                            "stdout": None, "error": None,
                            "thinking": decision.get("thinking", ""), "attempts": 0,
                            "terminal": True, "g1_satisfied": nav.g1_satisfied(log),
                            "synth_verdict": result["verdict"]})
                _persist()
                if on_step:
                    on_step(log[-1])
                break
            # Steps requested this turn are shown full on the NEXT turn.
            rehydrate = set(decision.get("rehydrate") or [])
            if rehydrate:
                logger.info("Step %d: Investigator requested rehydrate of steps %s.",
                            step, sorted(rehydrate))

            # Apply the ledger (pointer-based; a garbled block leaves the map intact).
            nav.apply_ledger_block(decision.get("ledger", ""))
            # Pin the target estimand the first time the Investigator names it, then
            # leave it fixed so the primary answer cannot silently drift to a proxy.
            est = (decision.get("estimand") or "").strip()
            if est and not nav.target_estimand:
                nav.target_estimand = est
                logger.info("Target estimand pinned: %s", est[:160])
            _persist()

            if decision["status"] == "SYNTHESIZE":
                # G1 HARD GATE. If the effect was never examined within a regime
                # and a candidate axis still exists, do NOT finish — push back and
                # make the Investigator do the stratification first.
                if (not nav.g1_satisfied(log) and nav.open_regimes()
                        and pushbacks < g1_pushback_budget):
                    pushbacks += 1
                    if stats:
                        stats.bump("g1_gate_overrides")
                    directive = DIRECTIVE_G1_GATE.format(axes=", ".join(nav.open_regimes()))
                    logger.info("Step %d: G1 gate pushback (%d/%d).",
                                step, pushbacks, g1_pushback_budget)
                    if ui:
                        ui.note(f"G1 gate — examine a regime first: {', '.join(nav.open_regimes())}",
                                "yellow")
                    continue

                # Synthesizer re-derives over raw evidence; it can also gate.
                if ui:
                    ui.agent("Synthesizer", synth_model)
                result = _final_synthesis()
                if ui:
                    ui.synthesis(result["verdict"], nav.g1_satisfied(log), result.get("reason"))
                if result["verdict"] == "NEEDS_MORE_WORK" and pushbacks < g1_pushback_budget:
                    pushbacks += 1
                    if stats:
                        stats.bump("synth_pushbacks")
                    directive = DIRECTIVE_SYNTH_GATE.format(reason=result["reason"])
                    logger.info("Step %d: synthesizer NEEDS_MORE_WORK pushback (%d/%d): %s",
                                step, pushbacks, g1_pushback_budget, result["reason"])
                    continue

                # Pushback budget spent and the synthesizer is still gating. Force a
                # final synthesis (which never gates — see Synthesizer.synthesize
                # final=True) so a run that did real work never ends without a
                # briefing. This is the same safety net as the iteration ceiling.
                if result["verdict"] == "NEEDS_MORE_WORK":
                    logger.info("Step %d: synthesis still gated with pushback budget "
                                "spent; forcing a provisional briefing.", step)
                    result = _final_synthesis(final=True)

                briefing = result.get("briefing", "")
                log.append({"step": step, "spec": "(synthesize)", "code": None,
                            "stdout": None, "error": None,
                            "thinking": decision["thinking"], "attempts": 0,
                            "terminal": True, "g1_satisfied": nav.g1_satisfied(log),
                            "synth_verdict": result["verdict"]})
                _persist()
                if on_step:
                    on_step(log[-1])
                logger.info("Investigation finished at step %d (verdict %s, G1 %s).",
                            step, result["verdict"], nav.g1_satisfied(log))
                break

            if decision["status"] == "SEARCH":
                query = (decision.get("search") or "").strip()
                if (not search_enabled) or searches_used >= search_budget or not query:
                    spent = searches_used >= search_budget
                    directive = DIRECTIVE_SEARCH_SPENT if spent else DIRECTIVE_SEARCH_FAILED
                    logger.info("Step %d: SEARCH requested but %s.", step,
                                "budget spent" if spent else "unavailable or empty query")
                    continue
                brief = (decision.get("thinking") or "(no stated reason)").strip()[:600]
                if ui:
                    ui.agent("Literature Search", search_model)
                try:
                    sresult = client.search_call(
                        [{"role": "user", "content":
                          SEARCH_MIDSTREAM_TEMPLATE.format(brief_context=brief, query=query)}],
                        search_model, max_tokens=4000, agent="Literature Search", max_uses=3)
                except Exception as exc:
                    logger.warning("Step %d: search failed: %s", step, exc)
                    directive = DIRECTIVE_SEARCH_FAILED
                    continue
                searches_used += 1
                if stats:
                    stats.bump("searches")
                analysis_dir = os.path.join(output_dir, "exploration", f"{step:02d}")
                entry = {"step": step, "kind": "search", "query": query,
                         "result": sresult or "(no result)", "spec": f"(web search) {query}",
                         "code": None, "stdout": None, "error": None, "attempts": 0,
                         "thinking": decision["thinking"]}
                log.append(entry)
                try:
                    os.makedirs(analysis_dir, exist_ok=True)
                    with open(os.path.join(analysis_dir, "search.md"), "w", encoding="utf-8") as f:
                        f.write(f"# Step {step} — Web search\n\n**Query:** {query}\n\n"
                                f"**Why:** {decision.get('thinking', '')}\n\n"
                                f"## Findings\n\n{sresult or ''}\n")
                except OSError as exc:
                    logger.warning("Could not write search artifact: %s", exc)
                _persist()
                if ui:
                    ui.searched(query, os.path.join(analysis_dir, "search.md"))
                if on_step:
                    on_step(entry)
                continue

            spec = decision["spec"]
            if not spec:
                logger.warning("Step %d: CONTINUE but empty spec; stopping.", step)
                break

            leaks = scan_spec_for_leakage(spec)
            if leaks:
                # Advisory only (fires on benign words like "clean"); INFO so it
                # stays out of the clean terminal but is visible with DELVE_VERBOSE.
                logger.info("Step %d spec contains leakage words %s — "
                            "an analytical choice may be leaking to the executor.",
                            step, leaks)

            if ui:
                ui.question(spec)
                ui.agent("Executor", executor_model)
            analysis_dir = os.path.join(output_dir, "exploration", f"{step:02d}")
            exec_registry = kernel.describe_namespace(names=_referenced_names(spec, kernel))
            result = executor.run(spec, kernel, exec_registry, analysis_dir=analysis_dir)

            entry = {
                "step": step, "spec": spec, "code": result["code"],
                "stdout": result["stdout"], "error": result["error"],
                "attempts": result["attempts"], "thinking": decision["thinking"],
                "leakage": leaks,
            }
            log.append(entry)
            _persist()
            _write_step_artifact(analysis_dir, entry, i, max_steps)
            if ui:
                ui.executed(entry, os.path.join(analysis_dir, "analysis.md"))
            if on_step:
                on_step(entry)

            # Periodic holistic re-derivation: snapshot the landscape and, if the
            # synthesizer flags a gap, course-correct the next turn.
            if periodic_every and i % periodic_every == 0:
                snap = synthesizer.synthesize(seed, schema_text, log, nav,
                                              prior_seeds=prior_seeds)
                try:
                    with open(os.path.join(output_dir, f"landscape_step{step:02d}.md"),
                              "w", encoding="utf-8") as f:
                        f.write(snap.get("briefing") or f"(NEEDS_MORE_WORK: {snap.get('reason')})")
                except OSError:
                    pass
                if snap["verdict"] == "NEEDS_MORE_WORK" and snap.get("reason"):
                    directive = DIRECTIVE_MIDPOINT.format(reason=snap["reason"])
    finally:
        # If we exhausted max_steps without a terminal synthesis, synthesize now
        # in FINAL mode — this always yields a (provisional) briefing rather than
        # gating, so a run that hits the ceiling never ends empty-handed.
        if not briefing and log and not any(e.get("terminal") for e in log):
            try:
                result = _final_synthesis(final=True)
                briefing = result.get("briefing", "")
                if stats:
                    stats.flag("provisional_briefing", True)
                logger.info("Reached max_steps; ran FINAL synthesis (provisional briefing).")
            except Exception as exc:
                logger.error("Final synthesis at max_steps failed: %s", exc)
        if own_kernel:
            kernel.cleanup()

    return log, kernel, nav, briefing