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
from llm import literature_search
from synthesis import (Editor, charts_for_editor, check_attributions,
                       check_coverage, check_numbers, format_sources,
                       parse_findings, render_chart_markers, render_citations,
                       strip_unverified_citations, technical_document)
from prompts import CHART_STYLE_DIRECTIVE
from kernel import PersistentKernel
from nav_state import NavState
from synthesis import Synthesizer
from logger_config import get_logger

logger = get_logger(__name__)
# Prompts now live in prompts.py (single source of truth; see the prompt audit).
from prompts import (
    INVESTIGATOR_TAIL_TEMPLATE,
    INVESTIGATOR_TASK_FIRST,
    INVESTIGATOR_TASK_LATER,
    WORKING_SET_HEADER,
    EXECUTOR_USER_TEMPLATE, EXECUTOR_RETRY_TEMPLATE,
    EXECUTOR_TRUNCATION_RETRY, DIRECTIVE_G1_GATE, DIRECTIVE_SYNTH_GATE,
    DIRECTIVE_MIDPOINT, DIRECTIVE_EXTEND_LEDGER, BUDGET_WRAPUP_TEMPLATE,
    SEARCH_INVESTIGATOR_INSTRUCTION, SEARCH_MIDSTREAM_TEMPLATE,
    DIRECTIVE_SEARCH_SPENT, DIRECTIVE_SEARCH_FAILED, DIRECTIVE_TRUNCATED_RETRY,
    DIRECTIVE_FORMAT_RETRY,
    DATA_MODE, mode_prompts,
)
from llm import (
    DEFAULT_MAX_TOKENS,
    call_with_ladder,
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


# Block extraction terminates ONLY at the next NAMED marker, never at an arbitrary
# "###": a model writing markdown "### Subsection" headers inside its THINKING or
# SPEC must not silently truncate the block (the Executor would receive half a
# spec, and the truncated spec is what the log — and so the Synthesizer — records
# for the step). This is the same live failure _parse_synth in synthesis.py was
# hardened against; that fix is ported here, together with its trailing-hash
# tolerance ("###SPEC##", "###SPEC" both open a block).
_INV_MARKERS = r"(?:ESTIMAND|THINKING|STATUS|SPEC|LEDGER|REHYDRATE|QUERY)"
_INV_BLOCK_END = rf"(?=###\s*{_INV_MARKERS}\b|\Z)"


def _parse_investigator(text):
    """Parse the Investigator's block output. Tolerant of minor drift."""
    def block(name):
        m = re.search(rf"###\s*{name}[ \t]*#*\s*(.*?){_INV_BLOCK_END}",
                      text, re.DOTALL | re.IGNORECASE)
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
    # Fallback: no THINKING and no SPEC. Keep the whole text as thinking so
    # nothing is lost, and keep SYNTHESIZE as the terminal fallback status. When
    # additionally NO known marker appears anywhere ("unparsed"), the turn is
    # flagged so the loop can retry the format once (see the retry loop in
    # run_investigation) instead of finalizing the run on a formatting lapse;
    # callers that ignore the flag degrade to the old behavior.
    unparsed = False
    if not thinking and not spec:
        unparsed = not re.search(rf"###\s*{_INV_MARKERS}\b", text or "", re.IGNORECASE)
        thinking = text.strip()
        status = "SYNTHESIZE"
    return {"thinking": thinking, "status": status, "spec": spec,
            "ledger": ledger, "rehydrate": rehydrate, "search": search,
            "estimand": estimand, "unparsed": unparsed}


def _write_step_artifact(analysis_dir, entry, iteration, max_steps):
    """Write a human-readable per-step record (the analytical move, the rationale,
    the code, and the raw output) to <exploration>/<NN>/analysis.md. This is a
    WRITE-ONLY audit artifact — it never enters any model's context, so it has no
    token cost; it just makes each step easy to review on disk. Plots (if any) land
    in the same folder.

    When the step needed more than one attempt (or ended in an error), an
    "Execution attempts" section records every attempt in order — the failed
    code, its traceback, and whether the kernel rolled it back — because the
    committed code alone no longer tells the whole story: failed attempts are
    transactionally rolled back and leave no trace in the namespace."""
    out = entry.get("stdout")
    body = out if out else f"[error after {entry.get('attempts','?')} attempt(s)]\n{entry.get('error','')}"
    attempt_log = entry.get("attempt_log") or []
    attempts_md = ""
    if len(attempt_log) > 1 or (attempt_log and attempt_log[-1].get("error")):
        sections = []
        for a in attempt_log:
            if not a.get("code"):
                status = "no code returned"
            elif a.get("error"):
                status = "FAILED — rolled back"
            else:
                status = "committed"
            body_a = a.get("error") or a.get("stdout") or "(no output)"
            sections.append(
                f"### Attempt {a.get('attempt', '?')} — {status}\n\n"
                f"```python\n{a.get('code', '')}\n```\n\n"
                f"```\n{body_a}\n```")
        attempts_md = "\n\n## Execution attempts\n\n" + "\n\n".join(sections) + "\n"
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
        f"{attempts_md}"
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

    def __init__(self, client, model, max_retries=2, max_tokens=None, prompts=None):
        self.client = client
        self.model = model
        self.max_retries = max_retries
        self.max_tokens = max_tokens or DEFAULT_MAX_TOKENS
        self.p = prompts or DATA_MODE

    def run(self, spec, kernel, registry_text, analysis_dir=None, step=None,
            commit=True):
        """Returns dict: {code, stdout, error, attempts, attempt_log}. Never
        raises on a bad executor response — if the model fails to produce
        runnable code, that is returned as an error dict so the Investigator can
        see it and adapt.

        attempt_log records EVERY attempt in order ({attempt, code, stdout,
        error}), including no-code responses and attempts the kernel rolled
        back, so the step artifact can show the full execution audit trail —
        the committed code alone no longer tells the whole story now that
        failed attempts are transactionally rolled back.

        commit=False (chart rendering) runs code against the live namespace
        without committing it to the kernel's replayable history; see
        PersistentKernel.execute."""
        messages = [
            {"role": "system", "content": self.p.exec_system},
            {"role": "user", "content": EXECUTOR_USER_TEMPLATE.format(
                spec=spec, registry=registry_text)},
        ]
        code = ""
        stdout = None
        error = None
        attempt_log = []
        for attempt in range(self.max_retries + 1):
            resp, _meta = call_with_ladder(self.client, messages, self.model,
                                           agent="Executor", max_tokens=self.max_tokens)
            code = extract_code(resp or "")
            if not code:
                # Most common cause: a reasoning model spent its whole token budget
                # on chain-of-thought and emitted no (or a truncated) code block.
                error = ("executor returned no runnable ```python``` code block "
                         f"(likely truncated after {self.max_tokens} output tokens of reasoning)")
                attempt_log.append({"attempt": attempt + 1, "code": "",
                                    "stdout": None, "error": error})
                messages.append({"role": "assistant", "content": (resp or "")[-2000:]})
                messages.append({"role": "user", "content": EXECUTOR_TRUNCATION_RETRY})
                continue
            stdout, error, _plots = kernel.execute(code, analysis_dir=analysis_dir,
                                                   step=step, commit=commit)
            attempt_log.append({"attempt": attempt + 1, "code": code,
                                "stdout": stdout, "error": error})
            if not error:
                return {"code": code, "stdout": stdout, "error": None,
                        "attempts": attempt + 1, "attempt_log": attempt_log}
            # Mechanical retry with the traceback. The kernel has already rolled
            # the failed attempt back to the last committed state, so the retry
            # runs against exactly the pre-step namespace (and the retry template
            # tells the model so).
            messages.append({"role": "assistant", "content": resp})
            messages.append({"role": "user", "content":
                             EXECUTOR_RETRY_TEMPLATE.format(traceback=error)})
        return {"code": code, "stdout": stdout,
                "error": error or "executor failed to produce runnable code",
                "attempts": self.max_retries + 1, "attempt_log": attempt_log}


def _render_charts(executor, kernel, charts, output_dir, ui=None, stats=None):
    """Run each chart spec through the Executor against the LIVE kernel
    namespace, landing files in <output_dir>/charts (the kernel's patched
    savefig routes bare filenames into the step's analysis_dir). Never raises:
    charts must not break the briefing. Returns the set of produced filenames
    and writes charts/manifest.json for provenance. Chart steps are
    deliberately NOT appended to the investigation log — and, since the
    isolation fix, not to the KERNEL either: chart code runs commit=False (no
    history append, no checkpoint) and the worker is restored to the last
    committed step afterwards, so chart-only variables never reach the
    Investigator's registry, kernel_history.json, or a later --resume/--extend
    replay. Before this, successful chart code silently entered the replayable
    history even though the log stayed clean."""
    charts_dir = os.path.join(output_dir, "charts")
    produced, manifest = set(), []
    try:
        for c in charts:
            name, spec = c["name"], c["spec"]
            full_spec = f"{spec}\n\n{CHART_STYLE_DIRECTIVE.format(name=name)}"
            try:
                registry_text = kernel.describe_namespace()
                res = executor.run(full_spec, kernel, registry_text,
                                   analysis_dir=charts_dir, commit=False)
                ok = os.path.exists(os.path.join(charts_dir, name))
                err = res.get("error")
            except Exception as exc:        # noqa: BLE001 - isolation is the contract
                ok, err = False, str(exc)
            if ok:
                produced.add(name)
            if stats:
                stats.bump("charts_rendered" if ok else "charts_failed")
            logger.info("Chart %s: %s", name, "saved" if ok else f"FAILED ({err})")
            manifest.append({"chart": name, "finding": c.get("finding"),
                             "caption": c.get("caption"), "spec": spec,
                             "produced": ok, "error": None if ok else err})
    finally:
        # Restore the worker to the last committed analytical step (no-op when
        # no chart code actually ran). Inside finally so even an unexpected
        # exception cannot leave chart state live.
        kernel.discard_uncommitted(reason="chart rendering finished; "
                                          "discarding chart-only kernel state")
    if manifest:
        try:
            os.makedirs(charts_dir, exist_ok=True)
            with open(os.path.join(charts_dir, "manifest.json"), "w",
                      encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        except OSError as exc:
            logger.warning("Could not write charts manifest: %s", exc)
    return produced


# The editorial pass may run at most this many literature searches, one round
# trip each. On ollama they are free but rate-limited, so the ceiling is calls.
LITERATURE_BUDGET = 3


# Investigator turns use the shared DEFAULT_MAX_TOKENS budget. The output-token cap
# is no longer the lever for a reasoning model that exhausts its budget thinking:
# the loop below steps the reasoning dial down (medium -> low -> none) on a truncated
# turn instead. Anthropic's non-streaming SDK limit is handled by a clamp inside
# llm.call, so the same budget is safe to pass on every provider.


class Investigator:
    """Premium thinker: reads raw evidence, integrates, decides + compiles next step."""

    def __init__(self, client, model, search_enabled=False, search_budget=0, prompts=None,
                 reasoning_effort="medium"):
        self.client = client
        self.model = model
        self.search_enabled = search_enabled
        self.search_budget = search_budget
        self.p = prompts or DATA_MODE
        self.reasoning_effort = reasoning_effort

    def decide(self, seed, schema, registry_text, log, nav, directive=None, rehydrate=None,
               budget_note=None, search_note=None, reasoning_effort=None):
        # No evidence index for the Investigator (audit 4.2): each step's spec
        # already sits in the history region as a full block, headline, or
        # archive line, so the index duplicated it — one line per step, growing
        # linearly, in the uncached tail. The Synthesizer still gets it (its
        # call site passes the log; its evidence assembly is a different payload).
        nav_text = nav.render_for_investigator()
        head = self.p.inv_head.format(seed=seed, schema=schema)
        # Append-only context (audit 5.2): the cached prefix carries one
        # PERMANENT block per step that has exited the recent window — a pure
        # function of the log, independent of protection and pins, so it never
        # rewrites and reads back from cache in full every turn. The FULL raws
        # of recent / live-thread / rehydrated steps ride in the volatile
        # working set at the top of the tail instead (see _render_context).
        prefix_blocks, working_blocks = _render_context(
            log, protected=nav.protected_steps(), pinned=set(rehydrate or []),
            prefix_budget=HISTORY_CHAR_BUDGET,
            working_budget=WORKING_SET_CHAR_BUDGET)
        # Search instruction lives in the cached prefix (static for the run) so it
        # adds no per-turn cost; present only when search is enabled.
        if self.search_enabled:
            stable_blocks = [head,
                             SEARCH_INVESTIGATOR_INSTRUCTION.format(budget=self.search_budget)] + prefix_blocks
        else:
            stable_blocks = [head] + prefix_blocks
        # The closing task is branched: until an estimand is pinned (normally
        # just the first turn), the tail carries the orientation task plus the
        # full ESTIMAND instructions relocated out of the permanent system
        # prompt; afterwards, a terse integrate-and-decide line. The tail is
        # volatile and uncached, so the branch is free (the HEAD stays
        # byte-stable — it is cached block 0).
        task = (INVESTIGATOR_TASK_FIRST.format(estimand_note=self.p.estimand_note)
                if not nav.target_estimand else INVESTIGATOR_TASK_LATER)
        tail = INVESTIGATOR_TAIL_TEMPLATE.format(registry=registry_text,
                                                 nav=nav_text, task=task)
        if working_blocks:
            tail = (WORKING_SET_HEADER + "\n\n"
                    + "\n\n".join(working_blocks) + "\n\n" + tail)
        # Late-window budget notice (volatile, uncached): present only in the run's
        # final stretch so the total budget never reads as a quota to fill.
        if budget_note:
            tail += ("\n\n" + budget_note)
        # Standing search-budget notice (volatile, uncached): present on every turn
        # once the budget is spent. The SEARCH advertisement lives in the CACHED
        # stable prefix and cannot be withdrawn without resetting the cache, so
        # this tail line overrides it — preempting the wasted premium turn where
        # the model requests a search the loop can only refuse.
        if search_note:
            tail += ("\n\n" + search_note)
        if directive:
            tail += ("\n\nDIRECTIVE (act on this now): " + directive)
        from llm import build_cached_messages
        messages = build_cached_messages(self.model, self.p.inv_system, stable_blocks, tail)
        # The budget is shared by a thinking model between its reasoning and its
        # structured decision. `truncated` lets the loop retry instead of mis-reading
        # an empty/cut-off turn as a decision. The effort defaults to the configured
        # starting level (the --reasoning-effort flag); the loop overrides it to 'none'
        # on the final truncation retry.
        eff = reasoning_effort if reasoning_effort is not None else self.reasoning_effort
        try:
            resp, meta = self.client.call(messages, self.model, agent="Investigator",
                                          max_tokens=DEFAULT_MAX_TOKENS, return_meta=True,
                                          reasoning_effort=eff)
        except TypeError:
            # Client predates return_meta/reasoning_effort (e.g. a lightweight test
            # stub): fall back to text-only; truncation can then only be inferred
            # from an empty response.
            resp = self.client.call(messages, self.model, agent="Investigator",
                                    max_tokens=DEFAULT_MAX_TOKENS)
            meta = {}
        decision = _parse_investigator(resp or "")
        decision["incomplete"] = (not (resp or "").strip()) or bool(meta.get("truncated"))
        return decision


# ── Context layout knobs ─────────────────────────────────────────────────────
# The cached PREFIX holds one PERMANENT block per step that has exited the
# recent window — written once, byte-identical forever (append-only by
# construction; audit 5.2). This budget bounds the prefix; past it, the oldest
# permanent blocks fold into the single archive block — the one remaining
# rewrite, a deliberate one-time cache reset. Permanent blocks run ~0.5-1.5k
# chars, so ordinary runs never approach it.
HISTORY_CHAR_BUDGET = 90_000
# The WORKING SET (volatile tail, uncached) carries the FULL raws of recent,
# protected (live-thread), and pinned (rehydrated) steps. Over this budget,
# protected residents trim to PROTECTED_SLIM_CEILING — attention hygiene AND
# the per-turn bill, since every working-set char is re-billed at full rate
# every turn; trimming is free cache-wise because the tail is never cached.
# Sized from the first live run on this layout: its working set ran 21-32k
# chars (heavy prints riding live threads for 3-6 turns) and a 60k budget
# never fired a single trim, while a print-disciplined run sits near 11k and
# never triggers. 30k catches exactly the observed excess.
WORKING_SET_CHAR_BUDGET = 30_000
# Over budget, the raw of an older PROTECTED resident (full only because it
# feeds a live thread, not because it is recent) is trimmed to this ceiling.
# Lowered 4,000 -> 2,000 after the third live run: protected raws measured
# 2.2-4.6k chars, so a 4k ceiling made every trim a no-op and a 39k working
# set sailed past the 30k budget untouched. 2,000 matches the print anchor —
# a protected resident was fully read while recent; past the budget it keeps
# its head plus the honest notice, and the full raw is one pinned REHYDRATE
# away.
PROTECTED_SLIM_CEILING = 2_000
# A permanent block's verbatim RESULT EXCERPT keeps the WHOLE raw up to
# KEEP_WHOLE chars (a print-budget-disciplined step fits, so archiving loses
# nothing), else this much of each end around an honest omission marker.
# HEAD/TAIL slimmed 400/200 -> 250/120 after the first live run on this
# layout: the permanent-block slope was ~1.9k chars/turn of cached prefix,
# and the band the slim drops contained zero later-cited numbers on that run
# (measured); KEEP_WHOLE deliberately stays at the old head+tail+80 so no raw
# kept whole before is truncated now.
RESULT_EXCERPT_KEEP_WHOLE = 680
RESULT_EXCERPT_HEAD = 250
RESULT_EXCERPT_TAIL = 120
# First sentence of the step's contemporaneous THINKING carried in its
# permanent block (audit 5.3: conclusions stay in the LOG, never the ledger).
NOTE_EXCERPT_CHARS = 160
# A REHYDRATE listing pins the step's full raw into the working set for this
# many turns (audit 5.3); re-listing refreshes. Before this, a rehydrated raw
# arrived for ONE turn and vanished unless re-requested every turn.
REHYDRATE_PIN_TURNS = 3
# Past the prefix budget, an archived line keeps only this much of its SPEC.
SLIM_SPEC_CHARS = 200


def _spec_excerpt(spec, limit=SLIM_SPEC_CHARS):
    """First sentence of a spec, capped: enough to recognize the step, cheap
    enough to keep forever."""
    s = (spec or "").strip()
    head = s.split(". ")[0]
    if len(head) < len(s):
        head = head.rstrip(".") + "."
    return head if len(head) <= limit else head[:limit].rstrip() + "..."


def _archive_line(e):
    return f"STEP {e['step']}: {_spec_excerpt(e.get('spec'))}"


def _result_excerpt(raw, head=RESULT_EXCERPT_HEAD, tail=RESULT_EXCERPT_TAIL,
                    keep_whole=RESULT_EXCERPT_KEEP_WHOLE):
    """Verbatim excerpt of a step's raw output for its permanent block. Short
    raws (up to keep_whole) are kept WHOLE — with the print budget observed,
    archiving a step loses no numbers at all. Long raws keep both ends around
    an honest omission marker. Deterministic bytes from the frozen entry
    alone: a permanent block must never change once written."""
    r = (raw or "(no output)").strip()
    if len(r) <= keep_whole:
        return r
    return (r[:head].rstrip()
            + f"\n[... {len(r) - head - tail:,} chars omitted; full raw on disk ...]\n"
            + r[-tail:].lstrip())


def _note_excerpt(thinking, limit=NOTE_EXCERPT_CHARS):
    """First sentence of the step's contemporaneous THINKING — the
    Investigator's own integration, recorded at the time it read the raw."""
    s = " ".join((thinking or "").split())
    if not s:
        return ""
    head = s.split(". ")[0]
    if len(head) < len(s):
        head = head.rstrip(".") + "."
    return head if len(head) <= limit else head[:limit].rstrip() + "..."


def _permanent_block(e):
    """The ONE form in which a step ever enters the cached prefix — written
    when it exits the recent window, byte-identical forever after (a pure
    function of the frozen log entry, independent of protection or pins).
    Searches are grandfathered in full: external calibration is small and
    load-bearing. Every other step keeps its full SPEC, a verbatim RESULT
    EXCERPT, and its note at the time — so archiving no longer strands every
    number behind a REHYDRATE round-trip the way the spec+pointer collapsed
    form did."""
    if e.get("kind") == "search":
        return _step_block(e)          # the search branch already renders full
    sid = e["step"]
    if e.get("error"):
        body = (f"[error after {e.get('attempts', '?')} attempt(s)]\n"
                f"{_result_excerpt(e.get('error'))}")
    else:
        body = _result_excerpt(e.get("stdout"))
    parts = [f"--- STEP {sid} (archived) ---",
             f"SPEC: {e.get('spec', '')}",
             f"RESULT EXCERPT (verbatim; full raw on disk — REHYDRATE {sid} to restore it):",
             body]
    note = _note_excerpt(e.get("thinking"))
    if note:
        parts.append(f"NOTE AT THE TIME: {note}")
    return "\n".join(parts)


def _step_block(e, hard_ceiling=20000, budget_trim=False):
    """Render ONE completed step FULL, as a deterministic, byte-stable block:
    SPEC + complete RAW output + the Investigator's prior note. Used for the
    working set (and for a search step's permanent form). The old collapsed
    mode is gone — a step that leaves the working set is rendered ONCE by
    _permanent_block into the append-only prefix instead.

    budget_trim=True marks a ceiling cut made by the working-set budget (an
    older protected resident trimmed to PROTECTED_SLIM_CEILING), which gets a
    calm recovery notice instead of the "unusual" one.
    """
    if e.get("kind") == "search":
        return (f"--- STEP {e['step']} (WEB SEARCH) ---\n"
                f"QUERY: {e.get('query', '')}\n"
                f"FINDINGS (external, for calibration):\n{e.get('result') or '(no result)'}")
    raw = (f"[error after {e.get('attempts', '?')} attempt(s)]\n{e['error']}"
           if e.get("error") else (e.get("stdout") or "(no output)"))
    if len(raw) > hard_ceiling:
        if budget_trim:
            raw = (raw[:hard_ceiling].rstrip()
                   + "\n[...older live-thread step trimmed here under the working-set "
                     "budget; the full raw is on disk and returns via ###REHYDRATE### "
                     "if you need its exact numbers...]")
        else:
            raw = (raw[:hard_ceiling].rstrip()
                   + "\n[...this step's raw output hit the per-step safety ceiling and was "
                     "cut here; unusual — the step likely dumped a very large table. Re-run "
                     "a summarized version if these numbers matter...]")
    parts = [f"--- STEP {e['step']} ---", f"SPEC: {e.get('spec', '')}",
             (f"RESULT: {raw}" if e.get("error") else f"RAW RESULT:\n{raw}")]
    if e.get("thinking"):
        parts.append(f"YOUR PRIOR NOTE: {e['thinking']}")
    return "\n".join(parts)


def _render_context(log, recent_full=3, protected=None, pinned=None,
                    prefix_budget=None, working_budget=None):
    """Split the completed steps into (prefix_blocks, working_blocks).

    PREFIX — append-only by construction (audit 5.2). Every step that has
    exited the recent window appears as its PERMANENT block (_permanent_block:
    spec + verbatim result excerpt + note), written once and never rewritten.
    The prefix is a pure function of the step list and `recent_full` — it does
    NOT depend on `protected` or `pinned` — so the two churn sources that
    invalidated the old cache (protection moving as threads advance, and this
    turn's rehydrates) can no longer touch deep history. Each turn appends at
    most the block(s) of the step(s) that just exited the window; the cached
    prefix reads back in full every turn.

    WORKING SET — the volatile, uncached tail. Full raws of: the last
    `recent_full` steps, protected steps (feeding live threads), and pinned
    steps (REHYDRATE, resident for REHYDRATE_PIN_TURNS). Chronological,
    labeled by step id. Over `working_budget`, protected residents trim to
    PROTECTED_SLIM_CEILING oldest-first — recents, pins, and searches are
    never trimmed; a working set whose untouchable core exceeds the budget
    simply exceeds it. Trimming here is free cache-wise.

    The one remaining prefix rewrite: past `prefix_budget`, the OLDEST
    permanent blocks fold into a single ARCHIVED STEPS block (one line each) —
    a deliberate, rare, one-time cache reset so the prefix asymptotes instead
    of growing with every completed step.

    Nothing is ever lost: full raws stay on disk, the Synthesizer sees every
    step in full, and REHYDRATE restores any step to the working set.
    """
    protected = protected or set()
    pinned = pinned or set()
    steps = [e for e in log if not e.get("terminal")]
    older, recent = steps[:-recent_full] or [], steps[-recent_full:]
    if not recent_full:
        older, recent = steps, []

    # ---- prefix: one permanent block per older step, oldest first ----
    perm = [_permanent_block(e) for e in older]
    prefix = perm
    if prefix_budget is not None and sum(map(len, perm)) > prefix_budget:
        header = ("--- ARCHIVED STEPS (oldest; spec lines only; raw on disk; any step "
                  "returns via ###REHYDRATE###) ---")
        n_arch = 0
        while n_arch < len(older):
            n_arch += 1
            arch = header + "\n" + "\n".join(_archive_line(e) for e in older[:n_arch])
            if len(arch) + sum(map(len, perm[n_arch:])) <= prefix_budget:
                break
        prefix = [arch] + perm[n_arch:]

    # ---- working set: full raws, chronological, labeled ----
    entries = []                       # (entry, kind) with kind deciding the label
    for e in older:
        sid = e["step"]
        if sid in pinned:
            entries.append((e, "pinned"))
        elif sid in protected and e.get("kind") != "search":
            entries.append((e, "protected"))
    for e in recent:
        entries.append((e, "recent"))

    def _label(e, kind):
        if kind == "pinned":
            return (f"[REHYDRATED step {e['step']} — full raw restored to the "
                    f"working set; its archived form remains above]\n")
        if kind == "protected":
            return (f"[LIVE-THREAD step {e['step']} — full raw resident while "
                    f"its thread stays open]\n")
        return ""

    def _render_working(trim_ids):
        out = []
        for e, kind in entries:
            if e["step"] in trim_ids:
                out.append(_label(e, kind) + _step_block(
                    e, hard_ceiling=PROTECTED_SLIM_CEILING, budget_trim=True))
            else:
                out.append(_label(e, kind) + _step_block(e))
        return out

    working = _render_working(set())
    if working_budget is not None and sum(map(len, working)) > working_budget:
        trim_ids = set()
        for e, kind in entries:       # oldest first; only protected residents trim
            if kind == "protected":
                trim_ids.add(e["step"])
                working = _render_working(trim_ids)
                if sum(map(len, working)) <= working_budget:
                    break
    return prefix, working


def _format_log(log):
    """Flat join of the full context (used for non-cached / debug rendering)."""
    prefix, working = _render_context(log)
    return "\n\n".join(prefix + working)


# How many times to re-call the Investigator when a turn comes back empty or
# token-capped (truncated) before giving up and emitting a provisional briefing.
INV_TRUNCATION_RETRIES = 2


def _budget_window(max_steps):
    """How many final iterations get the wrap-up notice: the last fifth of the
    budget, never fewer than the last 2 turns. The notice is absent before the
    window so the total budget is never visible as a quota to fill."""
    return max(2, -(-max_steps // 5))


def _referenced_names(text, kernel):
    """Names of existing derived objects that appear (as whole words) in `text`.
    Used to show the Executor only the objects its closed spec actually names."""
    ns = [it["name"] for it in kernel.registry.get("namespace", [])]
    text = text or ""
    # A spec may pin a step-versioned alias (records__s6). The registry lists the
    # bare name (carrying the ambiguity note that points at the aliases), so match
    # through the suffix or a pinned spec resolves nothing and trips the
    # self-containment tripwire for doing exactly the right thing.
    return {n for n in ns
            if re.search(rf"\b{re.escape(n)}(?:__s\d+)?\b", text)}




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
                      prior_seeds=None, search_model=None, search_budget=3, stats=None,
                      compute=False, reasoning_effort="medium", lit_search_model=None):
    """The inverted-core loop with synthesis.

    Each iteration: one premium Investigator call (integrate last raw result,
    update the nav ledger, decide next move), then one cheap Executor cycle
    (with mechanical retry) against the persistent kernel.

    When the Investigator requests SYNTHESIZE, the premium Synthesizer re-derives
    the answer over RAW evidence and either returns FINAL (briefing) or
    NEEDS_MORE_WORK (G1 hard gate / insufficient evidence), in which case the loop
    resumes with a directive instead of finishing. Optional periodic re-derivation
    writes landscape snapshots and can course-correct.

    `compute=True` selects dataset-free mode: `df` may be None, the agents use the
    compute prompt bundle, and the statistical G1 gate is bypassed (the compute
    synthesizer self-checks uncertainty/convergence/validity instead). Everything
    else (kernel, ledger, loop, nets) is identical.

    `max_steps` is the analysis-step budget. Model-chosen turns (analysis steps
    and SEARCH steps, including refused search requests) consume it; gate
    pushbacks (G1 pre-gate, Synthesizer NEEDS_MORE_WORK) are system-initiated
    and are refunded, so the run may make up to max_steps + g1_pushback_budget
    Investigator turns in total and a gate can never eat the final planned turn.

    To RESUME a prior run (--resume/--extend): pass the restored `kernel` (with
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

    p = mode_prompts(compute)
    synth_model = synth_model or investigator_model
    search_enabled = bool(search_model)
    investigator = Investigator(client, investigator_model, search_enabled=search_enabled,
                                search_budget=search_budget, prompts=p,
                                reasoning_effort=reasoning_effort)
    executor = Executor(client, executor_model, prompts=p)
    synthesizer = Synthesizer(client, synth_model, prompts=p,
                              reasoning_effort=reasoning_effort)
    editor = Editor(client, synth_model, prompts=p,
                    reasoning_effort=reasoning_effort)
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

    def _literature(technical):
        """Up to LITERATURE_BUDGET searches, chosen by the editor from the findings.
        Post-hoc by design: the findings are already fixed, so the literature can
        calibrate the write-up but cannot contaminate the analysis.

        Returns (sources, literature_text). Every source gets a stable [Sn] id here,
        in the harness, because the editor cannot be trusted to attribute one: given
        a page it will invent an author, and a wrong attribution on a real URL is
        invisible to any check. Same contract as the charts, for the same reason.

        Every outcome is recorded. The free ollama route makes no LLM call, so a
        search that fails there leaves NO trace in the run log; a live run had all
        three searches fail silently while the console still printed "searched"."""
        if not lit_search_model or not LITERATURE_BUDGET:
            return [], ""
        try:
            queries = editor.queries(seed, technical, LITERATURE_BUDGET)
        except Exception as exc:                                # noqa: BLE001
            logger.warning("Editor could not produce literature queries: %s", exc)
            return [], ""
        if not queries:
            logger.info("Editor asked for no literature searches.")
            return [], ""

        sources, notes, failures, seen = [], [], [], set()
        for q in queries[:LITERATURE_BUDGET]:
            if ui:
                ui.agent("Literature Search", lit_search_model)
            found, note, err = literature_search(client, lit_search_model, q)
            if err:
                failures.append((q, err))
                if stats:
                    stats.bump("literature_search_failed")
                logger.warning("Literature search failed (%s): %s", q, err)
                if ui:
                    ui.note(f"Literature search failed: {err}", "yellow")
                continue
            if stats:
                stats.bump("literature_searches")
            if ui:
                ui.searched(q, None)
            if note:
                notes.append(f"### QUERY: {q}\n{note}")
            for src in found:
                if src["url"] in seen:
                    continue
                seen.add(src["url"])
                src["id"] = f"S{len(sources) + 1}"
                sources.append(src)

        # Always leave the artifact: on a total failure it is the only place the
        # reason survives, and a briefing that ships without citations needs one.
        parts = list(notes)
        if sources:
            parts.append("## SOURCES\n\n" + "\n\n".join(
                f"[{s['id']}] {s['title']}\n{s['url']}" for s in sources))
        if failures:
            parts.append("## SEARCHES THAT FAILED\n\n" + "\n".join(
                f"- {q}\n  {err}" for q, err in failures))
        if parts:
            try:
                with open(os.path.join(output_dir, "literature.md"), "w",
                          encoding="utf-8") as f:
                    f.write("\n\n".join(parts) + "\n")
            except OSError:
                pass
        if not sources:
            msg = ("No literature was retrieved: every search on the %s seat failed. "
                   "The briefing ships without citations; see literature.md."
                   % lit_search_model)
            logger.warning(msg)
            if ui:
                ui.note(msg, "yellow")
            return [], ""
        return sources, "\n\n".join(notes + [format_sources(sources)])

    def _publish(result):
        """Technical record, then charts, then literature, then the deliverable.

        The technical briefing is the complete record and the --verify target;
        briefing.md is what a reader receives. Three gates run HERE and not in a
        prompt, because the deliverable now sits downstream of an extra call and a
        silently dropped finding, a fabricated number, or an invented citation
        would otherwise be invisible: nobody reads both artifacts."""
        findings = parse_findings(result["findings"])
        charts = result.get("charts") or []
        produced = set()
        if charts:
            if ui:
                ui.agent("Executor", executor.model)
            produced = _render_charts(executor, kernel, charts, output_dir,
                                      ui=ui, stats=stats)

        technical = technical_document(result)
        sources, literature = _literature(result["findings"])

        if ui:
            ui.agent("Editor", synth_model)
        briefing = editor.write(seed, result["findings"],
                                charts_for_editor(charts, produced), literature)

        # Gate 1, coverage: every decisive finding must reach the reader. One retry
        # naming what was dropped, then carry the survivors verbatim rather than
        # ever losing a finding the evidence earned.
        missing = check_coverage(briefing, findings)
        if missing and briefing:
            if stats:
                stats.bump("editor_coverage_retries")
            logger.warning("Editor dropped decisive finding(s) %s; retrying once.",
                           ", ".join(f["id"] for f in missing))
            briefing = editor.write(
                seed, result["findings"], charts_for_editor(charts, produced), literature,
                directive=("Your previous draft omitted these decisive findings: "
                           + ", ".join(f["id"] for f in missing)
                           + ". Every decisive finding must reach the reader, cited by "
                             "id. Carry them, or list them under a final "
                             "'## Not carried forward' heading with your reason."))
            missing = check_coverage(briefing, findings)
        if missing:
            if stats:
                stats.bump("editor_coverage_failed")
            logger.warning("Editor still dropped %s; appending them verbatim.",
                           ", ".join(f["id"] for f in missing))
            briefing = (briefing or "").rstrip() + "\n\n## Not carried forward\n\n" + "\n\n".join(
                f"**{f['id']}** {f['claim']}\n\n{f['numbers']}" for f in missing)

        # Gate 2, citations: the harness owns the label. [Sn] markers become the
        # reference list; a marker naming nothing is dropped, and any raw link the
        # editor wrote anyway is stripped unless the harness actually fetched it.
        briefing = render_citations(briefing, sources)
        before = briefing
        briefing = strip_unverified_citations(briefing, {s["url"] for s in sources})
        if stats and briefing != before:
            stats.bump("citations_stripped")
        invented = check_attributions(briefing)
        if invented:
            if stats:
                stats.bump("editor_invented_attributions")
            logger.warning("Editor attributed sources by name (%s); it was given "
                           "titles, not author lists, so these are guesses.",
                           ", ".join(invented[:5]))

        # Gate 3, numbers: a statistic in neither the record nor the literature is
        # either drift or invention. Warned, not silently rewritten: the technical
        # briefing sits alongside for anyone who wants to check.
        bad = check_numbers(briefing, result["findings"], literature)
        if bad:
            if stats:
                stats.bump("editor_unsourced_numbers")
            logger.warning("Editor briefing carries %d number(s) absent from the "
                           "technical record: %s", len(bad), ", ".join(bad[:8]))

        briefing = render_chart_markers(briefing, charts, produced)
        if not briefing.strip():
            logger.warning("Editor produced nothing; the technical record stands in "
                           "as the deliverable.")
            briefing = technical

        try:
            os.makedirs(output_dir, exist_ok=True)
            for name, text in (("technical_briefing.md", technical),
                               ("briefing.md", briefing)):
                with open(os.path.join(output_dir, name), "w", encoding="utf-8") as f:
                    f.write(text)
        except OSError as exc:
            logger.warning("Could not write the briefing artifacts: %s", exc)
        result["briefing"] = briefing
        result["technical"] = technical
        return result

    def _final_synthesis(final=False):
        result = synthesizer.synthesize(seed, schema_text, log, nav, final=final,
                                        prior_seeds=prior_seeds,
                                        registry_text=kernel.describe_namespace())
        if stats and result.get("format_repaired"):
            stats.bump("synth_format_retries")
        if result["verdict"] == "FINAL" and result["findings"]:
            _publish(result)
        return result

    # REHYDRATE pins (audit 5.3): step -> remaining turns of working-set
    # residency. A listing pins the step for REHYDRATE_PIN_TURNS; re-listing
    # refreshes. Before this, a rehydrated raw arrived for one turn and
    # vanished unless re-requested, taxing every live decision with a request.
    pins = {}
    rehydrate = set()  # the currently-pinned steps, passed to decide()
    try:
        # `budget` is the analysis-step budget for THIS invocation. Gate pushbacks
        # (the G1 pre-gate and a Synthesizer NEEDS_MORE_WORK) are system-initiated
        # turns the model did not plan for, so they no longer consume it: each
        # pushback extends the ceiling by one, bounded by g1_pushback_budget, so a
        # gate can never eat the model's final turn and force the ungated
        # provisional path. Model-chosen SEARCH turns (and refused search
        # requests) still consume budget, as documented.
        budget = max_steps
        i = 0
        while i < budget:
            i += 1
            step = step_offset + i
            registry_text = kernel.describe_namespace(names=_live_names(kernel, log))
            # Wrap-up notice only inside the final stretch of THIS invocation's
            # budget (resume/extend budgets are additional steps, so i is right).
            # `budget` reflects any pushback refunds, so the countdown stays honest.
            steps_left = budget - i + 1
            budget_note = None
            if steps_left <= _budget_window(max_steps):
                budget_note = BUDGET_WRAPUP_TEMPLATE.format(n=steps_left)
                if stats:
                    stats.bump("budget_wrapup_notices")
            # Once the search budget is spent, tell the model preemptively (every
            # turn, in the uncached tail) rather than letting it burn a full
            # premium turn on a SEARCH request the loop can only refuse.
            search_note = (DIRECTIVE_SEARCH_SPENT
                           if search_enabled and searches_used >= search_budget
                           else None)
            if ui:
                ui.iteration(step, budget, "ORIENTING" if i == 1 else "EXPLORING")
                ui.agent("Investigator", investigator_model)
            decision = investigator.decide(seed, schema_text, registry_text, log,
                                           nav, directive=directive, rehydrate=rehydrate,
                                           budget_note=budget_note,
                                           search_note=search_note)
            # An empty or token-capped (truncated) turn carries no real decision, and
            # a non-empty turn with NO ### blocks at all ("unparsed") carries one the
            # parser cannot see. Retry either rather than letting the markerless
            # fallback finalize the run prematurely.
            # Truncated: attempt 1 above ran at the chosen effort; a retry holds that
            # effort but adds the think-less directive, and the FINAL retry forces
            # reasoning off where the endpoint allows it (models that cannot disable
            # reasoning floor to their lowest accepted rung or a plain retry; see
            # _provider_effort), dropping the directive since 'none' is the fix
            # rather than the nudge.
            # Unparsed: the model finished its turn but skipped the format, so the
            # FORMAT directive is the fix — carried on every retry, at the chosen
            # effort (thinking less would not help a formatting lapse).
            # Only after the retries are exhausted do we fall through: a
            # still-truncated turn emits a provisional briefing (never end
            # empty-handed); a still-unparsed turn proceeds as the markerless
            # fallback (SYNTHESIZE) through the normal gate machinery.
            inv_retry = 0
            while ((decision.get("incomplete") or decision.get("unparsed"))
                   and inv_retry < INV_TRUNCATION_RETRIES):
                inv_retry += 1
                last = inv_retry == INV_TRUNCATION_RETRIES
                if decision.get("incomplete"):
                    kind = "empty/truncated"
                    retry_effort = "none" if last else None  # None keeps the chosen effort
                    if last:
                        retry_directive = directive
                    else:
                        retry_directive = (DIRECTIVE_TRUNCATED_RETRY if not directive
                                           else directive + "\n\n" + DIRECTIVE_TRUNCATED_RETRY)
                else:
                    kind = "unparsed (no ### blocks)"
                    retry_effort = None
                    retry_directive = (DIRECTIVE_FORMAT_RETRY if not directive
                                       else directive + "\n\n" + DIRECTIVE_FORMAT_RETRY)
                logger.info("Step %d: Investigator turn was %s "
                            "(retry %d/%d); reasoning_effort=%s.", step, kind, inv_retry,
                            INV_TRUNCATION_RETRIES, retry_effort or investigator.reasoning_effort)
                if ui:
                    ui.note("Investigator turn was cut off; retrying."
                            if kind == "empty/truncated"
                            else "Investigator reply had no parseable blocks; retrying.",
                            "yellow")
                if stats:
                    stats.bump("investigator_truncation_retries"
                               if kind == "empty/truncated"
                               else "investigator_format_retries")
                decision = investigator.decide(seed, schema_text, registry_text, log,
                                               nav, directive=retry_directive, rehydrate=rehydrate,
                                               budget_note=budget_note,
                                               search_note=search_note,
                                               reasoning_effort=retry_effort)
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
                            "synth_verdict": result["verdict"],
                            "gates_review": result.get("gates_review", "")})
                _persist()
                if on_step:
                    on_step(log[-1])
                break
            # Pin bookkeeping: decrement existing pins by one consumed decision,
            # then (re)pin this turn's requests for REHYDRATE_PIN_TURNS. The raws
            # arrive on the NEXT turn and stay resident until their pin expires.
            pins = {s: n - 1 for s, n in pins.items() if n > 1}
            requested = [int(s) for s in (decision.get("rehydrate") or [])]
            for s in requested:
                pins[s] = REHYDRATE_PIN_TURNS
            if requested:
                logger.info("Step %d: Investigator requested rehydrate of steps %s "
                            "(pinned for %d turns).", step, sorted(set(requested)),
                            REHYDRATE_PIN_TURNS)
            rehydrate = set(pins)

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
                # G1 HARD GATE (data mode only). If the effect was never examined
                # within a regime and a candidate axis still exists, do NOT finish:
                # push back and make the Investigator stratify first. In compute
                # mode there is no effect/regime to gate on, so it is skipped.
                if (not compute and not nav.g1_satisfied(log) and nav.open_regimes()
                        and pushbacks < g1_pushback_budget):
                    pushbacks += 1
                    budget += 1   # system-initiated turn: refund it (bounded by the pushback budget)
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
                    budget += 1   # system-initiated turn: refund it (bounded by the pushback budget)
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

                # FINAL with no parseable briefing is the one terminal state the
                # other nets do not cover (seen live: a malformed briefing marker
                # parsed to an empty briefing under a FINAL verdict). Re-run once
                # in finalization mode, which forces a briefing and salvages.
                if result["verdict"] == "FINAL" and not result.get("findings"):
                    logger.warning("Step %d: synthesizer returned FINAL with no "
                                   "parseable briefing; re-running in finalization "
                                   "mode to recover the deliverable.", step)
                    if stats:
                        stats.bump("synth_briefing_retries")
                    result = _final_synthesis(final=True)

                briefing = result.get("briefing", "")
                log.append({"step": step, "spec": "(synthesize)", "code": None,
                            "stdout": None, "error": None,
                            "thinking": decision["thinking"], "attempts": 0,
                            "terminal": True, "g1_satisfied": nav.g1_satisfied(log),
                            "synth_verdict": result["verdict"],
                            "gates_review": result.get("gates_review", "")})
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
                        search_model, max_tokens=4000, agent="Literature Search", max_uses=3,
                        query=query)
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
            refs = _referenced_names(spec, kernel)
            exec_registry = kernel.describe_namespace(names=refs)
            if not refs and re.search(r"\bstep\s+\d+\b", spec, re.IGNORECASE):
                # The executor sees only the spec plus the registry objects the
                # spec NAMES. A step-number reference with zero named objects
                # means it must rebuild the referenced method blind, and a
                # "change only X" contrast against a blind rebuild is invalid.
                logger.warning("Step %d: spec references a prior step by number "
                               "but names no registry objects; the executor "
                               "cannot see prior steps and will rebuild the "
                               "method from the spec text alone.", step)
                if stats:
                    stats.bump("blind_step_references")
            result = executor.run(spec, kernel, exec_registry, analysis_dir=analysis_dir,
                                  step=step)

            entry = {
                "step": step, "spec": spec, "code": result["code"],
                "stdout": result["stdout"], "error": result["error"],
                "attempts": result["attempts"], "thinking": decision["thinking"],
                "leakage": leaks,
            }
            # Attach the per-attempt audit only when it says more than the
            # committed code/output already do (a retry happened, or the step
            # ended in an error) — the common clean single-attempt entry stays
            # byte-identical to before.
            attempt_log = result.get("attempt_log") or []
            if len(attempt_log) > 1 or result["error"]:
                entry["attempt_log"] = attempt_log
            log.append(entry)
            _persist()
            _write_step_artifact(analysis_dir, entry, i, budget)
            if ui:
                ui.executed(entry, os.path.join(analysis_dir, "analysis.md"))
            if on_step:
                on_step(entry)

            # Periodic holistic re-derivation: snapshot the landscape and, if the
            # synthesizer flags a gap, course-correct the next turn.
            if periodic_every and i % periodic_every == 0:
                snap = synthesizer.synthesize(seed, schema_text, log, nav,
                                              prior_seeds=prior_seeds)
                if stats and snap.get("format_repaired"):
                    stats.bump("synth_format_retries")
                try:
                    with open(os.path.join(output_dir, f"landscape_step{step:02d}.md"),
                              "w", encoding="utf-8") as f:
                        f.write(snap.get("findings") or f"(NEEDS_MORE_WORK: {snap.get('reason')})")
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