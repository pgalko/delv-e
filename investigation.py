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
from synthesis import apply_chart_results
from prompts import CHART_STYLE_DIRECTIVE
from kernel import PersistentKernel
from nav_state import NavState
from synthesis import Synthesizer
from logger_config import get_logger

logger = get_logger(__name__)
# Prompts now live in prompts.py (single source of truth; see the prompt audit).
from prompts import (
    INVESTIGATOR_TAIL_TEMPLATE,
    EXECUTOR_USER_TEMPLATE, EXECUTOR_RETRY_TEMPLATE,
    EXECUTOR_TRUNCATION_RETRY, DIRECTIVE_G1_GATE, DIRECTIVE_SYNTH_GATE,
    DIRECTIVE_MIDPOINT, DIRECTIVE_EXTEND_LEDGER, BUDGET_WRAPUP_TEMPLATE,
    SEARCH_INVESTIGATOR_INSTRUCTION, SEARCH_MIDSTREAM_TEMPLATE,
    DIRECTIVE_SEARCH_SPENT, DIRECTIVE_SEARCH_FAILED, DIRECTIVE_TRUNCATED_RETRY,
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

    def __init__(self, client, model, max_retries=2, max_tokens=None, prompts=None):
        self.client = client
        self.model = model
        self.max_retries = max_retries
        self.max_tokens = max_tokens or DEFAULT_MAX_TOKENS
        self.p = prompts or DATA_MODE

    def run(self, spec, kernel, registry_text, analysis_dir=None):
        """Returns dict: {code, stdout, error, attempts}. Never raises on a bad
        executor response — if the model fails to produce runnable code, that is
        returned as an error dict so the Investigator can see it and adapt."""
        messages = [
            {"role": "system", "content": self.p.exec_system},
            {"role": "user", "content": EXECUTOR_USER_TEMPLATE.format(
                spec=spec, registry=registry_text)},
        ]
        code = ""
        stdout = None
        error = None
        for attempt in range(self.max_retries + 1):
            resp, _meta = call_with_ladder(self.client, messages, self.model,
                                           agent="Executor", max_tokens=self.max_tokens)
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


def _render_charts(executor, kernel, charts, output_dir, ui=None, stats=None):
    """Run each chart spec through the Executor against the LIVE kernel
    namespace, landing files in <output_dir>/charts (the kernel's patched
    savefig routes bare filenames into the step's analysis_dir). Never raises:
    charts must not break the briefing. Returns the set of produced filenames
    and writes charts/manifest.json for provenance. Chart steps are
    deliberately NOT appended to the investigation log: they run after
    synthesis, carry no analytical state, and must not enter Investigator
    history on a later --extend."""
    charts_dir = os.path.join(output_dir, "charts")
    produced, manifest = set(), []
    for c in charts:
        name, spec = c["name"], c["spec"]
        full_spec = f"{spec}\n\n{CHART_STYLE_DIRECTIVE.format(name=name)}"
        try:
            registry_text = kernel.describe_namespace()
            res = executor.run(full_spec, kernel, registry_text,
                               analysis_dir=charts_dir)
            ok = os.path.exists(os.path.join(charts_dir, name))
            err = res.get("error")
        except Exception as exc:        # noqa: BLE001 - isolation is the contract
            ok, err = False, str(exc)
        if ok:
            produced.add(name)
        if stats:
            stats.bump("charts_rendered" if ok else "charts_failed")
        logger.info("Chart %s: %s", name, "saved" if ok else f"FAILED ({err})")
        manifest.append({"chart": name, "section": c.get("section"),
                         "caption": c.get("caption"), "spec": spec,
                         "produced": ok, "error": None if ok else err})
    if manifest:
        try:
            os.makedirs(charts_dir, exist_ok=True)
            with open(os.path.join(charts_dir, "manifest.json"), "w",
                      encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        except OSError as exc:
            logger.warning("Could not write charts manifest: %s", exc)
    return produced


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
               budget_note=None, reasoning_effort=None):
        nav_text = nav.render_for_investigator(log)
        head = self.p.inv_head.format(seed=seed, schema=schema)
        # Tiered history: recent + live-thread (protected) + rehydrated stay full;
        # older closed-thread steps collapse to a headline + pointer. Past the
        # history budget, old headlines slim and fold into an archive and older
        # protected residents trim (see _history_blocks).
        blocks = _history_blocks(log, protected=nav.protected_steps(),
                                 forced_full=set(rehydrate or []),
                                 char_budget=HISTORY_CHAR_BUDGET)
        # Search instruction lives in the cached prefix (static for the run) so it
        # adds no per-turn cost; present only when search is enabled.
        if self.search_enabled:
            stable_blocks = [head,
                             SEARCH_INVESTIGATOR_INSTRUCTION.format(budget=self.search_budget)] + blocks
        else:
            stable_blocks = [head] + blocks
        tail = INVESTIGATOR_TAIL_TEMPLATE.format(registry=registry_text, nav=nav_text)
        # Late-window budget notice (volatile, uncached): present only in the run's
        # final stretch so the total budget never reads as a quota to fill.
        if budget_note:
            tail += ("\n\n" + budget_note)
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


# ── History-budget compaction knobs ─────────────────────────────────────────
# Tiered demotion engages only when the rendered per-step history exceeds this
# many characters (sum of block lengths), so short/typical runs render
# byte-identically to the unbudgeted behavior. Sized from measured heavy runs
# (the F1 seed): late-run history hit 95-145K chars; this budget holds it near
# 90K, roughly 30K prompt tokens once the head and volatile tail sit on top,
# while typical sub-13-step runs never cross it.
HISTORY_CHAR_BUDGET = 90_000
# Over budget, the raw of an older PROTECTED resident (full only because it
# feeds a live thread, not because it is recent) is trimmed to this ceiling.
PROTECTED_SLIM_CEILING = 4_000
# Over budget, an old collapsed headline keeps only this much of its SPEC (its
# first sentence, capped); the step's finding lives in the NAV MAP and the
# full raw remains one REHYDRATE away.
SLIM_SPEC_CHARS = 200


def _spec_excerpt(spec, limit=SLIM_SPEC_CHARS):
    """First sentence of a spec, capped: enough to recognize the step, cheap
    enough to keep forever."""
    s = (spec or "").strip()
    head = s.split(". ")[0]
    if len(head) < len(s):
        head = head.rstrip(".") + "."
    return head if len(head) <= limit else head[:limit].rstrip() + "..."


def _slim_headline(e):
    """A collapsed step demoted under the history budget: one-sentence SPEC
    plus the same recovery affordances as a normal headline."""
    return (f"--- STEP {e['step']} (collapsed) ---\n"
            f"SPEC: {_spec_excerpt(e.get('spec'))}\n"
            f"[Compacted under the history budget. Finding is in the NAV MAP; list "
            f"step {e['step']} in a ###REHYDRATE### block to restore the full raw.]")


def _archive_line(e):
    return f"STEP {e['step']}: {_spec_excerpt(e.get('spec'))}"


def _step_block(e, full=True, hard_ceiling=20000, budget_trim=False):
    """Render ONE completed step as a deterministic, byte-stable block.

    full=True  → SPEC + complete RAW output + the Investigator's prior note.
    full=False → COLLAPSED: SPEC + a pointer. The raw is NOT destroyed (it stays
                 in the on-disk log, is always given to the Synthesizer in full, and
                 can be pulled back via REHYDRATE); it is merely non-resident this
                 turn so stale tables don't crowd the model's attention. The step's
                 finding lives in the nav map.
    budget_trim=True marks a ceiling cut made by the history-budget policy (an
    older protected resident trimmed to PROTECTED_SLIM_CEILING), which gets a
    calm recovery notice instead of the "unusual" one.
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
        if budget_trim:
            raw = (raw[:hard_ceiling].rstrip()
                   + "\n[...older live-thread step trimmed here under the history "
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


def _history_blocks(log, recent_full=3, protected=None, forced_full=None,
                    char_budget=None):
    """Per-step blocks, one per completed step, append-only and stable.

    A step is shown FULL when it is recent (last `recent_full`), protected (feeds a
    live thread — see NavState.protected_steps), or explicitly rehydrated this turn
    (`forced_full`). Otherwise it is COLLAPSED to a headline+pointer. This keeps the
    working set small so long runs don't bury the signal (needle-in-haystack), while
    nothing is lost: collapsed raw stays on disk, goes to the Synthesizer in full,
    and is fetchable via REHYDRATE.

    Budgeted tiering (char_budget set; the Investigator passes
    HISTORY_CHAR_BUDGET): when the rendered blocks exceed the budget, steps are
    demoted oldest-first, cheapest-fidelity-first, until under budget:
      1. old collapsed headlines slim to a one-sentence SPEC;
      2. the oldest slim headlines fold into a single ARCHIVED STEPS block,
         one line each, so prompt size asymptotes instead of growing with
         every completed step;
      3. older PROTECTED residents get their raw trimmed to
         PROTECTED_SLIM_CEILING.
    The last `recent_full` steps, search blocks, and this turn's rehydrates are
    never demoted, whatever the budget, so a run whose untouchable core exceeds
    the budget simply exceeds it. Every demotion is recoverable (raw on disk,
    Synthesizer sees everything full, REHYDRATE restores any step), and below
    the budget the output is byte-identical to the unbudgeted rendering.
    Demotions rewrite deep history and so reset the cached prefix at the first
    changed block; measured runs already reset to the seed almost every turn
    under one-collapse-per-turn churn, so the marginal cache cost is small
    against the fresh-input savings.
    """
    protected = protected or set()
    forced_full = forced_full or set()
    steps = [e for e in log if not e.get("terminal")]
    recent_ids = {e["step"] for e in steps[-recent_full:]}

    def _base_full(e):
        sid = e["step"]
        return (e.get("kind") == "search") or (sid in recent_ids) \
            or (sid in protected) or (sid in forced_full)

    modes = ["full" if _base_full(e) else "collapsed" for e in steps]

    def _render():
        blocks, archive = [], []
        for e, m in zip(steps, modes):
            if m == "archived":
                archive.append(_archive_line(e))
            elif m == "slim_collapsed":
                blocks.append(_slim_headline(e))
            elif m == "slim_full":
                blocks.append(_step_block(e, full=True,
                                          hard_ceiling=PROTECTED_SLIM_CEILING,
                                          budget_trim=True))
            else:
                blocks.append(_step_block(e, full=(m == "full")))
        if archive:
            blocks.insert(0, "--- ARCHIVED STEPS (oldest; raw on disk; any step "
                             "returns via ###REHYDRATE###) ---\n" + "\n".join(archive))
        return blocks

    out = _render()
    if char_budget is None or sum(map(len, out)) <= char_budget:
        return out

    # Pass 1: slim old collapsed headlines, oldest first.
    for i, m in enumerate(modes):
        if m == "collapsed":
            modes[i] = "slim_collapsed"
            out = _render()
            if sum(map(len, out)) <= char_budget:
                return out
    # Pass 2: fold the oldest slim headlines into the archive block.
    for i, m in enumerate(modes):
        if m == "slim_collapsed":
            modes[i] = "archived"
            out = _render()
            if sum(map(len, out)) <= char_budget:
                return out
    # Pass 3: trim older protected residents (never recents, search blocks,
    # or this turn's rehydrates).
    for i, e in enumerate(steps):
        sid = e["step"]
        if modes[i] == "full" and e.get("kind") != "search" \
                and sid not in recent_ids and sid not in forced_full:
            modes[i] = "slim_full"
            out = _render()
            if sum(map(len, out)) <= char_budget:
                return out
    return out   # best effort: the untouchable core alone exceeds the budget


def _format_log(log):
    """Flat join of the per-step blocks (used for non-cached / debug rendering)."""
    return "\n\n".join(_history_blocks(log))


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
                      prior_seeds=None, search_model=None, search_budget=3, stats=None,
                      compute=False, reasoning_effort="medium"):
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
                                        prior_seeds=prior_seeds,
                                        registry_text=kernel.describe_namespace())
        if result["verdict"] == "FINAL" and result["briefing"]:
            charts = result.get("charts") or []
            if charts:
                # Post-synthesis chart pass: cheap Executor calls against the
                # still-live namespace. The briefing ships regardless of chart
                # fate; apply_chart_results guarantees no broken links.
                if ui:
                    ui.agent("Executor", executor.model)
                produced = _render_charts(executor, kernel, charts, output_dir,
                                          ui=ui, stats=stats)
                result["briefing"] = apply_chart_results(result["briefing"],
                                                         charts, produced)
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
            # Wrap-up notice only inside the final stretch of THIS invocation's
            # budget (resume/extend budgets are additional steps, so i is right).
            steps_left = max_steps - i + 1
            budget_note = None
            if steps_left <= _budget_window(max_steps):
                budget_note = BUDGET_WRAPUP_TEMPLATE.format(n=steps_left)
                if stats:
                    stats.bump("budget_wrapup_notices")
            if ui:
                ui.iteration(step, max_steps, "ORIENTING" if i == 1 else "EXPLORING")
                ui.agent("Investigator", investigator_model)
            decision = investigator.decide(seed, schema_text, registry_text, log,
                                           nav, directive=directive, rehydrate=rehydrate,
                                           budget_note=budget_note)
            # An empty or token-capped (truncated) turn carries no real decision.
            # Retry it rather than letting the parser's markerless fallback finalize
            # the run prematurely. Attempt 1 above ran at the chosen effort; a retry
            # holds that effort but adds the think-less directive, and the FINAL retry
            # forces reasoning off where the endpoint allows it (models that
            # cannot disable reasoning floor to their lowest accepted rung or a
            # plain retry; see _provider_effort),
            # dropping the directive since 'none' is the fix rather than the nudge.
            # Only after the retries are exhausted do we emit a provisional briefing
            # (never end empty-handed).
            inv_retry = 0
            while decision.get("incomplete") and inv_retry < INV_TRUNCATION_RETRIES:
                inv_retry += 1
                last = inv_retry == INV_TRUNCATION_RETRIES
                retry_effort = "none" if last else None  # None keeps the chosen effort
                if last:
                    retry_directive = directive
                else:
                    retry_directive = (DIRECTIVE_TRUNCATED_RETRY if not directive
                                       else directive + "\n\n" + DIRECTIVE_TRUNCATED_RETRY)
                logger.info("Step %d: Investigator turn was empty/truncated "
                            "(retry %d/%d); reasoning_effort=%s.", step, inv_retry,
                            INV_TRUNCATION_RETRIES, retry_effort or investigator.reasoning_effort)
                if ui:
                    ui.note("Investigator turn was cut off; retrying.", "yellow")
                if stats:
                    stats.bump("investigator_truncation_retries")
                decision = investigator.decide(seed, schema_text, registry_text, log,
                                               nav, directive=retry_directive, rehydrate=rehydrate,
                                               budget_note=budget_note,
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
                # G1 HARD GATE (data mode only). If the effect was never examined
                # within a regime and a candidate axis still exists, do NOT finish:
                # push back and make the Investigator stratify first. In compute
                # mode there is no effect/regime to gate on, so it is skipped.
                if (not compute and not nav.g1_satisfied(log) and nav.open_regimes()
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

                # FINAL with no parseable briefing is the one terminal state the
                # other nets do not cover (seen live: a malformed briefing marker
                # parsed to an empty briefing under a FINAL verdict). Re-run once
                # in finalization mode, which forces a briefing and salvages.
                if result["verdict"] == "FINAL" and not result.get("briefing"):
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