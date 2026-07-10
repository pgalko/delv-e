# delv-e: System Design & Handover

This is a complete handover document for the rebuilt **delv-e** system: an autonomous,
LLM-driven data-investigation engine. It is written to seed a fresh working session.
Read it top to bottom once; after that, sections 0 and the appendix are the day-to-day
references.

This document ships alongside:

- `delv-e.zip`: the complete, push-ready project (the 13 canonical modules, the `tests/`
  directory with its bundled httpx stub, the docs, and the repo scaffolding).
- `F1_GOAT_report.md`: the benchmark ground-truth report.
- `f1_driver_vs_car.csv`: the benchmark dataset (26,668 driver-races × 67 columns).
- the most recent `briefing.md` (output root) and the latest `logs/<timestamp>/run_log.json` and `run_telemetry.json`: the run's output, per-call trace, and rollup.

---

## 0. Quick start for a new session

**What the system does, in one line.** Given a tabular dataset and a seed question, it
runs an autonomous loop in which a strong model reasons over raw evidence and decides what
to investigate, a cheap model writes and runs the pandas code, and a strong model finally
re-derives the answer over the raw evidence and writes a briefing.

**Canonical code.** The system is 13 Python modules plus a `tests/` directory. The shipped,
authoritative archive is `delv-e.zip` (the complete project: the 13 modules, `tests/`, the
docs, and the repo scaffolding). The live working copy is `/home/claude/delve/delv-e/`. The
rebuilt system is exactly these 13 files: `kernel.py`, `investigation.py`, `nav_state.py`,
`synthesis.py`, `verify.py`, `dataio.py`, `executor.py`, `toolkit.py`, `llm.py`, `prompts.py`, `run_core.py`,
`ui.py`, `logger_config.py`. (The leftover original-delv-e files that used to sit in the working
directory, `auto_explore.py`, `engine.py`, `dashboard.py`, and the rest, were removed in the
pre-release cleanup; see 6.1. The live graph is now exactly these 13 modules.)

**How to run it.**

```
python3 run_core.py <dataset> "<seed question>" [flags]
```

Key flags (full list in the appendix): `--iterations N` (default 14), `--investigator-model`,
`--executor-model`, `--synth-model`, `--reasoning-effort LEVEL` (default medium), `--output DIR`, `--g1-pushback N` (default 2),
`--search-model` / `--search-budget N` (default 3, search off unless a search model is set),
`--resume`, `--extend`, `--periodic-every N`.

Model strings are `provider:model`, e.g. `anthropic:claude-...`, `ollama:glm-5.1:cloud`,
`openai:gpt-...`, `openrouter:...`. Premium defaults to Anthropic, cheap defaults to Ollama.

**How to validate without API keys.** The sandbox has no model API keys and no outbound
network. Validation is done two ways, both already in place:

1. **Mock LLM clients + real execution.** The tests inject a `Mock` client whose `call()`
   returns canned model text by `agent` role, while the *real* `PersistentKernel` executes
   *real* pandas against a small real DataFrame. This exercises the full control flow,
   kernel, ledger, compaction, and gates, end to end, deterministically.
2. **`httpx` stub.** A tiny `httpx` stub is bundled at `tests/stubs/httpx.py` so imports
   resolve without network access. Each test prepends it (and the repo root) to `sys.path`
   itself, so no environment setup is needed.

**Regression suite** (`tests/`, run with plain `python3 tests/test_X.py` from the repo root):
`test_happy`, `test_provisional`, `test_tier`, `test_executor`, `test_artifact`,
`test_continue`, `test_extend`, `test_search`, `test_context`, `test_g1_fixes`,
`test_truncation`, `test_ledger_parse`, `test_status`, `test_estimand`, `test_telemetry`,
`test_synth_gates`, `test_budget`, `test_toolkit`, `test_verify`, `test_compute`,
`test_compute_cli`, `test_executor_reasoning_effort`, `test_truncation_retry`,
`test_reasoning_ladder`, `test_function_reuse`, `test_checkpoint`, `test_compute_continue`,
`test_verify_compute`, `test_gpt56_cache`, `test_compaction_budget`,
`test_print_discipline`. All 31 pass as of this handover.
The dataset-backed tests (`test_continue`, `test_extend`, `test_search`) read their dataset from `datasets/` (or a path given via an environment variable) and skip cleanly if it is absent, so a fresh
clone runs the rest green. See `tests/README.md`.

**The ship workflow (used every change).** Edit source in `/home/claude/delve/delv-e/` → run
the suite → copy changed files to `/mnt/user-data/outputs/` → rebuild `delv-e.zip` from a clean
staging tree excluding `__pycache__` and gitignored artifacts → verify it imports from a fresh
extract → `present_files`, listing EVERY module changed in that round as a separate file
alongside the archive (the user deploys per-file; a change buried only in the zip gets missed).
Be honest about faithfulness versus fabrication in all work.

**House style.** Lean code, minimal dependencies, confirm before large changes. The user
dislikes em dashes and the "not X, it's Y" antithesis construction; avoid both in code
comments, docs, and prose.

---

## 1. Purpose of the system and key design decisions

### 1.1 The problem we set out to fix

The original delv-e was a large system (~11,900 LOC) that, on a hard analytical benchmark,
produced a **null** answer where the correct answer was a real, conditional effect (see
section 5). A single Claude model with code tools beat it. The rebuild has three goals:
match or beat the benchmark's accuracy, add analytical breadth, and drastically simplify.

### 1.2 The diagnosis: intelligence was allocated backwards

The original spent its strong model on orchestration and used weaker capacity on the actual
reasoning over evidence. The core insight of the rebuild is to **invert** that: the
expensive, capable model does the thinking over raw evidence, and the cheap model is
confined to mechanical translation of a fully specified plan into code. This is the
"inverted core."

### 1.3 Key design decisions

- **Inverted core.** The premium model (Investigator) integrates raw evidence, maintains
  the investigation state, and decides the next move. The cheap model (Executor) only turns
  a closed, prose specification into pandas and runs it. The premium model (Synthesizer)
  re-derives the final answer over the raw evidence.
- **One call per turn for the Investigator.** Each loop iteration is a single premium call
  that reads the (compacted) history, updates a structured ledger, and emits exactly one
  decision plus, when continuing, one spec.
- **Closed-spec contract.** The Investigator writes the spec **in words, never as runnable
  code**. If it includes a code block, it has done the Executor's job; the prompt forbids
  this explicitly. The Executor receives the spec plus a focused namespace registry and
  writes the Python.
- **Persistent kernel.** Code runs in one long-lived worker process holding a single
  namespace, so `df` is loaded once and derived objects and functions persist across steps. Crashes are
  isolated and the namespace is reconstructed from the most recent checkpoint plus a replay of only the steps after it (a pickled data checkpoint is written after each successful step; functions and other unpicklable objects fall back to a full replay, see 6.12).
- **Structured pointer ledger, not prose memory.** Investigation state lives in a
  machine-checkable ledger of short handles, status enums, and step pointers (`NavState`),
  re-emitted wholesale each turn. This is what the hard gates read, so a confident narrative
  cannot smuggle past a gate.
- **Guardrails as hard gates.** G1, G1b, and G2 (section 2.4) encode the analytical
  discipline that the original lacked, especially "do not declare a null before you have
  looked within subgroups."
- **Tiered context compaction.** Long runs would otherwise bury the signal and blow the
  context budget. Old steps collapse to headline pointers (rehydratable on demand) while
  recent and load-bearing steps stay full.
- **Web search is mid-stream, model-decided, and off by default.** When enabled it is for
  calibration only and is excluded from the benchmark protocol, to keep benchmark integrity.
- **Resume and extend.** A run can be resumed after interruption, or extended with a new
  question that reopens and reconciles the prior conclusion.

---

## 2. Key design structures and objects

### 2.1 Module catalog

| Module | Role |
|---|---|
| `run_core.py` | CLI entry point. Parses args, loads dataset and schema, wires the client and models, runs the loop, handles resume/extend and seed history. |
| `investigation.py` | The heart. Defines the `Investigator` and `Executor` agent classes, the `run_investigation` loop, the tiered history compaction (`_step_block`, `_history_blocks`), the registry-focus helpers (`_referenced_names`, `_live_names`, `_stratification_evidence`), and the synthesis gate logic. |
| `nav_state.py` | The structured pointer ledger (`NavState` + `Entry`), status vocabularies, ledger parsing (`apply_ledger_block`), the pinned `target_estimand` (named once, rendered at the top of the map every turn), the G1 predicate (`g1_satisfied`), protection/compaction helpers (`protected_steps`, `load_bearing_steps`), and the generic stratification detector (`code_shows_stratification`). |
| `synthesis.py` | The `Synthesizer` agent: assembles raw evidence, re-derives the answer, emits a verdict (FINAL or NEEDS_MORE_WORK) and a briefing, and applies the non-final G1 backstop. The G1b/G2/G3 disciplines (including the G3 estimand-coverage self-gate) live in its system prompt and route through the same NEEDS_MORE_WORK pushback. |
| `kernel.py` | `PersistentKernel`: the long-lived worker process, one namespace, crash isolation, checkpoint-anchored tail replay/restore, and the namespace registry (`describe_namespace`, which now also surfaces user-defined functions by signature). |
| `executor.py` | Stateless code-handling helpers reused by the kernel and loop: the security `BLACKLIST` for generated code, code-fence extraction (`extract_code`), DataFrame serialization, and temp-file management. It no longer executes code; live execution is owned by `PersistentKernel` (the old out-of-process `CodeExecutor` and its runner machinery were removed in the 6.1 cleanup). |
| `toolkit.py` | The vetted methods toolkit (section 6.8): three tested estimators (`paired_ability`, `cluster_bootstrap`, `rank_uncertainty`) preloaded into the kernel namespace at worker start. Pure numpy/pandas, print-free, defensive errors. Governed by the admission rule in its module docstring (evidence ticket required, cap of five). |
| `llm.py` | Provider abstraction: `AnthropicProvider`, `OpenAIProvider` (Responses API), `OpenAICompatProvider` (one chat-completions class behind the `_ollama_provider`/`_openrouter_provider` factories), the dispatching `LLMClient`, prompt caching (`build_cached_messages`: Anthropic `cache_control` blocks, GPT-5.6-via-OpenRouter `prompt_cache_breakpoint` parts, flattened strings elsewhere), `CostTracker` (cache reads and writes), `RunLogger` (writes `run_log.json` into `<output>/logs/<timestamp>/`), `RunStats` (a per-run events sink), `build_run_telemetry` (aggregates `run_telemetry.json`), `search_call` (Anthropic web search), and `parse_model_string`. |
| `prompts.py` | All static model-facing text: system prompts and templates for the three agents, plus every directive template. The single place to edit instruction wording. |
| `dataio.py` | Dataset loading (`load_dataset`) and data-derived schema construction (`build_schema`) with no domain hints. |
| `ui.py` | Terminal styling and the run's human-readable console output (iteration banners, agent lines, synthesis status, search notices). |
| `logger_config.py` | `get_logger`; the logging configuration used everywhere. |

### 2.2 How a single turn flows

1. `run_investigation` computes the **live** namespace registry for the Investigator
   (`describe_namespace(names=_live_names(kernel, log))`).
2. The **Investigator** call reads: the seed, the schema, the compacted per-step history,
   the live registry, and the rendered ledger. It emits `###THINKING###`, a `###STATUS###`
   (CONTINUE / SYNTHESIZE / SEARCH), a `###SPEC###` when continuing, a `###LEDGER###`
   (the whole ledger, re-emitted), and optionally `###REHYDRATE###` or `###QUERY###`.
3. The ledger block is applied to `NavState` (`apply_ledger_block`); a garbled block leaves
   the existing map intact.
4. On **CONTINUE**: the **Executor** receives the spec plus a **focused** registry
   (`describe_namespace(names=_referenced_names(spec, kernel))`), writes pandas, and the
   kernel executes it. The result becomes a new step in the log; an `analysis.md` artifact
   is written for that step.
5. On **SEARCH** (only if enabled and budget remains): the query runs once via `search_call`,
   the finding is recorded as a protected, always-full evidence item used for calibration.
6. On **SYNTHESIZE**: the gates run (section 2.4), then the **Synthesizer** re-derives over
   raw evidence and returns FINAL + briefing or NEEDS_MORE_WORK. The loop either finishes,
   pushes back, or (now) forces a provisional briefing.

### 2.3 Core objects and how they relate

- **`PersistentKernel`** owns the execution namespace. The Investigator and Executor never
  touch it directly except through specs and the registry it exposes; the kernel runs the
  Executor's code, persists derived objects, and reports state via `describe_namespace`.
- **`NavState` (+ `Entry`)** is the investigation's working memory. It holds four kinds of
  handle (frontier, regime, risk, breakdown), each with a status and a list of step pointers
  back into the log. It is the **only** state the hard gates consult, and it is re-emitted in
  full by the Investigator every turn. It serializes losslessly (`to_dict` / `from_dict`) for
  resume/extend.
- **The log** is an append-only list of step dicts (spec, code, stdout, error, attempts,
  thinking, terminal flag, and per-step metadata). It is the source of truth for the raw
  evidence; the Synthesizer always sees it in full, even for steps the Investigator sees only
  as collapsed pointers.
- **The spec** is the contract between Investigator and Executor: a closed, prose
  description of one analysis step. It references existing derived objects by their exact
  registry names.
- **Directives** are one-shot instructions the loop injects into the next Investigator turn
  (G1 gate, synthesis gate, midpoint course-correction, search-spent, extend-ledger). They
  are consumed after one turn.

### 2.4 The guardrails

- **G1, look within before declaring null/uniform.** The effect must be examined within at
  least one regime axis before a null or uniform answer is admissible. Enforced as a hard
  gate in two places (a pre-synthesis check in the loop and a backstop inside the
  Synthesizer). As of the latest work this is **code-grounded**: a regime counts as examined
  if the model marked it OR the executed code actually stratified the analysis (section 6).
- **G1b, when the effect varies, lead with the conditional estimate.** If the effect varies
  materially across the levels of an examined modifier, the headline must lead with the
  per-level estimate where it is identifiable; a pooled, averaged-over-levels figure may
  appear only as a clearly labeled secondary number. This exists to prevent a marginal
  average from masking or inverting a within-level effect.
- **G2, bound, do not pretend to a point.** When a confound cannot be resolved, report a
  signed bound, never a clean point estimate.
- **G3, a null needs direct estimand coverage.** Before a FINAL verdict of null, negligible, or
  unidentifiable on the primary question, the Synthesizer must confirm that some analysis directly
  estimated the named TARGET ESTIMAND (with the confound matched or stratified, not the data
  discarded) and that the null is reconciled against any retrieved calibration plus an explicit
  identifiability or power statement; otherwise it returns NEEDS_MORE_WORK through the existing
  pushback loop. The Investigator names that estimand once on the first step (pinned in the map),
  so the primary answer cannot drift to a proxy. Unlike G1 (code-grounded), G3 is a soft prompt
  discipline; it forces an attempt plus a justified null, never a fabricated number, and its
  real-run effect is still pending validation (section 6.1).
- **G4, do not defer the method that would decide the answer.** Before a FINAL verdict, if the
  answer makes a decisive claim (a superlative, ranking, outlier, or precise margin) that rests on
  an untested or weak-flagged assumption, and a feasible stronger method would test it, the
  Synthesizer returns NEEDS_MORE_WORK asking for that method together with its uncertainty, instead
  of filing it under future work. It is paired with a METHOD ADEQUACY instruction telling the
  Investigator to name the standard estimator for the data's structure up front. Like G3 it is a
  soft prompt discipline; it shares the same capped pushback budget, pushes to produce the better
  estimate (never a retreat to "undetermined", with G3 as the anti-null backstop), and finalizes
  once the named method has been attempted rather than hunting a more elaborate one. Its effect
  scales with the reasoning model and is still under validation (section 6.1).

All five are generic and name no dataset terms.

---

## 3. Agents, models, and their purpose

There are three agents. They are distinguished by role and by which model tier they use, not
by provider; any agent can run on any provider.

| Agent | Tier (default) | Calls per turn | Purpose |
|---|---|---|---|
| **Investigator** | premium (Anthropic) | one | Integrates raw evidence, updates the ledger, decides CONTINUE / SYNTHESIZE / SEARCH, and writes the closed prose spec. This is where the real thinking happens. |
| **Executor** | cheap (Ollama) | one per CONTINUE (plus mechanical retries) | Turns the closed spec into pandas and runs it in the kernel. No reasoning about the question; just faithful translation. |
| **Synthesizer** | premium (Anthropic) | once at finish (plus pushbacks / periodic snapshots) | Re-derives the answer over the full raw evidence, emits FINAL + briefing or NEEDS_MORE_WORK, and enforces the non-final G1 backstop. It never stores a prior conclusion as a premise, which is what lets an extension overturn an earlier finding. |

**Model wiring.** `--investigator-model`, `--executor-model`, and `--synth-model` set each
independently; defaults fall back to a premium model for Investigator and Synthesizer and a
cheap model for Executor. Model strings are parsed by `parse_model_string` into
`(provider, model)`.

**Providers (`llm.py`).** Anthropic, OpenAI, Ollama, OpenRouter, behind a common `LLMClient`.

- **Prompt caching is per provider** (`build_cached_messages`, 6.19). Anthropic models take
  `cache_control` breakpoints; `openrouter:openai/gpt-5.6*` models take the equivalent explicit
  `prompt_cache_breakpoint` markers on the first and last stable block (verified forwarded on
  chat completions, and billed: reads at the 90% discount, writes at 1.25x input, both
  accounted); xAI models rely on their automatic cache plus the per-run affinity header; and
  every OpenRouter request carries a per-run `session_id` so sticky routing keeps hitting the
  same cache. On Ollama there is no caching, so the full context is re-sent and re-billed every
  turn, the backdrop for several context-size observations in section 7.
- **`search_call`** is the Anthropic web-search path, pinned to a Haiku-class model and logged
  as "Literature Search."
- **`CostTracker`** accumulates token usage and cost; **`RunLogger`** writes the per-call
  trace to `run_log.json` (the artifact we analyze after every run).
- **The Executor runs with reasoning disabled on Ollama** (`reasoning_effort="none"`, 6.10).
  It is a mechanical transcriber, so a chain-of-thought buys it nothing and a reasoning model in
  that seat truncates by spending its whole output budget thinking. The effort is a per-agent
  module constant in `llm.py` (`EXECUTOR_REASONING_EFFORT`, `DEFAULT_REASONING_EFFORT`).
  OpenRouter treats `none` as a first-class rung (6.11) and receives it verbatim; glm on Ollama
  is the exception, where any explicit effort value triggers empty completions on their `/v1`
  endpoint, so the field is omitted for that combination (`_provider_effort` returns None) and
  the model thinks at its own default.
- **The Investigator and Synthesizer reasoning effort is set by `--reasoning-effort`** (default
  `medium`, 6.15), the one reasoning knob exposed as a run flag. The value is translated per
  provider in `llm.py` (`_provider_effort`): Ollama unchanged, OpenRouter `max` to `xhigh`, and a
  direct `anthropic:` model gets no effort field. The Executor is unaffected and stays at `none`.

---

## 4. Key shared and maintained storage objects

### 4.1 In-memory (live during a run)

- **Kernel namespace**: every derived object the Executor creates, held in the worker
  process. Surfaced to the agents through `describe_namespace`, which is now newest-capped
  and can be filtered to a subset of names (section 6).
- **`NavState`**: the ledger described in 2.3. Mutated each turn by `apply_ledger_block`.
- **The log**: the append-only evidence record. Drives compaction, the Synthesizer's
  evidence assembly, and the resume/extend machinery.
- **`briefing`**: the final (or provisional) briefing text, returned from
  `run_investigation` and written to disk.

### 4.2 On-disk (the output directory)

| File | What it is | Used for |
|---|---|---|
| `briefing.md` | The final or provisional briefing. | The deliverable; the answer to evaluate. |
| `logs/<timestamp>/run_log.json` | Per-call trace (agent, model, tokens, cache fields, timings, cost, input, output). One folder per invocation. | Post-run analysis: context growth, retries, gate behavior, which entry produced the briefing. |
| `nav_state.json` | Serialized `NavState`. | Resume/extend; inspecting the final ledger. |
| `log.json` / `kernel_history.json` | The step log and the kernel's replayable history. | Resume/extend; reconstructing the namespace. |
| `seeds.json` (with `seed.txt` back-compat) | The seed-question history. | Extend mode threads all prior questions co-equally. |
| `logs/<timestamp>/run_telemetry.json` | Per-invocation rollup: run, cost, tokens, calls, per-agent, reliability, gates, estimand, plus a human-readable `summary_text`. | Budgeting and dissecting a run (retries, gate hits, truncations, timing split). |
| `analysis/.../analysis.md` (per step) | Each step's move, rationale, code, and output. | Human audit trail of the investigation. |
| `landscape_stepNN.md` | Periodic holistic snapshots (only if `--periodic-every` > 0). | Mid-run course-correction record. |

**Resume vs extend.** `--resume` continues an interrupted or finished run from its saved
state (kernel history replayed via `restore_history`, `NavState` rehydrated via `from_dict`,
log appended, step numbering continued via an offset; a finished run's terminal entry is
popped so it can continue). `--extend` requires a new seed and **reopens and reconciles**:
because the Synthesizer never stored the prior conclusion as a premise, an extension can
overturn it. On extend, all questions are fed to the Synthesizer co-equally (original first)
to prevent the newest question from crowding out earlier ones.

---

## 5. The F1 GOAT benchmark and how we use it

### 5.1 The dataset and the question

`f1_driver_vs_car.csv`: 26,668 driver-race rows across 1950 to 2024, 67 columns, 861 drivers,
210 constructors, 1,125 races. Per-row fields include season and round, constructor, grid and
qualifying position, finish position and status, points, lap-time aggregates, a mechanical-DNF
flag, championship standing, and a set of pre-computed teammate-relative deltas: each driver's
grid, finish, qualifying-time, and lap-time delta versus the mean of their same-constructor peers
in that race, a `same_constructor_peer_count`, and a `beat_same_constructor_peer_mean_flag`.

**Seed question.** Identify the greatest F1 driver of all time. Raw career totals are confounded
by car quality and era length, so use teammate pairings as natural experiments to isolate driver
skill from machinery, find which independent dimensions best separate elite drivers from good
drivers in good cars, and determine whether one driver is a clear statistical outlier or whether
greatness is multidimensional.

### 5.2 The ground-truth answer (what a correct run should land on)

The reference is a revised expert analysis (`F1_GOAT_report.md`), a version 2 produced after an
independent stress-test retracted the version 1 claim of a single outlier. Its defensible
conclusions:

- **No single clear outlier.** Once measurement is on firmer footing, no driver is statistically
  separable as the lone greatest; the top is genuinely contested.
- **Era-separated headline.** In the high-validity modern era (1989 to 2024, genuine two-car
  teams), Max Verstappen is the most likely number one, P(#1) about 64% in a cluster bootstrap,
  with Norris and Leclerc the credible challengers. In the lower-validity classic era (1950 to
  1988), Juan Fangio is the top driver of his era, in a dead heat with Ayrton Senna's early-career
  record. An all-era bridge model puts Verstappen and Fangio as co-leaders within overlapping
  uncertainty.
- **Greatness is multidimensional, but modestly.** The best qualifier, racer, and most consistent
  driver are different people, yet the skill dimensions are less independent than a naive metric
  implies.
- **The central methodological facts.** Teammate comparisons isolate the car, but their validity
  is era-dependent. In early F1 a single constructor fielded many works, customer, and privateer
  cars in one race, so a same-constructor "teammate" is not a two-driver team. That causes
  pseudo-replication (a 13-car group can generate dozens of pseudo-independent pairs, inflating
  early drivers), concentrated exactly on the early greats (Fangio's comparisons are almost
  entirely multi-car groups, versus zero for Verstappen and Senna). The benchmark corrects this
  with a group-weighted Bradley-Terry model (each constructor-race contributes total weight one),
  era separation, and a cluster bootstrap resampling by constructor-race for uncertainty (P(#1),
  rank intervals). It excludes the Indianapolis 500 (an oval field that did not contest the
  championship) and keeps achievement (titles, win rate) out of the headline skill measure because
  it reintroduces the car/era confound.

### 5.3 Why it is hard

It requires rejecting raw totals, recognizing that the early-era same-constructor "teammate" is
not a true two-car teammate and correcting the resulting pseudo-replication, reaching for a
paired-comparison model (Bradley-Terry over the comparison network) rather than averaged deltas,
quantifying uncertainty before any "best" or "outlier" claim, and the discipline to refuse a single
cross-era number when the evidence does not support one. The version-1-to-version-2 retraction is
itself the cautionary tale: a confident single-outlier claim did not survive the stress-test.

### 5.4 How we evaluate runs

Each run's briefing is scored 1 to 100 against the benchmark (the benchmark is the 100 reference).
We read both the `briefing.md` and the `run_log.json`, never the briefing alone, because a run can
fail to write a fresh briefing; the Synthesizer's final entry in the run log carries the briefing
it actually produced. We read trajectory quality (which analyses ran, what was foreclosed, whether
the decisive method was reached) alongside the findings.

**Run history (scored vs the benchmark, 100 = reference; these are assessed scores, not a formal
harness).**

| Configuration | Score | One-line outcome |
|---|---|---|
| glm, no clauses | ~70 | right headline off composites; no formal model, no uncertainty |
| Opus, no clauses | ~76 | headline plus a real opposition-strength insight; deferred the formal model |
| Opus + method-adequacy clauses | ~82 | built Bradley-Terry plus a 300-replicate bootstrap; P(Verstappen #1) ~0.68, matching the benchmark's 64% |
| glm + clauses (two runs) | ~72-73 | named the decisive model both times, finalized on proxies; no uncertainty; gate never fired |
| glm + ONE MOVE rule | ~73 | decomposition fixed (0 of 14 multi-part); still deferred; regressed to a Clark three-sigma headline |
| Opus + ONE MOVE rule | ~82 | finer decomposition; both dimensions modeled with CI overlap; reliability confound resolved |
| glm + toolkit, run 1 | ~78 | toolkit invoked (BT, P(rank 1)); Indy 500 excluded (first run ever); hedged-outlier headline contradicting its own uncertainty |
| glm + toolkit, run 2 | ~82 | matches the benchmark headline; era-stratified; opponent-quality confound quantified; one-move held (0 of 12) |
| Opus + toolkit | ~83 | four ability models (finish, qualifying, modern-only, DNF-cleaned); cross-era non-identifiability stated; cost down 30% |

The trajectory tells one story. The method-adequacy clauses lifted only the capable brain (Opus 76
to 82) because they are soft self-gates; glm named the decisive method and deferred it regardless.
The ONE MOVE rule converged the two brains' spec mechanics while leaving their findings nine points
apart, isolating the residual as brain capability. The toolkit (section 6.8) then closed most of
that residual: the first run in which the decisive method cost one call is the first run glm
invoked it, confirming the activation-energy diagnosis, and glm moved from 73 to 78 then 82 while
Opus held at 83 with 30% lower cost and four fitted models instead of two. Both brains now land
benchmark-consistent conclusions on this dataset.

The residual gaps are shared across brains and recognition-bound rather than cost-bound:
`cluster_bootstrap` went unused in all four F1 toolkit runs because neither brain diagnoses the
diffuse pseudo-replication there (the Opus F1 run even settled for analytic CIs where its
pre-toolkit run had hand-built a bootstrap; see the substitution note in 6.8, which the
out-of-sample run later closed in the toolkit's favor), and opponent-quality weighting stayed an
open question in every run. glm also shows verdict-layer variance worth watching rather than
patching: run 1 led with an outlier claim its own P(rank 1) computation undermines and slipped on
one-move compliance (7 of 13 multi-part), while run 2 held both disciplines cleanly.

---

## 6. Recent fixes and enhancements

This section is the most useful for picking up where we left off. Items are newest first.

### 6.21 Print budget and kernel float format: attacking prompt weight at the source (shipped, validated on three live runs)

**Why.** An audit of a real 17-turn heavy run, reading the final Investigator prompt the way the model does, located the noise precisely: 65% of the prompt was the 4-5 full resident steps, 83% of that raw mass was numeric table rows (median step stdout 17,991 chars, near the 20K resident ceiling), and about 6% of the whole prompt was float digits past the fourth significant figure. Cross-step duplication (383 chars) and registry staleness (0 stale objects) were measured and ruled out as targets. Compaction (6.20) bounds accumulation but deliberately never touches the last three residents, so the remaining mass had to be attacked where it is created: the prints the specs ask for.

**Fix (files: `prompts.py`, `kernel.py`; test: `test_print_discipline.py`).** A PRINT BUDGET clause in both prompt modes' spec-writing rules, directly after the closure rule: decision-sufficient prints, not listings (top and bottom rows capped at ten per side, counts, summary statistics, correlations and shapes rather than every row), with the reminder that derived objects persist so any exact slice is one named print away. The clause itself passes the spec leakage audit it sits beside, which the test pins. The kernel worker prints floats at 4 significant digits (`display.float_format`), display-only, inside the existing anti-truncation block; stored values keep full precision.

**Status.** Suite 31 files green. Validated on three live runs of the F1 seed. Median step stdout fell 17,991 to 2,766 chars (max 37,663 to 7,189); over-precise floats 2,098 to 25; the late-run prompt 136K to 58K chars, with input flat at 20-21K tokens from turn 12 of a 19-turn run; the Synthesizer's full-history feed (11 steps) fit in 21K tokens; zero REHYDRATE requests (the under-printing signal never fired) and zero empty completions; complete-run cost $0.43-0.67 against $1.22 for the pre-change heavy run. The history budget (6.20) never engaged at this step weight, which is the intended end state: a backstop. Watch item: top-and-bottom-k sorted views may lean ranking-flavored briefings toward outlier headlines; across three F1 runs the conclusions stayed coherent (the pooled-modern and era-local lenses of one answer), but if the lean recurs, the clause gains a line asking for the CI-overlap summary alongside rankings.

### 6.20 History-budget compaction: tiered demotion with an inert default (shipped, verified live)

**Why.** Same-seed runs vary in length, and every marginal turn is the priciest of the run: per-turn cost rose about 3x from turn 3 to turn 16 on measured heavy runs because the prompt floor grows with depth (full residents near the 20K ceiling, a ~1.9K-char headline per collapsed step forever, an 11-18K-char volatile tail). Compaction itself was verified healthy against those runs (collapse counts climbing, residents capped, the modules hash-identical to baseline); the issue was that the floor it compacts to rises with every completed step.

**Fix (file: `investigation.py`; test: `test_compaction_budget.py`).** `_history_blocks` takes `char_budget` (default None, off; the Investigator's decide passes `HISTORY_CHAR_BUDGET = 90_000`). Under budget the rendering is byte-identical to the unbudgeted path. Over budget, steps demote oldest-first, cheapest-fidelity-first, re-measuring after each single demotion: collapsed headlines slim to a one-sentence SPEC (`SLIM_SPEC_CHARS`), the oldest slims fold into a single chronological ARCHIVED STEPS block (one line each, so size asymptotes instead of growing with every step), then older PROTECTED residents trim to `PROTECTED_SLIM_CEILING = 4_000` with a calm recovery notice. Red lines are never demoted whatever the budget: the last three recents, search blocks, and this turn's rehydrates; a run whose untouchable core exceeds the budget returns best effort. Every demoted form keeps the REHYDRATE affordance, the raw stays on disk, and the Synthesizer feed is untouched, so demotion is recoverable rather than lossy. Demotions rewrite deep history and reset the cached prefix at the first changed block; measured runs already reset to the seed almost every turn under one-collapse-per-turn churn, so the marginal cache cost is small against the fresh-input savings.

**Status.** Suite 30 files green at ship. Replaying the real 17-step F1 log: byte-identical through turn 14, engagement exactly at the crossing, 30-35% late-turn reductions in the protected-resident scenario that matches the heaviest measured run. Verified live twice: a 9-iteration run crossed nothing (no markers, $0.44 total), and after 6.21 landed, a 19-iteration run stayed under budget for its whole length.

### 6.19 GPT-5.6 explicit prompt caching via OpenRouter, cache-write accounting, and sticky sessions (shipped, validated live to the cent)

**Why.** `openrouter:openai/gpt-5.6-terra` cached only its first call. GPT-5.6's GA (July 9) replaced automatic prefix matching with breakpoint-based matching under a mandatory `prompt_cache_key` (implicit mode places one breakpoint on the latest message), and delv-e's non-Anthropic shape is one giant rebuilt user message, so only byte-identical full prompts could ever hit. OpenRouter's docs said the new explicit breakpoints were Responses-API-only on their platform; a three-call probe showed chat completions forwards them (call 2, same stable block with a different tail: cached 2,775 of 2,790), so the small fix was viable.

**Fix (files: `llm.py`; test: `test_gpt56_cache.py`).** `build_cached_messages` grew a branch for `openrouter:openai/gpt-5.6*` models with non-empty stable blocks: chat-parts content with `prompt_cache_breakpoint` on the first and last stable block (the Anthropic placement), staying in implicit mode so OpenAI's free latest-message breakpoint rides along. Part texts carry LEADING separators so appends never rewrite cached bytes, and the concatenated parts are byte-identical to the flattened prompt every other model receives (a pinned fidelity invariant). `_flatten_messages` passes marked part lists through untouched and flattens everything else byte-identically. Wire capture gained `_last_cache_write` beside `_last_cached` (both via a generic `_usage_detail`), flowing explicitly through `_sync_wire` into call meta, `CostTracker`, `RunLogger` rows, and the per-agent telemetry as `cached_write`. `compute_cost` takes `cache_write_tokens` with include-semantics (OpenAI's prompt_tokens INCLUDES the cached and written portions); the fallback write rate is 1.25x input (never under-report; the same premium the function already charges Anthropic writes). PRICING gained terra's `cached_write` 3.125 and the luna entry (1.00/6.0, cached 0.10, write 1.25). The per-run UUID was renamed `_XAI_CONV_ID` to `_RUN_ID` (it now feeds three affinity mechanisms), and every OpenRouter request carries `session_id`, because OpenRouter's default sticky-routing hash keys on the first non-system message, which delv-e rewrites every turn.

**Status / findings.** Validated live: all nine charged rows of a terra run matched OpenRouter's console to the cent (reads 0.25/M, writes 3.125/M); cached climbed 0, 7,220, 17,036, 26,002 across turns with 42% of Investigator input served from cache; on frontier misses cached pinned at exactly the seed prefix, proving the first-block fallback marker is honored (multiple explicit breakpoints work). $0.00 console rows are OpenRouter's zero completion insurance on empty or error completions, not missing charges: the ledger books insured empty finals (safe-side over-report, $0.10 on the measured run) and never books unlogged guard first attempts. Churn economics: under collapse-every-turn history, 5.6's frontier rarely survives and the run pays 1.25x rewrites beyond the seed (implicit mode would write them anyway); a heavy luna run still netted +11% input-side. Grok is untouched byte-for-byte (xAI reports no write accounting; per-row math identical before and after). The direct `openai:` path (Responses API) is deliberately excluded from the branch. gen_id capture stays out per the standing rule, but the concrete need has now arrived (row-level invoice reconciliation and empty-attempt forensics), so it is queued as a small opt-in on the non-streaming path when wanted.

### 6.18 llm.py structural refactor and two latent logging bugs (shipped, wire-diff-verified byte-identical)

**Why.** Three near-identical provider classes carried the OpenAI-compatible dispatch in triplicate, and the stream wrapper's empty-completion guard still held a duplicated (and once misindent-scarred) dispatch. Reading every method before rewriting also surfaced two live bugs: `RunLogger.log` had a duplicate `"reasoning_chars"` dict key whose second, dead entry (a getattr on the logger's own self) shadowed the explicit parameter, so every logged row recorded reasoning_chars=0; and `search_call` read `self._last_cached`/`_last_reasoning_chars` directly off the client, raising AttributeError when a search was the first call and otherwise attributing the previous call's wire stats to the search row.

**Fix (file: `llm.py`; tests: existing suite).** `OllamaProvider` and `OpenRouterProvider` collapsed into one `OpenAICompatProvider` (constructor takes a resolved base_url, api_key, an extras builder mapping (model, effort) to (extra_body, extra_headers), and optional default headers); zero-arg factories `_ollama_provider`/`_openrouter_provider` resolve env vars, and the registry keeps its name and shape (values are zero-arg callables, so test doubles drop in unchanged). Every provider quirk lives in the extras builders. `OpenAIProvider` stays separate deliberately: it speaks the Responses API (instructions/input, max_output_tokens, different stream events, no temperature), so folding it into a chat-completions class would have changed its wire behavior. Both LLMClient guard paths became a single for-attempt loop with the warning text intact. The logger fix deletes the dead shadowing line (rows now record real reasoning-channel sizes); search rows log honest zeros, since that path carries no cache or reasoning telemetry.

**Status.** Suite green; behavior proven byte-identical by a wire-diff harness driving the pre- and post-refactor modules side by side through a fake SDK across a 16-scenario effort-and-quirk matrix: every recorded constructor and request kwarg, return tuple, meta dict, and side-channel identical, with the shared run-id and timeout contracts asserted. llm.py went 1,230 to 1,199 lines at that point (the earlier 200-250 line estimate assumed more removable mass than the 221-line source region contained).

### 6.0 Toolkit legend, anchor-SE fix, budget notice, GATES block, and the `--verify` feature (shipped over one session, suite green)

A long single-session arc; the per-pass development log and the benchmark score tables it carried were removed in the compaction. What shipped, all still live:

- **Toolkit output schemas in the legend (`prompts.py`).** The TOOLKIT block lists the exact output columns/keys of each estimator and tells specs to reference them by name rather than by description; `tests/test_toolkit.py` runs the real functions and asserts the legend matches (the 6.2 three-way-contract discipline applied to the toolkit).
- **Reference-anchor SE (`toolkit.py`).** `paired_ability` returns `se=0.0` for the reference row instead of NaN, so the `paired_ability`-to-`rank_uncertainty` composition needs no hand-imputed SE and the anchor participates in every draw. Trade-off noted in 6.8: a zero-SE anchor makes `rank_uncertainty` freeze the anchor's rank.
- **Budget wrap-up notice (`prompts.py`, `investigation.py`, `llm.py`).** The Investigator sees no step counter through the body of a run, so the budget can never anchor as a quota; only in the final window (last fifth, minimum two turns; `_budget_window`) does a tail notice appear, paired with a standing license to synthesize as soon as the evidence supports it. Counted as `gates.budget_wrapup_notices`; `tests/test_budget.py`.
- **GATES block plus the estimand-discovery, rank-pool, composite, and grouped-uncertainty clauses (`prompts.py`, `synthesis.py`).** The Synthesizer deliberates each gate (pass/fail/na with a grounded reason) in a `###GATES###` block before the verdict; the clauses are generic disciplines (filter low-evidence pools before ranking; do not headline an asymmetric composite; respect a shared grouping axis in uncertainty; on discovery-type seeds the discovery is itself the estimand). `tests/test_synth_gates.py`.
- **The `--verify` audit feature (`verify.py`, `prompts.py`, `run_core.py`; `tests/test_verify.py`).** A fresh independent second pass over a finished run: claim extraction, audit under a hard-coded stress battery, reconciliation into one corrected briefing (originals kept alongside). Built deliberately not on `--extend` so the audit inherits no evidence chain and cannot launder the original's artifacts. Section 2 and the README describe the live behavior; the reconciliation prompt carries hard-won deference rules (treat a refutation as decisive only when the audit reproduces the discrepancy at the original's level of analysis and states each side's computation, and never adopt an unreproduced mechanism as fact). Each of the three verify prompts has a compute-mode variant (6.17); an audit runs in the mode of the run it audits, read from that run's `run_meta.json`, and its original question is the audited run's whole seed chain rather than just the last seed.
- **Ship-discipline rules adopted from a duplicate-implementation incident:** every scripted edit asserts its match count and writes nothing on failure; the ship checklist diffs against the staged snapshot before rebuild; CLI-path features get `main()`-level smoke tests, because function tests cannot see wiring chimeras.

Open pre-commitments still unfired (future-work triggers): a stock-and-flow sentence (ship on a second dataset that clears a flow channel while missing a stock-channel change); a coverage-sensitive-aggregate clause (sums and counts adjudicated under inclusive criteria need a coverage control); and a synthesizer-side grouped-uncertainty gate if a second primary run uses naive errors on group-structured inference.

### 6.1 Pre-release cleanup, STATUS hardening, executor cap, estimand coverage (G3), method adequacy (G4), one-move-per-spec, and telemetry (mixed status, see each)

Pre-GitHub work, shipped together.

- **Codebase cleanup (verified).** The working tree was trimmed from 20 modules (~12,973 lines, ~70% leftover original-delv-e code) to the 11 live modules plus `tests/`; nine unreachable modules, `executor.py`'s out-of-process layer (442 to 163 lines), and assorted dead files and helpers were removed. The portable `tests/` suite was added (each test self-bootstraps its `sys.path`, a tiny `httpx` stub is bundled, output goes to a tempdir, and the dataset-backed tests skip cleanly when the CSV is absent).
- **STATUS reader hardening (shipped, preventive).** `_decision_from_status` takes the decision from the leading token (whole-word fallback, CONTINUE wins ties), so a prose-wrapped status like "CONTINUE, not ready to synthesize" cannot misfire into a premature finalize or a spurious search. `tests/test_status.py`.
- **Executor cap 16k to 20k (shipped, since superseded).** Bought headroom against kimi truncation; the durable fix was disabling the Executor's Ollama reasoning (6.10), and the shared cap is now 64k (6.15).
- **Estimand coverage / G3 (shipped as prompt discipline; not yet fired on a real null).** The Investigator pins the TARGET ESTIMAND once on step 1 (`###ESTIMAND###`, `NavState.target_estimand`, rendered every turn; the question is pinned but the operationalization stays free to refine); G3 refuses a FINAL null/unidentifiable verdict on the primary question until some step directly estimated the named estimand (confound matched, answer-bearing data not discarded) and reconciled the null against an identifiability/power statement. Deliberately generic vocabulary; `tests/test_estimand.py`.
- **Method adequacy / G4 (shipped as prompt discipline; bites for a strong brain, under-fires for a weak one).** A METHOD ADEQUACY section tells the Investigator to name and use (or justify skipping) the standard estimator for a recognized data structure; G4 refuses to defer a feasible decisive method under FINAL. Three brakes against never-finishing: it shares the capped pushback budget, it pushes to produce the estimate (G3 backstops the null), and it must finalize once the named method is attempted.
- **One move per spec (shipped, decomposition verified).** A closure rule of one computation per spec, dependent moves sequenced across steps; on the post-clause glm rerun multi-part specs fell from 5/7 to 0/14 and mean spec length from 354 to 223 words, matching Opus's decomposition. It fixed reliability and decomposition, not the analytical ceiling (that is the toolkit, 6.8).
- **Run logging and telemetry (shipped, verified).** Per-invocation logs land under `<output>/logs/<UTC>/` (`run_log.json` plus the enriched `run_telemetry.json`, written once at run end); the checkpoint trio and the deliverables stay in the output root, so `--resume` and `--extend` are unaffected. `build_run_telemetry` reduces over the per-call, cost, and step data that already existed; a thread-safe `RunStats` sink counts the loop events that are not call numbers (gate overrides, pushbacks, truncation retries, searches, the provisional flag). `tests/test_telemetry.py`.

### 6.2 Ledger render/parse format alignment (shipped, verified)

**Why this matters.** A silent, unrecoverable context-corruption bug and the canonical example of a whole failure class. `render_for_investigator` showed the ledger as `label [status] steps:ids` (a bracket shape) while `apply_ledger_block` expected `KIND | label | status | steps` (a pipe shape); over the ledgers glm actually emitted, only three of nine parsed, so six turns of updates were dropped on the floor and the model navigated a stale map (completed work still showing not_examined). The raw evidence was never affected, which is why the system still produced reasoned briefings. This was the true root of the G1 false-gate and of glm's "poor ledger hygiene" reputation (glm echoed the rendered shape faithfully, and the rendered shape did not match the parser).

**The lesson (applies to every structured model exchange).** A model-facing format is a three-way contract: the legend you SHOW, the OBJECT your code renders as the live example the model imitates, and the PARSER that reads the output back must all agree exactly, or the model looks obedient while its output is silently discarded with no error raised. Write all three from one written description of the shape in the same change; review by taking real model output (or a real render), running it through the real parser, and confirming the round trip reproduces the object, never by reading templates and parser in isolation.

**Fix (files: `nav_state.py`, `prompts.py`; test: `test_ledger_parse.py`).** Layer 1 rewrote the `###LEDGER###` legend to the canonical bracket shape, so legend, render, and parser describe one shape. Layer 2 made `apply_ledger_block` tolerant: the kind-source and the field delimiter are orthogonal, so it accepts the bracket, legacy pipe, and bare-header-with-pipe variants, skips only individual malformed lines, requires a real status word (so stray prose and the evidence index are ignored), and leaves the prior map intact on a garbled block. Verified: the ledger now parses in flight and accumulates monotonically across turns.

### 6.3 Investigator truncation handling (shipped, since superseded by 6.11/6.15)

**Why.** A run finalized after three iterations: the fourth Investigator turn hit its 8,000-token output cap mid-reasoning and returned empty, and the parser's markerless fallback treated the empty turn as a decision to SYNTHESIZE.

**Fix (files: `investigation.py`, `llm.py`; test: `test_truncation.py`).** `LLMClient.call` gained an opt-in `return_meta` returning token usage plus a `truncated` flag; `Investigator.decide` marks an empty or truncated turn `incomplete`, and the loop retries it up to `INV_TRUNCATION_RETRIES` times before falling back to a forced provisional briefing (so it never loops forever and never ends empty-handed). The cap and the retry shape have since changed: a 64k shared cap and a hold-then-none retry with a dedicated directive (6.11, 6.15), and the Executor side of this problem was resolved by disabling its reasoning (6.10).

### 6.4 Context-growth and compaction fixes (shipped, verified)

**Diagnosis.** The Executor's task grew every iteration (input climbed from ~1.5k to ~26k chars while the spec plateaued at ~1.2-1.4k). The driver was the `CURRENT NAMESPACE REGISTRY` appended to both agents. Two defects: at the 120-object cap `describe_namespace` kept the oldest objects and dropped the newest (so the Executor went blind to variables it had just created), and the Investigator's tiered compaction was defeated because `protected_steps` kept every step of every open thread. Ollama has no caching, so all of this was re-sent each turn.

**Fixes (files: `kernel.py`, `investigation.py`, `nav_state.py`; test: `test_context.py`).** `describe_namespace` shows the newest at the cap ("+N older hidden"); the Executor receives only the objects its spec names (`_referenced_names`) plus the df columns and a count; the Investigator sees a live registry (`_live_names`: referenced in the last 4 steps or among the newest 30), implemented as a display filter rather than deletion so replay is unaffected; `protected_steps(max_protected=6)` keeps only the latest step of each live, non-foreclosed thread. Verified: Executor input went flat at ~2-6k with zero retries, and the Investigator context came down from a ~94k peak to ~70k as threads closed. The residual driver is now glm's large per-step outputs (the section-7 verbosity item) rather than the old unbounded registry.

### 6.5 Q2: code-grounded G1 backstop (shipped, verified)

**Why.** glm did the within-regime analysis but kept an unstable ledger, so G1 (which read only the self-reported ledger) fired on a run that had actually examined the relevant regimes, overriding a good FINAL to NEEDS_MORE_WORK.

**Fix (files: `nav_state.py`, `synthesis.py`, `investigation.py`; test: `test_g1_fixes.py`).** `g1_satisfied(log)` also counts G1 satisfied when the executed code actually stratified, via a generic `code_shows_stratification(log)` that scans real code for domain-agnostic idioms (`groupby`, `pivot_table`, `crosstab`, `cut`/`qcut`, per-level loops, formula interaction terms; no column names, nothing dataset-specific). It can only upgrade an unmarked ledger, never downgrade, so a sloppy ledger cannot cause a false gate. Intentionally permissive; see section 7.

### 6.6 Q3: never end empty-handed (shipped, tested; production path not yet exercised)

**Why.** When the Synthesizer was overridden to NEEDS_MORE_WORK with the G1 pushback budget already spent, the loop wrote a terminal NEEDS_MORE_WORK entry with an empty briefing; the ceiling net was then skipped (it only fires when no terminal entry exists), so the run returned nothing and the stale `briefing.md` on disk was never overwritten.

**Fix (file: `investigation.py`; test: `test_g1_fixes.py`).** When the synthesizer gates with the budget spent, the loop forces `_final_synthesis(final=True)`, which by construction never gates and always returns a clearly-flagged provisional briefing. A run that did real work can no longer end without one.

### 6.7 Earlier completed work (context for the codebase)

- **G1b guardrail** added to the Synthesizer system prompt (lead with the conditional
  estimate when the effect varies across an examined modifier).
- **`prompts.py` migration:** all static model-facing text consolidated into one module.
  Serializers (schema, namespace, ledger renderer) stay with their data.
- **Stop code-inlining in specs:** prompt tightened so the Investigator writes the spec in
  words and never as runnable code (glm had intermittently inlined whole programs).
- **`--resume` / `--extend`** continuation modes, with the extend-mode briefing-bias fix
  (feed all questions co-equally so the newest does not dominate the briefing).
- **Web search restored, off by default:** SEARCH as a third Investigator status, query in a
  `###QUERY###` block, mid-stream and model-decided, hard per-run budget (default 3),
  Anthropic-only, calibration-only, excluded from the benchmark protocol.
- **UI cleanup:** the ANSI box-drawing logo, removal of result-preview leakage, spec-leakage
  warning demoted to advisory.

### 6.8 Methods toolkit (shipped, validated on the F1 benchmark)

**The problem.** The one-move comparison (6.1) isolated the residual accuracy gap as brain capability: glm names the decisive method (an Elo-style network, a hierarchical model) in its own notes and then finalizes on a proxy, and soft G4 does not catch it. The diagnosis is activation energy rather than missing knowledge: the decisive method costs a 300-word error-prone executor spec while a defensible proxy costs one safe spec, so the brain keeps choosing the cheap branch. The fix collapses the decisive method's cost to roughly the proxy's.

**What shipped (`toolkit.py`).** Three tested, print-free estimators preloaded into the kernel namespace the way `df` is: `paired_ability` (Bradley-Terry via MM, or a network-adjusted linear fit on margins; connectivity and anchoring handled inside; accepts 0.5 as a tie), `cluster_bootstrap` (whole-cluster resampling for honest intervals under pseudo-replication; warns below ten clusters), and `rank_uncertainty` (P(rank 1) and rank intervals from estimates plus SEs or from bootstrap draws). They compose into the certificate chain (estimate, then honest draws, then a probability the leader is really first), the benchmark's headline statistic. Wiring: the Investigator's TOOLKIT block (signatures plus structural triggers, tied to METHOD ADEQUACY, available never mandatory; ~310 cached tokens) and one Executor line (call exactly as specified, never reimplement; ~60 tokens, re-sent). For standard models beyond the three, the Investigator specs the scipy/statsmodels call by name. `tests/test_toolkit.py` holds 57 known-answer checks.

**Governance (the standing discipline).** A function enters the toolkit only after a logged run in which a brain named its method class and then deferred it, and only when no common library offers a turnkey route; a hard cap of five (a sixth must merge with or replace an existing one); admission stays a human decision with known-answer tests, because the toolkit's value is the word "vetted" and self-promotion by the brain that needed the library would reintroduce the gap it closes. Open library-gap watch-list (no tickets yet): a wild-cluster mode, direct standardization, confounding sensitivity bounds (E-values), and matching. Kill criterion: if the queue fills with structurally unrelated methods across datasets, the toolkit is the wrong lever and the honest answer is a better brain.

**Scope, honestly bounded.** The toolkit moves the certificate layer only. The standing risk is availability bias (frictionless estimators lower the cost of fitting models on questions the data cannot answer by as much as on questions it can), and the defenses are the existing ones (METHOD ADEQUACY names the estimator from the structure first, the Synthesizer re-derives over raw evidence, G2 demands bounds). Validation across four F1 runs and the motivating clustered benchmark confirmed the diagnosis (glm invoked the estimators the first run they cost one call, zero reimplementation) and the scope boundary (the residual miss was upstream measure choice, which stays with the brain permanently); the per-run detail was removed in the compaction. One guardrail ticket came out of it: the code-grounded G1 backstop (6.5) can be satisfied by other stratifications while the estimand-named axis goes silently pooled, which is the estimand-substitution risk G3 owns (a G1 sharpening was tried against it and reverted as a logged loss; see section 7).

### 6.9 Dataset-free compute mode, plus three follow-on fixes (shipped, verified)

`--compute` runs the whole engine with no dataset loaded, for questions answered by computation rather than by analysis of a dataframe (Monte Carlo, numerical methods, derivations cross-checked against closed forms). The capability was already latent (the kernel tolerates `df=None`); three product layers blocked it, and this pass removed them without adding a parallel engine.

**Design.** A single `compute` boolean threads through `run_investigation` the way `search_enabled` does. The model-facing text is a bundle in `prompts.py` (`DATA_MODE`/`COMPUTE_MODE` via `mode_prompts(compute)`); everything else is reused (kernel, loop, `NavState`, the ledger render/parse, telemetry, the safety nets, `verify.py`). The compute prompts keep every output marker and the exact ledger shape, reinterpreting the four handles in prose only, so the 6.2 three-way contract is untouched. The statistical gates have no referent, so G1 is bypassed and the compute Synthesizer self-checks UNCERTAINTY/CONVERGENCE/VALIDITY in the same GATES/VERDICT/BRIEFING shape; the Investigator disciplines are C1-C4 (state the model, quantify with a Monte Carlo SE, cross-check a known case, check parameter dependence). About sixty lines of wiring across five modules, no new modules.

**Scope (v1).** Fresh compute runs only at the time; each would need the run mode persisted in the saved state. (`--resume`/`--extend` enabled in 6.16, `--verify` in 6.17.) `tests/test_compute.py` and `tests/test_compute_cli.py` drive `run_investigation` and the full `run_core.main()` respectively.

**Three follow-on fixes from a live compute run.** (1) A second unguarded `df.shape` read crashed post-run telemetry under `df=None` (the ledger bug in miniature: a value read in two places drifts when only one is guarded); fixed, and `test_compute_cli.py` now exercises the `run_core` path. (2) and (3) bounded the Investigator budget and gave the truncation retry a directive; both are superseded by the 64k cap and the hold-then-none retry (6.11, 6.15).

### 6.10 Executor reasoning disabled on Ollama (shipped, verified)

**Why.** A second truncation surfaced on the Executor: a reasoning Executor (kimi-k2.7-code) spent its whole output budget on chain-of-thought and emitted no code (on one run, 9 of 17 Executor calls, all 8 retries, both failed steps, ~12 of 31 minutes). The existing executor-truncation directive told it to emit only code and it ignored it, because a prompt steer does not suppress a reasoning model's thinking. The Executor is mechanical by design, so its reasoning buys nothing.

**Fix (file: `llm.py`; test: `test_executor_reasoning_effort.py`).** Ollama's native `think: false` does not propagate over the OpenAI-compatible `/v1` endpoint `OllamaProvider` uses, but the same endpoint honors `reasoning_effort` as a string and `"none"` turns thinking off (verified live). Two module constants are the knobs, `EXECUTOR_REASONING_EFFORT = "none"` and `DEFAULT_REASONING_EFFORT = "medium"`; the choice lives inside `LLMClient` keyed on the agent label (so the Executor call site and all mocks are unchanged) and is gated to Ollama, since OpenAI and OpenRouter reject `"none"`.

**Outcome.** On the same compute seed, Executor output collapsed from 242k tokens to 8.2k, its wall time from 23 minutes to 45 seconds, truncations from 9 to 0, and the whole run from 31.5 minutes to 9.1, with no scientific regression. A spec-versus-code audit confirmed the reasoning-off Executor still renders complex statistical specs faithfully; the residual executor retries are now genuine Python bugs whose rate reflects how fiddly the specs are rather than analytical quality. The controlled brain-ladder experiment this enabled, and its per-run scores, were removed in the compaction; the founding diagnosis that analytical judgment is brain-bound is recorded in section 1. If the Executor runs on a non-reasoning model (the Anthropic Haiku default) the change is inert and correct.

### 6.11 Token caps unified to 32k, reasoning-effort ladder, and the Anthropic clamp (shipped, verified)

**Why.** The previous regime pinned every agent near 20k and forbade raising it, because the Anthropic non-streaming SDK refuses a `max_tokens` that implies a run past its 10-minute timeout (6.1, 6.3, 6.9). That left no headroom for a heavy reasoning model and made 6.10's "turn reasoning off" the only lever. The durable fix separates the two concerns: a generous shared budget, with the Anthropic limit handled by a clamp rather than a hand-set cap, plus a graded retry on truncation.

**Fix (files: `llm.py`, `investigation.py`, `synthesis.py`, `verify.py`; tests: `test_reasoning_ladder.py`, `test_executor_reasoning_effort.py`, updated `test_compute.py` and `test_executor.py`).** `DEFAULT_MAX_TOKENS = 32000` is shared by all five agents (Investigator, Executor, Synthesizer, ClaimExtractor, Reconciler). `llm.call` clamps any direct `anthropic:` request to `ANTHROPIC_MAX_TOKENS = 20000` (the non-streaming guard); Ollama and OpenRouter take the full 32k, and an `openrouter:anthropic/...` model uses the OpenAI-compatible path and is not clamped. A module-level `call_with_ladder(client, ...)` steps the reasoning dial down (medium, low, none) whenever a turn returns empty or capped, for providers with a dial (Ollama, OpenRouter); other providers make a single call, and a client stub lacking the new kwargs falls back to one plain call, so the test mocks keep working. `default_reasoning_effort(agent)` starts the Executor at `none` and every other role at `medium`; `lower_reasoning_effort` walks the ladder. The Executor, Synthesizer, and the two verify agents route through `call_with_ladder`; the Investigator keeps its own truncation-retry loop (it owns the `ui`/stats side effects and re-sends `DIRECTIVE_TRUNCATED_RETRY`) and steps the same ladder per retry. `INVESTIGATOR_MAX_TOKENS` was removed. OpenRouter now threads the effort string through `{"reasoning": {"effort": ...}}` rather than hardcoding `"medium"`; per OpenRouter's dial, `none` is a first-class value, so the bottom rung needed no special case. The 6.10 "Executor reasoning off" behavior survives as the Executor's default rung, with the wiring ready if it is ever flipped to `medium`.

**Status.** Suite green. Confirmed on a live compute rerun: three truncations, all recovered, and the Synthesizer ladder fired for the first time (an empty first call de-escalated to a complete briefing). The cost wrinkle in section 7 (glm always truncates on the medium rung) is real and noted there.

### 6.12 Kernel reliability: checkpoint tail-replay, longer step timeout, and a compute-executor vectorization directive (shipped, verified)

**Why.** Two failure modes from the early compute runs. First, the Executor wrote scalar Python Monte Carlo loops (tens of thousands of iterations) that exceeded the step timeout, and on a kill the kernel replayed the ENTIRE history, which is O(n) and painful at deep iterations. Second, the timeout itself was tight for legitimate heavy compute.

**Fix (files: `kernel.py`, `prompts.py`; test: `test_checkpoint.py`).** `STEP_TIMEOUT` raised 300s to 600s. After each successful step the worker pickles its derived data objects to one checkpoint file (modules recorded by import name and re-imported on load; functions and other unpicklable objects can't pickle, so the completeness flag stays at the prior step). On a restart the worker loads the checkpoint and replays only `history[_snapshot_through:]`, the tail, falling back to a full replay if the tail errors. `_snapshot_through` advances only when the checkpoint is complete. Separately, `COMPUTE_EXECUTOR_SYSTEM` gained a vectorization directive mirroring data mode's "use vectorized pandas": vectorize simulations over independent replicates as numpy arrays (an active-sample mask plus per-sample accumulators), looping only over time steps, never in Python over individual samples.

**Status.** Suite green (`test_checkpoint.py` exercises zero-replay, tail-replay with a function present, and a real timeout). The vectorization directive is the prompt half; the live compute runs did produce vectorized code, though the Executor still introduces occasional code bugs (section 7).

### 6.13 Hidden-richness compute benchmark, the belief-structure failure, and the C5 discipline (shipped; encouraging, not yet a win)

**The benchmark.** A new dataset-free foraging problem, built and verified this session (ground-truth engine `forage_hidden_engine.py`, answer key `FORAGE_HIDDEN_GROUND_TRUTH.md`): a bee on a patch of unknown type (poor ~3 sips, rich ~20, equally likely) collects nectar in noisy Poisson sips whose rate tracks the draining nectar, and must infer the type from (sips so far, time) and decide when to abandon. The optimal policy is a graded (sips, time) abandonment boundary; the optimal long-run rate is 1.2212 sips/min, verified three ways (a belief-state DP equal to an exact forward pass to machine precision, and Monte Carlo within error). Blind (no learning) is 1.0793; the full-information ceiling is 1.2667. It is a genuine infer-and-decide-under-uncertainty test.

**The failure it exposed.** A live run (glm Investigator and Synthesizer, kimi Executor, on Ollama) returned 1.158 with tight error bars, confidently suboptimal. Root cause, stated in the model's own turn-1 reasoning: it concluded the belief depends on the sip count alone, "independent of time," because the rate ratio cancels at the instant of a sip. That drops the time-exposure term in the Poisson likelihood, the evidence in elapsed time with no sip. It then optimized a one-parameter "leave when the expected instantaneous rate falls below g" threshold by a 500k-visit simulation, never attempting the belief-state DP. Keeping its policy class but fixing the belief recovers 1.2095; the DP optimum is 1.2212. A from-first-principles reproduction confirmed all of this. The run was also expensive at ~48 minutes, mostly because an exact DP is cheap where a large policy sweep is not.

**Fix (file: `prompts.py`; test guard in `test_compute.py`).** A fifth compute-Investigator discipline, C5, gated to learning or partially-observed problems: name the information state and the minimal sufficient statistic (usually both the events seen AND the elapsed time); use the full likelihood including the evidence in what did not happen; check the belief responds to time; keep the generative model separate from the belief; and reach for an exact recursion over the information state before a large policy-search simulation. The header moved from FOUR to FIVE disciplines. A generic "compute economically" block drafted earlier was dropped in favor of this sharper, belief-targeted one.

**Status, stated carefully.** A rerun landed the correct 1.221 via a POMDP value iteration over an (n, T) belief state with the time-exposure term, Richardson-extrapolated, Monte-Carlo-checked, in half the wall-clock with the code overhead collapsing from 824s to 33s, and three truncations instead of eight. That is the right answer by the right method. But C5 supplies the two structural insights the benchmark was built to test, and the rerun's turn-1 reasoning visibly applies C5's framing (its terms, and the opposite-and-correct conclusion drawn from the same cancellation observation that misled the first run). So this is an integration test that the engine plus C5 produces the right answer, not a clean measure of unaided capability. The seed is now contaminated as a capability benchmark (see section 7). One run each way; model variance is not ruled out. Encouraging, not yet a win. A structurally different hidden-state problem is the next step.

### 6.14 Function reuse in the namespace registry (shipped, suite-verified, not yet exercised live)

**Why.** In the 1.158 run, five consecutive compute steps rebuilt the simulation from scratch, badly, instead of reusing the correct `simulate` function step 1 had defined and left live in the kernel. Two causes: the Investigator referenced "the same logic as step 1" instead of naming the function, and `_namespace_summary` dropped functions from the registry entirely (`elif t in ("function", "type", "module"): continue`), so even a named function could not reach the Executor, which sees only objects whose names appear in its spec (`_referenced_names`).

**Fix (files: `kernel.py`, `prompts.py`; test: `test_function_reuse.py`).** `_namespace_summary` now records a user-defined function with its call signature and first docstring line (for example `simulate(g, n_visits=200000, seed=42)`); imported modules and class definitions stay excluded, and the vetted toolkit functions stay out via `_INTERNAL`. The existing exposure machinery does the rest: the Investigator sees functions through the same `_live_names` liveness filter as data, so old throwaways age out, and the Executor sees a function only when its spec names it. The prompts gained a few words in both modes: the Investigator persistence line now reads "objects and functions you create persist," and each Executor prompt gained "when the spec names a function a prior step defined, call it; do not redefine it."

**Status.** Suite green. Not yet exercised in production: in the one compute rerun the Investigator never named a function for reuse, so the path has not fired live (section 7).

### 6.15 64K output cap, the `--reasoning-effort` flag, and a hold-then-none truncation retry (shipped, verified on a live glm-5.2 run)

**Why.** 6.11 unified the agent budget at 32k and added a step-down effort ladder (medium, then low, then none) for truncation. A run-log analysis of a glm-5.2 Investigator on a hard compute seed showed the ladder did not work on that model: both truncations hit exactly 32k of pure hidden reasoning with zero captured text, and the retries ran at `low` while still emitting 24k-30k reasoning tokens, because glm reads any effort below `high` as its maximum (confirmed against the vLLM and SGLang docs). So stepping medium to low was a no-op, and stepping high to medium was internally an UP-step; the ladder thrashed and each truncated turn burned a full 32k call before the productive retry. The same logs showed glm's reasoning fits in 24k-31k tokens when it completes, so a larger budget lets it finish in one call. Two durable changes follow: more headroom, and a retry shape that does not depend on intermediate effort steps a model may collapse.

**Fix (files: `llm.py`, `prompts.py`, `investigation.py`, `synthesis.py`, `run_core.py`; tests: rewrote `test_reasoning_ladder.py`, updated `test_compute.py`).** Three parts.

1. `DEFAULT_MAX_TOKENS` raised 32000 to 64000, shared by all five agents. The direct-Anthropic clamp is unchanged (`ANTHROPIC_MAX_TOKENS = 20000` on the non-streaming path); Ollama and OpenRouter take the full 64k.
2. A `--reasoning-effort` flag (values `max`/`high`/`medium`/`low`/`none`, default `medium`) sets the starting effort for the Investigator and Synthesizer; the Executor is untouched and stays at `none`. A new `_provider_effort(provider_name, effort)` helper in `llm.py` translates the value per provider: Ollama passes it through, OpenRouter renames the top rung (`max` to `xhigh`) and passes the rest through, and providers without an effort dial (direct Anthropic, OpenAI) receive no field. `run_investigation` threads the flag to both agents (`Investigator` and `Synthesizer` now carry a `reasoning_effort`); the Investigator's `decide` defaults each call to that level, and the Synthesizer passes it into `call_with_ladder`. A direct `anthropic:` model ignores the flag (it uses a thinking-token budget, a separate mechanism deferred to its own task); an `openrouter:anthropic/...` model still honors it.
3. The truncation retry was redesigned and the step-down ladder removed (`lower_reasoning_effort` and `_EFFORT_LADDER` deleted as dead code). The new shape, in both the Investigator's own loop and the shared `call_with_ladder`, is hold the chosen effort, then drop straight to `none`, the one value every dialed provider honors as off. For the Investigator (`INV_TRUNCATION_RETRIES = 2`, so three attempts): attempt one at the chosen effort, attempt two holds it and adds the strengthened `DIRECTIVE_TRUNCATED_RETRY`, the final attempt forces `none` and drops the directive since `none` is the fix rather than the nudge. The Synthesizer routes through `call_with_ladder`, which cannot inject a directive, so it collapses to two attempts (chosen, then `none`) with no wasted duplicate. The Executor starts at `none`, so it makes one call and never retries here. `DIRECTIVE_TRUNCATED_RETRY` was strengthened: it opens with "STOP.", states the prior turn was wasted, and demands the decision blocks immediately with thinking capped at two sentences (the distinctive phrase `test_truncation_retry.py` asserts is preserved).

**Status.** Suite green in the sandbox (26 files from a clean extract; the directly affected tests `test_reasoning_ladder`, `test_compute`, `test_truncation`, `test_truncation_retry`, and `test_executor_reasoning_effort` all exercise the new paths), and an end-to-end check confirms the flag reaches the Investigator's first call and the Synthesizer while the Executor stays at `none`. Confirmed on a live glm-5.2 Investigator/Synthesizer run on the hard marathon-peaking POMDP seed (kimi Executor, Ollama): verdict FINAL with no forced provisional, 8 iterations, every truncation recovered. The 64k cap earned its keep, five Investigator turns completed in a single call at 36k-63k output tokens (one 388 tokens under the ceiling), each of which would have truncated under the old 32k cap, so the headroom roughly halved the truncation events on this path. glm still truncates on its hardest turns (four events, each hitting 64k with zero visible output), as expected. Correcting the earlier guess: the strengthened directive is not cosmetic on glm. Three of the four truncation events recovered on the second attempt (chosen effort plus the directive), each completing at roughly 7k-12k hidden tokens rather than 64k, with only one event needing the `none` circuit-breaker on attempt three. So the three-attempt shape is justified rather than redundant on glm. One run, small N, held loosely. Cost is wall-clock, not dollars on Ollama: the five capped calls spent about 27 of the run's ~57 minutes of API time on reasoning that produced no decision, so each surviving truncation is pricey at 64k even though there are fewer of them. A model-aware starting effort (or adaptive start) is still deferred but is lower priority now that the directive carries most cases; see section 7.

### 6.16 Resume and extend under `--compute` (shipped, verified on a live run)

**Why.** 6.9 shipped compute mode as fresh-runs-only because nothing recorded whether a saved run was compute, so a `--resume`/`--extend` could not rehydrate the right prompts, schema, or gate behavior. With the truncation work settled (6.15), this was the natural next gap to close.

**Fix (file: `run_core.py`; test: `test_compute_continue.py`).** The wiring is small because every mode-dependent behavior keys off the single `compute` boolean that `run_investigation` already takes: once it is restored, `df=None`, the compute schema (`build_schema(None)`), the compute prompts (`mode_prompts`), and the G1 bypass all follow. So the only missing piece was persisting and restoring the mode. A new `run_meta.json` (`{"compute": bool}`) is written in the output root on every fresh run, in both modes (`_save_run_meta`/`_load_run_meta`). On a `--resume`/`--extend` the mode is restored from it before anything else, so a compute run is continued without re-passing `--compute` (a dataset run is not re-declared either); passing `--compute` against a saved dataset run is rejected with a clear message. The fresh-only guard is lifted for resume/extend; `--verify` stays blocked. Back-compat is clean: any pre-existing directory has no `run_meta.json` and loads as a dataset run, so existing resume/extend is unchanged. `kernel.restore_history` was already `df`-independent and `NavState` serialization already mode-agnostic, so neither needed changes.

**Status.** Suite 27 files green. Verified on a live glm-5.2 compute run: a fresh 18-morning migratory-stopover POMDP, extended to 36 mornings, then to 24. The step log accumulated 1-8 across the three invocations (steps 1-6 original, 7 the D=36 extend, 8 the D=24 extend), the kernel replayed all eight cells, the extend directive fired, and each extend added only its new horizon rather than re-deriving prior work (the D=24 extend was four API calls). Compaction held the Investigator input near 21k tokens with the older steps collapsed, and the final briefing covered all three horizons with correct per-step provenance (`[step 7]` for D=36, `[step 8]` for D=24). Two known follow-ons: a cross-invocation resume replays the full `kernel_history`, so a compute resume re-runs prior cells (heavy simulations re-execute; there is no namespace-snapshot reuse across invocations yet), and `--verify` under `--compute` is still deferred (its 6.9 token-cap blocker is now moot under 6.15's clamp, but it needs the same mode-persistence plus compute-aware audit templates).

### 6.17 Verify under `--compute` (shipped, validated on a live run)

**Why.** Verify was the last mode that did not work in compute mode (6.16 left it the one remaining gap). The audit pipeline (`verify.py`) is mode-agnostic plumbing, but its three model-facing prompts and the seed selection assumed a dataset, so a compute run could not be audited.

**Fix (files: `prompts.py`, `verify.py`, `run_core.py`; tests: `test_verify_compute.py`, `test_verify.py`).** Three compute variants of the verify prompts (`CLAIM_EXTRACTION_PROMPT_COMPUTE`, `AUDIT_SEED_TEMPLATE_COMPUTE`, `RECONCILIATION_PROMPT_COMPUTE`) sit beside the data versions; `verify.py`'s `extract_claims`/`compose_audit_seed`/`reconcile` take a `compute` flag and select the variant, and `run_core` passes `args.compute`. The substantive piece is the audit-seed stress battery: where the data battery stresses definitions, grouped uncertainty, coverage, and reference points, the compute battery stresses independent re-derivation (a different method, not the original's), convergence under refinement (finer grid, more samples, tighter tolerance), parameter and edge sensitivity, and a cross-check against known or limiting cases and the problem statement. The compute reconciliation uses the `## What the computation shows` header and adjudicates disputes on method, resolution, and error treatment; the rest of the deference logic carries over verbatim, since it is about epistemics, not data. Mode is not re-passed for an audit: `run_core` reads the prior run's `run_meta.json` (the file 6.16 added) and audits in that mode, so a compute run is audited dataset-free automatically, and `--compute` against a dataset prior is rejected. The old compute+verify guard is gone. The audit investigation already threaded `compute`, so it runs fully in compute mode on its own, and the independence invariant (fresh kernel, no inherited evidence) is unchanged, which is what lets the audit catch an implementation error by re-deriving from scratch.

**Seed-chain fix (same change, files: `verify.py`, `run_core.py`).** The first validation run exposed a latent bug in verify's seed selection, unrelated to compute: `original_seed` was `prior_saved[-1]`, the last seed of the audited run. For a fresh run that is the whole question, but the audited run here was an extend chain, so the last seed was only the instruction "also test it with 24 mornings" and carried none of the model. Handed no model, the auditor spent 25 steps reverse-engineering one across 11 candidate model families and never converged. A new `verify.original_question(seeds)` joins the whole chain (root problem first, then the extension instructions, consecutive duplicates collapsed), so an audited extend run carries its root problem; for a fresh run it is still the single seed. This fixes the audit of any extended run, dataset or compute.

**Status.** Suite 28 files green. Validated on a live glm-5.2 compute verify of the bird POMDP briefing (the 18/24/36 extend chain). With the model now in hand, the audit ran a clean nine steps (independent backward induction per horizon, grid-refinement cross-checks, a policy-structure pass, a Monte Carlo validation, and an edge-and-limiting-cases pass) rather than the 25-step model search the missing-model run produced. All four battery axes fired and added real value: it re-derived V(0,7,0.55)=0.4867203531 from scratch and confirmed all ten claims, cross-checked the 24- and 36-morning convergence the original explicitly never did (finding the errors smaller than the conservative claim), added a four-million-trial Monte Carlo cross-check by a genuinely different method (agreement to 0.31 SE), and caught a mislabeled "convexity violation," correctly re-identifying it as the cost of uncertainty. The reconciliation produced a clean corrected briefing with the values updated to the more accurate grid and a verification record. The failed-then-fixed contrast is itself evidence the audit is genuinely independent: it behaved completely differently with and without the model, which it could not do if it were leaning on the original. One open follow-on: every claim came back confirmed because the audited original was correct, so the refutation path (overturning a genuinely wrong decisive claim) is not yet exercised; a single run against a briefing with one corrupted value would close it.

## 7. Things to watch for

- **A model-facing structured format is a three-way contract: the legend you show, the object you
  render, and the parser that reads it back must all agree, exactly.** The ledger bug (6.2) was a
  silent, unrecoverable corruption caused by these three drifting apart, and it degraded an unknown
  number of benchmark runs before we found it: the model looked obedient, its output was discarded,
  no error was raised, and the state simply failed to update. Make this a standing discipline. When
  designing a block, write the renderer, the legend, and the parser from one written description of
  the shape, in the same change. When reviewing one, never conclude the templates and the parser
  agree by reading them separately; take live model output or a live render, run it through the
  real parser, and confirm the round trip reproduces the object. Always check the input format and
  the output format together, against real data. As a measure of how easily this drift creeps in:
  while documenting the fix we found the `nav_state.py` module docstring itself still printing the
  old pipe wire format in its header comment (now corrected). The session audit of every other
  injected block (THINKING, STATUS, SPEC, QUERY, REHYDRATE, VERDICT, BRIEFING, the schema, the
  evidence index, the namespace registry, and the Synthesizer evidence assembler) found no other
  strict-format mismatch. The one recoverable parse risk it surfaced, that the STATUS reader
  matched SYNTH and SEARCH as substrings (so prose like "CONTINUE, not ready to synthesize" could
  misfire), is now resolved (6.1): the decision is taken from the leading token, with a whole-word
  fallback and CONTINUE winning any tie, and `tests/test_status.py` covers the prose-wrapped cases.
- **Ledger carry-forward is fixed; confirm it stays fixed in each run.** What had looked like
  glm dropping regime lines and flipping examined/partial across turns was largely a render/parse
  format mismatch that silently discarded most of its ledger updates (see 6.2), so the map reset
  toward empty regardless of how well the model maintained it. With the legend, the renderer, and
  the parser now aligned, the ledger parses in flight and carries state forward (verified: pointers
  and statuses accumulate monotonically across turns). Keep checking it anyway, across models: in
  each new run, confirm that the nav rendered into a late Investigator input reflects the prior
  turn's emitted ledger, and that the G1 override fires only when the investigation genuinely never
  stratified.
- **Verify the briefing is fresh.** Always confirm `briefing.md` was written by the run whose
  `run_log.json` you are analyzing. We were once handed a stale piecewise briefing that did not
  match the run log (whose actual synthesizer output was different). Cross-check by searching
  the run log for the briefing's distinctive strings.
- **Ollama has no prompt caching.** Even a well-bounded context is re-sent and re-billed every
  turn on Ollama. The context fixes shrink the context and remove the correctness hazard, but
  they cannot remove the per-turn cost on Ollama.
- **Per-step output verbosity is the residual context driver.** With compaction now firing, the
  Investigator context height is dominated by large multi-part per-step outputs and the growing
  stack of collapsed headlines. If we want a lower plateau, the lever is truncating or
  summarizing raw inside the full blocks (per-step hard ceiling already exists at 64k), not more
  collapsing.
- **Stopping short of the decisive analytical move is the top accuracy risk.** A run can execute
  cleanly, stratify properly, and still fall short of the benchmark in one of two ways. It can peel
  confounds until the signal is gone and conclude a null, foreclosing the data that actually carries
  the answer (the failure G3 targets). Or it can settle for a defensible proxy where the data's
  structure calls for a formal method, naming the rigorous step but filing it under "future work"
  while the verdict leans on the proxy (the failure G4 and the Investigator's method-adequacy
  guidance target). Both are analysis-and-synthesis level rather than plumbing, and both are
  addressed by soft prompt disciplines rather than hard gates: G3 (estimand coverage, section 6.1)
  refuses a null until the requested contrast was directly estimated and reconciled; G4 (method
  adequacy, section 6.1) refuses to defer the method that would decide the answer, demanding it
  together with its uncertainty. The evidence and the fix are now recorded: the clauses alone
  lifted only the capable brain, and the methods toolkit (section 6.8) closed the weak-brain gap,
  validated across three F1 runs with both brains landing benchmark-consistent conclusions (run
  table in 5.4). What remains of this risk is recognition-bound and brain-shared: corrections
  neither brain reaches unprompted (`cluster_bootstrap` unused in all four toolkit runs,
  opponent-quality weighting always an open question), plus glm's verdict-layer variance (one of
  two runs led with a claim its own uncertainty computation undermines). Watch those rather than
  patch them; the revision criteria live in 6.8. The standing caveats: G3 has not been exercised
  on a real null since it shipped, the gates are soft self-reports rather than mechanical checks,
  and the toolkit still owes its third-dataset validation.
- **Verbose reasoning models truncate; a larger budget plus a hold-then-none retry is the durable fix (6.11, revised 6.15).** A
  reasoning model in any seat can spend its whole output budget on chain-of-thought and emit nothing
  parseable; glm and kimi both did this, glm as Investigator and kimi as Executor. The shared budget is
  now 64k (6.15), enough for a heavy reasoner such as glm-5.2 (whose reasoning fits in 24k-31k tokens
  when it completes) to finish in one call, and the Executor still starts at `none` since it
  transcribes a closed spec and gains nothing from a chain-of-thought. When a turn still comes back
  empty or capped, the retry no longer walks a step-down ladder: glm read every rung below `high` as
  its max, so stepping was a no-op and the ladder thrashed. Instead it holds the chosen effort once,
  with the strengthened `DIRECTIVE_TRUNCATED_RETRY`, then forces `none` on the final attempt, the one
  value every dialed provider honors as off. The starting effort is now a run flag, `--reasoning-effort`
  (6.15). Watch the logs for repeated "turn was cut off; retrying" notices: a retry that then succeeds
  is the system working, but a step that still yields near-empty output after the `none` attempt is a
  silent partial failure. A live glm run corrects the earlier guess that the directive is cosmetic:
  three of four truncation events recovered on the second attempt (chosen effort plus the strengthened
  directive), each finishing at ~7k-12k hidden tokens rather than 64k, with only one needing `none`.
  So keep the three-attempt shape and do not shortcut glm straight to `none`, which would discard those
  productive directive completions. Two things to watch. First, 64k is about the ceiling the hardest
  completing turns need (one landed 388 tokens under it on that run), so a turn that reasons past 64k
  still truncates, and raising the cap further trades diminishing returns for a bigger wasted call per
  truncation. Second, each surviving 64k truncation is expensive in wall-clock (roughly half the
  Investigator time on that run went to capped calls), so the deferred model-aware lever (a per-model
  starting effort, or an adaptive start once a model is seen to truncate) is worth revisiting if
  truncations climb, even though the directive handles most cases today. With reasoning
  off, residual executor failures are genuine code bugs whose rate tracks how fiddly the specs are, not
  analytical quality; the lever for a persistent case is a less verbose Executor model, not a higher cap.
- **The G1 code backstop is intentionally permissive.** It treats any stratifying construct as
  evidence of within-regime examination, so a trivial `groupby('x').size()` would satisfy it. It
  only ever upgrades an unmarked ledger, and the Synthesizer still re-derives over raw evidence,
  so the downside is bounded, but be aware it can let a thinly-stratified synthesis proceed. This
  produced its first measured live miss in the glm out-of-sample run (6.8): the backstop was
  satisfied by other stratifications while the one estimand-named axis carrying the answer went
  silently pooled, and the verdict went null-ward. An estimand-axis G1 clause was tried against
  this and reverted (6.8): it broke convergence and did not improve the deliverable, because the
  real failure is estimand substitution (a within-group proxy reported when the requested contrast
  returns null), which is G3's domain, not this backstop's. The permissive backstop stays as is;
  the open question is whether G3's direct-estimand-coverage gate should fire on a gradient-as-answer
  substitution, which in that run it did not. No machinery has been added for it.
- **Run count before ranking.** Per-model score ranges overlap and variance is 7 to 10 points;
  do not rank models on fewer than three runs each.
- **The shared 64k cap is clamped to 20k on the direct Anthropic non-streaming path (6.11, 6.15).** All
  agents share `DEFAULT_MAX_TOKENS = 64000`. The Anthropic SDK refuses a non-streaming
  `messages.create()` whose `max_tokens` implies a run over a 10-minute timeout ("Streaming is
  required for operations that may take longer than 10 minutes"), so `llm.call` clamps any direct
  `anthropic:` request to `ANTHROPIC_MAX_TOKENS = 20000` automatically. Ollama and OpenRouter take
  the full 64k; an `openrouter:anthropic/...` model goes through the OpenAI-compatible path and is
  not subject to the clamp, since OpenRouter handles the underlying model's own output limit. This
  replaces the earlier regime in which every cap was held at 20k by hand and raising it was
  forbidden. The cap is no longer the lever for a reasoning model that over-thinks; the hold-then-none
  retry (previous item) is. If a larger non-streaming Anthropic output is ever genuinely needed beyond the
  clamp, route that one call through `AnthropicProvider.stream`.
- **Compute mode is gate-free, and now fully continuable and auditable (6.9, 6.16, 6.17).** `--compute`
  disables the G1-G4 statistical gates (no dataset means no effect, regime, or confound to gate on); in
  their place the compute Synthesizer self-checks UNCERTAINTY/CONVERGENCE/VALIDITY in the same
  GATES/VERDICT/BRIEFING shape, so a compute briefing's rigor rests on those self-checks rather than on a
  hard backstop gate. `--resume`/`--extend` (6.16) and `--verify` (6.17) all work now, the mode restored
  from `run_meta.json` so it is never re-passed. Two things still worth watching: a cross-invocation
  resume replays the full `kernel_history`, so heavy compute cells re-run on every resume or extend (no
  namespace-snapshot reuse yet); and the verify-compute refutation path is unproven, since its validation
  original was correct and every claim came back confirmed, so the audit has caught gaps and an
  interpretation error but has not yet overturned a wrong decisive claim.
- **The hidden-richness compute benchmark is now contaminated as a clean capability test (6.13).**
  The C5 belief-structure discipline tells the Investigator that the sufficient statistic for a
  process watched over time is usually (events, elapsed time), not the event count alone, and to
  prefer an exact recursion over a policy-search simulation. Those are exactly the two structural
  insights the bee-foraging seed was built to test. So the rerun that landed the correct 1.221 (after
  the pre-C5 run landed a wrong 1.158 with a count-only belief) cannot be read as the model finding
  the information state unaided: turn-1 reasoning visibly applies C5's framing and uses its terms
  ("sufficient statistic", "information state"), and the model drew the opposite, correct conclusion
  from the same "the rate factors cancel" observation that had misled it before. C5 is generic in
  wording, with no bee-specific content or numbers, but it pre-answers this specific problem's
  challenge, and softening it would not un-contaminate the seed since the lesson is now known and
  encoded. Keep the bee seed as a regression check that the engine still lands 1.221; measure unaided
  information-state capability only on a fresh, structurally different hidden-state problem whose
  sufficient statistic is not (events, time). One run each way, model variance not ruled out:
  encouraging, not yet a win.
- **The function-reuse registry change is shipped but unexercised in a live run (6.14).** Functions a
  step defines now appear in the registry by signature and reach the Executor when the spec names
  them. But in the one compute rerun the Investigator never named a function for reuse; it let the
  Executor redefine helpers per step (14 `def`s across the run), so the path meant to stop the
  repeated-rebuild failure has not actually fired in production. That run also did less repeated work
  (an exact DP reuses less than a repeated simulation), so there was little to reuse. Watch a future
  run to confirm the Investigator names and calls a persisted function rather than ordering a rebuild.

---

- **Ledger versus invoice on OpenRouter.** The ledger books what the work would cost; OpenRouter's
  zero completion insurance refunds empty or error completions, so a run with guard-rescued
  empties over-reports by those rows (visible as rows with empty output and nonzero `cost_usd`),
  while insured guard FIRST attempts appear in neither ledger nor invoice (unlogged by design).
  If the numbers ever need row-level reconciliation, that is the gen_id capture queued in 6.19.
- **Watch ranking-flavored briefings for outlier lean under the print budget (6.21).**
  Top-and-bottom-k views foreground "there is a #1" more than full overlapping tables do. Three
  F1 runs stayed coherent (the pooled-modern and era-local lenses of one answer), but if the lean
  recurs on other ranking seeds, add a clause line asking for the CI-overlap summary alongside
  any ranking print.

## Appendix A: Path map

| What | Where |
|---|---|
| Live source (the 12 rebuilt modules + `tests/`) | `/home/claude/delve/delv-e/` |
| Outputs mirror (modules, docs, archive) | `/mnt/user-data/outputs/` |
| Shipped archive (complete project) | `/mnt/user-data/outputs/delv-e.zip` |
| Benchmark dataset | `datasets/f1_driver_vs_car.csv` (public; the only file tracked under `datasets/`) |
| Benchmark ground truth | `F1_GOAT_report.md` |
| `httpx` stub for offline validation | `tests/stubs/httpx.py` (bundled in the repo) |
| Regression tests | `tests/test_*.py` (run with plain `python3`) |
| Removed dead code (recoverable this session) | `/tmp/delve_attic/` |

## Appendix B: CLI reference (`run_core.py`)

| Flag | Default | Meaning |
|---|---|---|
| `dataset` (positional) | required unless `--compute` | Path to `.csv/.tsv/.xlsx/.parquet/.json/.jsonl`. Omitted in compute mode. |
| `question` (positional) | none | Seed question. In compute mode, the sole positional. |
| `--iterations N` | 14 | Max investigation steps. |
| `--investigator-model` | premium | Model string for the Investigator. |
| `--executor-model` | cheap | Model string for the Executor. |
| `--synth-model` | premium | Model string for the Synthesizer. |
| `--reasoning-effort` | medium | Starting reasoning effort for the Investigator and Synthesizer (`max`/`high`/`medium`/`low`/`none`); the Executor stays off. Mapped per provider; a direct Anthropic model ignores it. |
| `--output DIR` | `output` | Output directory. |
| `--data-dictionary` | none | Optional data dictionary file. |
| `--periodic-every N` | 0 | Holistic snapshot every N steps (0 = off). |
| `--g1-pushback N` | 2 | G1/synthesis pushback budget. |
| `--search-model` | none | Enables web search when set (Anthropic). |
| `--search-budget N` | 3 | Max searches per run. |
| `--resume` | off | Resume an interrupted/finished run from saved state. |
| `--extend` | off | Extend a finished run with a new seed (reopen-and-reconcile). |
| `--compute` | off | Dataset-free mode: no dataset loaded (`df` does not exist); runs simulations or pure computation. Fresh runs only (not with `--resume`/`--extend`/`--verify`). |

## Appendix C: Validation commands

```
# run the full regression suite (from the repo root; no PYTHONPATH needed)
for t in tests/test_*.py; do python3 "$t"; done
# the three dataset-backed tests target a private dataset that is not shipped here,
# so they skip cleanly on a public checkout.

# verify the project imports from a clean extract of the archive
cd /tmp && rm -rf _v && mkdir _v && cd _v && unzip -q /mnt/user-data/outputs/delv-e.zip && cd delv-e
python3 -c "import sys; sys.path[:0]=['tests/stubs','.']; import kernel, investigation, nav_state, synthesis, llm; print('OK')"
```

## Appendix D: Status vocabularies (NavState)

| Handle | Statuses |
|---|---|
| FRONTIER | `untested`, `in_progress`, `tested`, `foreclosed` |
| REGIME | `not_examined`, `partial`, `examined` |
| RISK | `open`, `resolved` |
| BREAKDOWN | `thin`, `holds`, and synonyms normalized on parse |

`g1_satisfied` is true when any REGIME is `examined`/`partial`, or (with `log`) when the code
shows stratification. `open_regimes` returns REGIMEs still `not_examined`. `protected_steps`
returns the latest step of each live frontier/open risk, capped at 6.