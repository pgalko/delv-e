# delv-e: System Design & Handover

This is a complete handover document for the rebuilt **delv-e** system: an autonomous,
LLM-driven data-investigation engine. It is written to seed a fresh working session.
Read it top to bottom once; after that, sections 0 and the appendix are the day-to-day
references.

This document ships alongside:

- `delv-e.zip`: the complete, push-ready project (the 11 canonical modules, the `tests/`
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
`--executor-model`, `--synth-model`, `--output DIR`, `--g1-pushback N` (default 2),
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
`test_reasoning_ladder`, `test_function_reuse`. All 25 pass as of this handover.
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
| `llm.py` | Provider abstraction: `AnthropicProvider`, `OpenAIProvider`, `OllamaProvider`, `OpenRouterProvider`, the dispatching `LLMClient`, prompt caching (`build_cached_messages`, Anthropic only), `CostTracker`, `RunLogger` (writes `run_log.json` into `<output>/logs/<timestamp>/`), `RunStats` (a per-run events sink), `build_run_telemetry` (aggregates `run_telemetry.json`), `search_call` (Anthropic web search), and `parse_model_string`. |
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

- **Prompt caching is Anthropic only** (`build_cached_messages`). On Ollama there is no
  caching, so the full (growing) context is re-sent and re-billed every turn. This is inherent
  to the provider and is the backdrop for several context-size observations in section 7.
- **`search_call`** is the Anthropic web-search path, pinned to a Haiku-class model and logged
  as "Literature Search."
- **`CostTracker`** accumulates token usage and cost; **`RunLogger`** writes the per-call
  trace to `run_log.json` (the artifact we analyze after every run).
- **The Executor runs with reasoning disabled on Ollama** (`reasoning_effort="none"`, 6.10).
  It is a mechanical transcriber, so a chain-of-thought buys it nothing and a reasoning model in
  that seat truncates by spending its whole output budget thinking. The effort is a per-agent
  module constant in `llm.py` (`EXECUTOR_REASONING_EFFORT`, `DEFAULT_REASONING_EFFORT`) and is
  gated to Ollama, since OpenAI and OpenRouter reject the value.

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

### 6.0 Kimi-as-brain run analysis; toolkit schema legend, anchor SE fix, and budget wrap-up notice (shipped, suite green)

**The run (kimi-k2.6 as Investigator, Executor, and Synthesizer; 20/20 iterations; forced provisional; assessed ~78-81).** First logged run with kimi as the brain. Conclusions were benchmark-consistent (no single outlier; era-conditional headline with Fangio leading the front-engine era and Verstappen the modern era with Norris/Leclerc chasing; finish-vs-qualifying r of 0.56 against the benchmark's corrected 0.54; achievement kept out of the headline; race pace correctly diagnosed as modern-only), with the campaign's shared recognition gaps recurring (Indy 500 never excluded, pseudo-replication and group weighting never surfaced, cluster_bootstrap idle, middle eras unmodeled). Mechanically the plumbing all held on a third brain: the ledger parsed in flight every turn and carried state forward (verified against raw turn inputs), the estimand pinned, compaction plateaued the Investigator at 20-23k input tokens, the Executor stayed flat at 0.7-2.8k, paired_ability ran five times and rank_uncertainty four with zero reimplementation, and the 0.5 tie convention drew zero retries. The run never self-finalized: kimi decomposes more finely than glm or Opus (the same milestone plan, but fit / filter-plus-rank / diagnostics as separate turns), needed about 22-23 steps against 20, and had no visibility of the budget, so the ceiling forced an ungated provisional briefing. All four executor retries plus the one failed step were a single class: KeyErrors from specs referencing toolkit output columns by description ("the column representing probability of rank 1") instead of by name, because the prompt advertised signatures without return schemas. The model also hand-imputed SEs for reference entities four times because paired_ability returned se NaN for the anchor and rank_uncertainty silently drops NaN-se entities from the ranking.

**Three levers shipped in response.**

1. **Toolkit output schemas in the legend (prompts.py).** The TOOLKIT block now lists the exact output columns of paired_ability and rank_uncertainty and the exact dict keys of cluster_bootstrap, and instructs specs to reference outputs by those names, never by a description. The per-function cluster_bootstrap line defers to this single list so two diverging lists cannot coexist. A new live drift check in tests/test_toolkit.py (test_prompt_schema_agreement) runs the real functions and asserts every real output column/key appears in INVESTIGATOR_SYSTEM, the 6.2 three-way-contract discipline applied to the toolkit legend; on its first execution it caught a real drift (the undocumented ci_level key, now in both the legend and the docstring).
2. **Reference anchor se 0 (toolkit.py).** paired_ability now returns se 0.0 (and degenerate CI) for the reference row instead of NaN, with the docstring updated: the reference is the anchor and all uncertainty is expressed relative to it. This ends the invented-SE workaround; the composition paired_ability to rank_uncertainty now needs no imputation and the reference participates in every draw (tested: test_reference_anchor_se_zero, test_ability_rank_composition_no_imputation; the toolkit suite is now 88 checks).
3. **Budget wrap-up notice, designed as a ceiling and never a quota (prompts.py, investigation.py, llm.py).** The Investigator sees no step counter at all through the body of a run, so the total budget can never anchor as a target to fill. Only inside the final stretch (the last fifth of the budget, minimum the last 2 turns; _budget_window) does an uncached tail notice appear: at most N turns remain, the ceiling force-synthesizes without the quality gates, use the remaining turns for the decisive moves and choose SYNTHESIZE on your own terms, untested FRONTIER items do not block synthesis. Paired with a standing license in the STATUS legend to synthesize as soon as the evidence supports a defensible answer and never to spend steps merely because budget remains, which applies at any step and also counterweights the pursue-the-frontier pressure. Notices are counted in telemetry (gates.budget_wrapup_notices). tests/test_budget.py drives the real loop and asserts the notice is absent on early turns, appears in the final window with the correct countdown, and never appears in a run that finishes early.

**Status.** The full suite (17 files) is green. The levers' live effect is unvalidated until the next real runs; the predictions to check are that the rank-printing retry class disappears, no run hand-imputes a reference SE, fine-grained brains self-finalize inside the budget (restoring gate coverage), and early-converging runs still finalize early rather than padding to the ceiling.

**Second pass (same session): two more live runs, then pool hygiene, the GATES block, and the estimand discovery clause (shipped, suite green).**

Runs 2 and 3 validated all three levers live. Run 2 (kimi as all three agents, 28 steps + synthesis at 29 of 30, FINAL, assessed ~70-74): zero rank-step retries (the run-1 class is gone), zero SE imputation (the anchor came out se 0 natively), the wrap-up notice fired on the last five turns and the model self-finalized inside the ceiling with the untested frontier item filed under open questions. Its failures were new and brain-bound: a composite-outlier headline built on coverage-asymmetric dimensions, rank_uncertainty run over the full unfiltered pool so the genuine leaders' p_rank1 collapsed to 0 against high-se noise entities, a G1 pass via an invented regime axis (performance_dimension) while era stayed a coverage table, and a Synthesizer that emitted FINAL as its first token with no visible gate deliberation. Run 3 (glm brain, kimi executor, 15 steps + synthesis of 30, FINAL, assessed ~76-79) gave the complementary lever evidence: the notice correctly stayed silent and the run still stopped early under the standing license, the composite headline carried propagated SEs over a filtered pool, G1 passed via genuine era-level estimation, and the same two cross-brain patterns recurred: verdict-first synthesis with no deliberation, and an estimand that enumerated candidate dimensions the seed asked it to discover. All three runs ranked a raw pool at least once; all three left Indy in, weighting and cluster_bootstrap untouched (still watched, not patched).

Three additions shipped in response, each evaluated against divergent use cases (league tables, A/B tests, causal digs, scouting, descriptive asks) before wording, generality the binding constraint:

1. **Rank-pool hygiene (prompts.py, toolkit.py).** One sentence in the rank_uncertainty legend: P(rank 1) is relative to the pool passed, low-evidence entities dominate rank-1 draws, name a minimum-evidence threshold and filter before ranking, unless ranking a mixed-evidence pool is itself the question (the scouting escape clause). Plus a mechanical, data-driven warning in estimates mode (max se at least 10x the median, 5+ entities) printed to stdout, the evidence channel, and stored in attrs['warning']; advisory only, never blocks. Evidence bar: three for three runs across both brains, the tie-note precedent.
2. **GATES block (prompts.py, synthesis.py, investigation.py).** The Synthesizer's output format is now three blocks: ###GATES### (one line per gate, pass, fail, or n/a with a grounded half-line reason, deliberated BEFORE the verdict), then VERDICT, then BRIEFING. The n/a option keeps it honest where a gate has no referent. The parser extracts gates_review tolerantly (absent, misplaced, or malformed blocks degrade gracefully; the briefing never absorbs gates text), and both terminal log entries persist it for post-run dissection. tests/test_synth_gates.py covers the legend, all four output shapes, and end-to-end plumbing, the 6.2 three-way-contract discipline applied to the new block.
3. **Estimand discovery clause (prompts.py).** When the seed asks which factors, dimensions, or causes matter, that discovery is itself the target: the estimand must not enumerate candidates the seed did not name; candidates belong on the FRONTIER. Binds only on discovery-type seeds; seed-named factors remain restatable.

Held under the watch-list with explicit criteria: composite-coverage-asymmetry guidance (trigger: a third FINAL run headlining an unnormalized asymmetric composite), regime-axis quality on G1 (trigger: recurrence of a non-effect-modifier axis satisfying the gate), and the standing trio. Suite now 18 files, toolkit at 94 checks. Predictions for the next runs: polluted-pool rankings stop appearing in briefings or arrive with the printed warning in evidence, synthesis logs show per-gate deliberation, and pinned estimands stop enumerating unnamed candidates.

**Third pass (same session): live GATES-block incident and the three-layer fix (shipped, suite green).**

Within hours of the GATES block shipping, a glm run (17 steps of 30, real era/teammate-quality/car-quality regime work, excellent per-gate deliberation including a genuine G2 bound) ended with final_verdict "none" and no briefing.md. Root cause: the model emitted the briefing marker one character short, "###BRIEFING##" with two trailing hashes. The parser's exact-marker regex silently returned an empty briefing while VERDICT parsed FINAL; _final_synthesis writes briefing.md only when the briefing is non-empty; FINAL-with-empty-briefing turned out to be the one terminal state none of the safety nets (G1 gate, pushback, forced-final, truncation, ceiling) cover, so the loop wrote a terminal FINAL entry with no deliverable and run_core mapped the empty briefing to "none". The model's complete ~6,900-char briefing was sitting in the raw output after the malformed marker. The 6.2 lesson compounded: the new block's tests covered absent and misplaced markers but not malformed ones, and a live model found the gap immediately.

Three layers fixed: (1) parser marker tolerance, the block regex now accepts any number of trailing hashes on the marker line (constrained to the same line so a zero-hash marker cannot swallow the content's own markdown headers); (2) a briefing salvage in _parse_synth, any malformed briefing-marker remnant (wrong leading hashes included) recovers the content after it, with the none/na filter intact so gated outputs stay empty; (3) a loop net, FINAL with no parseable briefing re-runs once in finalization mode (which forces a briefing and salvages), counted in telemetry as gates.synth_briefing_retries. tests/test_synth_gates.py grew to ten shapes including the live one-char malformation verbatim plus a loop-level recovery case, and the patched parser was validated against the actual failed run's raw output, recovering the full briefing. Standing discipline reinforced: every new model-facing marker ships with malformation tests, not just absence tests.

**Fourth pass (same session): the terminator gap, second live parser incident.**

The first Mt Buller benchmark run produced a FINAL verdict with a briefing.md cut off after its second header. All three block markers were well-formed this time; the briefing body contained "### Subsection" markdown headers, and the generic block terminator (?=###|\Z) ended the capture at the first one. This was a known limitation explicitly scoped out during the marker fix ("briefings use ##"), falsified by a live model within a day. Fix: blocks now terminate only at the named markers GATES, VERDICT, or BRIEFING (word-boundary guarded), never at arbitrary ###. Shapes 9 and 10 added: the live subsection miniature, and the interplay case proving named markers still terminate while subsections do not. The patched parser run on the actual output recovered the full 8,257-char briefing. The sharpened lesson: assumptions about model formatting habits are not load-bearing; named markers are. Telemetry note: the run also exercised the investigator truncation retry (token_caps_hit 1, recovered) and the empty-briefing net correctly stayed dormant since the truncated briefing was non-empty.

**Fifth pass (same session): campaign scoreboard, composite clause fired, grouped-uncertainty pre-commitment.**

Three benchmarks now exist with private answer keys held outside the repo: F1 (driver-vs-car), MLDA (age-21 regression discontinuity, stevedata mm_mlda with fitted columns stripped), and Mt Buller (fully private climate assembly; the gating variable is deliberately unnamed in the seed). Campaign results: F1 run 4 (glm) scored 82-85, the first run to beat the early bests, with nine within-era ability models, the era-compression diagnosis self-discovered, a car-dominance lower bound, and correct multidimensionality; misses remain the standing recognition trio (Indy exclusion, group weighting, cluster_bootstrap on home turf). MLDA (glm, 6 steps) scored 92-95, near ceiling: every estimate matched the key to the second decimal, the internal/external validity split arrived unprompted with a mechanism-consistency argument, plus an unrequested four-threshold placebo battery; the only soft spot was treating the two boundary NaN cells as routine missingness. Mt Buller (glm, 10 steps) scored ~70-74: the gate variable was discovered unprompted on a fully private dataset (the anti-recall design worked) with a real humidity refinement, but the headline threshold was evaluated at the all-days humidity median instead of storm-typical humidity, the keyed deep-pack tail collapse was missed (the step-change test was filed as an open question), the asserted snow-fraction decline reproduces only under a trace-day definition driven by early-record precipitation missingness (a data artifact, verified by reproduction), and the flagship daily Tasman significance dissolves under season clustering (clustered CI crosses zero, se inflation x2.8). The ground-truth key survived its first falsification attempt: the run's strongest novel claim was checked against the data and found artifactual, so the key stands unamended.

Lever scoreboard after three scored runs on the new prompts: P1 rank-pool hygiene confirmed live (filtered pools, meaningful p_rank1, warning correctly silent). P2 GATES block confirmed live in all three runs with grounded per-gate deliberation, and it stress-tested the delivery layer into two parser incidents now fixed (passes three and four). P3 estimand clause is non-binding on glm at n=2 (enumerated unnamed candidates in F1 run 4 and Mt Buller; in Mt Buller the enumerated conditions were then genuinely examined, so observed harm was nil); hold for cross-brain evidence before strengthening the wording. Budget notice: correct silence with early stops at 6, 10, and 17 of budget; the anti-quota property holds in both directions across four runs.

Fired this pass, per the pre-committed criterion (third FINAL run headlining an unnormalized asymmetric composite, met by letter on F1 run 4 in mitigated form): the composite-coverage clause, two generic sentences. Investigator (after the toolkit block): when combining several measures into one composite, check per-entity component coverage; under asymmetry, restrict to common coverage or normalize per entity over observed components, and print which was done. Synthesizer (appended to G1b): a composite built from components not observed for every entity compared must not be the headline; lead with per-component or common-coverage estimates and label any asymmetric composite secondary.

Recorded this pass, not fired (n=1, and the toolkit prompt already names cluster-robust regression as available, so the gap is recognition, not capability): the grouped-uncertainty pre-commitment. Criterion: if a second run on any brain presents inference on a group-level regressor from pooled within-group observations without clustered uncertainty, add one generic sentence to the TOOLKIT block on clustered uncertainty whenever observations share a grouping axis. The Mt Buller cost of this miss is the exemplar: a daily-level z of -4.08 on a season-level regressor whose season-clustered CI is [-1.42, +0.31].

Watch-list, consolidated: the F1 recognition trio (Indy, group weighting, cluster_bootstrap at home; F1-only, domain recognition not machinery); P3 estimand enumeration (n=2, glm only, harm so far nil); G1 regime-axis quality (no recurrence since); grouped-uncertainty criterion above. Removed: composite asymmetry (shipped).

**Sixth pass (same session): Mt Buller run 7 scored 80-83, first key amendment, terminator fix validated under fire.**

Rerun with the composite clause and terminator fix deployed (glm, 10 of 40, zero reliability events, early stop again). The briefing arrived complete despite containing ### subsection lines inside its body, the exact shape that truncated run 6, so the named-marker terminator is live-validated. Layer 1: the gate refound at -2.74 under a prcp > 0 inclusion rule, reproduced exactly and inside the now-widened definitional band; yield ladder reported as per-bin medians with the gauge skew (max ratio 75) handled honestly; the exposure split missing for the second straight run; threshold CI tight but unclustered. Layer 2 production: the campaign's best calibration language, an all-null battery plus a power analysis bounding detectable trends at 78-116% of the mean with explicit "not identified here, not no effect" framing, and a year x temperature interaction test establishing conversion stability; it also directly refutes run 6's artifact, tasman to snow fraction r = -0.04. Layer 2 retention: unopened again. Snow Pack was printed at orientation, "snow persistence" was named in the run's own pinned estimand, and no pack metric was ever computed; the G3 gate line passed by restating the estimand without the persistence and timing components. Attribution: correctly leveled at season grain with a Durbin-Watson self-diagnosis; the headline Tasman-to-precipitation-volume channel (+769 mm/C, p = 0.001) was verified and survives three missingness corrections at +450 to +750 mm/C, so the ground-truth key received its first amendment, crediting the channel and documenting the corr(tasman, missing-days) = -0.57 inflation engine. Run 6's novel claim died under verification; run 7's survives with attenuation noted: the falsification protocol works in both directions.

Criterion statuses after run 7. The grouped-uncertainty pre-commitment did NOT trigger: this run performed its group-level inference at group level (seasonal regressions, n = 37) and self-diagnosed the serial correlation, which is the behavior the criterion exists to protect. Two new pre-commitments recorded, neither fired at n=1 per discipline: (i) the G3 verbatim-component rule, to ship if a second run's gate review passes G3 while a pinned-estimand component is neither answered nor filed under open questions, wording: "G3 is evaluated against the pinned estimand verbatim; any named component neither answered nor explicitly filed as untested fails G3"; (ii) the stock-and-flow sentence, wording: "when the data contains both flow and accumulated-stock measures of the same quantity, examine both; change can appear in stocks without appearing in flows". P3 estimand enumeration is now n=3 on glm, observed harm still nil (the enumerated wind condition was then genuinely tested).

**Seventh pass (same session): standing configuration decision, criteria reworded for a glm-only world.**

Decision (user): glm is the standing investigator and synthesizer going forward; the executor is unchanged (kimi to date). Rationale: glm holds the campaign's three best scores and is the deployment target, so each benchmark run tests the configuration that will actually ship. Consequence accepted with eyes open: the prompt stack from P1 onward has not been validated on a non-glm investigator and that evidence freezes here; the prompts remain fully generic regardless (the standing house rule protects other users of the open-source repo), the kimi executor keeps half the prompt surface cross-model, and an outside report of misbehavior on another brain is the trigger to revisit.

Evidence bars amended accordingly, cross-brain replaced by cross-dataset within glm plus harm triggers: the stock-and-flow sentence now ships on a second DATASET that clears a flow channel while missing a stock-channel change (a stock/flow-structured benchmark, balances versus transactions or inventory versus sales, is a wanted roster addition to make this testable); P3 becomes purely harm-based, shipping a stronger wording only if an enumeration demonstrably narrows an investigation onto the named set while the key shows an unnamed axis mattered; the G3 verbatim-component rule and the grouped-uncertainty criterion were already brain-agnostic and stand unchanged. Next experiments, reordered for this world: an F1 rerun to grade the freshly shipped composite clause (prediction: the headline no longer leads with an unnormalized asymmetric composite), the obscure MLDA variant to close the recall question, and the stock/flow benchmark when convenient.

**Eighth pass (same session): F1 run 5 scored 86-89, composite clause confirmed, Indianapolis recognized at last.**

The F1 rerun under the full current stack (glm, 9 of 40, zero reliability events). The composite-clause prediction held on its first graded test: the opening sentence is the benchmark's own headline shape, no single clear outlier and greatness multidimensional, delivered for the first time in five F1 runs; the only composite in the document is mid-text, pace-only, justified, with the craft-inclusive composite explicitly ruled out; and the G1b gate line cites the rule back, "the headline will lead with per-era and per-dimension estimates, not a single pooled number", which is direct mechanism evidence the clause binds. Era conditioning arrives as within-era z-scores (Fangio 2.09, Senna 2.09 and 1.89, Schumacher 1.86, Verstappen 1.47 turbo-hybrid) with small-n era leaders flagged, and the modern claim is a filtered-pool p_rank1 of 60.4% framed as era-specific rather than absolute. Two genuinely new findings: the race-craft dimension exposed as structurally confounded with grid position, with its top scorers correctly identified as 1950s Indianapolis 500 specialists, verified against the step-5 evidence (Rathmann, Vukovich, Parsons, Hanks atop the craft table), the first Indianapolis recognition of the campaign, though dataset-level exclusion still was not performed; and the Hamilton anchor artifact diagnosed and bounded (anchor ability fixed at 0 with SE 0 makes rank_uncertainty freeze the anchor, p_rank1 0.0, median rank 12), handled exactly as G2 intends, bound rather than reported or discarded. A verification note for the record: a quick proxy check (raw mean positions gained) initially contradicted the Indianapolis claim; the run's own evidence table settled it in the run's favor, the standing lesson being to verify against what the system actually computed. Misses keeping it under 90: cluster_bootstrap and group weighting at n=5, no per-era rank probabilities (z-scores only), Indianapolis recognized but not excluded, and run 4's era-compression and car-dominance moves absent this time, run-to-run variance in which robustness moves appear, which is the live argument for the proposed verification stage. Toolkit observation recorded, no action: the anchor se=0 design choice (second-pass L2) trades NaN-dropping for rank-freezing of the anchor; the run's open question proposes re-anchoring or a sum-to-zero constraint, a candidate future toolkit refinement. Watch-list trio updated: Indianapolis now "recognized, not excluded" at n=5; weighting and cluster unchanged. P3 estimand enumeration n=4, harm still nil to positive (the enumerated conversion dimension was then genuinely built and proved informative). Campaign table: F1 runs 70-74, 76-79, 82-85, now 86-89; MLDA 92-95; Mt Buller 70-74 then 80-83.

**Ninth pass (same session): the serial verification experiment validated the concept, fired the grouped-uncertainty clause, and demonstrated the correlated-blind-spot limit.**

A zero-code audit run: fresh delv-e pass on the Mt Buller data whose seed carried run 6's six decisive claims plus a generic audit mandate (alternative definitions, grouping-aware uncertainty, coverage sensitivity, representative reference points, and a missed-channel probe), independence preserved by NOT using --extend, which would inherit the audited run's evidence chain. Graded against pre-registered predictions: B hit, the reference-point error caught with a sharper diagnosis than the scorer's own ("the number is right for the wrong reference point": the -3.3 boundary belongs at the precip-day median humidity of 85, not the all-days 77); C hit, the snow-fraction artifact dismantled with its full engine (proper-vs-broad metric distinction, the +1.59 precip-days/season coverage trend at p=0.001, coverage filtering collapsing the seasonal R-squared from 0.29 to 0.04, the 185 snow-without-precipitation days quantified and spot-verified exact); D hit, the genuine findings survived; A half, the attribution was refuted, the right outcome, but via the year-confound competition (anomaly p=0.746 once year is included, r=0.71) rather than clustered uncertainty, which appeared nowhere despite explicit seed instruction; E miss, the snowpack persistence channel stayed unopened even under an audit mandate, reinforcing the stock-and-flow pre-commitment (which still awaits a second dataset by its letter). Net: 2.5 of 3 known artifacts caught at zero code changes, nine steps, no cost; the verification-stage concept is validated. The auditor also produced one wrong verdict, claim 6 "confirmed" via an unclustered daily logistic (post-2006 era coefficient, p=0.011); the scorer's season-clustered bootstrap gives CI [-0.544, +0.262], crossing zero at x2.4 se inflation, so the key's era-wobble reading stands. The limitation predicted in the ensemble discussion is now demonstrated: a same-model auditor shares the generator's blind spots (clustering, the stock channel), so the eventual in-loop verification stage should carry a mandatory stress battery rather than rely on model judgment alone.

CRITERION FIRED: grouped uncertainty. The pre-committed letter (a second run presenting inference on a group-level regressor from pooled within-group observations without clustered uncertainty) was met twice inside the audit itself, the Model A daily p<0.001 on the season-grain anomaly and the claim-6 era confirmation, the latter materializing harm as a wrong verdict, and the gap proved instruction-resistant since the seed asked for grouping-aware uncertainty explicitly. Shipped, one generic sentence appended to the cluster_bootstrap bullet in the TOOLKIT block: "When observations share a grouping axis (the same season, subject, site, or repeated unit), uncertainty for any quantity that is constant within groups or pooled across them must respect that grouping: use cluster_bootstrap with that axis as cluster_col, or a cluster-robust model, rather than treating pooled rows as independent." Prediction for the next Mt Buller-class run: the campaign's first legitimate cluster_bootstrap call, and era or anomaly inferences arriving clustered or labeled as unclustered.

**Tenth pass (same session): the serial verification feature, built.**

--verify PRIOR_RUN_DIR automates the validated manual experiment as a follow-up command, deliberately not built on --extend: only the prior BRIEFING crosses over, never the evidence chain or kernel, because independence is what made the prototype catch the definitional artifact (a verifier reusing the prior kernel inherits the artifact-generating tables and can never find them). Three phases. Phase 1, claim extraction: one cheap call (agent ClaimExtractor) distills the prior briefing into at most ten numbered decisive claims; the parser is malformation-tolerant per the standing lesson (mixed numbering, markdown bold, continuation-line folding, header noise), and an empty extraction falls back to auditing a capped briefing excerpt rather than dying. Phase 2: the audit seed is composed from a fixed generic template, the original question plus the claims plus the manually validated mandate verbatim, its four stress axes hard-coded as a mandatory battery precisely because the experiment showed a same-model auditor left to choose weapons shares the generator's blind spots; the pipeline downstream is the standard one, untouched. Phase 3, reconciliation (agent Reconciler, before telemetry so the calls are counted): one synthesis-grade call merges the documents into the single user-facing briefing.md in the standard structure, each decisive claim carrying confirmed, attenuated, refuted, or contested status, verdicts changing only on decisive evidence, disagreements kept visible as contested rather than force-merged, plus a closing Verification record section; briefing_original.md, briefing_audit.md, and claims.md are preserved alongside, and an unusable reconciliation falls back to the audit briefing standing in (never end empty-handed). Guards: --verify excludes --resume and --extend, and refuses an --output equal to the audited directory. New module verify.py (~130 lines), three generic templates in prompts.py, thin wiring in run_core (deferred seed composition after the client exists), tests/test_verify.py with eleven checks including the end-to-end five-agent chain; suite now 19 files, telemetry rolls up the new agents through the existing generic groupby. Acceptance tests pre-registered for the first live use: pointed at Mt Buller run 6, the reconciled briefing must carry at least two of the three known artifact corrections with zero silent overwrites; pointed at F1 run 5, it must change almost nothing (the non-destructive negative control).

**Eleventh pass (same session): verify-mode UX round, a duplicate-implementation incident, and three new ship-discipline rules.**

UX round, user-requested, delivered: bare --verify resolves the audit target through a last-run pointer file (.delve_last_run), written only by completed primary runs that produced a briefing, so a bare audit can never self-select a previous audit; an absent pointer raises with guidance instead of guessing at ./output. The audit's output defaults to output_verify/ with --output retained for chained audits, and both the directory and the pointer are gitignored. The terminal makes an audit unmistakable: ui.MODE recolors the run magenta, the iteration banner relabels EXPLORING as VERIFYING, and the verify phases announce themselves in magenta notes.

The incident, recorded plainly because the handover is where honesty lives: while wiring this round, the working tree was found to already contain a complete parallel implementation of these exact features (the @last argparse const, inline pointer helpers in run_core, ui.note announcements, the README section, a helper test block, the ui.MODE machinery, the gitignore entries), present in the previous ship's archive and not attributable from inside this session's visible work. This turn's overlay edits had silently half-applied around that parallel code, plain str.replace no-ops on drifted anchors, leaving a live NameError on the verify path plus a duplicated announce-and-compose tail; the suite missed all of it because the tests drive functions, never main(). Consolidated to a single implementation with verify.py as the sole source of pointer logic (resolve_prior_dir, write_last_run_pointer, the @last sentinel), foreign helpers and the unconditional duplicate pointer write removed, one ui.MODE site, one announce, one compose. Behavior decisions: pointer writes are briefing-gated, and a pointer-absent bare --verify raises rather than silently auditing a default directory.

Three ship-discipline rules adopted from the incident: (1) every scripted edit asserts its match count and writes nothing on failure (an assert-or-die harness caught two would-be no-ops during the consolidation itself); (2) the ship checklist gains a diff against the staged snapshot before rebuild, so unexplained tree drift surfaces before it ships rather than after; (3) CLI-path features get main()-level smokes alongside function tests, because wiring chimeras live precisely where function tests cannot see, and three such smokes (explicit bad directory, bare flag without a pointer, pointer to a run without a saved question) are now permanent in tests/test_verify.py, exiting before any network or data load.

**Twelfth pass (same session): first --verify acceptance run (F1 negative control), adjudicated.**

The run: verify pointed at F1 run 5 (scored 86-89, pre-registered to change almost nothing). Two incidents and one discovery. Incident one, claim extraction: ClaimExtractor output landed at exactly 4000 tokens, its max_tokens cap, because glm burns reasoning tokens against the cap; zero claims parsed, the excerpt fallback fired as designed, and the audit fought half a document cut mid-sentence. The bloated seed produced the campaign's worst reliability profile (7 executor retries, step 3 failed, 6 investigator truncations, 36.8 minutes, 827k tokens). Fix shipped this pass: extractor cap 4000 to 16000. Incident two, reconciler deference: the Reconciler accepted every audit verdict wholesale (one Contested in ten dispositions) and stated the audit's hypothesized mechanisms as fact.

The adjudication, every disputed number recomputed against the dataset with the repo toolkit. Claim C (pooled BT pathology) reproduced EXACTLY: Lauda -1.80 vs audit's -1.79, Clark -1.48 vs -1.50, Schumacher -0.13 and Prost -1.07 to the digit, all with 59-84% raw teammate win rates; my wider scan found 51 drivers with win rates of 55% or more and 30 or more contests carrying negative pooled abilities, flagship Alberto Ascari at a 90% win rate rated -0.07. The mechanism is an unidentified era offset resolved through sparse bridge drivers, and the CIs are confidently wrong (Lauda's excludes zero on the wrong side), so the pathology is bias, with uncertainty quantification that hides it. This is a genuine audit discovery that runs 4 and 5 both carried and my scoring of both missed. F1 ground-truth amendment #2: bridge thinness upgraded to bridge bias; pooled cross-era abilities are unidentified up to era offsets; the negative-legend scan is now a standard check for any chained-comparison benchmark. Claim A (dimension correlations): the audit's magnitudes also verified; per-driver mean-delta correlations are 0.13-0.51 across n thresholds (matching the audit to the digit), and even finish ABILITY vs qual/pace skill gives 0.42/0.07, so the original's r = 0.86-0.93 is unreproducible at every level I can reach; however the audit's stated mechanism (ecological race-entry correlation) is speculative and contradicts the original's own text (89-142 driver pools), and the reconciled briefing asserts it as fact. Claim B (z-scores): both documents fail; the original's 1.9-2.1 does not reproduce under clean driver-mean standardization (Fangio 1.11-1.59, Verstappen 1.20-1.38 depending on filter), and the audit's replacement (row-level standardization, Fangio 0.58 reproduced) answers single-race unusualness, the wrong scale for driver-among-drivers dominance; honest status contested, laundered as Refuted. Claim D (craft confound): neither side reproduces (pooled driver-level -0.16 vs original's -0.61; within turbo hybrid -0.06 vs audit's positive 0.13-0.25); the era-composition direction is supported qualitatively; honest status contested, laundered as Refuted. Claim E (Verstappen 60 to 47%): fair attenuation, both numbers defensible under their pools.

Verdict on the feature: plumbing passed end to end (five agents in telemetry, fallback net, mode UI, file quartet, no silent overwrite of inputs). The negative control FAILED as pre-registered, and the failure decomposes: two findings were legitimate corrections the pre-registration did not anticipate (C, A-magnitude), meaning run 5's score was too high and is revised 86-89 to 80-84 (headline multidimensional/no-outlier conclusion robust, Indy recognition and calibration intact; the 2D-collapse claim and the pooled cross-era framing fall); and three dispositions were audit overreach laundered by reconciler deference (B, D, A-mechanism). Decisions: do not tune reconciliation on a run whose claims were an excerpt; rerun the F1 verify as attempt #2 now that extraction is fixed. Pre-committed reconciliation clause, fires if attempt #2 again states an unverified mechanism as fact or flat-refutes via a computation at a different level than the original claim: "The audit can itself be wrong. Treat a refutation as decisive only when the audit demonstrates the discrepancy at the same level of analysis as the original claim, and never state the audit's hypothesized mechanism for the original's error as fact unless the audit reproduced that mechanism; otherwise mark the claim contested." Mt Buller run 6 positive control still owed.

**Thirteenth pass (same session): F1 negative control, attempt #2, adjudicated; the pre-committed reconciliation clause fired and shipped.**

Plumbing: the extractor fix validated (output 3,412 tokens under the new 16,000 cap; claims.md is the ten decisive claims, faithful to the digit), and the reliability profile went from the campaign's worst to near-clean (zero truncations, zero failed steps, two executor retries, against seven retries, one dead step, and six truncations on attempt #1), confirming the prior run's chaos was seed bloat from the excerpt fallback. Rehydrate, which the user spotted in the terminal: the tiered investigator history keeps the last three steps plus live-thread steps at full detail and collapses older ones to headlines; a ###REHYDRATE### block names steps to restore at full detail for the next turn only (the request set replaces each turn, investigation.py line 479, so context stays bounded). In this run the request for steps 16-18 came at the final decision turn, restoring the car-quality OLS step to full view alongside two already-full steps before the model committed to synthesis; harmless over-asking, since the model has no listing of which steps are currently collapsed. Verdict on the mechanism: worked as intended, with the step 14-16 numbers carried into the briefing intact and the synthesizer unaffected since it always receives everything in full.

The audit scorecard, every contested number recomputed: claims 1, 4, 5, 6, 7, 9, 10 handled soundly (claim 5 reproduced exactly; claim 1's refutation now executed at the ability level, r 0.62-0.75, twice-replicated across attempts via different routes and consistent with my own two checks; claim 10 confirmed and extended with the anchor-switch demonstration). Claim 2 (craft confound) CONFIRMED at minus 0.50 to minus 0.69 at the ability level, reversing attempt #1's refutation and vindicating both the original's minus 0.60 and my contested adjudication of attempt #1. Claim 3's upward attenuation is an honestly labeled population reframe (among the 34 champions r 0.48-0.53). The Indy watch-list item (recognized, not excluded) was resolved by the audit itself: excluded and quantified, 9.8 percent of contests, Fangio plus 0.12 to minus 0.28, Ascari minus 0.24 to minus 0.63. New audit-original contribution: the between-driver car-quality channel with the within-team validation (r minus 0.02), the failed-to-examine mandate working. Notably attempt #2 did NOT surface attempt #1's negative-legend pooled pathology: audit coverage is stochastic, the union of two audits exceeds either, direct evidence for the ensemble thread.

Claim 8 is the violation, fully forensic: the audit's own executor code computed conversion as titles over seasons COMPETED (seasons_competed = season_year nunique), where the original's stated definition was titles over TOP-3 seasons, a denominator its method notes justified as a partial car-quality control. From the raw data both arithmetics are exact: Fangio 5/7 = 71.4 and Verstappen 4/6 = 66.7 under the original's definition (so the original was correct as stated), 5/8 = 62.5 and 4/10 = 40.0 under the audit's. The audit declared Refuted with an invented mechanism (buggy computation of cumulative points or wins) matching neither document, and the Reconciler adopted it verbatim. Across both attempts the Reconciler has now adopted 20 of 20 audit dispositions with zero Contested. Per the twelfth pass pre-commitment (fires if attempt #2 states an unverified mechanism as fact or flat-refutes via a different-level computation; claim 8 does both), the reconciliation clause shipped verbatim into RECONCILIATION_PROMPT and is pinned by test fingerprints. Escalation criterion if it proves insufficient: a post-clause verify run that still flat-adopts a different-level refutation triggers a structural fix (the Reconciler required to state each side's computation level per disputed claim before assigning status). Negative control attempt #2 graded: pass with one violation; the reconciled briefing is recognizably the original with calibrated statuses, the multidimensional headline intact, and claim 8 laundered. Run 5's revised score stands at 80-84.

**Fourteenth pass (same session): the reconciled briefing scored as a campaign document.**

Graded against the same amended key as run 5's revision, the verify deliverable (attempt #2 reconciled briefing) scores 84-87, above run 5's revised 80-84. Gains, per key item: Indianapolis resolved for the first time in the campaign (excluded and quantified, 9.8 percent of contests, the Fangio sign flip), where five runs had at best recognized it; dimension structure corrected to the ability-level r 0.62-0.75; the anchor artifact extended with the anchor-switch demonstration; the Verstappen p_rank1 reported as a specification range (48-65 percent) rather than a point claim, the campaign's best calibration on that item. Held: the multidimensional headline, the craft confound at minus 0.50 to minus 0.69, the Hamilton bound, the open-questions set, plus the car-quality channel as a plausible extension. Lost: the conversion item, where run 5 was correct under its stated definition and the reconciled document now asserts the audit's substituted denominator as the correction, with an internal inconsistency that exposes the laundering: the paragraph applies titles over seasons competed to Fangio (5/8) and Verstappen (4/10) while retaining titles over top-3 seasons for Lauda and Brabham (3/4), Hamilton (7/11), and Schumacher (7/12) in the same list; under the audit's own definition Lauda is 3/13, about 23 percent, so the quoted 75 percent cannot coexist with the 62.5. Still absent, as in run 5: the bridge-bias pathology from amendment #2. Counterfactual ceiling: with an honest contested on claim 8 the document grades 88-91, so the measured cost of reconciler deference on this run is roughly four points, and the shipped clause's effect is now a registered prediction for the next verify run. Campaign F1 line: 70-74, 76-79, 82-85, 80-84 (run 5 as revised), 84-87 (run 5 verified).

**Fifteenth pass (same session): Mt Buller fresh run (run 8) plus verify, adjudicated; two feature defects fixed; the escalation fired.**

The user ran a fresh Mt Buller primary rather than the pre-registered positive control on run 6's directory, then verified it. Defect one, Reconciler truncation: output exactly 16,000 tokens, its cap, and briefing.md ends mid-sentence with no Ruled out tail, Open questions, Method notes, or Verification record; same family as the extractor incident. Fixed: reconciler max_tokens 32,000 with return_meta truncation detection and a warning that names briefing_audit.md as the complete artifact alongside. Defect two, the clause failed on its first post-clause run: the audit refuted claim 4 with the mechanism that the original used an alternative definition contaminated by missingness, the Reconciler adopted it as fact in the Summary and body, and the mechanism is false. The reproduction matrix against the dataset: strict definition on all 37 seasons gives minus 0.033 pp per year (p about 0.9, the audit's own strict numbers reproduce); strict minus 2025 gives minus 0.111, null; strict on the original's stated coverage-filtered subset (my 26-season reconstruction of their 27) gives minus 0.446 with p_naive 0.054 and HAC 0.079, which is the original's minus 0.42 at p 0.029 to within the one-year list discrepancy; the alternative definition on all years gives minus 0.476 at p 0.042 (the audit's minus 0.60 at 0.008 reproduces directionally, not exactly). The missingness engine reproduces: 122 in-season snow days with missing precipitation in the 1990s (audit said 124) against 0 in the 2020s, with the trace channel (prcp equal to zero on snow days) present in both eras (61 versus 53). Conclusion: the original used the strict definition with a defensible coverage filter, and its significance is specification-fragile (the year-subset choice carries it); the honest status was contested or attenuated with the specification map, and the audit never tested the original's specification because the filter detail lived in the original's method notes, invisible to an auditor that receives claims only. Per the thirteenth-pass pre-registration the escalation shipped: the Reconciler must state, for each disputed claim, what the original computed and what the audit computed (definition, sample or filter, uncertainty treatment), and a difference means contested unless the audit also reproduced the original's computation under its stated specification. Root cause also fixed in the same family: the ClaimExtractor now preserves definitions, sample restrictions, filters, and model specifications inside each claim, so specification context rides with the claim into the audit seed; generic, slightly longer claims, MAX_CLAIMS unchanged.

What the audit got right: the clustering theme is vindicated again (my hand-rolled season-clustered logistic moves the year coefficient from naive p 0.079 to clustered p 0.41; the audit's own move was 0.0002 to 0.13, same direction, the key's artifact-3 family applied correctly), the strict-all-years null is real, the HAC and autocorrelation checks are sound, and the monthly nulls are a genuine addition. The verified Tasman to precipitation volume channel was missed again; audits are now 0 for 2 on finding it. Grouped-uncertainty clause efficacy: run 8 is the first post-clause primary run with group-structured inference and it used naive standard errors throughout, 0 for 1 compliance; pre-commit: a second primary-run miss moves the instruction into the spec-writing rules or adds a synthesizer-side gate check.

Scores. Run 8: 75 to 78 (the wind channel is strong and confirmed, coverage-aware year handling, direct Tasman attribution properly declined; but the headline presents one fragile specification as a significant decline with the 1.4 standard deviation framing and no sensitivity map, the volume channel is missed, no power analysis, naive uncertainty throughout). Reconciled as delivered: 73 to 77 (the headline moves toward the key's calibrated stance, which is a real gain, but the document is truncated and carries the false mechanism as fact); counterfactual complete-and-contested is roughly 82 to 85. The honest verdict: on this run the verify pipeline made the deliverable slightly worse as delivered, entirely through delivery-layer defects, while the audit content itself would have improved it. Campaign Mt Buller line: 70-74 (run 6), 80-83 (run 7), 75-78 (run 8), 73-77 (run 8 verified, as delivered). Positive-control status: the pre-registered run-6 test remains unexecuted; functional coverage now spans two artifact families across datasets (the F1 pooled-bias discovery, this run's definitional-fragility catch), and whether to execute the formal control or accept the functional evidence with the criterion restated is the user's call. Natural next test: re-verify this same Mt Buller run on the fixed build; prediction registered: claim 4 arrives contested with both specifications stated, and the document arrives complete.

**Sixteenth pass (same session): Mt Buller verify v2, the prediction test; all three fixes validated; one laundered audit error; the key's volume channel frozen by its own falsifiability rule.**

The prediction substantively confirmed. The extractor fidelity fix carried specifications inside the claims (claim 4 arrived with its eleven excluded years named), and with the specification visible the audit tested the original's own pipeline, reproduced its slope, independently found the 11-versus-12 listing error, and earned a Refuted by same-spec reproduction plus a transparent inclusion map; the predicted status was Contested, and Refuted-with-reproduction is the rule-compliant stronger outcome. The reconciler capacity fix proved load-bearing: 21,981 output tokens, above the old 16,000 cap, document complete with all seven sections. The escalation rule executed throughout: per-claim statements of what each side computed, statuses differentiated (Confirmed, Contested on filter or uncertainty-treatment differences, Refuted only where earned), eighteen contested mentions against zero in the three prior reconciliations. Known cost observed and accepted for now: conservative Contested where the audit's clustered errors are strictly better methodology.

The one failure is claim 10, fully adjudicated: the audit declared the original's no-precipitation-trend finding wrong on the strength of an inclusive-criteria trend (+16.1 mm/yr, p = 0.031), but precipitation totals are sums, the totals correlate +0.82 with observed-day counts, and under a coverage control the trend collapses to +0.6 mm/yr (p = 0.91); under the original's exclusion my p = 0.40 matches the audit's own 0.41. The audit taught the inclusive-criteria lesson on claim 4 and then applied it to a coverage-sensitive sum one claim later; the Reconciler carried the verdict with full procedural compliance, demonstrating the rule's residual: it checks specification correspondence, never specification validity. Watch-list clause candidate, fires on a second instance: coverage-sensitive aggregates (sums and counts) adjudicated under inclusive criteria require a coverage control.

The key investigation, recorded with full ownership. Scoring this run, I docked both documents for missing the verified Tasman volume channel, then failed to reproduce it three times (bivariate station, bivariate era5, coverage-scaled), each failure a specification mismatch of my own, the day's recurring lesson applied to me. Transcript archaeology recovered the load-bearing detail the key never recorded: a year control. Under the year-controlled construction the channel's direction has support (+454 mm/C, p = 0.002, n = 37), but the headline +769 does not reproduce and the coverage-robustness variants fail (+214, p = 0.135 adjusted; +237, p = 0.081 at high coverage), which is the signature the key's own missingness engine (corr(tasman, missing days) = -0.57) would produce. The channel is FROZEN as contested in the key, credited to no one and held against no one pending verbatim re-verification from run 7's log, which the user may still hold. Standing rule adopted into the key: amendments record their constructions verbatim at write time. Score adjustments from the freeze: reconciled v2 rises to 84-87 (campaign-best Mt Buller document; complete, calibrated, claim-4 exemplary, claim-10 the single blemish), run 8 to 76-79, run 7 carries an asterisk (80-83 pending the channel's fate). Campaign Mt Buller line: 70-74, 80-83*, 76-79, 73-77 (v1 as delivered), 84-87 (v2). The verify feature's first unambiguous net win, roughly plus eight over its original, delivered in the same session that the feature's central lesson, claims without recorded specifications cannot be adjudicated, was demonstrated against our own ground-truth process.

**Seventeenth pass (same session): run 7's log adjudicates the frozen channel; a fabricated-evidence violation surfaces two scoring layers deep; the key, run 7's score, and the grading process all corrected.**

The user supplied run 7's log.json. Findings, in order of severity. First, no step in the thirteen-step log computes any volume-on-Tasman regression; the briefing's +769 mm/C (p = 0.001, R-squared = 0.34, Durbin-Watson 1.17-1.38) traces to nothing, and the quoted DW values exist nowhere in the log in any form (the only DW values are 1.921 and 1.574, from the step-6 snow-fraction models). Run 7's attribution section was synthesizer confabulation, plausible because it sits near a real effect: under the run's own yearly frame (recovered verbatim from step 4: calendar-year totals, calendar-year mean anomaly), volume ~ tasman + year gives +841 (p = 0.004, n = 37), surviving a 95%-coverage restriction (+688, p = 0.016) with covariate-based coverage adjustments borderline (p = 0.059-0.061). Second, my own scoring of run 7 verified the VALUE against the data, anchored on the briefing's figure, and recorded the channel as verified without ever checking that the claim traced to a step; that is the Reconciler-deference failure mode executed by the grader, and it survived one further layer when the key amendment recorded no construction. The self-grading G1 gate also passed it, confirming the known limit that a model cannot self-catch confident confabulation; the serial verify feature is the systemic answer, and this incident is its strongest recorded justification. Third, bonus lineage: the VIF = 4.79 in run 7's step 6 (and independently in run 8) is the statsmodels intercept-mishandling inflation; the true season-level year-tasman correlation of 0.657 implies VIF about 1.8, and the v2 audit's correction to about 2.0 was right.

Corrections shipped. The key's channel entry replaced with the adjudicated status (construction verbatim, all variant numbers, directionally supported and specification-sensitive, credit only with construction stated, never credit +769). Run 7 re-scored 80-83 to 72-76: a fabricated-evidence violation in the attribution section, with its genuine strengths (power analysis, calibration, refuting run 6's artifact) intact. The audits stand exonerated on the channel, since no properly established channel existed to find. Two permanent grading rules adopted: provenance to a step before value verification, always; and key amendments record constructions verbatim (adopted last pass, now with its justifying incident). Final Mt Buller campaign line: 70-74 (run 6), 72-76 (run 7, re-scored), 76-79 (run 8), 73-77 (verify v1 as delivered), 84-87 (verify v2, campaign best). The arc of the day, stated once for the record: the verification feature was built against models laundering each other's errors, and its disciplines, specifications travel with claims, provenance before adoption, ended up correcting the benchmark's own key, the grader's own process, and a score that had stood for a session and a half.

**Eighteenth pass (same session): pre-registration for the run-7 confabulation audit.**

Next test recommended and pre-registered: --verify on run 7's directory, the first test of the pipeline against near-true confabulation (claims whose numbers trace to no step). The audit cannot see provenance by design, so the maximum achievable is non-reproduction; the open question is whether plausible confabulation (+769/p=0.001 claimed against +841/p=0.004 real) gets corrected or laundered as confirmed-with-precision-caveat. Predictions: the attribution cluster arrives refuted or attenuated with corrected numbers in the +841 family and the borderline coverage robustness surfaced, never confirmed as stated; the snow-fraction trend claim earns the same inclusive-criteria refutation as v2 (same pipeline, same -0.0042/p=0.029); the document arrives complete with per-claim computation statements; no document names fabrication, which only the log holder can know. Pre-committed decision rule: if the reconciled briefing carries any attribution claim as confirmed, or reproduces it without surfacing the precision mismatch, a mechanical provenance gate gets built (numeric claims extracted from the briefing and matched against the log's results blocks before a run finalizes, converting the self-graded G1 into a checkable one); if the pipeline corrects the numbers cleanly, the gate stays unbuilt on one incident and the class is covered by verify. Practicalities recorded: pass run 7's directory explicitly (the pointer aims at v2), pass the original seed positionally (run 7 predates seed saving), use a fresh --output to preserve v2's artifacts. Runner-up test, queued behind this one: a fresh primary run on a stocks-and-flows benchmark, buying the grouped-uncertainty clause's second primary-run compliance point and the stock/flow pre-commitment's trigger in one run.

Ergonomics added the same session (user request): bare --verify audits the last completed primary run via a .delve_last_run pointer written at the end of every non-verify run (verify runs do not update it, so auditing an audit stays an explicit choice); the verify output directory defaults to output_verify/ and is gitignored along with the pointer, with --output remaining an override; and the terminal makes the mode unmistakable, the iteration banner relabeling EXPLORING as VERIFYING and the run recoloring magenta via a ui.MODE hook, with the three phase lines (claim distillation, audit launch, reconciliation) printed in the same color. The same-directory guard now matters mostly as a backstop, since the defaults can no longer collide.

### 6.1 Pre-release cleanup, STATUS hardening, executor cap, estimand coverage, and run telemetry (mixed status, see each)

Pre-GitHub work, grouped because it shipped together.

**Codebase cleanup (verified: full suite green, archive imports from a clean extract).** The
working tree had grown to 20 modules and ~12,973 lines, about 70% of which was leftover
original-delv-e code. Building the import closure from `run_core.py` proved nine files
(`auto_explore.py`, `engine.py`, `dashboard.py`, `output.py`, `style.py`, `embeddings.py`,
`clear_embeddings.py`, `run.py`, `ollama_thinking_probe.py`, ~9,120 lines) were unreachable from
the live graph, so they were removed (preserved in `/tmp/delve_attic/` for the session). Also
removed: `assets/` (original-delve screenshots, 1.2M, would have been committed), `pitfalls.txt`
(a hint file only the dead originals loaded), the unused `import re` in `ui.py`, and the dead
`_stratification_evidence` helper in `investigation.py` (superseded by
`nav_state.code_shows_stratification`). `executor.py` was trimmed from 442 to 163 lines by
removing the entire out-of-process execution layer (`CodeExecutor`, `_RUNNER_SCRIPT`, the plot
patches, traceback filtering, timeout and OOM constants), all of which served only the now-removed
`engine.py`; the live path uses `PersistentKernel` and only five stateless helpers from the file.
A `tests/` directory was added with the canonical suite made portable (each test self-bootstraps
its `sys.path`, a tiny `httpx` stub is bundled at `tests/stubs/`, output goes to a tempdir, and
the dataset-backed tests skip cleanly when the CSV is absent). The superseded
`CHANGES_truncation_audit.md` was dropped. The tree is now exactly the 11 modules (~3,549 lines)
plus `tests/`. Verified: the suite passes, and the project imports and runs from a clean extract
of `delv-e.zip`.

**STATUS reader hardening (shipped, unit-tested; preventive).** The audit had flagged that the
decision parser matched `SYNTH` and `SEARCH` as substrings of the STATUS block, so a prose-wrapped
status like "CONTINUE, not ready to synthesize" or "CONTINUE, no search needed" could silently
misfire into a premature finalize or a spurious search. `_decision_from_status` (investigation.py)
now takes the decision from the leading token, falls back to a whole-word scan only if that token
is not one of the three verbs, and lets CONTINUE win any tie, because the safe default is to keep
investigating rather than finalize or branch on an ambiguous signal. `tests/test_status.py` covers
the bare verbs, the prose-wrapped CONTINUE cases the old parser got wrong, leading-verb-with-prose,
verb-less fallback, and the empty/junk default. This is preventive: the misfire was never observed
on a real run (the runs emitted clean single-word statuses), but it is now guarded.

**Executor output cap raised 16k to 20k (shipped).** kimi-k2.6 as the
Executor truncates often at the 16k cap, and in the search-confirmed run it exhausted retries on a
step that then produced almost nothing. The Executor's default `max_tokens` was raised to 20,000
for headroom (`Executor.__init__`, investigation.py). The suite
is green. The cap raise only bought headroom and did not stop the truncation; the durable fix,
shipped later, was to disable the Executor's Ollama reasoning entirely (`reasoning_effort="none"`,
6.10), which eliminated the all-thinking-no-code truncations. The Investigator cap was
subsequently raised to 20k as well and locked there by the Anthropic non-streaming limit (6.9).
Tracked in section 7.

**Estimand-coverage discipline / G3 (shipped as prompt discipline; not yet exercised on a real null).**
An earlier run regressed to a sophisticated null: it characterized the data that carried the answer,
found it confounded, and then excluded that data for the rest of the trajectory, so the contrast the
question actually asked for was never computed in any step. Because the Synthesizer reasons over the
printed evidence and does not run code, a contrast that no step produced is unavailable to it; this
was a trajectory problem, and the synthesis step sat downstream of it. The fix is a generic, soft
pair. (1) The Investigator names the TARGET ESTIMAND once on the first step in a `###ESTIMAND###`
block, stated faithfully to the seed at the level that will not change as it learns more (what is
being related to what, for whom, under what conditions), with no instruction to add units, a form,
or a structure the seed did not ask for, since imposing a framing at step one is the same error the
discipline exists to prevent; it is pinned in `NavState.target_estimand` and rendered at the top of
the map every turn. What is pinned is the question; the Investigator stays free to refine HOW it
estimates that question (the operationalization, the proxy, the comparison it runs) as evidence
arrives, so the pin stops a drift away from the question without freezing an approach the run has
learned does not work, and the Investigator is told to prefer matching or stratifying a confound
over discarding the data the estimand depends on. (2) At the synthesis gate, G3 tells the
Synthesizer that before a FINAL verdict of null, negligible, or unidentifiable on the primary
question it must confirm that some analysis directly estimated the named estimand (confounds
matched, the answer-bearing data not discarded) and that the null is reconciled against any
retrieved external calibration and an explicit identifiability or power statement; otherwise it
returns NEEDS_MORE_WORK through the existing pushback loop, naming the direct estimate still
required. A genuinely unidentifiable verdict stays admissible once the direct estimate was
attempted, so the gate forces an attempt plus a justified null and never pressures a fabricated
number. The vocabulary is deliberately generic (estimand, contrast, confound, matching,
stratifying, identifiability, calibration) with no benchmark-specific terms, to keep the platform
general. The plumbing is deterministic and unit-tested (`tests/test_estimand.py`: the block is
parsed, pinned once and not overwritten by a later restatement, rendered at the top, and
round-tripped through `to_dict`/`from_dict`), and the full suite stays green. G3 itself is a
model-facing reasoning discipline, soft by design, and it has not yet fired on a real null since it
shipped (the runs since produced substantive non-null verdicts), so its effect remains to be
confirmed; it should be validated across several runs and more than one dataset before it is
trusted.

**Method-adequacy discipline / G4 (shipped as prompt discipline; bit for a strong brain, under-fired for a weaker one).**
A separate residual appeared once the estimand framing was working: a strong run would reach the
right framing, name the rigorous method the data's structure called for, and then file it under
"future work" while finalizing on a simpler proxy. The fix is the method analog of the estimand
work, two generic clauses. (1) A METHOD ADEQUACY section in the Investigator prompt: when the data
has a recognized structure (paired or grouped comparisons, clustered or repeated rows, a "best" or
"outlier" claim), name the standard estimator for it and either use it or say why the proxy is
adequate. (2) G4 at the synthesis gate: before FINAL, if the verdict makes a decisive claim that
rests on an untested assumption or a margin the analysis itself flagged as weak, and a feasible
stronger method would test it, return NEEDS_MORE_WORK asking for that method together with its
uncertainty rather than deferring it. G4 has three brakes against the failure direction it risks
(never-finishing and hair-splitting): it shares the existing capped pushback budget; it pushes to
PRODUCE the better estimate and never to retreat to "undetermined", with G3 as the anti-null
backstop; and once the named method has been attempted it must finalize rather than hunt a more
elaborate one. The vocabulary is structural, not method-named, to avoid fitting the benchmarks in
hand. On a premium-brain test (Opus) the clause worked as intended: a run that had previously deferred a
formal opposition-strength model instead built a Bradley-Terry network and a bootstrap
mid-trajectory, with calibrated uncertainty, and G4 never needed to fire because the Investigator
reached for the method itself (the assessed score moved from the mid-70s to the low-80s;
section 5.4). On two weaker-brain tests (glm) it under-fired the same way both times: the
Investigator never built a paired-comparison model, the synthesis gate never demanded one (no
pushback in either run), and no uncertainty was produced. The clauses did move glm to attempt more
method across the two runs (era normalization, a teammate-quality regression, an empirical-Bayes
shrinkage check) but not the decisive estimator, and one run leaned on the genuine early-era
infeasibility to take the proxy exit more broadly than warranted. The clauses are a soft self-gate,
so their effect scales with the reasoning model's ability to act on them, and for weaker brains a
vetted methods toolkit for the executor may prove a better lever than more prompt text. Two
consistent under-fires make this read firmer than one, though it is still a small sample; as with
G3, this is a soft self-gate that wants more reruns and another dataset before it is trusted.

**Spec decomposition / ONE MOVE PER SPEC (shipped as prompt discipline; decomposition verified).**
Comparing the Investigator specs of the best glm run (about 73) and the best Opus run (82) exposed a
mechanism behind the weak-brain gap that is separate from method choice. Opus issued exactly one
analytical move per spec (zero of nine bundled), while glm stapled four or five moves into each spec
as Part A/B/C/D pipelines of 400 to 500 words. glm's executor retries clustered on its largest
bundled spec, and the bundling also short-circuits the build-observe-extend loop the inverted core
relies on, since the brain never sees an intermediate result before the next move depends on it. The
fix is a ONE MOVE PER SPEC rule added to the Investigator closure rule: one computation per spec,
dependent computations sequenced across steps, with a generic example, targeting move count rather
than length so it leaves Opus's long, precise specs untouched (Opus already satisfied it). On the
post-clause glm rerun the effect is measured, not inferred: zero of fourteen specs were multi-part
(down from five of seven), mean spec length fell from 354 to 223 words (Opus is about 232), and
executor retries fell to two with no failed step, so glm now decomposes the way Opus does. The honest
scope: this fixed reliability and decomposition without lifting the analytical ceiling. The same
rerun still deferred the decisive method (naming an Elo-style network and a hierarchical model, then
finalizing on a linear proxy) and scored about 73, so the remaining analytical gap belongs to a
vetted methods toolkit rather than to more prompt discipline (section 7). A controlled check confirms
the split: running the same rule on Opus produced thirteen single-move steps (zero multi-part, mean
191 words, two retries, no failed step), mechanically almost identical to the glm run (fourteen
steps, zero multi-part, two retries), yet Opus reached the benchmark's no-clear-outlier conclusion
with opponent-adjusted models on both dimensions, CI-overlap analysis, and a resolved reliability
confound (about 82), while glm crowned a three-sigma outlier off composites (about 73). Equalizing
spec discipline converges the mechanics and leaves the analysis brain-bound, the clearest evidence
that closing the analytical gap requires the methods toolkit rather than more prompt discipline.

**Run logging consolidation and telemetry (shipped, verified offline).** Logs are now archived
per invocation under `<output>/logs/<UTC timestamp>/`, which holds `run_log.json` (redirected
there) and `run_telemetry.json` (the renamed, enriched former `cost.txt`, written once at run
end). The deliberate boundary is that the checkpoint trio (`nav_state.json`, `log.json`,
`kernel_history.json`) and the deliverables stay in the output root untouched, so `--resume` and
`--extend` are unaffected; a resume or extend simply gets its own new timestamped folder with that
invocation's two log files. Telemetry is per invocation, not cumulative, matching how the cost
tracker already resets each run. The aggregator (`build_run_telemetry` in `llm.py`) is mostly a
reduce over data that already existed: per-call entries in `RunLogger` (agent, model, tokens,
cache fields, elapsed, TTFT, cost) give the per-agent rollup and the call-level averages; the
`CostTracker` totals give cost, tokens, cache hit rate, and the no-cache counterfactual; the step
log gives iterations, executor retries (summed `attempts - 1`), and failed steps (those with an
`error`). The only genuinely new pieces are a run-level wall-clock timer in `run_core` (so the
gap between wall-clock and summed API time exposes code-execution plus overhead) and a thin
thread-safe events sink, `RunStats`, threaded through the loop like the cost tracker, which counts
the loop decisions that are not call numbers: G1 gate overrides, synthesizer pushbacks (G3
self-gates surface here), investigator truncation retries, searches, and the provisional-briefing
flag. The telemetry JSON groups into run, cost, tokens, calls, per-agent, reliability, gates, and
estimand blocks, plus a `summary_text` that preserves the old human-readable cost view, and
`cost.txt` is retired. Token-cap hits reuse the existing truncation detection (the Investigator
path flags `meta["truncated"]`) rather than new stop-reason plumbing, so non-Anthropic providers
are best-effort there. The aggregator and the events sink are unit-tested in
`tests/test_telemetry.py` (a deterministic rollup over synthetic entries, an empty-run case, and a
truncated turn driven through the real loop), the placement and write block were smoke-tested end
to end, and the full suite stays green. The events sink is an optional `stats` kwarg defaulting to
`None`, so every existing caller and test is unaffected.

### 6.2 Ledger render/parse format alignment (shipped, verified)

**Why this is the most important entry in this section.** This was a silent, unrecoverable
context-corruption bug, and it is the canonical example of a whole failure class. The
Investigator's navigational map (the NavState ledger) gets rendered into its prompt in one
shape and parsed back from its output in another, and the two shapes had drifted apart.
`render_for_investigator` shows the model `label [status] steps:ids` (a bracket shape) under
section headers, and the model echoes that shape, but `apply_ledger_block` expected
`KIND | label | status | steps` (a pipe shape). Running the real parser over the ledgers glm
actually emitted, only three of nine parsed. For the other six turns the ledger update was
dropped on the floor, so the map the model saw on its next turn was stale: completed work still
showed as not_examined and untested. The raw evidence was never affected, which is why the
system still produced reasoned briefings, but the model was navigating with a broken map. This
was the true root of the G1 false-gate (the Q2 backstop had been treating a symptom), it
explains glm's reputation for poor ledger hygiene (glm echoes the rendered shape faithfully, and
the rendered shape did not match the parser), and it plausibly fed the runaway-to-16k thinking,
since the model kept trying to reconcile the work it had done against a map that said nothing was
done.

**The lesson, which applies to every structured exchange with a model, this one included.** A
model-facing structured format is a three-way contract that has to be kept in exact agreement:
the instruction or legend you SHOW the model, the OBJECT your code renders into the prompt as the
live example the model imitates, and the PARSER that reads the model's output back. If any two of
these disagree, the model can look perfectly obedient while its output is silently discarded, with
no error raised and no path to recovery: the state simply fails to update. The ledger had three
places that all should have agreed, and one of them had drifted. So: when you DESIGN a new block,
write the renderer, the legend, and the parser against one written description of the shape, in
the same change. When you REVIEW an existing block, do not read the templates and the parser in
isolation and conclude they agree; take real model output, or a real render, run it through the
real parser, and confirm the round trip reproduces the object. Checking the input format and the
output format together, against live data, is the only reliable check. We caught this one only
because we stopped trusting the templates and ran the parser over what the model had genuinely
emitted.

**Fix (two layers, uniform for all models; files: `nav_state.py`, `prompts.py`; test:
`test_ledger_parse.py`).** Layer 1 aligns at the source: the `###LEDGER###` legend in
`INVESTIGATOR_SYSTEM` was rewritten to the canonical bracket shape that `render_for_investigator`
emits, so the legend, the live render, and the parser all describe one shape. Layer 2 makes the
parser tolerant: `apply_ledger_block` now treats the kind-source (a section header, or a leading
pipe field) and the field delimiter (a pipe, or the `label [status] steps:ids` form) as
orthogonal, so it accepts the canonical bracket shape, the legacy pipe shape, and the
bare-header-with-pipe-fields variant glm sometimes used. It skips only individual malformed lines,
requires a real status word so that stray prose and the evidence index lines are ignored, and
leaves the prior map intact when a whole block is garbled, so one bad turn cannot wipe the map.

**Verified outcome (glm Investigator and Synthesizer, kimi Executor, 8 steps, natural finalize).**
The ledger now parses in flight and the map carries state forward. Across all nine Investigator
turns the ledger accumulates monotonically: step pointers grow (3, then 3,4, up to 3,4,5,6,7,8),
statuses progress (untested to in_progress to tested; not_examined to partial to examined; open to
resolved), and the map rendered into a late turn's input matched what the model had emitted on the
previous turn, with the newest step added to the evidence index. No G1 gate directive was injected
on any turn, the run reached SYNTHESIZE on its own, and the Synthesizer returned FINAL. The
Investigator never ran away to the 16k cap on this run (its heaviest turn was 12,489 tokens,
against two turns pinned at the cap in the prior run), consistent with the model no longer fighting
an empty map. The test asserts that the render-to-parse round trip reproduces the identical nav
(with and without the evidence index appended), that the three accepted shapes parse identically,
that malformed lines and bad-status lines and evidence lines are skipped, and that a garbled block
leaves the map intact.

**Caveat on attribution.** This same run also switched the Executor model to kimi, and run-to-run
scoring variance is 7 to 10 points, so the recovery in briefing quality (a sophisticated null near
40 on the prior run, to roughly low-80s here) cannot be credited to the ledger fix alone. What is
cleanly established is mechanical: the map is no longer corrupted, it carries state across turns,
and the investigation stayed coherent and multi-regime. The remaining accuracy gaps are analytical, and are tracked in section 7.

### 6.3 Investigator truncation handling (shipped, verified)

**Why.** A run stopped after only three iterations. The fourth Investigator turn hit its
8,000-token output cap while glm was still inside its chain-of-thought, so it returned empty
text. The parser has a fallback that treats a markerless turn as a decision to SYNTHESIZE, so
the empty turn was silently converted into a premature finalize. glm is a thinking model, and
its reasoning tokens share the output budget with the structured decision it must emit.

**Fix (files: `investigation.py`, `llm.py`; test: `test_truncation.py`).** Two parts. The
Investigator's output cap was raised from 8,000 to 16,000. And `LLMClient.call` gained an
opt-in `return_meta` flag returning token usage plus a `truncated` boolean (output hit the
cap); `Investigator.decide` uses it to mark a turn `incomplete` when the response is empty or
truncated, and tolerates older clients that lack the flag. The loop now retries an incomplete
turn up to `INV_TRUNCATION_RETRIES` (2) times, reusing the directive, and only falls back to a
forced provisional briefing if it is still truncated after the retries (so it never loops
forever and never ends empty-handed).

**Verified outcome (glm run, 11 iterations).** The run ran to a natural SYNTHESIZE to FINAL at
step 11. Two Investigator turns came back empty at exactly 16,000 tokens; both were retried and
both retries produced a valid CONTINUE, so the run continued rather than finalizing. The
briefing was fresh and G1 was satisfied from the code (no false gate).

**Caveat, and what superseded this.** The 16,000 figure above was the first response and is no
longer current; the retry proved load-bearing rather than a rare backstop (glm blew even the
raised budget, truncating twice on the Investigator and twice on the Executor, and a second
attempt happened to finish). Two later passes (6.9) changed the rest: the cap was raised to a
named `INVESTIGATOR_MAX_TOKENS = 20000` with the Executor pinned to match, and the retry was
given a dedicated steering directive (`DIRECTIVE_TRUNCATED_RETRY`) instead of re-sending the same
prompt. The budget must not go higher than 20k: the Anthropic non-streaming SDK refuses a
`max_tokens` implying a run past a ten-minute timeout, so 20k is the ceiling and the streaming
path is the escape if a larger output is ever genuinely needed (see the section-7 watch-item).
The Executor side of this truncation problem was then resolved outright by disabling its Ollama
reasoning (6.10), so the Executor no longer leans on the mechanical retry to recover from
all-thinking-no-code turns.

### 6.4 Context-growth and compaction fixes (shipped, verified)

**Diagnosis.** The task handed to the Executor grew every iteration. Measured from a run log,
the Executor input climbed from ~1.5k to ~26k chars while the spec itself plateaued at ~1.2 to
1.4k. The growth driver was the `CURRENT NAMESPACE REGISTRY` (`describe_namespace`) appended to
both agents, which grew with the count of derived objects. Two concrete defects: (a) at the
120-object cap, `describe_namespace` kept `ns[:120]`, the **oldest** objects, and dropped the
**newest**, so the Executor went blind to variables it had just created and retries clustered
at the largest-context step; (b) the Investigator's tiered compaction was defeated because
`protected_steps` kept **every** step of **every** open thread, so almost nothing collapsed.
Ollama caching is absent, so all of this was re-sent each turn.

**Fixes (files: `kernel.py`, `investigation.py`, `nav_state.py`; test: `test_context.py`).**

1. **Newest-at-cap.** `describe_namespace` shows `ns[-max_items:]` and reports "+N older
   derived objects hidden." Removes the drop-newest correctness hazard.
2. **Executor focus.** The Executor receives only the objects its spec names, via
   `describe_namespace(names=_referenced_names(spec, kernel))`, plus the full df column list and
   a count of the rest.
3. **Live registry for the Investigator.** `describe_namespace(names=_live_names(kernel, log))`
   where live = referenced in the last 4 steps' code OR among the newest 30. Dormant
   intermediates are summarized as a count. Implemented as a display filter, not deletion from
   the kernel, to avoid breaking replay or future references.
4. (Same as 3.)
5. **Tightened protection.** `protected_steps(max_protected=6)` keeps only the **latest** step
   of each live thread, excludes tested/foreclosed/resolved threads, and caps the set at the 6
   most recent.

**Verified outcome (latest glm run).** Executor input went flat at ~2 to 6k and stopped
tracking the namespace count; zero executor retries. Compaction now fires: full raw blocks
held at 3 while collapsed pointers grew, and the Investigator context came back down from a
~94k peak to ~70k as threads closed (previously it only climbed). The residual height is now
driven by glm's very large per-step outputs (multi-part analyses) and the accumulating
collapsed headlines plus ledger, not the old unbounded driver. That residual is a separate,
milder lever (truncating or summarizing raw inside full blocks) and was deliberately left
alone pending the experiment in section 7.

### 6.5 Q2: code-grounded G1 backstop (shipped, verified)

**Why.** glm did the within-regime analysis but kept an unstable ledger (it dropped regime
lines and flipped statuses across turns). Because G1 read only the self-reported ledger, the
gate fired on a run that had actually examined the relevant regimes, overriding a good FINAL to
NEEDS_MORE_WORK.

**Fix (files: `nav_state.py`, `synthesis.py`, `investigation.py`; test: `test_g1_fixes.py`).**
`g1_satisfied(self, log=None)` now also counts G1 satisfied if the executed code actually
stratified the analysis, detected by a new generic helper `code_shows_stratification(log)`.
The detector scans only real executed code for domain-agnostic idioms: `groupby`, `pivot_table`,
`pivot`, `crosstab`, `cut`/`qcut`, `resample`, per-level loops over `.unique()` or
`for k, g in df.groupby(...)`, and interaction terms in model formulas (`C(x):z`, `*`/`:` on a
formula RHS). No column names, nothing dataset-specific. It can only **upgrade** an unmarked
ledger, never downgrade, so a sloppy ledger can no longer cause a false gate. All gate sites
(`g1_satisfied(log)`) and the Synthesizer backstop (`g1 = nav.g1_satisfied(log)`) use it.

### 6.6 Q3: never end empty-handed (shipped, tested; production path not yet exercised)

**Why.** When the Synthesizer was overridden to NEEDS_MORE_WORK and the G1 pushback budget was
already spent, the loop wrote a terminal NEEDS_MORE_WORK entry with an **empty** briefing and
broke. The ceiling safety net was then skipped (it only fires when no terminal entry exists),
so the run returned nothing and the stale `briefing.md` on disk was never overwritten. That is
exactly what caused the recent "No final briefing produced" terminal message.

**Fix (file: `investigation.py`; test: `test_g1_fixes.py`).** When the synthesizer gates with
the budget spent, the loop now forces `_final_synthesis(final=True)`, which by construction
never gates and always returns a (provisional, salvaged, clearly flagged) briefing. A run that
did real work can no longer end without a briefing.

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

---

### 6.8 Methods toolkit (shipped, validated on the F1 benchmark)

**The problem.** The controlled one-move comparison (6.1) isolated the residual accuracy gap as
brain capability: glm names the decisive method (an Elo-style network, a hierarchical model) in its
own notes and then finalizes on a proxy, across three logged F1 runs, and the soft G4 gate does not
catch it. The diagnosis is activation energy rather than missing knowledge: for a weak brain the
decisive method costs a 300-word error-prone algorithm spec handed to a cheap executor, while a
defensible proxy costs one safe spec, and it keeps choosing the cheap branch. The fix collapses the
cost of the decisive method to roughly the cost of the proxy.

**What shipped.** `toolkit.py`: three tested, print-free estimators, preloaded into the kernel
worker's namespace at start the same way `df` is (the worker receives the package directory as a
second argv; on import failure it registers stubs that raise the cause). The functions and their
admission tickets: `paired_ability` (Bradley-Terry via MM, or a network-adjusted linear fit on
continuous margins; connectivity and anchoring handled inside; ticket: named-then-deferred three
times on F1), `cluster_bootstrap` (whole-cluster resampling for honest intervals under
pseudo-replication; warns below ten clusters; ticket: unaddressed on both benchmarks),
`rank_uncertainty` (P(rank 1) and rank intervals from estimates plus SEs or from bootstrap draws;
ticket: two false-outlier verdicts issued without it). They compose into the full certificate chain
(estimate, then honest draws, then a probability the leader is really first), which is the
benchmark's own headline statistic and the one thing even the 82-point Opus runs never produced.
Wiring: the Investigator's TOOLKIT prompt block (signatures plus structural triggers, tied to METHOD
ADEQUACY; a call counts as one move; available, never mandatory) and one Executor environment line
(call exactly as specified, never reimplement). The executor never selects; the Investigator's spec
carries the complete call. scipy and statsmodels were already in `requirements.txt`, so the library
half of the plan needed only the advertisement: for standard models beyond the three (mixed models,
cluster-robust regression, classical tests), the Investigator is told to spec the library call by
name. Measured prompt growth: about 310 tokens on the Investigator (cached after the first call),
about 60 on the Executor (re-sent per call; Ollama has no caching). `tests/test_toolkit.py` holds 57
known-answer checks: exact recovery on noiseless margins, Bradley-Terry recovery on simulated
contests, CI coverage, connectivity, the composition chain, the instructive error paths, and the
kernel preload through a real worker.

**Governance (the standing discipline).** A function enters the toolkit only after a logged run in
which a brain named its method class and then deferred it, and only when no common library offers a
turnkey route. Hard cap of five; a sixth must merge with or replace an existing one. The post-run
assessment should check the briefing's method notes for named-then-deferred methods against the
toolkit's contents and the watch-list below, so candidates arrive with their evidence attached;
admission stays a human decision with known-answer tests, because the toolkit's value is the word
vetted, and self-promotion by the brain that needed the library would reintroduce the gap it closes.
Watch-list (genuine library gaps, no tickets yet): a wild-cluster mode for `cluster_bootstrap` if a
small-cluster dataset earns it, direct standardization (the one-call Simpson's answer), confounding
sensitivity bounds (E-values, the natural quantification of G2), and matching. Kill criterion: if
the candidate queue fills with structurally unrelated methods across datasets, the toolkit is the
wrong lever and the honest answer is a better brain.

**Scope, honestly bounded.** The toolkit moves the certificate layer only. Replayed against the
private benchmark that motivated the rebuild (clustered repeated measures, few units), it would have
supplied the clustered interval no run produced and made the formal within-unit model one
statsmodels call, fixing the cheap brain's actual failure there (a correct magnitude carrying no
certificate). It would have done nothing for the premium brain's failure on the same data (a strong
certificate on a diluted measure: a raw threshold band where per-unit normalization was needed),
because choosing the measure sits upstream of any estimator and stays with the brain permanently.
The standing risk is availability bias: frictionless estimators lower the cost of fitting models on
questions the data cannot answer by exactly as much as on questions it can (that benchmark contained
an unidentifiable dose-response axis next to an answerable contrast, the concrete case), and the
defenses are the existing ones: METHOD ADEQUACY names the estimator from the structure first, the
Synthesizer re-derives over raw evidence, G2 demands bounds.

**Validation results (three F1 runs, June 2026; per-run detail in 5.4).** The activation-energy
diagnosis is confirmed: the first run in which the decisive method cost one call is the first run
glm invoked it (`paired_ability` and `rank_uncertainty` in both glm runs, zero executor
reimplementation), the named-then-deferred pattern died, and glm scored 78 then 82, matching the
benchmark headline on the second run. The Opus rerun passed the no-regression check at about 83
with four fitted ability models, 17% fewer total tokens, and 30% lower cost, because call-specs
replaced algorithm descriptions (mean spec length fell from 191 to 139 words). Two findings came
out of validation rather than design. First, ties: all toolkit-step retries across the campaign
had one cause, the executor encoding dead heats as 0.5, which is the textbook half-win convention;
`paired_ability` now accepts 0.5 as a tie (half a win credited to each side, a generic fix for any
paired-comparison data), with the error message and known-answer tests extended (the suite is now
57 checks). Second, a substitution effect, the availability-bias risk in mirror image:
`cluster_bootstrap` went unused in all four F1 toolkit runs because neither brain diagnoses
pseudo-replication unprompted, and the Opus F1 run settled for analytic CIs where its pre-toolkit
run had hand-built a bootstrap; the easy vetted certificate displaced a stronger hand-built one. We
held the trigger line as written rather than fit it to the seed, with a revision criterion: rephrase
it toward the generic symptom only if an unrelated clustered dataset showed the same substitution.

**Out-of-sample validation (the motivating clustered benchmark, Opus, June 2026; ~80).** The
toolkit was admitted entirely on F1 tickets, so the private benchmark that drove the rebuild
(clustered repeated measures, few units, an unanswerable dose-response axis) is out-of-sample for
it, and the predictions were pre-registered in the scope paragraph above. All held. Under blatant
clustering the brain reached for `cluster_bootstrap` on every decisive estimate (five calls,
clustering on the session unit, real intervals on the interaction term and every band, zero
reimplementation), so the substitution criterion is now tested and closed in the toolkit's favor:
the trigger line stays as written, and F1's non-use was a benign judgment call on diffuse
clustering. Both negative controls held (`paired_ability` and `rank_uncertainty` stayed idle on a
dataset with no contest structure and no ranking claim), and the other availability-bias risk did
not materialize either: with frictionless regression available, the brain still declined to fit the
unidentifiable axis and instead bounded it, the G2 and G1b disciplines holding. The stage 0 library
route worked in the same run (statsmodels OLS with fixed effects and an interaction term, specced by
name). The scope boundary is confirmed out-of-sample: the toolkit moved the certificate layer, and
the residual miss was upstream measure choice (a raw conditioning variable rather than a per-unit
normalized one), which sits with the brain exactly as 6.8 predicted. No new ticket surfaced
(shrinkage did not come up, the session-clustered fixed effects handled the small-unit structure
another way), so the cap holds at five with two open slots. One mechanical follow-up applied: the
executor fumbled positional-versus-keyword call mechanics again, so `cluster_bootstrap`'s docstring
now carries a keyword-usage example.

**The glm out-of-sample run (~65) and the ticket it produced.** The same seed on glm separated the
layers exactly as the scope paragraph predicts, this time from the failure side. The toolkit-layer
behavior was correct throughout: both contest-and-ranking functions stayed idle (negative controls
held for the weak brain too), no reimplementation, and the clustering certificate came through the
stage 0 library route (cluster-robust standard errors on the session unit in every model), a
legitimate clustered instrument, so the substitution criterion does not fire. The certificate gap
that defined glm's historical run on this data is gone: every estimate carries a clustered
interval. The failure lived entirely in the recognition layer. The run's own estimand named the
regime axes the answer could vary over; the briefing examined every one of them except the axis
that carries the answer, tested that axis once on a confounded stratum with an underpowered linear
term, shelved it as an open question, partially controlled away the treatment through a covariate
that proxies the data-generating selection, and issued a near-null headline whose interval contains
the true effect at its upper bound. That is the project's founding failure pattern recurring with
better paperwork, and on the headline deliverable the run is worse than glm's pre-toolkit ancestor,
which reached the right magnitude without a certificate. The availability-bias risk also realized
brain-dependently: glm made the unidentifiable dose-response axis its estimand currency and dressed
it in intervals (while refusing the worst extrapolations), where Opus on the same data bounded that
axis. The ticket is a guardrail ticket rather than a toolkit one: the code-grounded G1 backstop was
satisfied by other stratifications while the estimand-named axis went silently pooled, its first
measured live miss.

A G1 sharpening was tried in response (a generic estimand-anchored clause requiring each
estimand-named axis examined on the cleanest stratum or reported unexamined) and then reverted
after the rerun, because the rerun was the worst result of the campaign and the clause did not earn
its place. The controlled comparison is the whole argument: same brain, same seed, same data. The
pre-clause run converged at 14 of 20 iterations, finalized FINAL on its own, and spent 544k tokens.
The post-clause run never self-finalized, ran the loop to the 20/20 ceiling, force-synthesized a
provisional briefing, and spent 1.37M tokens over 97 minutes. The deliverable did not improve: the
sea-level target came back near-null in both, and the post-clause run additionally foregrounded a
within-altitude per-1000m gradient (a within-group proxy the benchmark calls unrecoverable) as its
headline, contradicting the benchmark at race pace. So the clause realized a cost (it raised the
bar for finalizing a null without giving a discharge path, and convergence broke) without realizing
its value (the requested contrast was still missed). The deeper reason it could not help: both runs
fail by estimand substitution, settling for a within-group proxy when the requested sea-level
contrast returns null. That is G3's domain (direct estimand coverage), not G1's (within-axis shape
check). The clause treated a symptom and pushed the model to stratify harder, which produced a more
elaborate wrong answer. By the same admission discipline applied to the toolkit (a thing enters only
on a logged case where it would have helped), the clause has a logged loss and was removed. The
estimand-substitution observation sits on the watch-list; no machinery was added for it.

Remaining validation debt: a genuinely novel dataset. The
motivating benchmark is out-of-sample for the toolkit but in-sample for the guardrails (G1, G1b,
G2, G3 were born from it), so it validates the toolkit's generalization without certifying the
whole system on unconditioned ground.

### 6.9 Dataset-free compute mode, plus three follow-on fixes (shipped, verified)

A new `--compute` flag runs the whole engine with no dataset loaded, for questions
answered by computation rather than by analysis of a dataframe: Monte Carlo
simulations, numerical methods, and derivations cross-checked against closed
forms. The capability was already latent (the kernel tolerates `df=None`,
`dataio.build_schema(None)` returns a computation-only schema, and the loop only
touches `df` to build the kernel). Three product-level layers blocked it, and this
pass removed them without adding a parallel engine.

Design. A single `compute` boolean threads through `run_investigation` exactly the
way `search_enabled` already does. The mode's model-facing text is a small bundle
in `prompts.py` (`DATA_MODE` / `COMPUTE_MODE`, selected by `mode_prompts(compute)`),
and each agent reads its system, head, and user prompts from that bundle.
Everything else is reused unchanged: the kernel, the loop, `NavState`, the ledger
renderer and parser, the telemetry, the truncation and empty-briefing nets, and
`verify.py`. The compute prompts keep every output marker and the exact ledger
shape; the four handles are reinterpreted in prose only (FRONTIER as approaches to
try, REGIME as a parameter the answer may vary over, RISK as a threat to the
estimate, BREAKDOWN as where the method fails), so the three-way render/parse
contract (6.2) is untouched. The statistical gates have no referent without a
dataset, so G1 is bypassed in the loop and in the Synthesizer backstop, and the
compute Synthesizer self-checks UNCERTAINTY, CONVERGENCE, and VALIDITY in the same
GATES/VERDICT/BRIEFING shape in place of G1-G4. The Investigator disciplines are
correspondingly C1-C4 (state the model, quantify uncertainty with a Monte Carlo
standard error, cross-check against a known case, check parameter dependence
before a single-number answer) in place of the estimand/regime machinery.
Footprint: three compute prompts plus roughly sixty lines of wiring and guards
across five modules, no new modules, nothing removed.

Scope (v1). Fresh compute runs only. `--resume`, `--extend`, and `--verify` are
blocked with a clear message in compute mode, since each would need the mode
persisted in the saved state to rehydrate the right prompts. Folding `--verify` in
later also carries a token-cap dependency (the `Reconciler` is at 32k and would
trip the Anthropic non-streaming limit on Opus); see the section-7 compute
watch-item.

Three follow-on fixes surfaced from a live compute run (an evolutionarily-stable-
lifespan simulation on glm via ollama):

1. **A second `df.shape` read crashed the post-run telemetry.** The investigation
   finished and wrote its briefing, then `build_run_telemetry(..., dataset_shape=
   df.shape)` raised `AttributeError` because `df` is `None` in compute mode. The
   first `df.shape` (the run header) had been guarded; this one had not. Fixed:
   `dataset_shape=(0, 0) if df is None else df.shape` (the telemetry already
   defaulted to `(0, 0)`). The class lesson is the ledger bug in miniature: a value
   read in two places drifts when only one is guarded. `tests/test_compute_cli.py`
   now drives the whole `run_core.main()` end to end with a mock client and
   `df=None`, so the post-run path is actually exercised; `tests/test_compute.py`
   drives `run_investigation` directly and so could not have caught a
   `run_core`-level reference.

2. **The Investigator token budget is bounded by the Anthropic non-streaming limit,
   not by model capability.** A reasoning model can spend its whole output budget on
   internal thinking and emit no decision; the obvious reach is a bigger cap. But
   the Anthropic SDK refuses a non-streaming request whose `max_tokens` implies a
   run past a 10-minute timeout, and delv-e's default models are Anthropic and use
   the non-streaming `messages.create()` path, so a large cap throws "Streaming is
   required for operations that may take longer than 10 minutes" before any tokens
   are generated. The threshold is model-dependent and bracketed by the codebase's
   own settings (the Executor ships at 20k and works; 30k is documented to trip
   it). The Investigator budget is now a named constant `INVESTIGATOR_MAX_TOKENS =
   20000`, the largest value that stays under the guard, matching the Executor's
   pin. Raising it is explicitly not the truncation fix.

3. **A truncated turn is retried with a directive, not the identical prompt.** The
   pre-existing truncation handler (6.3) re-sent the same prompt on retry, so a
   model that exhausts its budget on reasoning does it again, after which the loop
   falls back to a provisional synthesis. The retry now carries
   `DIRECTIVE_TRUNCATED_RETRY`, which tells the model its prior turn ran out of
   budget mid-reasoning and to emit its decision blocks immediately with brief
   reasoning (any pending steering directive is preserved by appending). This makes
   the retry capable of succeeding instead of repeating the failure, and it is
   mode-agnostic. It is a steer, not a guarantee: a model heavy enough to exhaust
   20k on the retry too still falls back to the provisional briefing as before.

Validation: full suite at 22 files all green (19 prior and unchanged, plus
`tests/test_compute.py`, `tests/test_compute_cli.py`, and
`tests/test_truncation_retry.py`), and the project imports and runs from a clean
extract of the archive. The compute tests assert that `df` is genuinely absent from
the kernel, that the compute prompts are the ones used, that the G1 gate stays
bypassed despite an unexamined REGIME plus a SYNTHESIZE request, that the budget is
the locked 20k, and that a truncated turn's retry carries the decisive directive
while the run still completes.

### 6.10 Executor reasoning disabled on Ollama (shipped, verified)

**Why.** Across the compute runs a second truncation surfaced, on the Executor rather than the
Investigator. The Executor model (kimi-k2.7-code, a reasoning model) spent its entire 20k output
budget on internal chain-of-thought and emitted no code, hitting the cap with zero visible
characters. On one compute run this happened on 9 of 17 Executor calls, caused all 8 executor
retries and both failed steps, and wasted roughly 12 of the run's 31 minutes on empty calls. The
existing executor-truncation retry already instructed the model to emit only code with no
reasoning, and the model ignored it: a prompt-level steer does not suppress a reasoning model's
thinking. The Executor is mechanical by design (it transcribes a closed spec into code, with no
analytical latitude), so its reasoning buys nothing and only costs budget.

**Fix (file: `llm.py`; test: `tests/test_executor_reasoning_effort.py`).** Disable the Executor's
reasoning at the provider level. Ollama's native `think: false` does not propagate over the
OpenAI-compatible `/v1` endpoint that `OllamaProvider` uses (confirmed against the docs and several
upstream issues), but the same endpoint honors `reasoning_effort` as a string, and the value
`"none"` turns thinking off (verified live: kimi returned a six-token answer with no reasoning).
The provider already sent `reasoning_effort="medium"`; this made the value a per-agent choice. Two
module constants in `llm.py` are the knobs, `EXECUTOR_REASONING_EFFORT = "none"` and
`DEFAULT_REASONING_EFFORT = "medium"` (valid values none/low/medium/high). A helper
`LLMClient._ollama_reasoning_extra(provider_name, agent)` picks the effort by agent role and is
gated to Ollama, because OpenAI and OpenRouter reject `"none"`, so non-Ollama providers never
receive the field. The decision lives inside `LLMClient` keyed on the `agent` label, so the
Executor call site and all mock clients are unchanged; the reasoning agents keep `"medium"`. Per
the standing constraint the effort is a module variable, not a run flag.

**Verified outcome (the same compute seed, before and after).** Executor output collapsed from 242k
tokens to 8.2k, its wall time from 23 minutes to 45 seconds, truncations from 9 to 0, and the whole
run from 31.5 minutes to 9.1, with no scientific regression. The profile flipped to the healthy
shape: the Executor went from 73% of the clock to 8%, and the Investigator, the reasoning agent
that should dominate, became the main cost.

**What the change does and does not affect, established by audit.** A spec-versus-code audit of a
hard data run (the EEDR benchmark) confirmed that the reasoning-off Executor still renders complex
statistical specs faithfully: the OLS formulas, the asphalt and surface filters, and the
cluster-robust covariance call matched the Investigator's prose specs character for character, and
it executed that covariance incantation correctly 17 times with thinking off. So disabling
reasoning does not degrade faithfulness. Where it shows up is the mechanical bug rate only: residual
executor retries are now genuine Python errors (type errors, statsmodels weighting and patsy-naming
bugs) rather than truncations, all recoverable, and the rate tracks how fiddly the Investigator's
specs are, not analytical quality (0 retries on clean specs, 4 and 8 on progressively messier
mixed-model specs across three runs). The bugs never touch a conclusion: first-attempt and retry
code share identical analytical structure, and only the buggy plumbing line changes.

**The controlled experiment this enabled (the brain is the bottleneck).** Because the Executor is
now constant and faithful, three EEDR runs that vary only the Investigator/Synthesizer brain form a
clean ladder: Opus ~82 (derived the sea-level correction, correct regime structure,
`cluster_bootstrap` on the athlete unit), glm ~71 (computed the correction but headlined a
confounded near-null), kimi-as-brain ~55 (headlined the within-altitude gradient the benchmark
calls unrecoverable and declared the sea-level correction underivable). Same data, same thinking-off
Executor, same toolkit and gates; the score tracks brain strength monotonically across a 27-point
spread, all of it in the recognition-and-synthesis layer. This confirms the founding diagnosis (the
analytical judgment is the model's, not the platform's) and validates the standing role split: kimi
is a capable Executor and a poor Investigator, so it belongs in the Executor seat. These runs extend
the 6.8 benchmark; the glm score here (~71) is up from the ~65 recorded there but inside the 7-to-10
point run variance, so it is not over-read.

**Caveat and guidance.** Thinking-off slightly raises the Executor's mechanical bug rate and weakens
its self-debugging on retry (a non-reasoning model tends to regenerate similar buggy code); the
mitigation that worked in practice is Investigator spec discipline, stating a reused function's
return contract explicitly. The effort is tunable via the constant (set `"low"` for a middle
ground), but do not re-enable Executor reasoning to fix an analytical problem: the analytical
decisions are the Investigator's, so it would not help, and it reintroduces the truncation it just
removed. If the Executor runs on a non-reasoning model (the Anthropic Haiku default) the change is
inert and correct, since that model does not truncate this way. Suite is now 23 files with the
wiring test added; it asserts the Executor's effort reaches an Ollama provider as the configured
value, a reasoning agent gets the default, and non-Ollama providers never receive the field, on both
the call and stream paths.

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
  summarizing raw inside the full blocks (per-step hard ceiling already exists at 32k), not more
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
- **Verbose reasoning models truncate; the reasoning-effort ladder is the durable fix (6.11).** A
  reasoning model in any seat can spend its whole output budget on chain-of-thought and emit nothing
  parseable; glm and kimi both did this, glm as Investigator and kimi as Executor. The output cap is
  no longer the lever. All agents now share a 32k budget and a truncation-triggered effort ladder
  (medium, then low, then none, via `call_with_ladder`): a turn that comes back empty or capped is
  retried with the reasoning dial stepped down, and the Executor starts at `none` outright since it
  transcribes a closed spec and gains nothing from a chain-of-thought. The Investigator keeps its own
  retry loop that steps the same ladder and re-sends the decisive `DIRECTIVE_TRUNCATED_RETRY`. Watch
  the logs for repeated "turn was cut off; retrying" notices: a retry that then succeeds is the
  system working, but a step that exhausts the ladder and yields near-empty output is a silent partial
  failure. One cost wrinkle the live compute runs surfaced: glm reliably truncates on the FIRST
  (medium) rung on a hard problem, so each such turn pays a full wasted 32k call before the productive
  retry, roughly half the wall-clock in the worst run and still three wasted calls in a clean one. If
  that recurs, lower this model's starting rung or make the start adaptive within a run once medium is
  observed to truncate. With reasoning stepped off, the residual executor failures are genuine code
  bugs (variable scoping, a wrong constant) whose rate tracks how fiddly the specs are, not analytical
  quality; the lever for a persistent case is a less verbose Executor model, not a higher cap.
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
- **The shared 32k cap is clamped to 20k on the direct Anthropic non-streaming path (6.11).** All
  agents share `DEFAULT_MAX_TOKENS = 32000`. The Anthropic SDK refuses a non-streaming
  `messages.create()` whose `max_tokens` implies a run over a 10-minute timeout ("Streaming is
  required for operations that may take longer than 10 minutes"), so `llm.call` clamps any direct
  `anthropic:` request to `ANTHROPIC_MAX_TOKENS = 20000` automatically. Ollama and OpenRouter take
  the full 32k; an `openrouter:anthropic/...` model goes through the OpenAI-compatible path and is
  not subject to the clamp, since OpenRouter handles the underlying model's own output limit. This
  replaces the earlier regime in which every cap was held at 20k by hand and raising it was
  forbidden. The cap is no longer the lever for a reasoning model that over-thinks; the effort ladder
  (previous item) is. If a larger non-streaming Anthropic output is ever genuinely needed beyond the
  clamp, route that one call through `AnthropicProvider.stream`.
- **Compute mode is fresh-runs-only and gate-free (6.9).** `--compute` disables the G1-G4
  statistical gates (no dataset means no effect, regime, or confound to gate on) and blocks
  `--resume`/`--extend`/`--verify`. If those are wanted in compute mode later, the mode must be
  persisted in the saved state so a resumed run rehydrates the compute prompt bundle rather than
  the data one; nothing else in the loop needs to change. One former token-cap dependency here is
  now resolved: the verify-pipeline agents (`ClaimExtractor`, `Reconciler`) route through
  `call_with_ladder` at the shared 32k, and the same `llm.call` clamp brings any Anthropic call
  among them down to 20k automatically, so they no longer need separate cap handling before
  `--verify` is wired to compute. The compute Synthesizer self-checks (UNCERTAINTY/CONVERGENCE/
  VALIDITY) also mean an audit of a computation has no dataset to recompute against, so how a
  verify pass reasons about a dataset-free run is its own design question, separate from the cap.
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