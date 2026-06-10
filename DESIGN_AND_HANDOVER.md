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

**Canonical code.** The system is 11 Python modules plus a `tests/` directory. The shipped,
authoritative archive is `delv-e.zip` (the complete project: the 11 modules, `tests/`, the
docs, and the repo scaffolding). The live working copy is `/home/claude/delve/delv-e/`. The
rebuilt system is exactly these 11 files: `kernel.py`, `investigation.py`, `nav_state.py`,
`synthesis.py`, `dataio.py`, `executor.py`, `llm.py`, `prompts.py`, `run_core.py`, `ui.py`,
`logger_config.py`. (The leftover original-delv-e files that used to sit in the working
directory, `auto_explore.py`, `engine.py`, `dashboard.py`, and the rest, were removed in the
pre-release cleanup; see 6.1. The live graph is now exactly these 11 modules.)

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
`test_truncation`, `test_ledger_parse`, `test_status`, `test_estimand`, `test_telemetry`. All fifteen pass as of this handover.
The three dataset-backed tests (`test_continue`, `test_extend`, `test_search`) read their dataset from `datasets/` (or a path given via an environment variable) and skip cleanly if it is absent, so a fresh
clone gets nine run plus three skipped, all green. See `tests/README.md`.

**The ship workflow (used every change).** Edit source in `/home/claude/delve/delv-e/` → run
the suite → copy changed files to `/mnt/user-data/outputs/` → rebuild `delv-e.zip` from a clean
staging tree excluding `__pycache__` and gitignored artifacts → verify it imports from a fresh
extract → `present_files`. Be honest about faithfulness versus fabrication in all work.

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
  namespace, so `df` is loaded once and derived objects persist across steps. Crashes are
  isolated and the namespace is reconstructed by replaying history.
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
| `kernel.py` | `PersistentKernel`: the long-lived worker process, one namespace, crash isolation, history replay/restore, and the namespace registry (`describe_namespace`). |
| `executor.py` | Stateless code-handling helpers reused by the kernel and loop: the security `BLACKLIST` for generated code, code-fence extraction (`extract_code`), DataFrame serialization, and temp-file management. It no longer executes code; live execution is owned by `PersistentKernel` (the old out-of-process `CodeExecutor` and its runner machinery were removed in the 6.1 cleanup). |
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

- glm-5.1 brain, kimi-k2.6 executor: about 70. Correct headline (no outlier, multidimensional,
  Verstappen the modern standout) and good confound awareness, but it leaned on z-scored composites
  plus a Grubbs outlier test rather than a paired-comparison model, left the early-era multi-car
  pseudo-replication uncorrected, and ranked Clark over Fangio while not surfacing Senna.
- Opus 4.8 brain, same executor: about 76. Same headline, plus a genuine qualitative
  opposition-strength insight (Verstappen's raw teammate margin was earned against average
  teammates, Hamilton's against the strongest slate of any top driver), which is conceptually what
  the benchmark's model does formally. But it filed the formal model under "future work" and
  finalized on the proxy, and it over-corrected toward Alonso on small-sample duels.
- Opus 4.8 brain with the method-adequacy clauses (G4 and the Investigator's method-adequacy
  guidance, section 6.1): about 82. It built a connected Bradley-Terry network over teammate
  qualifying and ran a 300-replicate bootstrap producing P(Verstappen #1) about 0.68 and a thin
  top-gap interval, matching the benchmark's instrument and its 64% figure while keeping calibrated
  uncertainty. This is the closest any run has come.

- glm-5.1 brain with the same clauses, same executor: about 72 to 73 across two runs. Neither run
  changed the decisive method: no paired-comparison model and no computed uncertainty (no interval,
  no P(#1)) in either, and the synthesis self-gate fired in neither (`synth_pushbacks` 0 both
  times). The clauses did push glm to attempt more method over the two runs (the first added a
  coverage/era split and a cross-tier check; the second added PCA, within-era normalization, a
  teammate-quality regression, and an empirical-Bayes shrinkage check, which is the right instinct
  for small samples), but it never landed on the estimator the structure calls for, and the
  uncertainty G4 asks for was absent both times. The second run also let weak-teammate artifacts
  into the headline ranking (Mika Salo #2, Jonathan Palmer #7) and over-corrected the classic greats
  downward (Clark to #10), and Senna was not surfaced in either. Both gains sit within run-to-run
  variance of the 70.

Read together, these four points locate the lift to 82 in the interaction between the clauses and a
capable brain. The same clauses moved Opus from 76 to 82 by pushing it to build the formal model it
had previously deferred, but moved glm only into the low 70s across two runs, because glm did not act on the
method-adequacy nudge where it counts: it attempted more elaborate method each time but never the
decisive estimator, and produced no uncertainty in either run. The clauses are a soft self-gate, so
their effect scales with the reasoning model's ability to act on them; for a weaker brain the
binding constraint looks like reasoning and execution capability rather than the prompt itself. Two
glm runs showing the same ceiling make that read firmer than one would, though two is still a small
sample; the implication is that the next lever for weaker brains is a vetted methods toolkit for the
executor rather than more prompt text. The residual gaps that keep even the best run below the high 80s are structural: the
Bradley-Terry network was built on qualifying only, so the classic greats cannot enter the headline
dimension; the early-era multi-car group-weighting is sidestepped rather than solved; the
Indianapolis 500 is not excluded; and the cross-era finish dimension stays reliability-contaminated.
Those are a smaller, more specific class of gap than the earlier "never reached for the method."

The ONE MOVE PER SPEC rule (section 6.1) was then run on both brains. glm scored about 73 again, but
the spec problem it targeted was fixed: 14 single-move steps, zero multi-part (down from five of
seven), mean spec length 223 words, retries down to two with no failed step. glm also computed
uncertainty for the first time, yet still deferred the decisive method (naming an Elo-style network
and a hierarchical model while finalizing on a linear proxy) and regressed to a single-outlier
headline (Clark at three sigma, which the benchmark contradicts). Opus under the same rule scored
about 82, on par with its pre-rule run: it decomposed even finer (13 single-move steps, mean 191
words), modeled both qualifying and finishing with CI-overlap analysis, and resolved the reliability
confound the 82 run had left open. Under the identical rule the two runs' spec mechanics converge
while their findings stay nine points apart, which isolates the residual gap as brain capability
rather than spec discipline (section 6.1). Decomposition fixed reliability without lifting the
analytical ceiling, pointing the next build at a vetted methods toolkit for the executor.

---

## 6. Recent fixes and enhancements

This section is the most useful for picking up where we left off. Items are newest first.

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

**Executor output cap raised 16k to 20k (shipped; effect pending a real run).** kimi-k2.6 as the
Executor truncates often at the 16k cap, and in the search-confirmed run it exhausted retries on a
step that then produced almost nothing. The Executor's default `max_tokens` was raised to 20,000
for headroom (`Executor.__init__`, investigation.py; the Investigator cap stays at 16k). The suite
is green, but whether 20k materially reduces kimi truncation is not yet confirmed on a live run;
the more durable fix may be a less verbose Executor model. Tracked in section 7.

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

**Caveat that matters.** glm still hits the 16,000 cap. It truncated twice on the Investigator
and twice on the Executor (the Executor recovered via its existing mechanical retry). So 16k did
not eliminate glm's truncation; the retry is load-bearing, not a rare backstop, because glm is
a heavy enough thinker to blow even the raised budget, and a second attempt happens to finish.
If a model truncates persistently even with retries, the budget may need to go higher again.

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
  summarizing raw inside the full blocks (per-step hard ceiling already exists at 20k), not more
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
  together with its uncertainty. The evidence on this lever is now fairly clear.
  With a capable brain (Opus) the clauses work: it builds the deciding model and quantifies
  uncertainty rather than deferring, which lifted it from the mid-70s to the low-80s. With a weaker
  brain (glm) they do not: across several runs glm names the deciding model and finalizes on a proxy
  anyway, and the soft gate does not catch it. The ONE MOVE PER SPEC rule (section 6.1) fixes the
  spec-decomposition half of the problem for both brains while leaving the analytical half untouched;
  a controlled run of that rule on both brains converged the mechanics while leaving an Opus-versus-glm
  gap of nine points, locating the residual in brain capability. The honest caveats: G3 has not been
  exercised on a real null since it shipped, the gates are soft self-reports rather than mechanical
  checks, and the method-adequacy work needs more reruns and another dataset before it is trusted.
  The indicated next lever for weak brains is a vetted methods toolkit for the executor.
- **Verbose executors truncate; the cap is now 20k for the Executor.** The truncation retry is
  load-bearing for heavy models, not a rare backstop. glm-as-Executor blew the old 16k cap; so does
  kimi-k2.6, which truncated three times in a row on a single step in the search-confirmed run and
  left that step with only 68 characters of output (the backstop prevented a crash, but the step
  did not accomplish its analysis). The Executor output cap was raised from 16k to 20k (6.1) to give
  these models more headroom; whether that materially reduces kimi truncation is pending the next
  real run. The Investigator cap stays at 16k (it has not truncated since the ledger fix). Watch the
  logs for repeated "turn was cut off; retrying" notices: a retry that then succeeds is the system
  working, but a step that exhausts retries and yields near-empty output is a silent partial failure,
  and the lever there is a less verbose Executor model rather than a still-higher cap.
- **The G1 code backstop is intentionally permissive.** It treats any stratifying construct as
  evidence of within-regime examination, so a trivial `groupby('x').size()` would satisfy it. It
  only ever upgrades an unmarked ledger, and the Synthesizer still re-derives over raw evidence,
  so the downside is bounded, but be aware it can let a thinly-stratified synthesis proceed.
- **Run count before ranking.** Per-model score ranges overlap and variance is 7 to 10 points;
  do not rank models on fewer than three runs each.

---

## Appendix A: Path map

| What | Where |
|---|---|
| Live source (the 11 rebuilt modules + `tests/`) | `/home/claude/delve/delv-e/` |
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
| `dataset` (positional) | required | Path to `.csv/.tsv/.xlsx/.parquet/.json/.jsonl`. |
| `question` (positional) | none | Seed question. |
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