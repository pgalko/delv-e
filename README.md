# delv-e

Autonomous data investigation driven by large language models.

delv-e takes a dataset and a research question, then runs an iterative analysis
loop until it can answer the question. A premium model does the reasoning and
decides each analytical move. A cheaper model turns each move into Python code
and runs it. A premium model writes the final briefing from the evidence that was
gathered. Every step is recorded, so the whole investigation can be audited.

It also runs without a dataset, in a computation-only mode, for questions answered
by simulation or numerical work rather than by analysis of a file (see
[Computation-only mode](#computation-only-mode)).

## What you get

You give delv-e two things: a data file and a question in plain language. It
returns a written briefing that answers the question, backed by a full trail of
the code it ran, the intermediate results, and the reasoning behind each step.

## How it works

delv-e runs three roles over a shared, long-lived Python kernel.

The **Investigator** is the premium model. Each turn it reads the question, the
data schema, a registry of the objects currently in the kernel, and the evidence
collected so far. It writes its reasoning, decides whether to continue or to
synthesize, produces a closed specification for the next computation, and updates
a pointer ledger that tracks the state of the investigation.

The **Executor** is the cheaper model. It receives the closed specification and
writes pandas code for that one computation, then runs it in the kernel. It
retries on mechanical errors such as a wrong column name.

The **Synthesizer** is a premium model. When the Investigator decides the
question is answered, the Synthesizer reads the raw evidence and writes the
briefing. It can also send the investigation back for more work when the evidence
does not yet support a conclusion.

The kernel also preloads a small **vetted toolkit** of tested estimators
(`paired_ability`, `cluster_bootstrap`, `rank_uncertainty`), so the Investigator
can request a standard method as a single call instead of describing the
algorithm. The Investigator chooses the estimator; the Executor only transcribes
the call; the implementation lives in `toolkit.py` under known-answer tests.

### The persistent kernel

All code runs in one long-lived Python worker process with a single namespace.
The dataset loads once at the start. Objects and functions created in one step
remain available to later steps, so the investigation builds on its own
intermediate work. When a step crashes, the failure is isolated and the namespace
is rebuilt from the most recent checkpoint and a replay of only the steps after it.

### The pointer ledger

The Investigator maintains a structured ledger called the nav state. It holds
handles for the open frontier, candidate regimes, risks, and breakdowns, each
with a status and references to the steps that bear on it. The ledger records
where the investigation stands and which steps support which claims. Conclusions
live in the evidence and the briefing. The ledger stays a compact map of state.

### Guardrails

A small family of rules keeps conclusions honest.

**G1** governs null and uniform findings. Before delv-e reports that an effect is
absent or constant, it has to examine that effect within the levels of at least
one candidate effect modifier. When a relevant axis is still unexamined, the run
is pushed back to stratify first. G1 is code-grounded: a regime counts as examined
if the executed code actually stratified the analysis, not only if the model said
it did.

**G1b** governs varying effects. When an effect changes materially across the
levels of an examined modifier, the briefing leads with the per-level estimate; a
pooled average may appear only as a clearly labeled secondary figure, so a
marginal average cannot mask a within-level effect.

**G2** governs unresolved confounding. When a confounder cannot be removed,
delv-e reports the result as a bound and states the direction of the bias. A
bound stays a bound through to the briefing.

**G3** governs estimand coverage. Before a null or unidentifiable verdict on the
primary question, some analysis must have directly estimated the question that was
asked, matching or stratifying the confound rather than discarding the data, with
the null reconciled against an explicit identifiability statement. The
Investigator names the target estimand on the first step, so the answer cannot
quietly drift to a proxy.

**G4** governs method adequacy. When the answer makes a decisive claim that rests
on an untested assumption and a feasible stronger method could test it, delv-e is
pushed to run that method, with its uncertainty, rather than file it under future
work. It pushes toward a better estimate and never toward "no answer".

G1 and G2 are enforced strictly; G3 and G4 are soft prompt disciplines whose
effect depends on the reasoning model. The number of gate pushbacks allowed before
a verdict is accepted is shared across G1, G3, and G4 and is configurable.

### Long runs and cost

Stable context is cached wherever the provider allows it: Anthropic models use
`cache_control` breakpoints, OpenAI GPT-5.6 models routed through OpenRouter use
the equivalent explicit `prompt_cache_breakpoint` markers, xAI models get a
per-run affinity header, and every OpenRouter request carries a per-run session
id so sticky routing keeps hitting the same cache. Cache reads and cache writes
are both accounted (GPT-5.6 bills writes at 1.25x input; the report shows savings
net of that premium). As evidence accumulates, older steps that no longer bear on
an open question collapse to a pointer, the Investigator can rehydrate any of
them, and the Synthesizer always reads the full raw evidence. Two disciplines
keep long runs from growing without bound: specs are written under a print
budget (decision-sufficient prints rather than full table dumps, floats shown at
four significant digits), and a history character budget
(`HISTORY_CHAR_BUDGET` in `investigation.py`) acts as a backstop that slims,
archives, and trims the oldest material if the rendered history ever exceeds it,
staying byte-identical below the line. A cost report summarizes token usage and
dollar cost per provider, including the savings from caching.

## Install

delv-e needs Python 3.9 or newer.

```
pip install -r requirements.txt
```

Create a `.env` file with the keys for whichever providers you use:

```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
OPEN_ROUTER_API_KEY=...
```

A local Ollama server needs no key.

## Usage

```
python run_core.py DATA_FILE "YOUR QUESTION"
```

To audit a finished run with a fresh, independent second pass (claims are
distilled from its briefing, re-derived under a fixed stress battery, and the
two documents are reconciled into one corrected briefing):

```
python run_core.py DATA_FILE --verify                 # audits the last run
python run_core.py DATA_FILE --verify SOME_RUN_DIR    # audits a specific run
```

Results land in `output_verify/` (override with `--output`). The audit's
`briefing.md` carries each decisive claim with its verification status
(confirmed, attenuated, refuted, or contested); the original and the raw
audit are preserved alongside as `briefing_original.md` and `briefing_audit.md`.
Roughly doubles runtime; worth it when the conclusions matter.

A compute run is audited the same way, with no data file
(`python run_core.py --verify SOME_RUN_DIR`). The audit detects the run's mode
from its saved state and re-derives the computation independently, by a different
method where possible, rather than re-reading a dataset.

Example:

```
python run_core.py datasets/f1_driver_vs_car.csv "Identify the greatest Formula 1 driver of all time. Use teammate pairings as natural experiments to isolate driver skill from car quality, and determine whether one driver is a clear statistical outlier or whether greatness is multidimensional."
```

A benchmark dataset, `datasets/f1_driver_vs_car.csv`, is included so the example
runs as-is.

When you omit the question, delv-e prompts for it. Supported data formats are
csv, tsv, xlsx, parquet, json, and jsonl.

### Options

| Flag | Default | Purpose |
| --- | --- | --- |
| `--iterations N` | 14 | Maximum number of steps. The run stops earlier when the Investigator synthesizes. With `--resume` or `--extend`, this is the number of additional steps. |
| `--investigator-model M` | anthropic:claude-opus-4-8 | Premium model for reasoning and synthesis. |
| `--executor-model M` | anthropic:claude-haiku-4-5-20251001 | Cheaper model for writing and running code. |
| `--synth-model M` | investigator model | Optional separate model for synthesis. |
| `--reasoning-effort LEVEL` | medium | How hard the Investigator and Synthesizer think: `max`, `high`, `medium`, `low`, or `none`. The Executor always runs with reasoning off. Mapped per provider; a direct Anthropic model ignores it. See [Reasoning effort](#reasoning-effort). |
| `--output DIR` | output/ | Where results are written. |
| `--data-dictionary FILE` | none | Markdown file describing columns and caveats, appended to the schema. |
| `--periodic-every N` | 0 | Take a holistic re-derivation snapshot every N steps. 0 turns it off. |
| `--g1-pushback N` | 2 | How many times the synthesis gates (G1, G3, G4) may collectively force more work before a verdict is accepted. |
| `--no-search` | off | Disables mid-stream web search. Search is otherwise auto-seated on the first run model whose provider can search. |
| `--search-budget N` | 3 | Hard cap on web searches per run when search is enabled. |
| `--resume` | off | Continue an interrupted or finished run from the saved state in the output directory. |
| `--extend` | off | Continue a finished run with a new question that can revise the earlier conclusion. Requires a new question. |
| `--verify [DIR]` | off | Audit a finished run with a fresh independent second pass: its decisive claims are re-derived under a fixed stress battery and reconciled into a corrected briefing (originals kept alongside). Bare `--verify` audits the last run; pass a run directory to audit a specific one. Writes to `output_verify/` by default. The audit runs in the same mode as the run it audits, so a compute run is audited dataset-free automatically. Cannot combine with `--resume` or `--extend`. |
| `--compute` | off | Run without a dataset (computation-only mode); the sole positional argument is the question. Can be resumed, extended, and verified. See below. |

### Models and providers

A model is named as `provider:model`. The supported providers are `anthropic`,
`openai`, `openrouter`, and `ollama`. You can mix providers, for example a premium
Anthropic Investigator with a local Ollama Executor:

```
python run_core.py data.csv "..." \
  --investigator-model anthropic:claude-opus-4-8 \
  --executor-model ollama:kimi-k2.6:cloud
```

### Reasoning effort

`--reasoning-effort` sets how hard the premium models think. It applies to the
Investigator and Synthesizer; the Executor always runs with reasoning off, since it
only transcribes a closed specification. The levels are `max`, `high`, `medium`
(the default), `low`, and `none`.

Each level is mapped to the provider's own dial: Ollama takes it unchanged,
OpenRouter's top level is `xhigh` so `max` maps to that, and a direct `anthropic:`
model ignores the flag (it has no effort dial; an `openrouter:anthropic/...` model
still honors it). Reasoning models also read the levels differently. GLM-5.2 is special-cased:
on Ollama the field is omitted entirely, because any explicit value triggers empty
completions on their `/v1` endpoint, so the flag is a no-op there and the model
thinks at its own default; via OpenRouter, `none` is floored to `high` (their
documented enum for GLM).

### Web search

Search is on by default and the Investigator alone decides when to use it. A
search step is an independent call, so it never runs on the run's premium model:
each provider has one designated search model (`SEARCH_MODELS` in `llm.py`), and
the search follows the Investigator's provider.

| Investigator | Search runs on |
| --- | --- |
| OpenRouter model (grok, terra, luna, glm, kimi, ...) | `x-ai/grok-4.3`: xAI's native web and X search, agentic (it chooses how many searches to run), at a fraction of a frontier model's token rate |
| Anthropic model | `claude-haiku-4-5` with the native web_search tool |
| Ollama model, with `OLLAMA_API_KEY` | Ollama's hosted search endpoint, distilled by the run's own model. Free on any ollama account (rate-limited, never billed), so an all-Ollama run stays free |

When the Investigator's own provider cannot serve search (an Ollama seat with no
`OLLAMA_API_KEY`, or a direct OpenAI seat, which has no search route), it falls
through to OpenRouter, then Anthropic, whichever is credentialed. If none is,
search is disabled from the start, the run says so, and the Investigator is never
told search exists. `--no-search` disables it explicitly. `--no-search` disables it, and
`--search-budget` caps how many searches one run may use. Each search is recorded
in the evidence log and in `exploration/NN/search.md`. Search results serve as
calibration context; the analysis itself always comes from the data.

### Computation-only mode

For questions answered by computation rather than by analysis of a file,
`--compute` runs the same loop with no dataset loaded:

```
python run_core.py --compute "Estimate the probability that two fair six-sided dice sum to 9 or more, by Monte Carlo, and cross-check it against the exact value."
```

The roles are unchanged. The Investigator decides each move and writes a closed
specification; the Executor writes and runs the code for that move (a simulation, a
numerical method, or a derivation) against the same persistent kernel; the
Synthesizer writes the briefing from the results. Because there is no dataset, the
data-oriented guardrails do not apply. In their place the Synthesizer checks that
the answer carries its uncertainty (a Monte Carlo standard error or an interval),
that it converged or was cross-checked against a known case, and that its
assumptions and the regime where it holds are stated.

This mode can be resumed, extended, and verified: the run's mode is restored from the
saved state, so you continue or audit a compute run without re-passing `--compute`, and
a `--verify` audit of a compute run re-derives the computation independently. Pass the
question as the single positional argument and omit the data file.

## Output

Everything is written under the output directory.

- `briefing.md` is the final report.
- `log.json` is the full evidence log of every step.
- `nav_state.json` is the pointer ledger.
- `kernel_history.json` is the code history, used to resume.
- `exploration/NN/analysis.md` holds the move, the reasoning, the code, and the output for step NN. Web-search steps write `search.md` there instead.
- `landscape_stepNN.md` holds the periodic snapshots when `--periodic-every` is on.
- `seeds.json` stores the question history for `--resume` and `--extend`; `seed.txt` mirrors the latest question for quick reading.
- `logs/<timestamp>/` holds one folder per run: `run_log.json` (the full API-call log) and `run_telemetry.json` (token usage, dollar cost per provider, cache savings, and per-agent timing, written once at the end of the run).

## Resuming and extending a run

To continue an investigation after an interruption, point `--resume` at the same
output directory:

```
python run_core.py data.csv --resume --iterations 8
```

The kernel is rebuilt from its history, the ledger and evidence are reloaded, and
the new iteration count is added on top of the prior steps. A finished run can be
resumed too; its closing entry is reopened so the investigation continues.

To take a finished investigation further, `--extend` accepts a new question:

```
python run_core.py data.csv "Is the effect explained by era alone?" --extend --iterations 8
```

An extension rehydrates the prior state, pursues the new question alongside the
original one, and writes a single combined briefing that reconciles both lines.
The Synthesizer re-derives everything from the raw evidence, so an extension can
revise the original conclusion when the new work warrants it.

Compute runs (`--compute`) are resumed and extended the same way, without a data
file. The mode is restored from the saved state, so you do not re-pass `--compute`:
`python run_core.py --resume --output DIR` continues a compute run, and
`python run_core.py "new question" --extend --output DIR` extends it.

## Terminal output

delv-e prints a styled progress view while it runs. Library warnings and routine
logs are kept off the console so the view stays readable, and the full record
remains in `run_log.json`. To keep logs and warnings on the console, set
`DELVE_VERBOSE=1`. Styling turns itself off when output is piped to a file, and
`NO_COLOR` disables it as well.

## Project layout

| Module | Role |
| --- | --- |
| `run_core.py` | Command line entry point and run setup. |
| `investigation.py` | The main loop, the Investigator, and the Executor. |
| `synthesis.py` | The Synthesizer and the briefing. |
| `verify.py` | The independent audit pass behind `--verify`: claim extraction, re-derivation under a stress battery, and reconciliation into a corrected briefing. |
| `nav_state.py` | The pointer ledger. |
| `kernel.py` | The persistent Python kernel. |
| `llm.py` | Provider clients, prompt caching, cost tracking, and run telemetry. |
| `prompts.py` | All model-facing prompt text: system prompts, templates, and directives. |
| `executor.py` | Shared code helpers: the security blacklist for generated code, code extraction, and temp-file utilities. |
| `toolkit.py` | Vetted statistical estimators preloaded into the kernel (paired-comparison ability models, cluster bootstrap, rank uncertainty). |
| `dataio.py` | Dataset loading and schema building. |
| `ui.py` | Terminal styling and progress output. |
| `logger_config.py` | Logging setup. |

## License

delv-e is released under the MIT License. See the [`LICENCE`](LICENCE) file for the
full text. Copyright (c) 2025 Pavel Galko.