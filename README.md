# delv-e

Autonomous data investigation driven by large language models.

delv-e takes a dataset and a research question, then runs an iterative analysis
loop until it can answer the question. A premium model does the reasoning and
decides each analytical move. A cheaper model turns each move into Python code
and runs it. A premium model writes the final briefing from the evidence that was
gathered. Every step is recorded, so the whole investigation can be audited.

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

### The persistent kernel

All code runs in one long-lived Python worker process with a single namespace.
The dataset loads once at the start. Objects created in one step remain available
to later steps, so the investigation builds on its own intermediate work. When a
step crashes, the failure is isolated and the namespace is rebuilt by replaying
the prior code.

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

The premium model uses prompt caching so that stable context is not paid for
again on every turn. As evidence accumulates, older steps that no longer bear on
an open question collapse to a pointer, which keeps the working context small and
focused. The Investigator can rehydrate any collapsed step when it needs the
detail again. The Synthesizer always reads the full raw evidence. A cost report
summarizes token usage and dollar cost per provider, including the savings from
caching.

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
| `--iterations N` | 14 | Maximum number of steps. The run stops earlier when the Investigator synthesizes. With `--continue`, this is the number of additional steps. |
| `--investigator-model M` | anthropic:claude-opus-4-8 | Premium model for reasoning and synthesis. |
| `--executor-model M` | anthropic:claude-haiku-4-5-20251001 | Cheaper model for writing and running code. |
| `--synth-model M` | investigator model | Optional separate model for synthesis. |
| `--output DIR` | output/ | Where results are written. |
| `--data-dictionary FILE` | none | Markdown file describing columns and caveats, appended to the schema. |
| `--periodic-every N` | 0 | Take a holistic re-derivation snapshot every N steps. 0 turns it off. |
| `--g1-pushback N` | 2 | How many times the synthesis gates (G1, G3, G4) may collectively force more work before a verdict is accepted. |
| `--continue` | off | Resume from saved state in the output directory. Iterations are additive. |

### Models and providers

A model is named as `provider:model`. The supported providers are `anthropic`,
`openai`, `openrouter`, and `ollama`. You can mix providers, for example a premium
Anthropic Investigator with a local Ollama Executor:

```
python run_core.py data.csv "..." \
  --investigator-model anthropic:claude-opus-4-8 \
  --executor-model ollama:kimi-k2.6:cloud
```

## Output

Everything is written under the output directory.

- `briefing.md` is the final report.
- `log.json` is the full evidence log of every step.
- `nav_state.json` is the pointer ledger.
- `kernel_history.json` is the code history, used to resume.
- `exploration/NN/analysis.md` holds the move, the reasoning, the code, and the output for step NN.
- `seed.txt` stores the question for resuming.
- `logs/<timestamp>/` holds one folder per run: `run_log.json` (the full API-call log) and `run_telemetry.json` (token usage, dollar cost per provider, cache savings, and per-agent timing, written once at the end of the run).

## Resuming a run

To continue an investigation, point `--continue` at the same output directory:

```
python run_core.py data.csv --continue --iterations 8
```

The kernel is rebuilt from its history, the ledger and evidence are reloaded, and
the new iteration count is added on top.

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
| `nav_state.py` | The pointer ledger. |
| `kernel.py` | The persistent Python kernel. |
| `llm.py` | Provider clients, prompt caching, and cost tracking. |
| `executor.py` | Shared helpers for code extraction, plot capture, and kernel support. |
| `dataio.py` | Dataset loading and schema building. |
| `ui.py` | Terminal styling and progress output. |
| `logger_config.py` | Logging setup. |

## License

delv-e is released under the MIT License. See the [`LICENCE`](LICENCE) file for the
full text. Copyright (c) 2025 Pavel Galko.