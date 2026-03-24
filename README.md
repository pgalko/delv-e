# delv-e

Autonomous data investigation powered by LLMs. Give it a dataset and a question — it recursively generates hypotheses, writes and executes analysis code, evaluates results, and adapts its exploration strategy based on what it discovers.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Set your API key(s)
cp .env.example .env
# Edit .env with your API keys

# Run
python run.py data.csv "What factors drive churn?" --iterations 10
```

## How It Works

Before the main loop, an **orientation phase** profiles the dataset's analytical landscape — column coverage, group sizes, confounders, power boundaries, derivable variables, and sparse-column artifacts. This produces a compact brief pinned into every agent's context for the entire run.

Each iteration:

1. **Generate questions** — LLM proposes analytical questions guided by the research model. During PURSUING, all questions target a single finding's next maturity stage
2. **Write & execute code** — code model writes Python, runs it against your DataFrame
3. **Evaluate results** — LLM scores parallel solutions, selects the best, and recommends the next phase. Considers both score trajectory and finding maturity
4. **Update research model** — living document of hypotheses, findings, maturity tracking, cross-finding connections, and exploration health
5. **Transition phase** — system follows the evaluator's recommendation; thread completion is the only structural override

Periodically, a **connection explorer** generates questions testing interactions between established findings — compounding effects, mediating relationships, and conditional dependencies.

### Phase System

Two phases with model-driven transitions. The evaluator recommends a phase after every iteration based on the full exploration context, research model, and finding maturity state.

| Phase | Mode | When the evaluator recommends it |
|---|---|---|
| **MAPPING** | Broad survey, screening | Recent analyses concentrated on same topic, large unexplored territory, thread concluded, or exploration is early-stage |
| **PURSUING** | Deep dive, validation | Latest result opened genuinely new territory, finding needs verification, or an active finding hasn't reached DECOMPOSED maturity |

During PURSUING, all parallel question slots target the same finding — the system doesn't split attention across topics until it switches back to MAPPING.

### Finding Maturity

Significant findings (score 7+) are tracked through an analytical arc:

| Stage | What it means | Next step |
|---|---|---|
| **DETECTED** | Signal found, direction known | Quantify: rate, magnitude, significance |
| **QUANTIFIED** | Effect size precisely established | Decompose: subgroups, percentiles, time periods |
| **DECOMPOSED** | Distribution characterised | Regime-test: structural breaks, rolling windows |
| **REGIME-TESTED** | Temporal stability checked | Connect: test interactions with other findings |
| **COMPLETE** | Operationally interpretable | Graduate; eligible for cross-finding connections |

The maturity tracker prevents premature abandonment — the evaluator keeps the system in PURSUING until the active finding reaches at least DECOMPOSED. A finding that gets contradicted at any stage is dropped rather than forced through remaining stages.

### Cross-Finding Connections

After enough established findings accumulate (≥4), a dedicated connection explorer periodically generates questions testing whether independent findings interact:

- **Compounding** — do they amplify each other?
- **Mediating** — does one explain the other?
- **Conditional** — does one modify the other?
- **Contradicting** — do they point in opposite directions?

Connection results are tracked in the research model and graduated to established findings when confirmed. The question generator also discovers connections organically during PURSUING — the dedicated explorer provides systematic coverage on top of that.

## Usage

```
python run.py <dataset> ["<question>"] [options]
```

| Option | Default | Description |
|---|---|---|
| `--iterations N` | 5 | Exploration iterations |
| `--parallel N` | 2 | Parallel solutions per iteration |
| `--output DIR` | output/ | Output directory |
| `--continue` | | Resume from previous run's checkpoint |
| `--no-orientation` | | Skip the orientation phase (data profiling) |
| `--agent-model` | anthropic:claude-haiku-4-5-20251001 | Model for agents (evaluator, QG, RI, selector) |
| `--code-model` | anthropic:claude-haiku-4-5-20251001 | Model for code generation |
| `--premium-model` | same as code-model | Model for orientation, connections, and synthesis |

### Providers

Model format: `provider:model_name`

| Provider | Example | Requires |
|---|---|---|
| Anthropic | `anthropic:claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai:gpt-5.4` | `OPENAI_API_KEY` |
| OpenRouter | `openrouter:moonshotai/kimi-k2.5` | `OPEN_ROUTER_API_KEY` |
| Ollama | `ollama:qwen3:30b` | Local Ollama server |

OpenRouter provides access to hundreds of models (DeepSeek, Qwen, Kimi, GLM, Gemini, Llama, etc.) via a single API key. See [openrouter.ai/models](https://openrouter.ai/models) for available models and pricing.

### Examples

```bash
# All Haiku (cheapest cloud option)
python run.py data.csv "Analyze trends" --iterations 15

# Quick 3-iteration run — skip orientation
python run.py data.csv "What's the class balance?" --iterations 3 --no-orientation

# OSS models via OpenRouter
python run.py data.csv "What predicts price?" \
    --agent-model openrouter:moonshotai/kimi-k2.5 \
    --code-model openrouter:moonshotai/kimi-k2.5

# Mix providers — OSS agents, Anthropic code
python run.py data.csv "Deep analysis" \
    --agent-model openrouter:z-ai/glm-5 \
    --code-model anthropic:claude-haiku-4-5-20251001

# Local Ollama
python run.py data.csv "Quick look" \
    --agent-model ollama:qwen3:30b \
    --code-model ollama:qwen3:30b

# Cheap run with premium orientation + synthesis — run locally, use Opus for bookends
python run.py data.csv "Deep analysis" \
    --agent-model ollama:qwen3:30b \
    --code-model ollama:qwen3:30b \
    --premium-model anthropic:claude-opus-4-6
```

## Resuming Runs

Checkpoint saved after every iteration. Resume with `--continue`:

```bash
# Initial run
python run.py shoes.csv "Analyze shoe efficiency" --iterations 25

# Continue with a new direction (iterations are additive)
python run.py shoes.csv "Pursue the cardiovascular paradox" --continue --iterations 30
```

The seed question on `--continue` becomes the first analysis in the resumed run. The DataFrame, research model, insight tree, phase history, and all context are preserved. You can switch models between runs.

## Output

```
output/
├── dashboard.html           # Live dashboard — open in browser, auto-refreshes
├── synthesis_report.md      # Final report with citations
├── research_model.md        # Hypotheses, findings, maturity, connections, gaps
├── run_log.json             # Full log of every LLM call
├── state.json               # Checkpoint for --continue
├── dataframe.parquet        # Preserved DataFrame
├── cost.txt                 # Cost breakdown
├── orientation/
│   └── analysis.md          # Dataset analytical profile
└── exploration/
    ├── 01_MAPPING/
    │   ├── _summary.md      # Iteration evaluation + phase decision
    │   └── 1773198695/
    │       ├── analysis.md  # Question + code + output
    │       └── plot_001.png
    ├── 02_PURSUING/
    └── ...
```

## Memory Architecture

LLMs have no memory between calls. delv-e manages context through five layers:

**Data Profile** — produced by the orientation phase before iteration 1. A compact analytical brief (~500-1000 tokens) covering column coverage, group sizes, confounders, power boundaries, derivable variables, and sparse-column artifacts. The orientation is aware of coverage-driven correlation artifacts (e.g., two sparse columns appearing correlated because they're both only recorded during the same operational period). Pinned into every agent's context for the entire run.

**Insight Tree** — every analysis is a node with question, results, score, and summaries. Agents see a tiered view: recent entries with RI-curated key numbers (result_digest), older entries compressed to one-sentence summaries (finding_summary from the evaluator). Nothing is deleted — the system manages visibility, not existence.

**Research Model** — a structured document updated after every iteration and read by every agent:
- *Active Hypotheses* (max 4) — testable claims the next analysis could change
- *Established Findings* (max 10) — confirmed facts with quantitative anchors
- *Finding Maturity* (max 5) — significant findings tracked through DETECTED → QUANTIFIED → DECOMPOSED → REGIME-TESTED → COMPLETE, each with a specific next analytical step
- *Threads* (max 3) — active lines of inquiry with open questions
- *Cross-Finding Connections* (max 5) — tested and untested interactions between findings
- *Attention Flags* — findings where later analyses produced contradictory results
- *Biggest Gap* — the most important thing not yet investigated (flags when stuck)
- *Exploration Health* — honest self-assessment of breadth, recent topic concentration, and unexplored territory. This section drives strategic direction: when the RI reports low breadth, the evaluator recommends MAPPING, and the question generator pivots to new territory

**Q&A Pairs** — the Code Generator sees the 20 most recent question-result pairs plus the dataset schema. A deliberate sliding window — the code writer needs tactical context, not the full exploration history.

**Full Results Store** — untruncated results from every analysis, never shown to agents during exploration. Used only by the Synthesis Generator, which selects up to 40 analyses via score-weighted selection (top-scoring from the entire run + most recent 15 for continuity). Both orientation and synthesis can optionally use a stronger model via `--premium-model`. The connection explorer also uses the premium model when set. These three are the highest-leverage calls in the run — orientation sets the analytical brief, connections discover compound effects between findings, and synthesis produces the final report. In a 100-iteration run they account for ~14 calls total while the remaining ~635 use the cheaper models.

### Context Management

The system uses two schema modes: a full schema for the Code Generator, and a slim schema (column names, types, and unique counts only) for all other agents. For datasets with more than 50 columns, `head()` and `describe()` are omitted from the full schema. This reduces code generator input by up to 70% on wide datasets.

The evaluator generates one-sentence summaries for all parallel solutions (not just the winner), giving every node in the tree an LLM-curated finding_summary. The RI generates a 3-5 line result_digest of key numbers for winning nodes only. Non-winning nodes marked dormant get hypothesis labels combining the question and finding summary, which direct the QG on branch switches.

## Cost

| Configuration | ~Cost per 10 iterations |
|---|---|
| All Haiku | $0.50–1.50 |
| Haiku agents + Opus code | $2–4 |
| OpenRouter OSS (kimi/glm) | $0.50–1.00 |
| All Ollama (local) | Free |
| Ollama + Opus premium | ~$0.60 (orientation + connections + synthesis) |

Check `output/cost.txt` after each run for exact breakdown by agent.

## Architecture

```
run.py               CLI — dataset loading, --continue handling, --premium-model override
engine.py            ExplorationEngine — runtime, code execution, orientation, file output
auto_explore.py      Core loop — phases, maturity tracking, connections, research model
dashboard.py         Live HTML dashboard — written after each iteration, auto-refreshes
llm.py               Multi-provider LLM client (Anthropic, OpenAI, OpenRouter, Ollama)
executor.py          Local code execution with security guards
prompts.py           All prompt templates (agents, code generation, orientation, connections)
style.py             Terminal formatting
output.py            Print routing
logger_config.py     Logging configuration
```

## Security

Generated code runs locally via `exec()`. A module blacklist blocks dangerous operations (subprocess, socket, file deletion, network access) but this is **not a sandbox**. See `executor.py` for the full blacklist. API keys are read from environment variables only, never logged or stored.

## Origin

Standalone extraction of the auto-explore module from [BambooAI](https://github.com/pgalko/BambooAI). Core exploration logic preserved; web UI, database, billing, and multi-tenant routing replaced with minimal local equivalents.