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

Before the main loop, an **orientation phase** profiles the dataset's analytical landscape — group sizes, confounders, power boundaries, and variable structure. This produces a compact brief that is pinned into every agent's context for the entire run, preventing wasted iterations on underpowered comparisons or rediscovering confounds.

Each iteration:

1. **Generate questions** — LLM proposes analytical questions guided by the research model's Exploration Health section, which tracks what's been explored, what's unexplored, and whether breadth is adequate
2. **Write & execute code** — code model writes Python, runs it against your DataFrame
3. **Evaluate results** — LLM scores parallel solutions, selects the best, and recommends the next exploration phase (MAPPING or PURSUING) based on the full trajectory
4. **Update research model** — living document of hypotheses, findings, open gaps, and an honest assessment of exploration breadth
5. **Transition phase** — system follows the evaluator's phase recommendation; the only structural override is thread completion, which triggers MAPPING

### Phase System

delv-e uses two phases with model-driven transitions. The evaluator recommends a phase after every iteration based on the full exploration context, research model, and Exploration Health assessment.

| Phase | Mode | When the evaluator recommends it |
|---|---|---|
| **MAPPING** | Broad survey, screening | Recent analyses concentrated on same topic, large unexplored territory, thread concluded, or exploration is early-stage |
| **PURSUING** | Deep dive, validation | Latest result opened genuinely new territory, finding needs verification or robustness testing |

The system's purpose is **discovery** — surveying a dataset's full landscape to find what is interesting. The evaluator is instructed that breadth is more valuable than depth: a run covering 8 topics at moderate depth is better than one covering 2 topics exhaustively. Phase transitions have no hardcoded rules — the models see the exploration state and decide.

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
| `--code-model` | anthropic:claude-haiku-4-5-20251001 | Model for code generation and synthesis |

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
├── synthesis_report.md      # Final report with citations
├── research_model.md        # Hypotheses, findings, gaps
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

**Data Profile** — produced by the orientation phase before iteration 1. A compact analytical brief (~500-1000 tokens) covering group sizes, confounders, power boundaries, and variable structure. Pinned into every agent's context for the entire run — never truncated or compacted. This is how the system knows "LumA-chemo has only n=26" at iteration 50 without having to rediscover it.

**Insight Tree** — every analysis is a node with question, results, score, and summaries. Agents see a tiered view: recent entries with RI-curated key numbers (result_digest), older entries compressed to one-sentence summaries (finding_summary from the evaluator). Nothing is deleted — the system manages visibility, not existence.

**Research Model** — a structured document with six sections, updated after every iteration and read by every agent:
- *Active Hypotheses* (max 4) — testable claims the next analysis could change
- *Established Findings* (max 6) — confirmed facts with quantitative anchors
- *Threads* (max 3) — active lines of inquiry with open questions
- *Biggest Gap* — the most important thing not yet investigated
- *Exploration Health* — honest self-assessment of breadth, recent topic concentration, and unexplored territory. This section drives the exploration's strategic direction: when the RI reports low breadth, the evaluator recommends MAPPING, and the question generator pivots to new territory. Breadth is assessed based on *recent concentration* (what the last 5-8 analyses focused on), not total topic count.
- *Narrative* — 2-3 sentences connecting the latest result to prior understanding

**Q&A Pairs** — the Code Generator sees the 20 most recent question-result pairs plus the dataset schema. A deliberate sliding window — the code writer needs tactical context, not the full exploration history.

**Full Results Store** — untruncated results from every analysis, never shown to agents during exploration. Used only by the Synthesis Generator, which selects up to 40 analyses via score-weighted selection (top-scoring from the entire run + most recent 15 for continuity).

### Context Management

The system uses two schema modes: a full schema for the Code Generator, and a slim schema (column names, types, and unique counts only) for all other agents. For datasets with more than 50 columns, `head()` and `describe()` are omitted from the full schema — these become unreadable noise at high column counts and the column-level metadata (types, ranges, sample values) provides what the code model needs. This reduces code generator input by up to 70% on wide datasets.

The evaluator generates one-sentence summaries for all parallel solutions (not just the winner), giving every node in the tree an LLM-curated finding_summary. The RI generates a 3-5 line result_digest of key numbers for winning nodes only. Non-winning nodes that are marked dormant get hypothesis labels combining the question and finding summary, which direct the QG on branch switches.

## Cost

| Configuration | ~Cost per 10 iterations |
|---|---|
| All Haiku | $0.50–1.50 |
| Haiku agents + Opus code | $2–4 |
| OpenRouter OSS (kimi/glm) | $0.50–1.00 |
| All Ollama (local) | Free |

Check `output/cost.txt` after each run for exact breakdown by agent.

## Architecture

```
run.py               CLI — dataset loading, --continue handling
engine.py            ExplorationEngine — runtime, code execution, orientation, file output
auto_explore.py      Core loop — model-driven phases, research model, insight tree
llm.py               Multi-provider LLM client (Anthropic, OpenAI, OpenRouter, Ollama)
executor.py          Local code execution with security guards
prompts.py           All prompt templates
style.py             Terminal formatting
output.py            Print routing
logger_config.py     Logging configuration
```

## Security

Generated code runs locally via `exec()`. A module blacklist blocks dangerous operations (subprocess, socket, file deletion, network access) but this is **not a sandbox**. See `executor.py` for the full blacklist. API keys are read from environment variables only, never logged or stored.

## Origin

Standalone extraction of the auto-explore module from [BambooAI](https://github.com/pgalko/BambooAI). Core exploration logic preserved; web UI, database, billing, and multi-tenant routing replaced with minimal local equivalents.