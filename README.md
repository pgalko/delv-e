# delv-e

Autonomous data investigation powered by LLMs, designed as a **pre-filter for deeper analytical work**. Give it a dataset and a question — or just a question — and it recursively generates hypotheses, writes and executes analysis code, and accumulates a structured record of what is and is not answerable with the available evidence. The output is a handoff briefing: a structural map of the investigation's terrain, findings tagged by confidence, directions that have been foreclosed with evidence, and questions that remain open. The briefing is intended as input to a subsequent phase of investigation (human or AI), not as a final publication.

Works with datasets (CSV, Excel, Parquet) or in computation-only mode for simulations, mathematical exploration, and numerical experiments.

The system implements a three-rung division of labor inspired by the [Knuth/Stappers/Claude collaboration](https://www-cs-faculty.stanford.edu/~knuth/papers/claude-cycles.pdf) and [analysed by Vishal Misra](https://medium.com/@vishalmisra/knuth-just-showed-us-where-to-put-the-human-013c0330ef0a) through the lens of Pearl's Causal Hierarchy. Cheap, fast models handle high-throughput pattern matching. A premium model provides strategic oversight every iteration. A final premium-model pass renders the accumulated state into a handoff briefing. The role the system does *not* attempt to play is the Rung-3 counterfactual reasoning that would convert "provisional finding" into "proven construction" — that work remains for whoever picks up the briefing.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Set your API key(s)
cp .env.example .env
# Edit .env with your API keys

# Run with a dataset (OSS agents with premium strategic oversight)
python run.py data.csv "What factors drive churn?" \
    --agent-model openrouter:moonshotai/kimi-k2.5 \
    --code-model openrouter:moonshotai/kimi-k2.5 \
    --premium-model anthropic:claude-opus-4-6 \
    --iterations 50

# Run with a data dictionary (strongly recommended for domain-specific data)
python run.py data.csv "What factors drive churn?" \
    --data-dictionary data_dictionary.md \
    --premium-model anthropic:claude-opus-4-6 \
    --iterations 50

# Run without a dataset (computation-only mode)
python run.py "Simulate the evolution of cooperation using iterated Prisoner's Dilemma" \
    --agent-model openrouter:moonshotai/kimi-k2.5 \
    --code-model openrouter:moonshotai/kimi-k2.5 \
    --premium-model anthropic:claude-opus-4-6 \
    --iterations 50
```

## What delv-e is (and isn't) for

**delv-e is a hypothesis-search pre-filter.** It explores quickly, tries many angles, and accumulates a structured map of the analytical terrain: what's identifiable with the available evidence, what's blocked and by what mechanism, what's been foreclosed with diagnostics, what remains open. The handoff briefing is meant to be read by a downstream investigator who will take the work further.

**delv-e is not a deliverable.** The briefing is not a report to publish. It is a structured hand-off that saves a downstream investigator from re-discovering foreclosed directions, lets them see where identifiability is blocked and why, and points them at the highest-value entry points. The investigator is expected to bring domain knowledge, theoretical framing, or additional data that delv-e does not have.

A concrete shape of what this looks like in practice: on a multi-site dataset with per-site metadata, delv-e might discover that two variables of interest (say, a measurement-regime parameter and a site-level categorical attribute) are 1:1 confounded — each site has exactly one value of the attribute. It would foreclose several approaches that look natural (controlling for the attribute in a pooled regression, stratifying analyses by attribute) with specific diagnostics (VIF, coefficient reversals under fixed effects), and identify the one or two identification strategies that remain tractable given the data structure. It would not solve the underlying scientific question — that requires domain knowledge delv-e has no channel to receive, or data the run did not have access to. What the briefing does is save the next investigator from re-discovering the block, hand them the entry points, and make their first hour productive instead of exploratory.

## Design Philosophy

When Claude Opus 4.6 solved an open combinatorics problem for Donald Knuth in March 2026, it didn't work alone. Filip Stappers coached the model through 31 explorations, forcing it to document progress, redirecting unproductive strategies, and maintaining the arc of the investigation across context losses. Knuth then proved why the construction works. Three participants, three distinct roles.

Vishal Misra framed this through Pearl's Causal Hierarchy. Claude's contribution was **Rung 1**: extraordinary associative pattern matching within each exploration. Stappers' contribution was **Rung 2**: intervention, changing tools, redirecting strategy, maintaining coherence across context losses. Knuth's was **Rung 3**: counterfactual reasoning that proves the construction works for values never computed.

delv-e implements Rungs 1 and 2 as an autonomous system and **deliberately stops short of Rung 3**, producing a briefing that a Rung-3 investigator can pick up:

**Rung 1: cheap models at speed.** Code generation, question generation, result evaluation, and research-model interpretation are handled by fast, inexpensive models doing what LLMs do best: pattern matching, code writing, structured summarization. They run 4-5 calls per iteration.

**Rung 2: premium model as strategic overseer.** A stronger model runs a strategic review every iteration. It reads the research model and the data profile, decides whether to hold commitment on the current arc, pivot to a new direction, or abandon an exhausted line. When it pivots, it names the specific next direction as a binding constraint that the question generator must follow. The premium model also maintains two structurally-protected sections of the research model that the cheap models cannot touch: the Strategic Trajectory (why pivots happened, what's committed) and the Structural Landscape (what's identifiable, what's blocked, what's foreclosed, what remains open). These sections accumulate across the run and become the primary source material for the final handoff briefing. This is the Stappers role: strategic coherence with override authority and a memory of what's been decided.

**Rung 2, final pass: briefing rendering.** When exploration ends, the premium model renders the accumulated research model into a handoff briefing with a fixed seven-section structure: scope, structural landscape, findings with STATUS tags, foreclosed directions, open questions, suggested entry points, methodological notes. This is *rendering*, not re-derivation — the structural content has been accumulated across the run, and the briefing's job is to present it cleanly to the downstream investigator rather than synthesize it from scratch.

**Rung 3 is deliberately absent.** The briefing does not attempt to prove, generalize beyond what was tested, or bring domain knowledge the system has no channel to receive. That work is the downstream investigator's job. The briefing is structured to make their first hour productive.

## How It Works

Before the main loop, an **orientation phase** (premium model) profiles the dataset's analytical landscape — column coverage, group sizes, confounders, power boundaries. Critically, orientation also produces an initial **Structural Landscape** block: identifiability constraints, coverage asymmetries, and any directions the structural diagnostics already foreclose. This seeds the research model's Structural Landscape section, which strategic review will extend as structural discoveries accumulate during exploration.

When a data dictionary is provided via `--data-dictionary`, orientation reads it as authoritative context and emits a KEY CONSTRAINTS block in the profile (see [Data Dictionary](#data-dictionary)).

In computation-only mode, orientation is skipped — the system begins directly with seed decomposition.

A **seed decomposition** step (premium model) converts the user's research agenda into a focused first analysis and an initial Strategic Trajectory.

Each iteration:

1. **Generate questions**: Cheap model proposes analytical questions guided by the research model. Questions that would re-try foreclosed directions are explicitly avoided; questions that would resolve open identifiability gaps are preferred.
2. **Write & execute code**: Cheap code model writes Python, runs it against your DataFrame. A `pitfalls.txt` file of known API issues is loaded fresh on every call. Runtime error patterns are recorded and injected into future prompts to prevent repeated failures.
3. **Evaluate results**: Cheap model scores parallel solutions, selects the best, and generates finding summaries. Clean foreclosures of blocked approaches (with specific diagnostics) are scored as high-value findings, not as failures.
4. **Update research model**: Cheap model (Research Interpreter) rewrites the four RI-maintained sections of the research model. Established Findings are tagged with a STATUS prefix that the cheap model is responsible for maintaining across iterations as evidence accumulates.
5. **Strategic review**: Premium model reads the full research model and recent results. Issues a commitment — HOLD, PIVOT, or ABANDON. Updates the Strategic Trajectory. When structural discoveries have accumulated since the last review, extends the Structural Landscape. When it judges a second look at raw results would be valuable, triggers a reframing probe.

A live dashboard (`output/dashboard.html`) updates after each iteration. Open it in a browser to monitor progress, scores, the accumulated structural map, and finding STATUS breakdown in real time.

![delv-e dashboard](assets/dashboard.png)

### The Research Model

The research model is the central shared state, read by every agent and updated every iteration. It has six sections:

**RI-maintained (cheap model)**:

- **Established Findings** (max 10): confirmed facts with `[STATUS]` prefix tags and quantitative anchors. Four STATUS tags, assigned and maintained by the RI:
  - `[ESTABLISHED]` — survived multiple lines of analysis, confound-controlled where available
  - `[PROVISIONAL]` — signal present but a specific identifiability or coverage gap remains
  - `[SHRINKS]` — initially apparent effect deflated under controls; progression of estimates is reported
  - `[CONTRADICTED]` — earlier finding reversed by a later analysis; both estimates cited
- **Active Hypotheses** (max 4): testable claims the next analysis could strengthen or refute.
- **Attention Flags**: findings where a later analysis produced a different direction or >30% magnitude change. Drives STATUS retagging.
- **Exploration Health**: minimal — Breadth assessment (LOW/MEDIUM/HIGH) and an Unexplored list of column/variable groups not yet touched.

**Opus-maintained (structurally protected from cheap-model corruption)**:

- **Strategic Trajectory**: commitment state, arc history, what direction has highest expected value next. The premium model's strategic memory.
- **Structural Landscape**: the investigation's terrain. Four sub-sections:
  - *Identifiability* — what's separable from what, with diagnostic evidence
  - *Coverage* — where the evidence thins
  - *Foreclosed Directions* — approaches tried and ruled out, with DO-NOT operational guidance
  - *Open Questions* — structural questions the investigation cannot resolve with available data

Both protected sections use a save-before-RI / re-splice-after-RI pattern: before each RI call, they are extracted; after the RI rewrites the model, they are spliced back. The cheap model literally cannot corrupt them regardless of what it emits.

### Commitment System

The strategic review issues one of three commitments every iteration:

| Commitment | Meaning | Effect |
|---|---|---|
| **HOLD** | Current arc is productive | Deepen: questions drill into the same finding |
| **PIVOT** | New direction identified | Redirect: next_direction becomes a binding constraint |
| **ABANDON** | Current arc exhausted | Move on: system transitions to a new arc |

The commitment is enforced structurally until the next strategic review changes it. Commitment control lives entirely in the premium model, preventing the oscillation that occurs when a cheap model pattern-matches "needs breadth" one iteration and "needs depth" the next.

The dashboard displays commitment posture as a colored bar per iteration: cyan for EXPLORING, yellow for HOLD, green for PIVOT, red for ABANDON. Commitment history is recorded in the checkpoint.

### Reframing Probe

On any iteration, the strategic review can request a **reframing probe** — a premium-model second look at the raw analytical output (not the compressed digest) from the last three analyses. The probe receives full uncompressed stdout and is asked: what pattern, threshold, or distributional feature does the headline test not capture? If it finds an alternative framing, that becomes the binding direction for the next iteration. This mechanism exists because the most valuable reframings typically come from reading raw numbers with fresh eyes, not from compressed digests.

Expected frequency: roughly 5-8 times per 100 iterations.

### Perspective Rotation

When an original arc completes (ABANDON with ARC_COMPLETE), a **perspective rotation** generates 2-3 fundamentally different analytical lenses on the same phenomenon, ranked from most to least promising. An arc that investigated "what factors cause Y to decline" might spawn perspectives like "how frequently do positive vs negative events occur" or "has the system's capacity to convert input into outcome changed." The top-ranked perspective is automatically pursued for 1-2 iterations. Perspective arcs do not spawn further rotations. This mechanism addresses a structural blind spot: the system naturally deepens each topic through one lens but never switches lenses unless forced.

### Auto-Stop

When `--auto-stop` is enabled, the system can terminate before `max_iterations` on two signals:

1. **Strategic review explicit request**: the premium model sets EARLY_STOP: YES when there is no unexplored territory, all major findings are tagged ESTABLISHED or resolved as SHRINKS/CONTRADICTED, the last 5+ iterations have been unproductive ABANDONs, and Structural Landscape open questions require external data.
2. **Mechanical backstop**: 8 consecutive ABANDONs with mean score < 6.0 forces termination regardless of the review's recommendation.

Never triggers before iteration 15. Default off — full iteration budget is always used unless explicitly opted in.

```bash
python run.py data.csv "Analyze trends" --iterations 100 --auto-stop
```

### Data Dictionary

When `--data-dictionary PATH` points to a markdown file, delv-e treats its contents as authoritative context for column meanings and known caveats. This matters whenever the schema alone cannot express everything an analyst needs to know — event-specific or entity-specific columns (a metric whose definition varies across subjects in ways the column name doesn't reveal), categorical values with non-obvious semantics, sparse columns with structured missingness, derived columns that carry constraints about comparability. Without a dictionary, the Question Generator will sometimes propose questions that violate these constraints.

**The dictionary feeds orientation, not the Code Generator.** Single-channel architecture: dictionary → orientation → profile → all downstream agents. Orientation is required to emit a KEY CONSTRAINTS block at the top of the profile, restating the dictionary's non-obvious rules as numbered guardrails with specifics preserved verbatim. Each constraint ends with a "→" practical action.

**File format**: a plain markdown file, read verbatim. Conventionally useful sections: a one-paragraph overview, a "Must-read constraints" list, a column reference organized by role, and any source-data quirks. Soft cap: 20 KB.

**When to use**: datasets with domain-specific semantics (clinical studies, sports science, financial instruments), event-heterogeneous comparability, categorical values encoding things beyond what names suggest, structured missingness. For generic datasets with self-explanatory columns, a dictionary adds little.

```bash
python run.py data.csv "<question>" \
    --data-dictionary data_dictionary.md \
    --iterations 50 --auto-stop
```

### Literature Search

When `--search-model` is provided with an Anthropic model, the system can search published literature and integrate findings into the research model. Prevents rediscovering established results; helps validate novel findings.

**Pre-loop search** (computation-only mode): before the first iteration, searches for established research on the seed question and integrates `[PUBLISHED]` findings. Fills the role orientation plays for dataset mode — grounding the investigation.

**Mid-stream search**: the strategic review can request a search at any PIVOT or ABANDON iteration by outputting `SEARCH_NEEDED: <query>`. Typically fires when pivoting into an unfamiliar domain, when a surprising finding might already be established, or when nearing completion and top findings should be validated.

`[PUBLISHED]` entries are structurally protected from RI modification (like Strategic Trajectory and Structural Landscape). They enter as testable predictions, not assumptions — when simulation results contradict published claims, this is flagged as potentially novel rather than treated as error.

```bash
python run.py "Simulate evolution of cooperation" \
    --search-model anthropic:claude-sonnet-4-6
```

Adds ~$0.50-1.20 per 100-iteration run. Default disabled.

### Error Patterns and Pitfalls

**Static pitfalls** (`pitfalls.txt`): user-maintained file of known code-generation traps. Loaded fresh on every code-generation call, so edits take effect mid-run without restart.

**Runtime error patterns**: when code execution fails, library-specific errors (AttributeErrors, ImportErrors) are recorded and injected into all future code-generation prompts. Generic errors are filtered out.

Both sources are appended to the code-generator's prompt to prevent errors recurring across iterations.

## Output

The primary deliverables are three handoff artefacts, with chain_id citations linking back to the individual analyses that produced each number:

```
output/
├── briefing.md                 # Primary handoff — 7-section structured briefing
├── briefing.html               # Styled HTML version with clickable citations
├── findings_index.md           # All winning analyses with scores, methods, summaries
├── findings_index.html         # Styled HTML version
├── structural_map.md           # Structural Landscape (live-written during run)
├── structural_map.html         # Styled HTML version
├── synthesis_charts/           # Publication-quality charts per finding in §2
│   ├── 01_first_finding_name.png
│   └── ...
├── research_model.md           # Final research model (all 6 sections)
├── dashboard.html              # Live dashboard
├── run_log.json                # Full log of every LLM call
├── state.json                  # Checkpoint for --continue
├── dataframe.parquet           # Preserved DataFrame (dataset mode)
├── cost.txt                    # Cost breakdown by agent
├── orientation/
│   └── analysis.md             # Dataset analytical profile
└── exploration/
    ├── 01/
    │   ├── _summary.md         # Iteration evaluation + commitment
    │   └── 1773198695/
    │       ├── analysis.md     # Question + code + output
    │       ├── analysis.html   # Styled HTML with back-link to briefing
    │       └── plot_001.png
    └── ...
```

### The Briefing

`briefing.md` follows a fixed seven-section structure:

- **§0 Investigation Scope** — what was asked, what was available, what this briefing covers
- **§1 Structural Landscape** — rendered from the research model's accumulated landscape: identifiability (often as a table), coverage asymmetries, available identification strategies, blocked approaches
- **§2 Findings** — each as its own H3 block with STATUS tag, key numbers with citations, CONFOUND-STATUS, and NEXT actions. Ordered by substantive importance for the downstream investigator
- **§3 Foreclosed Directions** — evidence-backed dead ends, each with a DO-NOT operational instruction
- **§4 Open Questions** — things the investigation could not resolve with available evidence, each with BLOCKER and WOULD-RESOLVE-WITH
- **§5 Suggested Entry Points** — ranked concrete starting directions for the next phase, with RATIONALE and FIRST STEP
- **§6 Methodological Notes** — filtering, sample sizes, method-dependent results

Every numerical claim includes a `[[chain_id]]` citation that becomes a clickable link in the HTML version, opening the original analysis with its code, output, and plots. Each analysis page has a "← Back to Briefing" link.

`findings_index.md` is a complete one-line-per-analysis catalogue: score, iteration, chain_id, method, question, finding summary. `structural_map.md` is the Structural Landscape rendered as a standalone artefact — written live during the run, updated each time strategic review extends the landscape, so it's visible on the dashboard as it grows.

Charts in `synthesis_charts/` are publication-quality matplotlib figures generated for each non-CONTRADICTED H3 finding in §2. Each chart is adapted from the original analysis code that produced the finding, ensuring the numbers on the chart match the numbers in the text.

## Memory Architecture

LLMs have no memory between calls. delv-e manages context through five layers:

**Data Profile** produced by orientation. An analytical brief (capped at 12 KB, ~3K tokens) covering coverage, groups, confounders, and — critically — a KEY CONSTRAINTS block when a dictionary is provided and an initial Structural Landscape block that seeds the research model. Pinned into every agent's context for the entire run.

**Research Model** — six sections, read by every agent, updated every iteration. Four sections maintained by the cheap Research Interpreter. Two sections (Strategic Trajectory, Structural Landscape) maintained exclusively by the premium strategic review and structurally protected from cheap-model corruption via extract-before / re-splice-after. The cheap model literally cannot corrupt them regardless of what it emits.

**Insight Tree** — every analysis is a node with question, code, results, score, and summaries. Agents see a tiered view: recent entries with RI-curated key numbers, older entries compressed to one-sentence summaries. Nothing is deleted; the system manages visibility, not existence.

**Q&A Pairs** — a summary-based format for the Code Generator: recent pairs (last 40) with finding summaries and chain IDs. Keeps tactical context present without overwhelming the prompt.

**Full Results Store** — untruncated results from every analysis, never shown during exploration. Read only by the briefing generator, which receives top-10 winning analyses at full detail plus digest-only summaries for the remaining score-≥6 winners.

### Context Budget for Cheap Agents

The research model is the central shared state. Its shape was optimized in the November 2026 refactor to reduce cheap-model load: removed sections (Finding Maturity's 5-stage tracker, Cross-Finding Connections, Biggest Gap) that served the old narrative-synthesis path but added noise for cheap agents. The result is that cheap agents see less structural clutter and more actionable guidance (Structural Landscape's Foreclosed Directions list tells them what not to re-try; Open Questions list tells them what's valuable to resolve).

## Cost

| Configuration | ~Cost per 10 iterations |
|---|---|
| All Haiku | $0.50-1.50 |
| Haiku agents + Opus code | $2-4 |
| OpenRouter OSS (kimi/glm) | $0.50-1.00 |
| All Ollama (local) | Free |
| Ollama + Opus premium | ~$1.00 (orientation + strategic review + briefing) |

The strategic review (premium model) runs every iteration but is lightweight — roughly 6K input tokens and 600 output tokens per call. Over 100 iterations this adds roughly $7 at Opus pricing. Briefing generation adds ~$0.50 and chart generation adds ~$0.50-0.80 (one premium-model call per non-CONTRADICTED H3 finding). Total premium model cost for a 100-iteration run is typically $8-10.

A data dictionary adds negligible cost — it enters the prompt only during orientation (one call per run).

Check `output/cost.txt` after each run for exact breakdown by agent.

## Usage

```
python run.py [<dataset>] ["<question>"] [options]
```

The dataset is optional. If the first argument is not a file path, it is treated as the question and the system runs in computation-only mode.

| Option | Default | Description |
|---|---|---|
| `--iterations N` | 5 | Exploration iterations |
| `--parallel N` | 2 | Parallel solutions per iteration |
| `--output DIR` | output/ | Output directory |
| `--continue` | | Resume from previous run's checkpoint |
| `--no-orientation` | | Skip the orientation phase |
| `--auto-stop` | | Allow early termination when investigation is complete |
| `--data-dictionary PATH` | none | Markdown file with column semantics and caveats |
| `--agent-model` | anthropic:claude-haiku-4-5-20251001 | Cheap model for evaluator, QG, RI, selector |
| `--code-model` | anthropic:claude-haiku-4-5-20251001 | Model for code generation |
| `--premium-model` | same as code-model | Model for orientation, strategic review, briefing, seed decomposition |
| `--search-model` | disabled | Anthropic model for literature search |

### Providers

Model format: `provider:model_name`

| Provider | Example | Requires |
|---|---|---|
| Anthropic | `anthropic:claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai:gpt-5.4` | `OPENAI_API_KEY` |
| OpenRouter | `openrouter:moonshotai/kimi-k2.5` | `OPEN_ROUTER_API_KEY` |
| Ollama | `ollama:qwen3:30b` | Local Ollama server |

### Examples

```bash
# All Haiku (cheapest cloud option)
python run.py data.csv "Analyze trends" --iterations 15

# Quick 3-iteration scan, skip orientation
python run.py data.csv "What's the class balance?" --iterations 3 --no-orientation

# Domain-specific dataset with dictionary
python run.py clinical_trial.csv "What predicts response to treatment?" \
    --data-dictionary trial_dictionary.md \
    --premium-model anthropic:claude-opus-4-6 \
    --iterations 50 --auto-stop

# Full 100-iteration run with auto-stop
python run.py data.csv "What drives peak snowpack decline?" \
    --agent-model openrouter:moonshotai/kimi-k2.5 \
    --code-model openrouter:moonshotai/kimi-k2.5 \
    --premium-model anthropic:claude-opus-4-6 \
    --iterations 100 --auto-stop

# Local Ollama with premium strategic oversight
python run.py data.csv "Deep analysis" \
    --agent-model ollama:qwen3:30b \
    --code-model ollama:qwen3:30b \
    --premium-model anthropic:claude-opus-4-6
```

### Computation-Only Mode

When no dataset is provided, the system runs in computation-only mode. The code generator has access to the full scientific Python stack (numpy, scipy, sympy, pandas, networkx, statsmodels, scikit-learn) and generates code that creates, simulates, or computes rather than analysing an existing DataFrame.

```bash
# Evolutionary game theory simulation
python run.py "Simulate the evolution of cooperation using iterated Prisoner's Dilemma" \
    --iterations 100

# Number theory exploration
python run.py "Explore the distribution of twin prime gaps up to 10^8" \
    --iterations 50
```

The intelligence loop — research model, strategic review, question generation, briefing rendering — works identically in both modes. The only difference is how each iteration's analysis step executes.

## Resuming Runs

Checkpoint saved after every iteration. Resume with `--continue`:

```bash
# Initial run
python run.py data.csv "Analyze X" --iterations 25

# Continue with a new direction (iterations are additive)
python run.py data.csv "Pursue the Y paradox" --continue --iterations 30
```

The seed question on `--continue` becomes the first analysis in the resumed run. DataFrame, research model, insight tree, commitment history, and all context are preserved. You can switch models between runs. State format is versioned — older checkpoints (pre-v6, before the Structural Landscape refactor) are migrated automatically on load.

## Architecture

```
run.py               CLI: dataset loading, --continue handling, --auto-stop, --data-dictionary
engine.py            ExplorationEngine: LLM pipeline, code execution, orientation
auto_explore.py      Core loop: commitment system, strategic review, Structural Landscape
                     maintenance, research model management, briefing + artefact writing
output.py            OutputManager: terminal display, analysis markdown, briefing HTML,
                     findings_index HTML, structural_map HTML, PDF export
dashboard.py         Live HTML dashboard with commitment bands, three-artefact buttons
llm.py               Multi-provider LLM client (Anthropic, OpenAI, OpenRouter, Ollama)
executor.py          Local code execution with security guards and traceback filtering
prompts.py           All prompt templates
style.py             Terminal formatting: colored commitment bars, exploration tree, spinners
pitfalls.txt         Static code generation hints (user-editable, live-reloaded)
logger_config.py     Logging configuration
```

## Security

Generated code runs locally via `exec()`. A module blacklist blocks dangerous operations (subprocess, socket, file deletion, network access) but this is **not a sandbox**. See `executor.py` for the full blacklist. API keys are read from environment variables only, never logged or stored.

## Origin

Standalone extraction of the auto-explore module from [BambooAI](https://github.com/pgalko/BambooAI). Core exploration logic preserved; web UI, database, billing, and multi-tenant routing replaced with minimal local equivalents. Substantially refactored in November 2026 to move from narrative-synthesis output to structured-briefing output, reflecting the system's role as a Rung-1+2 hypothesis-search pre-filter rather than an end-to-end analytical deliverable.