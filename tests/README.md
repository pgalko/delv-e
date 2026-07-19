# Tests

The canonical regression suite for delv-e. Each test is a standalone script that
bootstraps its own import paths (the repo root, plus a bundled `httpx` stub under
`tests/stubs/`), so no setup beyond the project dependencies is required and no
network access is needed.

## Running

From the repository root:

    pip install -r requirements.txt
    for t in tests/test_*.py; do python3 "$t"; done

Each script prints its own assertions and exits non-zero on failure.

## Dataset-backed tests

`test_continue.py`, `test_extend.py`, and `test_search.py` exercise the resume,
extend, and mid-stream search paths against the EEDR benchmark dataset. They look
for the CSV at `datasets/EEDR_sessions_laps_enriched.csv` (override with the
`EEDR_DATASET` environment variable) and skip cleanly if it is absent, so the rest
of the suite still runs in a fresh clone. To run them, provide the dataset:

    EEDR_DATASET=/path/to/EEDR_sessions_laps_enriched.csv python3 tests/test_search.py

## Coverage

- `test_happy` — natural CONTINUE to SYNTHESIZE to FINAL on synthetic data
- `test_provisional` — provisional briefing at the iteration ceiling when synthesis gates
- `test_tier` — model-tier selection and cost-tracking wiring
- `test_executor` — kernel crash-then-fix retry path; the error variable stays bound
- `test_artifact` — artifact and output-file production
- `test_continue` — `--continue` resume from a saved run (dataset-backed)
- `test_extend` — `--extend` adds steps to a finished run (dataset-backed)
- `test_search` — mid-stream SEARCH path and budget cap (dataset-backed)
- `test_context` — context-growth and compaction behavior
- `test_g1_fixes` — Q2 code-grounded G1 backstop and Q3 provisional-on-gate
- `test_truncation` — Investigator truncation retry, then provisional fallback
- `test_ledger_parse` — nav ledger render/parse round trip and parser tolerance
- `test_status` — STATUS reader hardening: decision from the leading token, prose-wrapped cases
- `test_estimand` — estimand pinning and the G3 gate on null verdicts
- `test_telemetry` — `run_telemetry.json` aggregation and the `RunStats` event counters
- `test_synth_gates` — Synthesizer GATES deliberation and the verdict clauses
- `test_budget` — budget wrap-up notice window
- `test_toolkit` — known-answer checks for the three preloaded estimators
- `test_verify` — the `--verify` audit pipeline: extraction, audit seed, reconciliation
- `test_compute` — dataset-free compute mode through `run_investigation`
- `test_compute_cli` — compute mode through the full `run_core.main()` CLI
- `test_compute_continue` — resume/extend of compute runs via `run_meta.json`
- `test_verify_compute` — verify in compute mode: prompt variants and the seed chain
- `test_executor_reasoning_effort` — per-agent reasoning-effort wiring across providers
- `test_truncation_retry` — the hold-then-none truncation retry ladder
- `test_reasoning_ladder` — effort mappings per provider, the ladder, the empty-completion guard, the xAI affinity header
- `test_function_reuse` — function reuse through the namespace registry
- `test_checkpoint` — kernel checkpoint tail-replay
- `test_gpt56_cache` — GPT-5.6 explicit caching: breakpoint emission, flatten pass-through, write-cost math pinned to a live probe
- `test_compaction_budget` — history-budget tiered demotion: inert default, pass order, red lines
- `test_print_discipline` — the PRINT BUDGET clause (both modes, leakage-clean) and the kernel float format
- `test_namespace` — the shared mutable namespace: registry return contracts and element shapes (so a blind Executor never guesses what a persisted object contains), step-versioned aliases (`records__s6`) that survive a later rebind, ambiguity flagging on the bare name, and pinned-alias resolution in specs
- `test_charts` — charts: CHART/FINDING/CAPTION/SPEC parsing, harness-owned placement by finding id, no broken links and no lost charts
- `test_reasoning_floors` — per-model effort floors (glm, x-ai) and the reasoning-rejection circuit breaker
- `test_spec_selfcontainment` — the executor-visibility spec rule (both modes), the compute persist-as-function rule, and the blind-step-reference tripwire
- `test_search_providers` — provider-native web search: dispatch, the per-call plugin opt-in (searchless Executor/Synthesizer pin), the ollama REST path, and auto seating with the fallback chain
- `test_provider_cost` — OpenRouter's `usage.cost` as the authoritative ledger figure: capture, absent-vs-zero, tool-fee reconciliation, unchanged fallback
- `test_synth_audience` — the synthesis prompt contract after the split: the standard of proof governs the technical pass (and carries no audience rule), the audience standard governs the editor, reconciliation adjudicates findings, plus the permanent domain-neutrality guard and the f-string brace guard
- `test_two_pass` — the two-pass machinery: findings parsing, the three harness gates (coverage with its retry-then-append path driven through the real loop, unsourced numbers, unverified citations), chart markers keyed on findings, and the editor's isolation from raw evidence
