#!/usr/bin/env python3
"""
delv-e: Autonomous Data Investigation

Usage:
    python run.py data.csv                          # interactive — prompts for question
    python run.py data.csv "What patterns exist?"   # inline question
    python run.py data.csv "Explore revenue" --iterations 10
    python run.py data.csv --code-model openai:gpt-5.3-codex

    # Resume a previous run with 20 additional iterations
    python run.py data.csv "Pursue the MSS3 finding" --continue --iterations 20

Model format: provider:model_name
    anthropic:claude-opus-4-6       (requires ANTHROPIC_API_KEY)
    openai:gpt-5.4                  (requires OPENAI_API_KEY)
    ollama:qwen3:30b                (requires local Ollama server)
"""

import argparse
import json
import os
import sys

def main():
    parser = argparse.ArgumentParser(
        description="delv-e: Autonomous data investigation powered by LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dataset", help="Path to data file (.csv, .tsv, .xlsx, .parquet, .json, .jsonl)")
    parser.add_argument("question", nargs="?", default=None,
                        help="Seed question (if omitted, prompts interactively)")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Exploration iterations to run (default: 5)")
    parser.add_argument("--parallel", type=int, default=2,
                        help="Parallel solutions per iteration (default: 2)")
    parser.add_argument("--output", default="output",
                        help="Output directory (default: output/)")
    parser.add_argument("--agent-model", default=None,
                        help="provider:model for agents (default: anthropic:claude-haiku-4-5-20251001)")
    parser.add_argument("--code-model", default=None,
                        help="provider:model for code gen and synthesis (default: anthropic:claude-opus-4-6)")
    parser.add_argument("--continue", dest="continue_run", action="store_true",
                        help="Resume from a previous run's saved state. "
                             "Iterations are additive (e.g. 25 completed + --iterations 30 = 30 more).")
    args = parser.parse_args()

    # Load environment
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    import pandas as pd
    resumed_state = None

    if args.continue_run:
        # ── CONTINUE: load saved state and DataFrame ──
        state_path = os.path.join(args.output, "state.json")
        df_path = os.path.join(args.output, "dataframe.parquet")

        if not os.path.exists(state_path):
            print(f"Error: No saved state found at {state_path}", file=sys.stderr)
            print("Run without --continue to start a fresh exploration.", file=sys.stderr)
            sys.exit(1)

        with open(state_path) as f:
            resumed_state = json.load(f)

        completed = resumed_state.get('iterations_completed', 0)
        print(f"Resuming from iteration {completed} ({completed} completed)")

        # Load DataFrame: prefer saved parquet (may have added columns), fall back to dataset arg
        if os.path.exists(df_path):
            df = pd.read_parquet(df_path)
        elif os.path.exists(args.dataset):
            df = _load_dataset(args.dataset)
        else:
            print(f"Error: Neither saved DataFrame nor {args.dataset} found", file=sys.stderr)
            sys.exit(1)
    else:
        # ── FRESH: load dataset from file ──
        if not os.path.exists(args.dataset):
            print(f"Error: File not found: {args.dataset}", file=sys.stderr)
            sys.exit(1)
        df = _load_dataset(args.dataset)

    # Interactive question prompt if not provided
    question = args.question
    if not question:
        from style import LOGO, VERSION, TAGLINE, DIM, WHITE, RESET, CYAN
        if not args.continue_run:
            print(LOGO)
            print(f"    {DIM}{VERSION} — {TAGLINE}{RESET}")
            print()
            print(f"    {DIM}Loaded{RESET} {WHITE}{len(df):,} rows × {len(df.columns)} cols{RESET} {DIM}from{RESET} {WHITE}{os.path.basename(args.dataset)}{RESET}")
        else:
            print(f"    {DIM}Enter a direction for the continued exploration:{RESET}")
        print()
        try:
            question = input(f"    {CYAN}>{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            sys.exit(0)
        if not question:
            print("No question provided.", file=sys.stderr)
            sys.exit(1)
        print()

    # Build engine
    from engine import ExplorationEngine
    engine = ExplorationEngine(
        df=df,
        output_dir=args.output,
        agent_model=args.agent_model,
        code_model=args.code_model,
        continue_run=args.continue_run,
    )

    # Run exploration
    from auto_explore import AutoExplorer
    explorer = AutoExplorer(engine)
    explorer.run(
        seed_question=question,
        max_iterations=args.iterations,
        num_parallel_solutions=args.parallel,
        interactive=args.question is None,
        resumed_state=resumed_state,
    )


def _load_dataset(path):
    """Load a dataset from file, inferring format from extension."""
    import pandas as pd
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in ('.tsv',):
            return pd.read_csv(path, sep='\t')
        elif ext in ('.xlsx', '.xls'):
            return pd.read_excel(path)
        elif ext in ('.parquet', '.pq'):
            return pd.read_parquet(path)
        elif ext in ('.json',):
            return pd.read_json(path)
        elif ext in ('.jsonl',):
            return pd.read_json(path, lines=True)
        else:
            return pd.read_csv(path)
    except Exception as e:
        print(f"Error loading {ext or 'file'}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()