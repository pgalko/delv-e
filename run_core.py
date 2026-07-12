#!/usr/bin/env python3
"""
delv-e (inverted core) — entry point.

Two model tiers, three roles:
  --investigator-model  premium; does all analytical thinking (and synthesis,
                        unless --synth-model is given).
  --executor-model      cheap; writes pandas for closed specs, zero judgment.

Usage:
    python run_core.py data.csv "Your seed question"
    python run_core.py data.csv "Seed" --iterations 16
    python run_core.py data.csv "Seed" \
        --investigator-model anthropic:claude-opus-4-8 \
        --executor-model ollama:kimi-k2.6:cloud
    python run_core.py data.csv --resume --iterations 10
    python run_core.py data.csv "Is the threshold effect a surface artifact?" --extend --iterations 10

Model format: provider:model_name (anthropic:..., openai:..., ollama:..., openrouter:...).
Requires the relevant provider key in the environment / .env (same as the old run.py).
"""

import argparse
import glob
import json
import os
import shutil
import sys
import time

# Sensible two-tier defaults: works with only an Anthropic key. Swap the executor
# to a local/cheaper model (e.g. ollama:kimi-k2.6:cloud) to cut cost.
DEFAULT_INVESTIGATOR_MODEL = "anthropic:claude-opus-4-8"
DEFAULT_EXECUTOR_MODEL = "anthropic:claude-haiku-4-5-20251001"


def main():
    parser = argparse.ArgumentParser(
        description="delv-e inverted-core: autonomous data investigation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dataset", nargs="?", default=None,
                        help="Path to data file (.csv/.tsv/.xlsx/.parquet/.json/.jsonl). "
                             "Omit when using --compute.")
    parser.add_argument("question", nargs="?", default=None, help="Seed question.")
    parser.add_argument("--compute", action="store_true",
                        help="Dataset-free mode: run simulations or pure computation "
                             "with no dataset loaded (`df` does not exist). Pass the "
                             "question as the sole positional argument.")
    parser.add_argument("--iterations", type=int, default=14,
                        help="Max steps (a ceiling; the run stops earlier when the "
                             "Investigator synthesizes). With --resume/--extend, this "
                             "is ADDITIONAL steps. Default 14.")
    parser.add_argument("--investigator-model", default=None,
                        help=f"Premium model: thinking + synthesis (default {DEFAULT_INVESTIGATOR_MODEL}).")
    parser.add_argument("--executor-model", default=None,
                        help=f"Cheap model: code only (default {DEFAULT_EXECUTOR_MODEL}).")
    parser.add_argument("--synth-model", default=None,
                        help="Optional separate model for synthesis (default: investigator model).")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: output/; under --verify the "
                             "default is output_verify/ so the audit never lands in "
                             "the run it audits).")
    parser.add_argument("--data-dictionary", default=None,
                        help="Optional markdown file describing columns/caveats; appended to schema.")
    parser.add_argument("--periodic-every", type=int, default=0,
                        help="Run a holistic re-derivation snapshot every N steps "
                             "(0 = off; it adds cost). Default 0.")
    parser.add_argument("--g1-pushback", type=int, default=2,
                        help="How many times the G1 gate may force more work before "
                             "allowing synthesis. Default 2.")
    parser.add_argument("--no-search", action="store_true",
                        help="Disable mid-stream web search. By default search is "
                             "auto-seated on the first run model (investigator, then "
                             "synthesizer, then executor) whose provider can search: "
                             "anthropic (native tool), openrouter (web plugin), or "
                             "ollama (hosted endpoint, requires OLLAMA_API_KEY).")
    parser.add_argument("--search-budget", type=int, default=3,
                        help="Hard cap on web searches per run when search is "
                             "enabled. Default 3.")
    parser.add_argument("--resume", dest="resume_run", action="store_true",
                        help="Recover an interrupted run: rehydrate the saved state in "
                             "--output (kernel history, nav ledger, log) and finish the "
                             "SAME seed question. Iterations are additive.")
    parser.add_argument("--extend", dest="extend_run", action="store_true",
                        help="Extend a finished run with a NEW seed question (required): "
                             "rehydrate the prior state, pursue the new question in light "
                             "of it, and synthesize one combined briefing that reconciles "
                             "both lines, revising the original conclusion if warranted.")
    parser.add_argument("--verify", dest="verify_run", default=None,
                        nargs="?", const="@last", metavar="PRIOR_RUN_DIR",
                        help="Audit a prior run: distill the decisive claims from its "
                             "briefing.md, run a FRESH independent investigation that "
                             "adjudicates them against the data, then write one "
                             "reconciled briefing (originals preserved alongside) into "
                             "output_verify/ by default. Bare --verify audits the last "
                             "completed run (tracked in .delve_last_run). The positional "
                             "question, if given, replaces the prior run's saved "
                             "question in the audit framing.")
    parser.add_argument("--reasoning-effort",
                        choices=["max", "high", "medium", "low", "none"], default="medium",
                        help="Starting reasoning effort for the Investigator and Synthesizer "
                             "(default medium). The Executor always runs with reasoning off. "
                             "Mapped per provider; direct Anthropic ignores it (no effort dial). "
                             "Note GLM-5.2 already runs at its max for medium/low; pass 'high' to "
                             "make it reason less.")
    args = parser.parse_args()
    if args.output is None:
        args.output = "output_verify" if args.verify_run is not None else "output"

    investigator_model = args.investigator_model or DEFAULT_INVESTIGATOR_MODEL
    executor_model = args.executor_model or DEFAULT_EXECUTOR_MODEL
    synth_model = args.synth_model or investigator_model

    # Local imports so --help is fast and import errors are actionable.
    from dotenv import load_dotenv
    load_dotenv()

    # Web search seat. MUST come after load_dotenv(): the seat is credential-gated
    # (see llm.resolve_search_seat), so resolving it before .env is read would find
    # no keys and silently disable search on every run. Search follows the
    # Investigator's provider: openrouter -> grok-4.3, anthropic -> Haiku, ollama ->
    # its own model + Ollama's free hosted endpoint; when the investigator's provider
    # cannot serve search, it falls through to OpenRouter, then Anthropic.
    from llm import resolve_search_seat, parse_model_string
    search_model = None if args.no_search else resolve_search_seat(investigator_model)
    if not args.no_search:
        if search_model is None:
            print("Web search disabled: no provider can serve it (an ollama-seated "
                  "run needs OLLAMA_API_KEY for its free search; otherwise set "
                  "OPEN_ROUTER_API_KEY or ANTHROPIC_API_KEY).", file=sys.stderr)
        elif (parse_model_string(search_model)[0]
              != parse_model_string(investigator_model)[0]):
            # The investigator's own provider cannot search, so search fell through
            # to another provider and will bill it. Say so once.
            print(f"Web search runs on {search_model} (the investigator's provider "
                  f"cannot serve search).", file=sys.stderr)

    # Quiet the run: suppress library warnings and keep INFO logs off the console
    # so the styled output stays clean. (Real warnings/errors still surface; the
    # full record is in run_log.json. Set DELVE_VERBOSE=1 to keep logs on.)
    if not os.environ.get("DELVE_VERBOSE"):
        import warnings
        import logging
        warnings.filterwarnings("ignore")
        os.environ.setdefault("PYTHONWARNINGS", "ignore")
        logging.disable(logging.INFO)
        for noisy in ("httpx", "httpcore", "anthropic", "openai", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    from llm import LLMClient, CostTracker, RunLogger, RunStats, build_run_telemetry
    from kernel import PersistentKernel
    from nav_state import NavState
    from dataio import load_dataset, build_schema
    from investigation import run_investigation

    if args.resume_run and args.extend_run:
        print("Use either --resume or --extend, not both.", file=sys.stderr)
        sys.exit(1)
    if args.verify_run and (args.resume_run or args.extend_run):
        print("--verify starts a fresh independent audit; it cannot be combined "
              "with --resume or --extend.", file=sys.stderr)
        sys.exit(1)
    is_continue = args.resume_run or args.extend_run

    # Resume/extend restore the run's mode from run_meta.json, so a compute run can be
    # continued without re-passing --compute (a dataset run is not re-declared either).
    # Passing --compute against a saved dataset run is a mismatch and is rejected.
    if is_continue:
        saved_compute = _load_run_meta(args.output).get("compute", False)
        if args.compute and not saved_compute:
            parser.error(f"{args.output} is a dataset run; --compute cannot be "
                         "applied to a --resume/--extend.")
        args.compute = saved_compute

    # A verify run audits the prior run in that run's own mode: a compute run is
    # audited in compute mode (no dataset), a dataset run in dataset mode. The mode is
    # read from the prior run's run_meta.json, so --compute is not re-passed for an
    # audit, and passing it against a dataset run is rejected.
    prior_dir = None
    if args.verify_run:
        import verify
        prior_dir = verify.resolve_prior_dir(args.verify_run)
        audit_compute = _load_run_meta(prior_dir).get("compute", False)
        if args.compute and not audit_compute:
            parser.error("the run being verified is a dataset run; --compute does "
                         "not apply to its audit.")
        args.compute = audit_compute

    if args.compute:
        # There is no dataset, so the sole positional is the question.
        if args.question is None and args.dataset is not None:
            args.question = args.dataset
            args.dataset = None
    elif not args.dataset:
        parser.error("a dataset path is required (or pass --compute for a "
                     "dataset-free run).")

    # Verify mode: validate the prior run, load its briefing and question now;
    # the audit seed itself is composed after the client exists (claim
    # extraction is an LLM call). Independence is deliberate: only the prior
    # BRIEFING crosses over, never the prior evidence chain or kernel.
    verify_ctx = None
    if args.verify_run:
        prior_briefing_path = verify.check_dirs(prior_dir, args.output)
        with open(prior_briefing_path, encoding="utf-8") as f:
            original_briefing = f.read()
        prior_saved = _load_seeds(prior_dir)
        original_seed = (args.question or verify.original_question(prior_saved)
                         or _load_saved_seed(prior_dir) or "")
        if not original_seed:
            print(f"--verify: no saved question found in {prior_dir}; "
                  f"pass the original question as the positional argument.",
                  file=sys.stderr)
            sys.exit(1)
        verify_ctx = {"original_seed": original_seed,
                      "original_briefing": original_briefing,
                      "prior_dir": prior_dir}
        seed = None                      # composed after the client exists

    # Resolve the active seed and the prior-seed history.
    # - fresh:  seed from arg/prompt; history = [seed]
    # - resume: reuse the saved seed; history unchanged; prior_seeds = None
    # - extend: NEW seed required; prior_seeds = saved history; append new seed
    prior_seeds = None
    saved_seeds = _load_seeds(args.output) if is_continue else []
    if args.extend_run:
        seed = args.question
        if not seed and sys.stdin.isatty():
            try:
                seed = input("Extension seed question: ").strip()
            except (EOFError, KeyboardInterrupt):
                sys.exit(1)
        if not seed:
            print("--extend requires a NEW seed question.", file=sys.stderr)
            sys.exit(1)
        prior_seeds = saved_seeds or ([_load_saved_seed(args.output)]
                                      if _load_saved_seed(args.output) else [])
        _save_seeds(args.output, prior_seeds + [seed])
    elif args.resume_run:
        seed = args.question or (saved_seeds[-1] if saved_seeds else
                                 _load_saved_seed(args.output) or "")
        if not seed:
            print("Nothing to resume: no saved seed in --output.", file=sys.stderr)
            sys.exit(1)
        if not saved_seeds:                      # older run had only seed.txt
            _save_seeds(args.output, [seed])
    elif not args.verify_run:
        seed = args.question
        if not seed:
            try:
                seed = input("Seed question: ").strip()
            except (EOFError, KeyboardInterrupt):
                sys.exit(1)
        if not seed:
            print("No seed question provided. Aborting.", file=sys.stderr)
            sys.exit(1)

    df = None if args.compute else load_dataset(args.dataset)
    data_dict = None
    if args.data_dictionary and os.path.exists(args.data_dictionary):
        with open(args.data_dictionary, encoding="utf-8") as f:
            data_dict = f.read()
    schema_text = build_schema(df, data_dictionary=data_dict)

    os.makedirs(args.output, exist_ok=True)
    import ui
    if verify_ctx is not None:
        ui.MODE = "verify"
    cost_tracker = CostTracker()
    # Per-invocation log archive: <output>/logs/<UTC timestamp>/ holds run_log.json
    # and (at run end) run_telemetry.json. The checkpoint trio (nav_state.json,
    # log.json, kernel_history.json) and the deliverables stay in the output root,
    # so --resume and --extend are unaffected; a resume or extend simply gets its
    # own new timestamped folder with that invocation's two log files.
    run_ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    logs_dir = os.path.join(args.output, "logs", run_ts)
    os.makedirs(logs_dir, exist_ok=True)
    run_logger = RunLogger(os.path.join(logs_dir, "run_log.json"), append=False)
    run_stats = RunStats()
    client = LLMClient(cost_tracker=cost_tracker, run_logger=run_logger,
                       progress=ui.ENABLED)

    # Verify mode, phase 1: distill the prior briefing into decisive claims and
    # compose the audit seed. From here on the pipeline is the standard one;
    # the audit is just a run whose seed happens to carry claims to adjudicate.
    if verify_ctx is not None:
        ui.note(f"Verification audit of '{verify_ctx['prior_dir']}'; the "
                f"reconciled briefing will land in '{args.output}/'.", "magenta")
        ui.note("Verify: distilling the prior briefing into decisive claims…",
                "magenta")
        verify_ctx["claims"] = verify.extract_claims(
            client, synth_model, verify_ctx["original_briefing"],
            compute=args.compute)
        ui.note(f"Verify: auditing "
                f"{len(verify_ctx['claims']) or 'an excerpt of'} claim(s).",
                "magenta")
        seed = verify.compose_audit_seed(
            verify_ctx["original_seed"], verify_ctx["claims"],
            verify_ctx["original_briefing"], compute=args.compute)

    # Build / restore kernel, nav, prior log.
    kernel = PersistentKernel(df=df)
    nav = NavState()
    prior_log = None
    if is_continue:
        nav, prior_log, history = _load_saved_state(args.output)
        if history:
            verb = "Extending" if args.extend_run else "Resuming"
            print(f"{verb}: replaying {len(history)} prior step(s) into the kernel…")
            kernel.restore_history(history)
    else:
        _save_seeds(args.output, [seed])
        _save_run_meta(args.output, args.compute)
        # Fresh run: clear stale per-step artifacts from any previous run in this
        # output dir so old exploration/NN folders and landscape files don't
        # accumulate. (--resume/--extend intentionally skip this to preserve the run.)
        stale = os.path.join(args.output, "exploration")
        if os.path.isdir(stale):
            shutil.rmtree(stale, ignore_errors=True)
        for f in glob.glob(os.path.join(args.output, "landscape_step*.md")):
            try:
                os.remove(f)
            except OSError:
                pass

    ui.banner()
    ui.run_header(seed=seed,
                  rows=0 if df is None else df.shape[0],
                  cols=0 if df is None else df.shape[1],
                  iterations=args.iterations, code_model=executor_model,
                  brain_model=investigator_model, output=args.output,
                  compute=args.compute)

    run_t0 = time.time()
    try:
        log, kernel, nav, briefing = run_investigation(
            seed=seed, df=df, client=client,
            investigator_model=investigator_model,
            executor_model=executor_model, synth_model=synth_model,
            schema_text=schema_text, max_steps=args.iterations,
            output_dir=args.output, kernel=kernel, nav=nav, log=prior_log,
            periodic_every=args.periodic_every,
            g1_pushback_budget=args.g1_pushback, ui=ui,
            prior_seeds=prior_seeds,
            search_model=search_model, search_budget=args.search_budget,
            stats=run_stats, compute=args.compute,
            reasoning_effort=args.reasoning_effort,
        )
    finally:
        kernel.cleanup()
    run_wall = time.time() - run_t0

    # Verify mode, phase 3: reconcile the audit with the original briefing into
    # the single user-facing document. Runs before telemetry so the extra calls
    # are counted. The audit's own briefing is preserved as briefing_audit.md;
    # if reconciliation yields nothing usable, it stands in (never end
    # empty-handed).
    if verify_ctx is not None and briefing:
        ui.note("Verify: reconciling the audit with the original briefing…",
                "magenta")
        reconciled = verify.reconcile(
            client, synth_model, verify_ctx["original_seed"],
            verify_ctx["original_briefing"], briefing, compute=args.compute)
        _, used_fallback = verify.finalize_verify_outputs(
            args.output, verify_ctx["original_briefing"], briefing,
            reconciled, verify_ctx.get("claims", []))
        if not used_fallback:
            briefing = reconciled

    # Run telemetry (supersedes the old cost.txt). Written once, at run end, into
    # the timestamped log folder. The console still prints the cost line for live
    # feedback; the JSON carries the full rollup for dissecting the run.
    cost_line = cost_tracker.report()
    final_verdict = ("provisional" if run_stats.flags.get("provisional_briefing")
                     else "FINAL" if briefing else "none")
    telemetry = build_run_telemetry(
        run_logger, cost_tracker, run_stats, log,
        seed=seed, dataset_shape=(0, 0) if df is None else df.shape,
        models={"investigator": investigator_model, "executor": executor_model,
                "synthesizer": synth_model, "search": search_model},
        max_iters=args.iterations, wall_clock_s=run_wall,
        target_estimand=getattr(nav, "target_estimand", ""),
        final_verdict=final_verdict)
    with open(os.path.join(logs_dir, "run_telemetry.json"), "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2, default=str)
    print()
    print(ui.c("  " + cost_line.replace("\n", "\n  "), "dim"))
    print(ui.c(f"  telemetry: {os.path.join(logs_dir, 'run_telemetry.json')}", "dim"))
    if briefing and verify_ctx is None:
        import verify as _verify
        _verify.write_last_run_pointer(args.output)
    if briefing:
        ui.done(os.path.join(args.output, "briefing.md"))
    else:
        ui.note("No final briefing produced (check logs / nav_state.json).", "yellow")



def _save_seeds(output_dir, seeds):
    """Persist the full seed history as seeds.json (list, original first). Also
    mirror the active (last) seed to seed.txt for human-readability and back-compat."""
    try:
        with open(os.path.join(output_dir, "seeds.json"), "w", encoding="utf-8") as f:
            json.dump(list(seeds), f, indent=2)
        if seeds:
            with open(os.path.join(output_dir, "seed.txt"), "w", encoding="utf-8") as f:
                f.write(seeds[-1])
    except OSError:
        pass


def _load_seeds(output_dir):
    """Return the seed history list. Falls back to a legacy single seed.txt
    (runs created before seeds.json existed) so old output dirs still extend."""
    try:
        with open(os.path.join(output_dir, "seeds.json"), encoding="utf-8") as f:
            seeds = json.load(f)
            if isinstance(seeds, list) and seeds:
                return seeds
    except (OSError, json.JSONDecodeError):
        pass
    legacy = _load_saved_seed(output_dir)
    return [legacy] if legacy else []


def _load_saved_seed(output_dir):
    try:
        with open(os.path.join(output_dir, "seed.txt"), encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def _save_run_meta(output_dir, compute):
    """Persist run-level mode as run_meta.json so --resume/--extend can restore it.
    Written for every fresh run, dataset and compute alike."""
    try:
        with open(os.path.join(output_dir, "run_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"compute": bool(compute)}, f, indent=2)
    except OSError:
        pass


def _load_run_meta(output_dir):
    """Return saved run metadata, or {} if absent (older dirs are dataset runs)."""
    try:
        with open(os.path.join(output_dir, "run_meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
            if isinstance(meta, dict):
                return meta
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _load_saved_state(output_dir):
    from nav_state import NavState
    nav = NavState()
    try:
        with open(os.path.join(output_dir, "nav_state.json"), encoding="utf-8") as f:
            nav = NavState.from_dict(json.load(f))
    except (OSError, json.JSONDecodeError):
        pass
    log = None
    try:
        with open(os.path.join(output_dir, "log.json"), encoding="utf-8") as f:
            log = json.load(f)
    except (OSError, json.JSONDecodeError):
        pass
    history = []
    try:
        with open(os.path.join(output_dir, "kernel_history.json"), encoding="utf-8") as f:
            history = json.load(f)
    except (OSError, json.JSONDecodeError):
        pass
    return nav, log, history


if __name__ == "__main__":
    main()