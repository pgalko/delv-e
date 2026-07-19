#!/usr/bin/env python3
"""Run the eval matrix over 1-3 named configurations (model setups and/or
repo checkouts) x N seeds x R replicates.

A configuration is a NAME plus a repo path plus model flags. Results
accumulate under <out>/<name>/<seed_id>/rep<k>/ and are resumable, so the
default workflow is incremental:

  # today: score one configuration (the default single-config mode)
  python3 eval/run_matrix.py --repo . --dataset f1.csv --seeds eval/seeds_f1.json \
      --name grok45 --investigator-model openrouter:x-ai/grok-4.5 \
      --executor-model ollama:glm-5.2:cloud --out eval/results

  # next week: same repo, different models, SAME --out
  python3 eval/run_matrix.py --repo . --dataset f1.csv --seeds eval/seeds_f1.json \
      --name opus --investigator-model anthropic:claude-opus-4-8 \
      --executor-model ollama:glm-5.2:cloud --out eval/results
  # -> two configurations now exist; the end-of-run report compares them.

Or declare 2-3 configurations up front and get the comparison in one go:

  python3 eval/run_matrix.py --conditions eval/conditions.json \
      --dataset f1.csv --seeds eval/seeds_f1.json --out eval/results \
      [--judge-model openrouter:strong/judge]

conditions.json:
  [{"name": "grok45", "repo": ".", "investigator_model": "...",
    "executor_model": "...", "synth_model": "..."},
   {"name": "opus",   "repo": ".", "investigator_model": "...", ...}]

After the runs, this script always runs the free mechanical scorer, runs the
judge automatically if --judge-model is given, and prints the report
(absolute scorecard for one configuration; comparison for two or more).
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def build_conditions(args):
    if args.conditions:
        conds = json.load(open(args.conditions))
        if not isinstance(conds, list) or not 1 <= len(conds) <= 3:
            sys.exit("--conditions must be a JSON list of 1-3 configurations")
        for c in conds:
            for k in ("name", "repo", "investigator_model", "executor_model"):
                if k not in c:
                    sys.exit(f"configuration missing '{k}': {c}")
        return conds
    if not (args.repo and args.investigator_model and args.executor_model):
        sys.exit("single-config mode needs --repo, --investigator-model, "
                 "--executor-model (or use --conditions FILE)")
    name = args.name or re.sub(r"[^A-Za-z0-9_.-]+", "-",
                               f"{args.investigator_model}_{args.executor_model}")[:60]
    return [{"name": name, "repo": args.repo,
             "investigator_model": args.investigator_model,
             "executor_model": args.executor_model,
             "synth_model": args.synth_model}]


def dataset_fingerprint(path):
    """Content hash of the dataset actually used. Regenerating the synthetic
    data with different parameters changes this, so a stale results tree can
    never silently mix two datasets under one configuration name."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def check_provenance(out_root, cond, dataset, seeds_path, truth_path=None):
    """A condition name is a contract: the same name must mean the same setup —
    same repo, same models, AND the same data and questions. Mixing any of
    these under one name makes every downstream number uninterpretable."""
    cdir = os.path.join(out_root, cond["name"])
    meta_p = os.path.join(cdir, "condition.json")
    meta = {k: cond.get(k) for k in
            ("name", "repo", "investigator_model", "executor_model", "synth_model")}
    meta["repo"] = os.path.abspath(meta["repo"])
    meta["dataset"] = os.path.basename(dataset)
    meta["dataset_sha"] = dataset_fingerprint(dataset)
    meta["seeds"] = os.path.basename(seeds_path)
    if truth_path and os.path.exists(truth_path):
        try:
            meta["dgp_version"] = json.load(open(truth_path)).get("dgp_version")
        except (ValueError, OSError):
            pass
    if os.path.exists(meta_p):
        prev = json.load(open(meta_p))
        # Keys absent from a legacy condition.json cannot be verified; backfill
        # them and say so rather than failing on the absence itself.
        legacy = [k for k in meta if k not in prev]
        diffs = sorted(k for k in meta if k in prev and prev[k] != meta[k])
        # The repo path legitimately differs across machines (the same checkout
        # synced via Dropbox mounts at different absolute paths on Mac vs
        # Windows); warn, record the new path, but never refuse on it alone.
        if diffs == ["repo"]:
            print(f"note: configuration '{cond['name']}' was recorded from a "
                  f"different repo path ({prev['repo']}); continuing — models, "
                  f"dataset, and seeds all match. Ensure both checkouts carry "
                  f"the same code version.")
            prev["repo"] = meta["repo"]
            with open(meta_p, "w") as f:
                json.dump(prev, f, indent=2)
            diffs = []
        if diffs:
            what = ("dataset" if "dataset_sha" in diffs else "setup")
            was = {k: prev.get(k) for k in diffs}
            now = {k: meta[k] for k in diffs}
            sys.exit(
                f"\nREFUSING TO RUN: configuration '{cond['name']}' already exists in\n"
                f"  {out_root}\nwith a different {what} ({diffs}).\n"
                f"  existing:  {was}\n"
                f"  requested: {now}\n\n"
                f"Mixing these under one name would make the scores uninterpretable.\n"
                f"Use a fresh --out directory (or a new --name).\n")
        if legacy:
            print(f"note: configuration '{cond['name']}' predates provenance fields "
                  f"{legacy}; backfilling from this invocation (earlier cells in this "
                  f"tree were NOT verified against them)")
            prev.update({k: meta[k] for k in legacy})
            with open(meta_p, "w") as f:
                json.dump(prev, f, indent=2)
    else:
        os.makedirs(cdir, exist_ok=True)
        with open(meta_p, "w") as f:
            json.dump(meta, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", help="JSON file with 1-3 configurations")
    ap.add_argument("--repo", help="single-config mode: repo checkout")
    ap.add_argument("--name", help="single-config mode: configuration name")
    ap.add_argument("--investigator-model")
    ap.add_argument("--executor-model")
    ap.add_argument("--synth-model", default=None)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=24)
    ap.add_argument("--out", default=os.path.join("eval", "results"))
    ap.add_argument("--allow-search", action="store_true")
    ap.add_argument("--only-seed", default=None)
    ap.add_argument("--judge-model", default=None,
                    help="if set, run the blinded judge automatically after the runs")
    ap.add_argument("--truth", default=None,
                    help="truth.json for planted seeds; if set, ground-truth "
                         "recovery is scored automatically after the runs")
    ap.add_argument("--baseline", default=None,
                    help="condition name treated as baseline in the comparison "
                         "(default: alphabetically first condition)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    conds = build_conditions(args)
    seeds = json.load(open(args.seeds))["seeds"]
    if args.only_seed:
        seeds = [s for s in seeds if s["id"] == args.only_seed]
    out_root = os.path.abspath(args.out)
    dataset = os.path.abspath(args.dataset)
    os.makedirs(out_root, exist_ok=True)
    for c in conds:
        check_provenance(out_root, c, dataset, args.seeds, args.truth)

    plan, skipped = [], 0
    for s in seeds:                       # interleave conditions within seed
        for rep in range(1, args.reps + 1):
            for c in conds:
                cell = os.path.join(out_root, c["name"], s["id"], f"rep{rep}")
                if os.path.exists(os.path.join(cell, "briefing.md")):
                    skipped += 1
                    continue
                plan.append((c, s, rep, cell))
    print(f"matrix: {len(plan)} cells to run, {skipped} already complete "
          f"({len(conds)} configuration(s): {[c['name'] for c in conds]})")
    if not plan and skipped:
        print("note: every requested cell already has a briefing — nothing to run. "
              "Raise --reps to add replicates, or use a fresh --out for a new dataset.")
    if args.dry_run:
        for c, s, rep, _ in plan:
            print(f"  {c['name']} {s['id']} rep{rep}")
        return

    failures = []
    for i, (c, s, rep, cell) in enumerate(plan, 1):
        os.makedirs(cell, exist_ok=True)
        cmd = [sys.executable, "run_core.py", dataset, s["question"],
               "--iterations", str(args.iterations),
               "--investigator-model", c["investigator_model"],
               "--executor-model", c["executor_model"],
               "--output", cell]
        if c.get("synth_model"):
            cmd += ["--synth-model", c["synth_model"]]
        if not args.allow_search:
            cmd += ["--no-search"]
        print(f"[{i}/{len(plan)}] {c['name']} {s['id']} rep{rep}")
        t0 = time.time()
        # Force UTF-8 in the child regardless of platform: Windows pipes
        # default to a legacy codepage that cannot carry the UI's glyphs
        # (measured crash: banner() under cp1252), and decode the capture as
        # UTF-8 so stderr artifacts stay readable cross-platform.
        env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        r = subprocess.run(cmd, cwd=os.path.abspath(c["repo"]),
                           capture_output=True, encoding="utf-8",
                           errors="replace", env=env)
        ok = os.path.exists(os.path.join(cell, "briefing.md"))
        print(f"    {'ok' if ok else 'FAILED'} in {time.time()-t0:.0f}s")
        if not ok:
            failures.append(cell)
            with open(os.path.join(cell, "runner_stderr.txt"), "w") as f:
                f.write(r.stdout[-4000:] + "\n----\n" + r.stderr[-8000:])
    if failures:
        print(f"\n{len(failures)} failed cells (stderr captured in each):")
        for c in failures:
            print(" ", c)

    # ---- automatic scoring & comparison ----
    print("\n--- mechanical scoring (free) ---")
    subprocess.run([sys.executable, os.path.join(HERE, "score_mechanical.py"),
                    out_root])
    if args.judge_model:
        print("\n--- blinded judge ---")
        jcmd = [sys.executable, os.path.join(HERE, "judge.py"), out_root,
                "--seeds", os.path.abspath(args.seeds),
                "--judge-model", args.judge_model]
        if args.baseline:
            jcmd += ["--baseline", args.baseline]
        subprocess.run(jcmd)
    def run_and_save(title, cmd, fname):
        print(f"\n--- {title} ---")
        r = subprocess.run(cmd, capture_output=True, encoding="utf-8",
                           errors="replace",
                           env=dict(os.environ, PYTHONUTF8="1",
                                    PYTHONIOENCODING="utf-8"))
        out = r.stdout + (("\n" + r.stderr) if r.returncode else "")
        print(out)
        with open(os.path.join(out_root, fname), "w") as f:
            f.write(out)
        print(f"(saved to {os.path.join(out_root, fname)})")

    if args.truth:
        run_and_save("ground-truth recovery",
                     [sys.executable, os.path.join(HERE, "score_recovery.py"),
                      out_root, "--truth", os.path.abspath(args.truth)],
                     "recovery_report.txt")
    rcmd = [sys.executable, os.path.join(HERE, "report.py"), out_root]
    if args.baseline:
        rcmd += ["--baseline", args.baseline]
    run_and_save("report", rcmd, "report.txt")
    if not args.judge_model:
        print("\n(judged axes absent — run eval/judge.py with a strong model, then "
              "eval/report.py again, for the blinded quality comparison)")


if __name__ == "__main__":
    main()