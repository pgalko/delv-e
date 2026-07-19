#!/usr/bin/env python3
"""Report over 1-3 configurations in a results tree.

- 1 configuration: absolute scorecard (judged axes if present, mechanical
  always, replication agreement).
- 2-3 configurations: scorecard per configuration, then paired per-seed
  deltas of each candidate vs the baseline with bootstrap CIs and the H1/H2
  verdicts from eval/DESIGN.md, plus pairwise win rates if judge pairs exist.
- Works without judge scores (mechanical-only) so a comparison is available
  immediately after runs finish; judged axes appear after eval/judge.py.

  python3 eval/report.py eval/results [--baseline NAME]
"""
import argparse
import glob
import json
import os
import random
from collections import defaultdict

AXES = ("estimand", "confounders", "calibration", "verdict_clarity")


def load_tree(root):
    cells = defaultdict(dict)
    for cell in sorted(glob.glob(os.path.join(root, "*", "*", "rep*"))):
        cond, sid, rep = cell.split(os.sep)[-3:]
        for kind, fn in (("judge", "judge_scores.json"), ("mech", "mech_scores.json")):
            p = os.path.join(cell, fn)
            if os.path.exists(p):
                cells[(cond, sid, rep)][kind] = json.load(open(p))
    return cells


def bootstrap_ci(deltas, n=10000, lo=5, hi=95, seed=7):
    rng = random.Random(seed)
    if not deltas:
        return (float("nan"),) * 3
    means = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas)
                   for _ in range(n))
    return (sum(deltas) / len(deltas), means[n * lo // 100], means[n * hi // 100])


def headline_agreement(cells, cond, sid):
    hs = [v["judge"].get("headline") for (c, s, r), v in cells.items()
          if c == cond and s == sid and "judge" in v]
    hs = [h for h in hs if h]
    if len(hs) < 2:
        return None
    pairs = agree = 0
    for i in range(len(hs)):
        for j in range(i + 1, len(hs)):
            pairs += 1
            d_ok = (hs[i].get("direction") or "").strip().lower() == \
                   (hs[j].get("direction") or "").strip().lower()
            a, b = hs[i].get("key_number"), hs[j].get("key_number")
            n_ok = (a is None and b is None) or (
                isinstance(a, (int, float)) and isinstance(b, (int, float))
                and (a == b == 0 or abs(a - b) <= 0.15 * max(abs(a), abs(b), 1e-9)))
            agree += (d_ok and n_ok)
    return agree / pairs


def judged_means(cells, cond, sid):
    vals = defaultdict(list)
    for (c, s, r), v in cells.items():
        if c == cond and s == sid and "judge" in v:
            for ax in AXES:
                if isinstance(v["judge"].get(ax), (int, float)):
                    vals[ax].append(v["judge"][ax])
    return {ax: sum(xs) / len(xs) for ax, xs in vals.items() if xs}


def scorecard(cells, cond, seeds):
    print(f"\n=== scorecard: {cond} ===")
    have_judge = any("judge" in v for k, v in cells.items() if k[0] == cond)
    if have_judge:
        print(f"{'seed':<22}" + "".join(f"{ax[:10]:>12}" for ax in AXES))
        for sid in seeds:
            m = judged_means(cells, cond, sid)
            if m:
                print(f"{sid:<22}" + "".join(f"{m.get(ax, float('nan')):>12.2f}" for ax in AXES))
        overall = [sum(judged_means(cells, cond, sid).values()) / len(AXES)
                   for sid in seeds if judged_means(cells, cond, sid)]
        if overall:
            print(f"{'MEAN (over seeds)':<22}{sum(overall)/len(overall):>12.2f} (axis-avg)")
        vals = [v for sid in seeds if (v := headline_agreement(cells, cond, sid)) is not None]
        if vals:
            print(f"replication agreement: {sum(vals)/len(vals):.0%} over {len(vals)} seeds")
    ms = [v["mech"] for k, v in cells.items() if k[0] == cond and "mech" in v]
    if ms:
        gf = sum(m["grounding_findings"][0] for m in ms) / len(ms)
        gt = sum(m["grounding_thinking"][0] for m in ms) / len(ms)
        red = sum(m["redundancy"] for m in ms) / len(ms)
        it = sum(m["iterations"] for m in ms) / len(ms)
        cost = sum(m["cost_usd"] for m in ms) / len(ms)
        pb = sum(m["integrity"]["pushbacks"] for m in ms) / len(ms)
        print(f"mechanical: grounding {gf:.2f}/{gt:.2f}  redundancy {red:.1f}  "
              f"pushbacks {pb:.1f}  iters {it:.1f}  ${cost:.2f}/run  (n={len(ms)})")


def compare(cells, base, cand, seeds):
    print(f"\n=== comparison: {cand} vs baseline {base} (delta = {cand} - {base}) ===")
    deltas_overall, deltas_ax = [], defaultdict(list)
    header = f"{'seed':<22}" + "".join(f"{ax[:10]:>12}" for ax in AXES) + f"{'overall':>10}"
    rows = []
    for sid in seeds:
        pa, pb = judged_means(cells, base, sid), judged_means(cells, cand, sid)
        if not pa or not pb:
            continue
        row, ds = f"{sid:<22}", []
        for ax in AXES:
            if ax in pa and ax in pb:
                d = pb[ax] - pa[ax]
                deltas_ax[ax].append(d)
                ds.append(d)
                row += f"{d:>+12.2f}"
            else:
                row += f"{'—':>12}"
        if ds:
            deltas_overall.append(sum(ds) / len(ds))
            row += f"{deltas_overall[-1]:>+10.2f}"
        rows.append(row)
    if rows:
        print(header)
        print("\n".join(rows))
        m, lo, hi = bootstrap_ci(deltas_overall)
        print(f"\nH1 (non-inferiority of {cand}): overall delta = {m:+.2f}, "
              f"90% CI [{lo:+.2f}, {hi:+.2f}] over {len(deltas_overall)} seeds")
        print("  verdict:", "NON-INFERIOR (CI excludes -0.5)" if lo > -0.5 else
              "NOT established (CI reaches -0.5)")
        for ax in ("confounders", "calibration"):
            m, lo, hi = bootstrap_ci(deltas_ax[ax])
            sup = "SUPERIOR (CI excludes 0)" if lo > 0 else "not established"
            print(f"H2 {ax}: delta = {m:+.2f}, 90% CI [{lo:+.2f}, {hi:+.2f}] -> {sup}")
    else:
        print("(no judged scores yet on both sides — mechanical scorecards above; "
              "run eval/judge.py for the blinded comparison)")
    pw = [json.load(open(p)) for p in
          glob.glob(os.path.join(ROOT, "pairwise_*.json"))]
    pw = [j for j in pw if j.get("baseline") == base and j.get("candidate") == cand]
    if pw:
        wins = defaultdict(lambda: defaultdict(int))
        for j in pw:
            for ax in AXES + ("overall",):
                wins[ax][j.get(ax, "tie")] += 1
        print(f"\npairwise ({len(pw)} comparisons):")
        for ax in AXES + ("overall",):
            w = wins[ax]
            print(f"  {ax:<16} {cand}: {w[cand]:>2}  {base}: {w[base]:>2}  tie: {w['tie']:>2}")


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default=os.path.join("eval", "results"))
    ap.add_argument("--baseline", default=None)
    args = ap.parse_args()
    ROOT = args.results
    cells = load_tree(args.results)
    conds = sorted({c for c, _, _ in cells})
    seeds = sorted({s for _, s, _ in cells})
    if not conds:
        print("no scored cells found (run score_mechanical.py / judge.py first)")
        return
    print(f"configurations: {conds}")
    for cond in conds:
        scorecard(cells, cond, seeds)
    if len(conds) >= 2:
        base = args.baseline or conds[0]
        if base not in conds:
            print(f"--baseline {base} not found; using {conds[0]}")
            base = conds[0]
        for cand in [c for c in conds if c != base]:
            compare(cells, base, cand, seeds)


if __name__ == "__main__":
    main()
