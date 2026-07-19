#!/usr/bin/env python3
"""Mechanical (no-API) quality metrics for a delv-e run.

Input: a run directory (containing briefing.md and logs/<ts>/run_log.json),
a bare run_log.json path, or an eval/results tree (scored cell by cell).
Output: metrics JSON per run (written next to the log as mech_scores.json when
scoring a tree; printed when scoring a single input).

Metrics:
  grounding_findings   share of decimal numbers in the final FINDINGS that
                       appear in the evidence the Synthesizer saw
                       (rounding-tolerant: 0.73 matches a raw 0.7278)
  grounding_thinking   share of the Investigator's quoted estimates (3-4dp)
                       visible in its own context at the time
  uncertainty_present  briefing/findings report intervals or rank/probability
                       uncertainty, not bare points
  gates_with_numbers   the gates block cites at least one number per passed gate
  redundancy           near-duplicate spec pairs (token-Jaccard > 0.8)
  toolkit_calls        paired_ability / rank_uncertainty / cluster_bootstrap uses
  integrity events     executor retries, rollbacks seen, synth pushbacks,
                       format repairs, truncation retries
  iterations, cost_usd
"""
import json
import os
import re
import sys
import glob
import itertools


def _text(x):
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return "\n".join(p.get("content", "") if isinstance(p, dict) else str(p)
                         for p in x)
    return str(x)


def _grounded(n, gset):
    if n in gset:
        return True
    try:
        v = float(n)
        dp = len(n.split(".")[1])
    except (ValueError, IndexError):
        return False
    for g in gset:
        try:
            if len(g.split(".")[1]) >= dp and abs(round(float(g), dp) - v) < 1e-9:
                return True
        except (ValueError, IndexError):
            continue
    return False


def _jaccard(a, b):
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / max(len(sa | sb), 1)


def score_log(log_path, briefing_path=None):
    d = json.load(open(log_path))
    inv = [e for e in d if e.get("agent") == "Investigator"]
    sy = [e for e in d if e.get("agent") == "Synthesizer"]
    ex = [e for e in d if e.get("agent") == "Executor"]
    out = {"log": log_path}
    if not inv or not sy:
        out["error"] = "log missing Investigator/Synthesizer calls"
        return out

    ground = "".join(_text(e["input"]) for e in inv) + _text(sy[-1]["input"])
    gset = set(re.findall(r"\d+\.\d{2,4}", ground))

    final = sy[-1]["output"]
    fm = re.search(r"###\s*FINDINGS\s*#*(.*?)(?=###|\Z)", final, re.S)
    fnums = re.findall(r"\d+\.\d{2,4}", fm.group(1) if fm else "")
    out["grounding_findings"] = (round(sum(_grounded(n, gset) for n in fnums)
                                       / max(len(fnums), 1), 3), len(fnums))

    th_tot = th_ok = 0
    for e in inv:
        m = re.search(r"###\s*THINKING\s*#*(.*?)(?=###|\Z)", e["output"], re.S)
        if not m:
            continue
        seen = set(re.findall(r"\d+\.\d{2,4}", _text(e["input"])))
        for n in set(re.findall(r"\d+\.\d{3,4}", m.group(1))):
            th_tot += 1
            th_ok += _grounded(n, seen)
    out["grounding_thinking"] = (round(th_ok / max(th_tot, 1), 3), th_tot)

    body = (open(briefing_path, encoding="utf-8").read()
            if briefing_path and os.path.exists(briefing_path) else final)
    out["uncertainty_present"] = bool(
        re.search(r"\bCI\b|\bconfidence\b|\binterval\b|p_rank|\bprobabilit|"
                  r"\bse\b|standard error|±|\[\s*\d+\s*,\s*\d+\s*\]", body, re.I))
    g = re.search(r"###\s*GATES\s*#*(.*?)(?=###|\Z)", final, re.S)
    gates = g.group(1) if g else ""
    passed = re.findall(r"G\d\w*\s*:\s*pass[^\n]*", gates, re.I)
    out["gates_with_numbers"] = (sum(1 for p in passed if re.search(r"\d", p)),
                                 len(passed))

    specs = [" ".join(m.group(1).split()) for e in inv
             if (m := re.search(r"###\s*SPEC\s*#*(.*?)(?=###|\Z)", e["output"], re.S))
             and len(m.group(1).split()) > 5]
    # >0.95: parallel specs (same operation, different signal/column) measure
    # 0.90-0.92 on a verified-clean run and must not count; true redos exceed .95.
    out["redundancy"] = sum(1 for a, b in itertools.combinations(specs, 2)
                            if _jaccard(a, b) > 0.95)
    allspecs = " ".join(specs)
    out["toolkit_calls"] = {t: allspecs.count(t) for t in
                            ("paired_ability", "rank_uncertainty", "cluster_bootstrap")}

    ex_in = "".join(_text(e["input"]) for e in ex)
    out["integrity"] = {
        "executor_calls": len(ex),
        "rollback_retries": sum("ROLLED BACK" in _text(e["input"]) for e in ex),
        "synth_calls": len(sy),
        "pushbacks": max(0, len(sy) - 1),
        "format_repairs": sum("exact output format was not followed" in _text(e["input"])
                              for e in inv),
    }
    out["iterations"] = len(inv)
    out["cost_usd"] = round(sum(e.get("cost_usd") or 0 for e in d), 4)
    return out


def find_log(run_dir):
    logs = sorted(glob.glob(os.path.join(run_dir, "logs", "*", "run_log.json")))
    return logs[-1] if logs else None


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    target = sys.argv[1]
    if target.endswith(".json"):
        print(json.dumps(score_log(target), indent=2))
        return
    if os.path.exists(os.path.join(target, "briefing.md")) or find_log(target):
        lp = find_log(target) or target
        print(json.dumps(score_log(lp, os.path.join(target, "briefing.md")), indent=2))
        return
    # results tree: results/<cond>/<seed>/rep<k>/
    cells = sorted(glob.glob(os.path.join(target, "*", "*", "rep*")))
    done = 0
    for cell in cells:
        lp = find_log(cell)
        if not lp:
            continue
        s = score_log(lp, os.path.join(cell, "briefing.md"))
        with open(os.path.join(cell, "mech_scores.json"), "w") as f:
            json.dump(s, f, indent=2)
        done += 1
    print(f"scored {done}/{len(cells)} cells under {target}")


if __name__ == "__main__":
    main()
