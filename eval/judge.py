#!/usr/bin/env python3
"""Blinded judge for the delv-e quality A/B.

Reads an eval/results tree (results/<cond>/<seed>/rep<k>/briefing.md), scores
each briefing on the four axes (absolute, anchored 0-5), extracts its headline
claim as structured JSON, and runs pairwise comparisons between conditions for
the same seed/replicate in randomized order. Writes judge_scores.json per cell
and pairwise_<seed>_rep<k>.json at the tree root.

The judge sees ONLY the seed question, the seed's confounder checklist, and
briefing text. Never condition names, paths, telemetry, or costs.

Usage:
  python3 eval/judge.py eval/results --seeds eval/seeds_f1.json \
      --judge-model openrouter:some/strong-model [--pairwise-only]

Uses the repo's LLMClient; requires the same env/credentials as a normal run.
"""
import argparse
import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The pipeline's entry point (run_core.py) loads .env itself; the judge is a
# separate entry point and must do the same, or provider keys defined only in
# the repo's .env are invisible here (the exact failure: judge crashes with
# "ANTHROPIC_API_KEY not found" while runs worked fine).
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
    load_dotenv()  # also honor a CWD .env, matching run_core's behavior
except ImportError:
    pass
from llm import LLMClient  # noqa: E402

ABS_PROMPT = """You are grading a data-analysis briefing against the question it was asked to answer. Grade ONLY what is on the page; do not reward confidence or length.

QUESTION:
{question}

ANALYSIS REQUIREMENTS for this question (a competent analysis must address these):
{checklist}

BRIEFING:
<<<
{briefing}
>>>

Score each axis 0-5 using the anchors. Respond ONLY with JSON, no prose:

{{"estimand": <0-5>, "confounders": <0-5>, "calibration": <0-5>, "verdict_clarity": <0-5>,
  "headline": {{"claim": "<one sentence>", "direction": "<e.g. 'Verstappen highest' / 'positive effect' / 'null'>",
               "key_number": <number or null>, "conditions": "<the population/era/pool the claim holds under>"}},
  "notes": "<one sentence on the single biggest weakness>"}}

Anchors:
- estimand: 5 = answers exactly the question asked (right target quantity, right population); 3 = answers a related but shifted question; 0 = answers something else.
- confounders: 5 = every listed requirement addressed with evidence; 3 = about half addressed; 0 = confounded analysis presented as clean.
- calibration: 5 = uncertainty quantified where it matters, limits stated, no overclaiming; 3 = some intervals but overreaching conclusions; 0 = bare point estimates presented as certain.
- verdict_clarity: 5 = a decision-ready answer with its conditions; 3 = hedged summary the reader must decode; 0 = no usable answer."""

EXTRACT_PROMPT = """Read this data-analysis briefing and state what it actually CONCLUDED, in one structured line. Do not grade it.

QUESTION IT ANSWERED:
{question}

BRIEFING:
<<<
{briefing}
>>>

Respond ONLY with JSON, no prose:
{{"headline": {{"claim": "<one sentence: the briefing's main answer>",
              "direction": "<e.g. 'Driver_X highest' / 'positive effect, larger in early era' / 'null'>",
              "key_number": <number or null>,
              "conditions": "<the population/era/pool the claim holds under>"}}}}"""

PAIR_PROMPT = """Two independent analysis briefings answered the same question. Judge which is the better ANSWER TO THE QUESTION on each axis. Do not reward length or confidence.

QUESTION:
{question}

ANALYSIS REQUIREMENTS:
{checklist}

BRIEFING 1:
<<<
{b1}
>>>

BRIEFING 2:
<<<
{b2}
>>>

Respond ONLY with JSON:
{{"estimand": <1|2|0 for tie>, "confounders": <1|2|0>, "calibration": <1|2|0>, "verdict_clarity": <1|2|0>, "overall": <1|2|0>, "reason": "<one sentence>"}}"""


def _parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else None


def _call(client, model, prompt):
    r = client.call([{"role": "user", "content": prompt}], model,
                    max_tokens=1200, temperature=0, agent="Judge")
    return r[0] if isinstance(r, tuple) else r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--seeds", default=os.path.join(os.path.dirname(__file__), "seeds_f1.json"))
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--pairwise-only", action="store_true")
    ap.add_argument("--mode", choices=("auto", "full", "extract"), default="auto",
                    help="full = rubric scores + headline; extract = headline only "
                         "(cheap; all the truth-grading needs). auto = extract for "
                         "seeds whose id starts with 'planted_', full otherwise.")
    ap.add_argument("--baseline", default=None,
                    help="condition name to pair others against (default: alphabetically first)")
    ap.add_argument("--pair-seed", type=int, default=17, help="RNG seed for order randomization")
    args = ap.parse_args()

    seeds = {s["id"]: s for s in json.load(open(args.seeds))["seeds"]}
    client = LLMClient()
    rng = random.Random(args.pair_seed)

    cells = sorted(glob.glob(os.path.join(args.results, "*", "*", "rep*")))
    conds = sorted({c.split(os.sep)[-3] for c in cells})

    if not args.pairwise_only:
        for cell in cells:
            bp = os.path.join(cell, "briefing.md")
            outp = os.path.join(cell, "judge_scores.json")
            if not os.path.exists(bp) or os.path.exists(outp):
                continue
            sid = cell.split(os.sep)[-2]
            s = seeds[sid]
            extract_only = (args.mode == "extract" or
                            (args.mode == "auto" and sid.startswith("planted_")))
            if extract_only:
                prompt = EXTRACT_PROMPT.format(
                    question=s["question"],
                    briefing=open(bp, encoding="utf-8").read()[:60000])
            else:
                prompt = ABS_PROMPT.format(
                    question=s["question"],
                    checklist="\n".join(f"- {c}" for c in s["checklist"]),
                    briefing=open(bp, encoding="utf-8").read()[:60000])
            parsed = _parse_json(_call(client, args.judge_model, prompt))
            if parsed:
                with open(outp, "w") as f:
                    json.dump(parsed, f, indent=2)
                print("scored", cell)

    # Pairwise: each non-baseline configuration against the baseline, per
    # seed/replicate, in randomized presentation order.
    if len(conds) >= 2:
        base = args.baseline or conds[0]
        if base not in conds:
            sys.exit(f"--baseline {base} not among conditions {conds}")
        for cand in [c for c in conds if c != base]:
            for cell_a in sorted(glob.glob(os.path.join(args.results, base, "*", "rep*"))):
                sid, rep = cell_a.split(os.sep)[-2], cell_a.split(os.sep)[-1]
                cell_b = os.path.join(args.results, cand, sid, rep)
                pa, pb = os.path.join(cell_a, "briefing.md"), os.path.join(cell_b, "briefing.md")
                outp = os.path.join(args.results, f"pairwise_{base}__{cand}__{sid}_{rep}.json")
                if not (os.path.exists(pa) and os.path.exists(pb)) or os.path.exists(outp):
                    continue
                s = seeds[sid]
                flip = rng.random() < 0.5
                b1, b2 = (pb, pa) if flip else (pa, pb)
                prompt = PAIR_PROMPT.format(question=s["question"],
                                            checklist="\n".join(f"- {c}" for c in s["checklist"]),
                                            b1=open(b1, encoding="utf-8").read()[:50000],
                                            b2=open(b2, encoding="utf-8").read()[:50000])
                parsed = _parse_json(_call(client, args.judge_model, prompt))
                if parsed:
                    mapping = {0: "tie", 1: (cand if flip else base), 2: (base if flip else cand)}
                    result = {ax: mapping.get(parsed.get(ax), "tie")
                              for ax in ("estimand", "confounders", "calibration",
                                         "verdict_clarity", "overall")}
                    result["baseline"], result["candidate"] = base, cand
                    result["reason"] = parsed.get("reason", "")
                    result["presented_first"] = cand if flip else base
                    with open(outp, "w") as f:
                        json.dump(result, f, indent=2)
                    print("paired", base, "vs", cand, sid, rep, "->", result["overall"])


if __name__ == "__main__":
    main()
