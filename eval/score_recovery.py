#!/usr/bin/env python3
"""Score planted-truth recovery: compare each briefing against truth.json.

Requires judge.py to have run first (uses its extracted headline JSON), plus
the briefing text for name/direction checks. Purely mechanical from there —
no further API calls.

  python3 eval/score_recovery.py eval/results_planted --truth eval/planted/truth.json

Writes recovery.json per cell and prints a per-condition table:
  planted_ranking      ace_found (the true best driver — short career, modest
                       cars — is named), top_pair_ok (ace and the long-career
                       star both named), star_trap_taken (the star presented
                       while the ace is absent: the volume trap), top5_overlap
  planted_home_adv     direction_pos (positive home effect claimed),
                       interaction_ok (early era claimed larger than late)
  planted_nation_null  null_ok (no confident superior-nation claim; a null or
                       uncertainty-dominated verdict)
  planted_reliability  trend_ok (declining failures), car_link (worse cars
                       fail more, stated)
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict


def load(cell, name):
    p = os.path.join(cell, name)
    return json.load(open(p)) if os.path.exists(p) else None


def briefing_text(cell):
    p = os.path.join(cell, "briefing.md")
    return open(p, encoding="utf-8").read() if os.path.exists(p) else ""


NULL_PAT = re.compile(r"no (credible|significant|systematic|robust|reliable|clear) "
                      r"(national|nation|country|nationality)|"
                      r"(consistent|compatible) with (chance|sampling|noise)|"
                      r"null (result|finding|effect)|no evidence (of|for|that)", re.I)
CONFIDENT_NATION = re.compile(r"(drivers from|nationality|nation|Iberi|Galli|Teuton|"
                              r"Auson|Batav|Nordi|Lusitan|Pannon|Hiberni|Britanni)\w*"
                              r"[^.\n]{0,120}\b(systematically|significantly|clearly|"
                              r"genuinely|robustly)\s+(better|superior|more skilled|"
                              r"out-?perform\w*|over-?perform\w*)", re.I)
DECLINE_PAT = re.compile(r"(declin|improv|fell|dropp|decreas)\w*", re.I)
CAR_LINK_PAT = re.compile(r"(worse|weaker|less competitive|slower|lower[- ]quality) "
                          r"(cars?|teams?|constructors?)[^.\n]{0,80}"
                          r"(fail|retire|dnf|break)", re.I)


def score_cell(cell, sid, truth):
    b = briefing_text(cell)
    j = load(cell, "judge_scores.json") or {}
    head = (j.get("headline") or {})
    hd = ((head.get("direction") or "") + " " + (head.get("claim") or "")).lower()
    out = {}
    if sid == "planted_ranking":
        ace = truth["top_structure"]["ace"]["name"]
        star = truth["top_structure"]["star"]["name"]
        # CONCLUSION-level grading uses the judge's blinded headline extraction
        # when present — measured necessity, not preference: briefings that
        # crowned the wrong driver still MENTIONED the ace somewhere, so pure
        # string presence graded 3/3 clean on runs whose actual verdicts were
        # 0/3 correct. Semantic headline first; strings only as fallback.
        if hd.strip():
            # headlines may abbreviate "Driver_ACE" to "ACE" (live judge output
            # did exactly that and a correct conclusion was graded as a miss):
            # match the full name or its bare suffix as a word.
            def named_in_hd(full):
                suffix = full.split("_", 1)[-1].lower()
                return bool(re.search(rf"{re.escape(full.lower())}|\b{re.escape(suffix)}\b", hd))
            ace_in, star_in = named_in_hd(ace), named_in_hd(star)
            out["ace_concluded"] = ace_in
            out["star_trap_taken"] = star_in and not ace_in
            out["wrong_driver_concluded"] = not ace_in
        else:
            first = b[:800]     # fallback: the briefing's opening verdict
            out["ace_concluded"] = ace in first
            out["star_trap_taken"] = (star in first) and (ace not in first)
            out["wrong_driver_concluded"] = ace not in first
        out["ace_mentioned"] = ace in b
        out["top_pair_ok"] = ace in b and star in b
        out["top5_overlap"] = sum(d in b for d in truth["top5"])
        out["decoy_flagged"] = bool(re.search(
            r"fan[_ ]rating[^.\n]{0,120}(confound|car|unreliab|not .{0,12}skill|"
            r"decoy|circular|proxy|caution|distrust|misleading)", b, re.I))
        out["dupes_caught"] = bool(re.search(
            r"duplicat|double[- ](entered|counted)|season 2[^.\n]{0,60}(twice|2x|double)",
            b, re.I))
    elif sid == "planted_home_adv":
        # Direction: trust the judge's extracted DIRECTION field when present
        # (a correct briefing may legitimately say "no home advantage in the
        # late era" — an era-scoped negation that string matching on the full
        # text misreads as a global null). Strings are the fallback only.
        direction = (load(cell, "judge_scores.json") or {}).get("headline", {}) or {}
        dtext = (direction.get("direction") or "").lower()
        if dtext.strip():
            out["direction_pos"] = bool(
                re.search(r"home[^,;]{0,40}(better|advantage|boost|positive)|"
                          r"(better|advantage|positive)[^,;]{0,40}home", dtext)
                and not re.search(r"^\s*(no |null|none)", dtext))
        else:
            out["direction_pos"] = bool(re.search(
                r"(home|native)[^.\n]{0,100}(advantage|better|boost|improve|gain)", b, re.I)
                or re.search(r"(better|improve\w*|gain|advantage)[^.\n]{0,60}(at|in|during) home", b, re.I))
        early_first = re.search(r"early[^.\n]{0,140}(larger|stronger|bigger|greater|twice|2x|double|concentrated)", b, re.I)
        late_shrunk = re.search(r"(late|recent)[^.\n]{0,140}(smaller|weaker|halv|declin|shrunk|shrink|indistinguishable)", b, re.I)
        out["interaction_ok"] = bool(early_first or late_shrunk)
    elif sid == "planted_nation_null":
        if hd.strip():
            # judge headline: does the extracted verdict lean null or name a
            # nation as genuinely better? (A briefing may DESCRIBE the naive
            # significant result before rejecting it — string matching flags
            # that description; the headline reflects the actual verdict.)
            nullish = bool(re.search(r"null|no (credible|systematic|real|nation)|"
                                     r"chance|noise|not distinguish", hd))
            # both directions are false discoveries against a planted zero:
            # naming a nation as better OR worse (live grok runs claimed
            # "Teutonia underperforms" and escaped the positive-only list)
            named = False
            for m in re.finditer(r"(iberi|galli|teuton|auson|batav|nordi|"
                                 r"lusitan|pannon|hiberni|britanni)\w*"
                                 r"[^,;]{0,60}(better|superior|out-?perform\w*|"
                                 r"over-?perform\w*|under-?perform\w*|higher|"
                                 r"lower|worse|deficit|top|best|advantage)", hd):
                # naming a group is only a false discovery when asserted as
                # real: a naming attributed to chance/artifact in the same
                # clause is CORRECT reporting (a live correct null said
                # "Lusitania's top ranking cannot be distinguished from
                # chance" and was wrongly flagged)
                window = hd[m.start():m.end() + 90]
                if not re.search(r"artifact|chance|noise|cannot be disting|"
                                 r"not (statistically )?significan|n\.s\.|"
                                 r"consistent with (chance|sampling|noise)", window):
                    named = True
                    break
            # Naming specific nations as better OR worse is a false discovery
            # against a planted zero, even when the verdict appends "all
            # others null" (live runs did exactly that).
            out["null_ok"] = nullish and not named
            out["overclaimed_nation"] = named
        else:
            confident = bool(CONFIDENT_NATION.search(b))
            nullish = bool(NULL_PAT.search(b))
            out["null_ok"] = nullish and not confident
            out["overclaimed_nation"] = confident
    elif sid == "planted_reliability":
        out["trend_ok"] = bool(DECLINE_PAT.search(b)) and bool(
            re.search(r"(retire|dnf|fail|mechanical)", b, re.I))
        out["car_link"] = bool(CAR_LINK_PAT.search(b))
        out["dupes_caught"] = bool(re.search(
            r"duplicat|double[- ](entered|counted)|season 2[^.\n]{0,60}(twice|2x|double)",
            b, re.I))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--truth", required=True)
    args = ap.parse_args()
    truth = json.load(open(args.truth))

    agg = defaultdict(lambda: defaultdict(list))
    for cell in sorted(glob.glob(os.path.join(args.results, "*", "*", "rep*"))):
        cond, sid, _ = cell.split(os.sep)[-3:]
        if not sid.startswith("planted_") or not briefing_text(cell):
            continue
        s = score_cell(cell, sid, truth)
        with open(os.path.join(cell, "recovery.json"), "w") as f:
            json.dump(s, f, indent=2)
        for k, v in s.items():
            agg[(cond, sid)][k].append(v)

    print("=== planted-truth recovery (share of runs; top5_overlap is a count /5) ===")
    for (cond, sid), metrics in sorted(agg.items()):
        row = f"{cond} {sid:<22}"
        for k, vs in sorted(metrics.items()):
            m = sum(vs) / len(vs)
            row += f"  {k}={m:.2f}" if isinstance(vs[0], bool) else f"  {k}={m:.1f}"
        print(row + f"   (n={len(next(iter(metrics.values())))})")

    # Composite summary per condition: answer-correctness checks vs the
    # data-hygiene traps (kept separate — they measure different skills).
    ANSWER = {"planted_home_adv": ["direction_pos", "interaction_ok"],
              "planted_nation_null": ["null_ok"],
              "planted_ranking": ["ace_concluded"],
              "planted_reliability": ["trend_ok", "car_link"]}
    HYGIENE = {"planted_ranking": ["decoy_flagged", "dupes_caught"],
               "planted_reliability": ["dupes_caught"]}
    print("\n=== composite per configuration ===")
    for cond in sorted({c for c, _ in agg}):
        a_pass = a_tot = h_pass = h_tot = 0
        for (c, sid), metrics in agg.items():
            if c != cond:
                continue
            for k in ANSWER.get(sid, []):
                if k in metrics:
                    a_pass += sum(metrics[k]); a_tot += len(metrics[k])
            for k in HYGIENE.get(sid, []):
                if k in metrics:
                    h_pass += sum(metrics[k]); h_tot += len(metrics[k])
        tot_p, tot_t = a_pass + h_pass, a_tot + h_tot
        print(f"  {cond}: answers {a_pass}/{a_tot} ({100*a_pass/max(a_tot,1):.0f}%) | "
              f"hygiene traps {h_pass}/{h_tot} ({100*h_pass/max(h_tot,1):.0f}%) | "
              f"combined {tot_p}/{tot_t} ({100*tot_p/max(tot_t,1):.0f}%)")
    print("\nNote: name/direction checks are string-based; skim recovery.json against"
          "\nthe briefings for any cell that looks misgraded before quoting results.")


if __name__ == "__main__":
    main()