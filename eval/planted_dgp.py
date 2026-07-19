#!/usr/bin/env python3
"""Generate a synthetic F1-like dataset with KNOWN ground truth, for
planted-truth evaluation (eval/seeds_planted.json + eval/score_recovery.py).

The generating process (all parameters written to truth.json):
- D drivers with true latent abilities ~ N(0, 1). Ability is the ONLY
  persistent driver trait.
- Constructors with car quality per (constructor, season), drifting AR(1):
  car quality moves results as much as or more than ability, exactly the
  confound the teammate design must remove.
- Two drivers per constructor per race -> teammate pairing is well-defined
  and paired_ability-style analysis is possible by construction.
- Finish order per race: rank of (ability + car + home_adv * is_home + noise);
  mechanical DNFs at an era-varying, constructor-dependent rate.
- PLANTED effects (each is checkable ground truth):
  * home_advantage: latent boost when driver_nationality == race_country,
    era-dependent: strong in the early era, HALF that in the late era.
  * nationality -> skill: EXACTLY ZERO. Nationalities are assigned
    independently of ability; any "nation X produces better drivers" claim
    is a false discovery (the trap: 4-9 drivers per nation makes naive group
    means spread widely). Calibrated: true-ability ANOVA across nations
    p=0.38 on the default seed.
  * reliability: mechanical-DNF rate declines 0.22 -> 0.10 across seasons,
    plus a worse-car -> more-DNFs term.
  NOTE a momentum/hot-streak null was considered and rejected: lagged-outcome
  regressors are biased under both naive controls (errors-in-variables, +)
  and fixed effects (Nickell, -), so a "find the null" grade would punish
  correct-but-standard methodology. Do not add dynamic-null seeds casually.
- Grid position: same latent + independent noise (a second, noisier signal).

Usage:
  python3 eval/planted_dgp.py --out eval/planted --rows-target 24000 --seed 11
Writes: eval/planted/synth_f1.csv and eval/planted/truth.json
"""
import argparse
import json
import os
import numpy as np
import pandas as pd

NATIONS = ["Britannia", "Gallia", "Iberia", "Teutonia", "Ausonia", "Batavia",
           "Nordia", "Lusitania", "Pannonia", "Hibernia"]


def generate(seed=17, n_drivers=60, n_constructors=10, n_seasons=25,
             races_per_season=20, home_adv_a=1.2, home_adv_b=0.25,
             ability_sd=1.0, car_sd=1.2, race_noise_sd=1.6,
             ace_ability=2.0, star_ability=1.6, ace_seasons=6,
             seat_ability_corr=True, dnf_car_coef=0.045):
    """Difficulty is PLANTED, not left to the ability-draw lottery:
    - ace: the true best driver (+ace_ability), but a SHORT career
      (ace_seasons) in mid-tier cars -> few contests, wide uncertainty.
    - star: nearly as good (+star_ability), a LONG career that gravitates to
      the best cars -> maximum visibility. Naive methods crown the star;
      a correct analysis finds the ace (or reports the ace/star top pair)
      with the ace's thin record flagged.
    - seat_ability_corr: better drivers tend to get better cars, so raw
      results are confounded by construction; only within-team designs
      identify skill.
    The remaining draws stay random; these two names are the graded top."""
    rng = np.random.default_rng(seed)
    drivers = [f"Driver_{i:02d}" for i in range(n_drivers)]
    ability = dict(zip(drivers, rng.normal(0, ability_sd, n_drivers)))
    nat = dict(zip(drivers, rng.choice(NATIONS, n_drivers)))
    constructors = [f"Team_{c}" for c in "ABCDEFGHIJ"[:n_constructors]]
    # plant ace and star above the field
    ordered = sorted(ability, key=ability.get, reverse=True)
    field_max = ability[ordered[2]] if len(ordered) > 2 else 0.0
    ace, star = "Driver_ACE", "Driver_STAR"
    drivers += [ace, star]
    ability[ace] = max(ace_ability, field_max + 0.6)
    ability[star] = max(star_ability, field_max + 0.3)
    nat[ace], nat[star] = rng.choice(NATIONS, 2)

    # car quality AR(1) per constructor across seasons
    car = {}
    for c in constructors:
        q = rng.normal(0, car_sd)
        for s in range(1, n_seasons + 1):
            q = 0.8 * q + rng.normal(0, car_sd * 0.6)
            car[(c, s)] = q

    # seat assignment: 2 drivers per constructor per season, reshuffled with
    # some persistence so careers span teams (needed for graph connectivity).
    # With seat_ability_corr, seats are filled best-car-first from an
    # ability-sorted (noisily) candidate list: skill and machinery correlate,
    # as in reality, so raw results are confounded by construction.
    rows = []
    base_pool = [d for d in drivers if d not in ("Driver_ACE", "Driver_STAR")]
    seats_prev = None
    era_mid = n_seasons // 2
    star_span = set(range(3, min(n_seasons, 3 + 18)))          # long career
    ace_start = max(4, era_mid - ace_seasons // 2)
    ace_span = set(range(ace_start, ace_start + ace_seasons))  # short career
    for s in range(1, n_seasons + 1):
        pool = base_pool[:]
        rng.shuffle(pool)
        active = pool[: 2 * n_constructors]
        if seat_ability_corr:
            noisy = {d: ability[d] + rng.normal(0, 0.8) for d in active}
            active = sorted(active, key=lambda d: -noisy[d])
        team_order = sorted(constructors, key=lambda c: -car[(c, s)])
        seats = {c: [active[2 * i], active[2 * i + 1]]
                 for i, c in enumerate(team_order)}
        if seats_prev and rng.random() < 0.6:
            # partial persistence: 60% of seasons keep ~2/3 of last year's seats
            keep = rng.choice(constructors, max(1, 2 * n_constructors // 3),
                              replace=False)
            placed = set()
            for c in keep:
                prev = [d for d in seats_prev.get(c, []) if d in set(active)]
                for k, d in enumerate(prev[:2]):
                    seats[c][k] = d
                    placed.add(d)
            leftovers = [d for d in active if d not in placed
                         and sum(d in p for p in seats.values()) == 0]
            for c in constructors:
                for k in range(2):
                    if sum(seats[c][k] in p for p in seats.values()) > 1 and leftovers:
                        seats[c][k] = leftovers.pop()
        # plant the careers: star rides a top-2 car through a long span,
        # the ace a mid-grid car for a short span (bumping the incumbent)
        if s in star_span:
            c_star = team_order[int(rng.integers(0, 2))]
            seats[c_star][0] = "Driver_STAR"
        if s in ace_span:
            c_ace = team_order[n_constructors // 2]
            seats[c_ace][0] = "Driver_ACE"
        seats_prev = seats
        countries = rng.choice(NATIONS, races_per_season)
        for r in range(1, races_per_season + 1):
            country = countries[r - 1]
            era = "early" if s <= era_mid else "late"
            h = home_adv_a if era == "early" else home_adv_b
            latent, entries = {}, []
            for c, (d1, d2) in seats.items():
                for d in (d1, d2):
                    home = int(nat[d] == country)
                    latent[d] = (ability[d] + car[(c, s)] + h * home
                                 + rng.normal(0, race_noise_sd))
                    entries.append((d, c, home))
            grid_latent = {d: ability[d] + car[dict((x[0], x[1]) for x in entries)[d], s]
                           + rng.normal(0, race_noise_sd * 1.3) for d, _, _ in entries}
            grid_rank = {d: i + 1 for i, d in
                         enumerate(sorted(grid_latent, key=grid_latent.get, reverse=True))}
            fin_rank = {d: i + 1 for i, d in
                        enumerate(sorted(latent, key=latent.get, reverse=True))}
            for d, c, home in entries:
                dnf_p = 0.22 - 0.12 * (s / n_seasons) + max(0, -car[(c, s)]) * dnf_car_coef
                dnf = rng.random() < dnf_p
                rows.append({
                    "season": s, "round": r, "race_country": country,
                    "driver": d, "constructor": c,
                    "driver_nationality": nat[d], "is_home_race": home,
                    "grid_position": grid_rank[d],
                    "finish_position": np.nan if dnf else fin_rank[d],
                    "status": "mechanical_dnf" if dnf else "classified",
                    "era_label": era,
                })
    df = pd.DataFrame(rows)

    # ---- data traps (each is recorded in truth.json) ----
    # 1) DECOY column: a per-driver "fan_rating" that actually tracks the
    #    quality of cars the driver sat in (plus noise) — a tempting one-line
    #    shortcut to "skill" that is confounded by construction. An analysis
    #    that trusts it inherits the volume/star trap.
    car_exposure = (df.assign(cq=[car[(c, s)] for c, s in
                                  zip(df.constructor, df.season)])
                      .groupby("driver").cq.mean())
    fan = 70 + 10 * car_exposure + rng.normal(0, 2.0, len(car_exposure))
    fan_map = dict(zip(car_exposure.index, np.round(fan, 1)))
    df["fan_rating"] = df.driver.map(fan_map)
    # 2) LEAKY column: career totals constant on every row of a driver —
    #    future information relative to any given race, and a volume metric.
    podium = (df.finish_position <= 3).groupby(df.driver).sum()
    df["career_podiums_total"] = df.driver.map(podium).astype(int)
    # 3) DUPLICATED season: every row of season 2 appears twice (a classic
    #    ingestion fault). Entry counts for that season are visibly 2x; naive
    #    aggregates over-weight it. A data-profiling pass catches this.
    dup = df[df.season == 2]
    df = pd.concat([df, dup], ignore_index=True)
    df = df.sample(frac=1.0, random_state=int(rng.integers(1e9))).reset_index(drop=True)

    truth = {
        "dgp_version": 2,   # v2: era interaction re-powered (v1's 2x latent
                            # ratio compressed to ~5pp of win probability and
                            # was swamped by sampling noise at ~200 home
                            # contests per era; interaction grading against a
                            # v1 dataset is invalid)
        "generator_seed": seed,
        "true_abilities": {d: round(float(a), 4) for d, a in
                           sorted(ability.items(), key=lambda kv: -kv[1])},
        "top_driver": max(ability, key=ability.get),
        "top_pair": ["Driver_ACE", "Driver_STAR"],
        "top_structure": {
            "ace": {"name": "Driver_ACE", "ability": round(ability["Driver_ACE"], 3),
                    "career": "short (%d seasons), mid-grid cars" % ace_seasons},
            "star": {"name": "Driver_STAR", "ability": round(ability["Driver_STAR"], 3),
                     "career": "long, top cars"},
            "grading": "correct = ace named best (with uncertainty over the thin "
                       "record) or ace+star reported as the inseparable-or-close top "
                       "pair; wrong = star declared uniquely best on volume/results"},
        "top5": [d for d, _ in sorted(ability.items(), key=lambda kv: -kv[1])[:5]],
        "home_advantage_latent": {"early": home_adv_a, "late": home_adv_b,
                                  "units": "latent performance (sd of ability = 1); "
                                           "detectable as a finish/grid position shift"},
        "nationality_effect": 0.0,
        "nationality_truth": "no systematic nation->skill effect exists; observed "
                             "group differences are sampling noise over 4-9 drivers "
                             "per nation; correct conclusion is a null / "
                             "uncertainty-dominated answer",
        "reliability_trend": {"dnf_rate_season1": 0.22, "dnf_rate_final": 0.10,
                              "car_quality_link": "worse cars fail more"},
        "era_interaction": "home advantage in the early era is 2x the late era",
        "data_traps": {
            "fan_rating": "DECOY - tracks average car quality of the driver's "
                          "seats, not skill; trusting it inherits the star trap",
            "career_podiums_total": "LEAK - per-driver career total on every "
                                    "row; future information for any per-race "
                                    "analysis and a pure volume metric",
            "duplicated_season": 2,
            "duplicated_season_note": "all season-2 rows appear exactly twice "
                                      "(ingestion fault); counts for that "
                                      "season are 2x and naive aggregates "
                                      "over-weight it"},
        "notes": "race noise sd %.1f, car sd %.1f: car quality dominates single results; "
                 "teammate designs are required and sufficient." % (race_noise_sd, car_sd),
    }
    return df, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("eval", "planted"))
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--seasons", type=int, default=25)
    args = ap.parse_args()
    df, truth = generate(seed=args.seed, n_seasons=args.seasons)
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "synth_f1.csv"), index=False)
    with open(os.path.join(args.out, "truth.json"), "w") as f:
        json.dump(truth, f, indent=2)
    print(f"wrote {len(df):,} rows -> {args.out}/synth_f1.csv (+ truth.json)")
    print("KEEP truth.json OUT of the dataset directory given to runs.")


if __name__ == "__main__":
    main()
