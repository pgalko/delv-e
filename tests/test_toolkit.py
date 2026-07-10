"""
Known-answer tests for toolkit.py: the three vetted estimators and their
kernel preload. Pure numpy/pandas; no network, no API keys.

Run from the repo root:  python3 tests/test_toolkit.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from toolkit import paired_ability, cluster_bootstrap, rank_uncertainty

PASS = []


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name}: {detail}")
    PASS.append(name)


# --------------------------------------------------------------------------
# paired_ability: margin network, exact recovery on noiseless data
# --------------------------------------------------------------------------

def test_margin_exact():
    true = {"a": 3.0, "b": 1.5, "c": 0.0, "d": -2.0}
    rows = []
    ents = list(true)
    for i in range(len(ents)):
        for j in range(len(ents)):
            if i != j:
                for _ in range(4):
                    rows.append((ents[i], ents[j],
                                 true[ents[i]] - true[ents[j]]))
    df = pd.DataFrame(rows, columns=["A", "B", "margin"])
    res = paired_ability(df, "A", "B", margin_col="margin", ref="c")
    got = dict(zip(res["entity"], res["ability"]))
    for e, v in true.items():
        check("margin_exact_recovery", abs(got[e] - v) < 1e-8,
              f"{e}: got {got[e]}, want {v}")
    check("margin_exact_ref_zero", abs(got["c"]) < 1e-12)
    check("margin_exact_se_zero",
          float(res.loc[res.entity != "c", "se"].max()) < 1e-6,
          "noiseless data should give ~zero SEs")
    check("margin_exact_order",
          list(res["entity"]) == ["a", "b", "c", "d"],
          f"got order {list(res['entity'])}")
    check("margin_exact_attrs", res.attrs["model"] == "margin_ols"
          and res.attrs["reference"] == "c")


def test_margin_noisy_ci():
    rng = np.random.default_rng(7)
    true = {"a": 2.0, "b": 0.5, "c": 0.0}
    rows = []
    for i in "abc":
        for j in "abc":
            if i != j:
                for _ in range(60):
                    rows.append((i, j, true[i] - true[j] + rng.normal(0, 1.0)))
    df = pd.DataFrame(rows, columns=["A", "B", "m"])
    res = paired_ability(df, "A", "B", margin_col="m", ref="c")
    row_a = res[res.entity == "a"].iloc[0]
    check("margin_noisy_ci_covers",
          row_a.ci_low < true["a"] < row_a.ci_high,
          f"CI [{row_a.ci_low:.3f}, {row_a.ci_high:.3f}] misses {true['a']}")
    check("margin_noisy_se_positive", row_a.se > 0)


# --------------------------------------------------------------------------
# paired_ability: Bradley-Terry recovery on simulated contests
# --------------------------------------------------------------------------

def test_bradley_terry():
    rng = np.random.default_rng(11)
    true_beta = {"p": 2.0, "q": 1.0, "r": 0.0, "s": -1.0}
    ents = list(true_beta)
    rows = []
    for i in range(len(ents)):
        for j in range(i + 1, len(ents)):
            a, b = ents[i], ents[j]
            pa = 1.0 / (1.0 + np.exp(-(true_beta[a] - true_beta[b])))
            for _ in range(300):
                rows.append((a, b, int(rng.random() < pa)))
    df = pd.DataFrame(rows, columns=["A", "B", "a_won"])
    res = paired_ability(df, "A", "B", win_col="a_won", ref="r")
    got = dict(zip(res["entity"], res["ability"]))
    check("bt_order", list(res["entity"]) == ["p", "q", "r", "s"],
          f"got order {list(res['entity'])}")
    for e, v in true_beta.items():
        check("bt_recovery", abs(got[e] - v) < 0.35,
              f"{e}: got {got[e]:.3f}, want {v}")
    check("bt_ref_zero", abs(got["r"]) < 1e-9)
    p_row = res[res.entity == "p"].iloc[0]
    check("bt_ci_covers", p_row.ci_low < true_beta["p"] < p_row.ci_high)
    check("bt_attrs", res.attrs["model"] == "bradley_terry")


def test_bt_perfect_record_finite():
    # an entity that wins every contest must not blow up to infinity
    df = pd.DataFrame([("x", "y", 1)] * 10 + [("y", "z", 1)] * 10,
                      columns=["A", "B", "w"])
    res = paired_ability(df, "A", "B", win_col="w", ref="z")
    check("bt_perfect_finite", np.isfinite(res["ability"]).all(),
          f"abilities {res['ability'].tolist()}")


def test_bt_ties():
    # all-tie pair: half-win convention must leave the two abilities equal
    df = pd.DataFrame([("x", "y", 0.5)] * 20, columns=["A", "B", "w"])
    res = paired_ability(df, "A", "B", win_col="w", ref="x")
    got = dict(zip(res["entity"], res["ability"]))
    check("bt_ties_equal", abs(got["x"] - got["y"]) < 1e-6,
          f"x={got['x']}, y={got['y']}")
    # ties soften a one-sided record: A beating B 10-0 must give A a larger
    # edge than A beating B 10-0 with 10 extra ties mixed in
    one_sided = pd.DataFrame([("a", "b", 1)] * 10, columns=["A", "B", "w"])
    softened = pd.DataFrame([("a", "b", 1)] * 10 + [("a", "b", 0.5)] * 10,
                            columns=["A", "B", "w"])
    e1 = paired_ability(one_sided, "A", "B", win_col="w", ref="b")
    e2 = paired_ability(softened, "A", "B", win_col="w", ref="b")
    a1 = float(e1.loc[e1.entity == "a", "ability"].iloc[0])
    a2 = float(e2.loc[e2.entity == "a", "ability"].iloc[0])
    check("bt_ties_soften", 0 < a2 < a1, f"pure={a1:.3f}, with ties={a2:.3f}")
    # ties alone connect the comparison graph
    chain = pd.DataFrame([("p", "q", 1)] * 6 + [("q", "r", 0.5)] * 6,
                         columns=["A", "B", "w"])
    rc = paired_ability(chain, "A", "B", win_col="w", ref="r")
    check("bt_ties_connect", set(rc["entity"]) == {"p", "q", "r"})
    # invalid fractional values other than 0.5 still raise
    badf = pd.DataFrame({"A": ["a"], "B": ["b"], "w": [0.7]})
    try:
        paired_ability(badf, "A", "B", win_col="w")
        raise AssertionError("0.7 win value did not raise")
    except ValueError as e:
        check("bt_ties_invalid_rejected", "0.5" in str(e) and "tie" in str(e),
              str(e))


# --------------------------------------------------------------------------
# paired_ability: connectivity and validation errors
# --------------------------------------------------------------------------

def test_connectivity():
    rows = ([("a", "b", 1.0)] * 5 + [("b", "c", 0.5)] * 5
            + [("x", "y", 2.0)] * 9)  # island {x,y} smaller than {a,b,c}
    df = pd.DataFrame(rows, columns=["A", "B", "m"])
    res = paired_ability(df, "A", "B", margin_col="m")
    check("connectivity_keeps_largest",
          set(res["entity"]) == {"a", "b", "c"},
          f"kept {set(res['entity'])}")
    check("connectivity_reports_dropped",
          res.attrs["dropped_disconnected"] == ["x", "y"])


def test_errors_instructive():
    df = pd.DataFrame({"A": ["a"], "B": ["b"], "m": [1.0]})
    try:
        paired_ability(df, "A", "WRONG", margin_col="m")
        raise AssertionError("missing column did not raise")
    except ValueError as e:
        check("error_lists_columns", "available columns" in str(e), str(e))
    try:
        paired_ability(df, "A", "B")
        raise AssertionError("neither margin nor win did not raise")
    except ValueError as e:
        check("error_one_of", "exactly one" in str(e), str(e))
    try:
        paired_ability(df, "A", "B", margin_col="m", win_col="m")
        raise AssertionError("both margin and win did not raise")
    except ValueError as e:
        check("error_both", "exactly one" in str(e), str(e))
    try:
        paired_ability(pd.DataFrame({"A": ["a"], "B": ["b"], "w": [2]}),
                       "A", "B", win_col="w")
        raise AssertionError("non-binary win col did not raise")
    except ValueError as e:
        check("error_nonbinary_win", "0, 1, or" in str(e), str(e))


# --------------------------------------------------------------------------
# cluster_bootstrap: scalar stat, Series stat, warning, reproducibility
# --------------------------------------------------------------------------

def _clustered_frame(n_clusters=20, per=30, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_clusters):
        mu = rng.normal(5.0, 2.0)  # strong cluster effect
        for _ in range(per):
            rows.append((f"c{c}", mu + rng.normal(0, 0.5)))
    return pd.DataFrame(rows, columns=["cluster", "y"])


def test_cluster_bootstrap_scalar():
    df = _clustered_frame()
    res = cluster_bootstrap(df, "cluster", lambda d: d["y"].mean(),
                            n_boot=400, seed=1)
    check("cb_estimate_matches", abs(res["estimate"] - df["y"].mean()) < 1e-12)
    check("cb_ci_brackets", res["ci_low"] < res["estimate"] < res["ci_high"])
    check("cb_truth_covered", res["ci_low"] < 5.0 < res["ci_high"],
          f"[{res['ci_low']:.2f}, {res['ci_high']:.2f}]")
    check("cb_n_clusters", res["n_clusters"] == 20)
    check("cb_no_warning", res["warning"] is None)
    check("cb_wider_than_naive",
          (res["ci_high"] - res["ci_low"])
          > 2 * 1.96 * df["y"].std() / np.sqrt(len(df)),
          "cluster CI should exceed the naive iid CI under cluster effects")
    res2 = cluster_bootstrap(df, "cluster", lambda d: d["y"].mean(),
                             n_boot=400, seed=1)
    check("cb_seed_reproducible", res2["ci_low"] == res["ci_low"]
          and res2["ci_high"] == res["ci_high"])


def test_cluster_bootstrap_small_cluster_warning():
    df = _clustered_frame(n_clusters=5)
    res = cluster_bootstrap(df, "cluster", lambda d: d["y"].mean(),
                            n_boot=200, seed=2)
    check("cb_small_warns", res["warning"] is not None
          and "5 clusters" in res["warning"], str(res["warning"]))


def test_cluster_bootstrap_series_and_compose():
    rng = np.random.default_rng(5)
    rows = []
    true_mu = {"alpha": 6.0, "beta": 5.0, "gamma": 1.0}
    for c in range(24):
        for ent, mu in true_mu.items():
            for _ in range(8):
                rows.append((f"c{c}", ent, mu + rng.normal(0, 1.5)))
    df = pd.DataFrame(rows, columns=["cluster", "entity", "y"])
    res = cluster_bootstrap(df, "cluster",
                            lambda d: d.groupby("entity")["y"].mean(),
                            n_boot=300, seed=4)
    check("cb_series_estimate", isinstance(res["estimate"], pd.Series))
    check("cb_series_draws_shape",
          res["draws"].shape == (300, 3), str(res["draws"].shape))
    # compose with rank_uncertainty mode B
    ranks = rank_uncertainty(draws=res["draws"])
    top = ranks.iloc[0]
    check("compose_leader", top.entity == "alpha", str(ranks))
    check("compose_p_sane", 0.5 < top.p_rank1 <= 1.0, f"{top.p_rank1}")
    check("compose_p_sums",
          abs(ranks["p_rank1"].sum() - 1.0) < 1e-9,
          f"sum {ranks['p_rank1'].sum()}")
    check("compose_loser_low",
          float(ranks[ranks.entity == "gamma"]["p_rank1"].iloc[0]) < 0.01)


def test_cluster_bootstrap_errors():
    df = _clustered_frame()
    try:
        cluster_bootstrap(df, "nope", lambda d: 0.0)
        raise AssertionError("bad cluster col did not raise")
    except ValueError as e:
        check("cb_error_lists_columns", "available columns" in str(e), str(e))
    try:
        cluster_bootstrap(df, "cluster", lambda d: 0.0, n_boot=10)
        raise AssertionError("tiny n_boot did not raise")
    except ValueError as e:
        check("cb_error_n_boot", "at least 100" in str(e), str(e))
    try:
        cluster_bootstrap(df, "cluster",
                          lambda d: 1 / 0, n_boot=200)
        raise AssertionError("always-failing stat did not raise")
    except RuntimeError as e:
        check("cb_error_reports_last", "ZeroDivisionError" in str(e), str(e))


# --------------------------------------------------------------------------
# rank_uncertainty: mode A separation, direction flag, validation
# --------------------------------------------------------------------------

def test_rank_uncertainty_normal_mode():
    est = pd.DataFrame({
        "who": ["lead", "mid", "tail"],
        "est": [10.0, 5.0, 0.0],
        "se": [0.5, 0.5, 0.5],
    })
    r = rank_uncertainty(estimates=est, est_col="est", se_col="se",
                         entity_col="who", seed=9)
    check("ru_clear_leader",
          r.iloc[0].entity == "lead" and r.iloc[0].p_rank1 > 0.999, str(r))
    check("ru_rank_interval_tight",
          r.iloc[0].rank_ci_low == 1.0 and r.iloc[0].rank_ci_high == 1.0)
    flipped = rank_uncertainty(estimates=est, est_col="est", se_col="se",
                               entity_col="who", higher_is_better=False,
                               seed=9)
    check("ru_direction_flag",
          flipped.iloc[0].entity == "tail" and flipped.iloc[0].p_rank1 > 0.999)

    near = pd.DataFrame({"who": ["x", "y"], "est": [1.0, 1.0],
                         "se": [1.0, 1.0]})
    rt = rank_uncertainty(estimates=near, est_col="est", se_col="se",
                          entity_col="who", seed=2)
    check("ru_tie_near_half", abs(float(rt["p_rank1"].iloc[0]) - 0.5) < 0.03,
          str(rt))


def test_rank_uncertainty_errors():
    est = pd.DataFrame({"e": [1.0], "s": [0.1]})
    try:
        rank_uncertainty()
        raise AssertionError("no mode did not raise")
    except ValueError as e:
        check("ru_error_mode", "exactly one" in str(e), str(e))
    try:
        rank_uncertainty(estimates=est, est_col="e")
        raise AssertionError("missing se_col did not raise")
    except ValueError as e:
        check("ru_error_needs_se", "se_col" in str(e), str(e))
    try:
        rank_uncertainty(estimates=pd.DataFrame({"e": [1.0, 2.0],
                                                 "s": [0.1, -0.1]}),
                         est_col="e", se_col="s")
        raise AssertionError("negative se did not raise")
    except ValueError as e:
        check("ru_error_negative_se", "negative" in str(e), str(e))


# --------------------------------------------------------------------------
# reference anchor: se 0 (not NaN), and the fit-to-rank composition needs no
# hand-imputed SE for the reference (the failure a live kimi run worked around)
# --------------------------------------------------------------------------

def test_rank_pool_dispersion_warning():
    """A pool mixing precisions warns (printed + attrs); clean pools stay silent."""
    import io
    from contextlib import redirect_stdout

    mixed = pd.DataFrame({
        "who": list("abcdef"),
        "est": [2.0, 1.5, 1.0, 0.5, 0.0, 1.8],
        "se":  [0.1, 0.1, 0.1, 0.1, 0.1, 5.0],   # max/median = 50
    })
    buf = io.StringIO()
    with redirect_stdout(buf):
        r = rank_uncertainty(estimates=mixed, est_col="est", se_col="se",
                             entity_col="who", n_sim=500, seed=0)
    check("pool_warn_attrs", r.attrs.get("warning") is not None,
          "extreme se spread should set attrs['warning']")
    check("pool_warn_printed", "WARNING [rank_uncertainty]" in buf.getvalue(),
          "warning must reach stdout (the evidence channel)")
    check("pool_warn_advisory_only", len(r) == 6,
          "the warning is advisory; all entities still ranked")

    clean = mixed.assign(se=[0.4, 0.5, 0.5, 0.5, 0.6, 0.5])
    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        r2 = rank_uncertainty(estimates=clean, est_col="est", se_col="se",
                              entity_col="who", n_sim=500, seed=0)
    check("pool_clean_silent", r2.attrs.get("warning") is None
          and "WARNING" not in buf2.getvalue(),
          "homogeneous pool must not warn")

    small = mixed.head(4)  # below the 5-entity floor: never warn
    buf3 = io.StringIO()
    with redirect_stdout(buf3):
        r3 = rank_uncertainty(estimates=small, est_col="est", se_col="se",
                              entity_col="who", n_sim=300, seed=0)
    check("pool_small_silent", r3.attrs.get("warning") is None,
          "tiny pools are exempt from the warning")

    dr = pd.DataFrame(np.random.default_rng(0).normal(size=(200, 3)),
                      columns=["a", "b", "c"])
    r4 = rank_uncertainty(draws=dr)
    check("pool_draws_attrs_none", r4.attrs.get("warning") is None,
          "draws mode has no se; warning stays None")


def _simple_contests(seed=0, n_rep=30):
    """Noisy margin contests among four entities with known ordering."""
    rng = np.random.default_rng(seed)
    true = {"a": 2.0, "b": 1.0, "c": 0.0, "d": -1.0}
    ents = list(true)
    rows = []
    for i in range(len(ents)):
        for j in range(i + 1, len(ents)):
            for _ in range(n_rep):
                rows.append((ents[i], ents[j],
                             true[ents[i]] - true[ents[j]] + rng.normal(0, 0.5)))
    return pd.DataFrame(rows, columns=["A", "B", "margin"])


def test_reference_anchor_se_zero():
    df = _simple_contests()
    # margin mode
    res_m = paired_ability(df, "A", "B", margin_col="margin", ref="c")
    ref_m = res_m[res_m.entity == "c"].iloc[0]
    check("anchor_se_zero_margin", ref_m.se == 0.0, f"got se={ref_m.se}")
    check("anchor_ci_zero_margin",
          ref_m.ci_low == 0.0 and ref_m.ci_high == 0.0,
          f"got ci [{ref_m.ci_low}, {ref_m.ci_high}]")
    check("anchor_no_nan_se_margin", not res_m["se"].isna().any(),
          "no entity should carry a NaN se")
    # Bradley-Terry mode
    df_bt = df.copy()
    df_bt["win"] = (df_bt["margin"] > 0).astype(float)
    res_b = paired_ability(df_bt, "A", "B", win_col="win", ref="c")
    ref_b = res_b[res_b.entity == "c"].iloc[0]
    check("anchor_se_zero_bt", ref_b.se == 0.0, f"got se={ref_b.se}")
    check("anchor_no_nan_se_bt", not res_b["se"].isna().any(),
          "no entity should carry a NaN se")


def test_ability_rank_composition_no_imputation():
    df = _simple_contests(seed=1, n_rep=40)
    res = paired_ability(df, "A", "B", margin_col="margin", ref="c")
    ranks = rank_uncertainty(estimates=res, est_col="ability", se_col="se",
                             entity_col="entity", higher_is_better=True,
                             n_sim=4000, seed=0)
    ref_row = ranks[ranks.entity == "c"].iloc[0]
    check("compose_ref_included", ref_row.n_draws == 4000,
          f"reference must participate in every draw, got {ref_row.n_draws}")
    check("compose_ref_p_finite", not np.isnan(ref_row.p_rank1),
          "reference p_rank1 must be a number, not NaN")
    top = ranks.iloc[0]
    check("compose_leader_is_a", top.entity == "a",
          f"clear leader should top p_rank1, got {top.entity}")
    check("compose_leader_p_high", top.p_rank1 > 0.9,
          f"clear leader should dominate, got {top.p_rank1}")


# --------------------------------------------------------------------------
# prompt-schema agreement: the TOOLKIT legend in the Investigator prompt must
# name the REAL output columns/keys (the 6.2 three-way-contract discipline,
# checked against live objects rather than by reading the texts side by side)
# --------------------------------------------------------------------------

def test_prompt_schema_agreement():
    import prompts
    df = _simple_contests(seed=2, n_rep=10)
    res = paired_ability(df, "A", "B", margin_col="margin", ref="c")
    ranks = rank_uncertainty(estimates=res, est_col="ability", se_col="se",
                             entity_col="entity", n_sim=200, seed=0)
    boot = cluster_bootstrap(
        _clustered_frame(n_clusters=12, per=8), "cluster",
        lambda d: d["y"].mean(), n_boot=120, seed=0)
    sys_text = prompts.INVESTIGATOR_SYSTEM
    for col in res.columns:
        check(f"prompt_names_paired_ability.{col}", col in sys_text,
              f"paired_ability output column '{col}' missing from the prompt legend")
    for col in ranks.columns:
        check(f"prompt_names_rank_uncertainty.{col}", col in sys_text,
              f"rank_uncertainty output column '{col}' missing from the prompt legend")
    for key in boot:
        check(f"prompt_names_cluster_bootstrap.{key}", key in sys_text,
              f"cluster_bootstrap key '{key}' missing from the prompt legend")


# --------------------------------------------------------------------------
# kernel preload: the functions exist inside the persistent worker namespace
# --------------------------------------------------------------------------

def test_kernel_preload():
    sys.path.insert(0, os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), "tests", "stubs"))
    from kernel import PersistentKernel
    df = _clustered_frame(n_clusters=12, per=10)
    k = PersistentKernel(df=df, analysis_root="/tmp/toolkit_kernel_test")
    try:
        code = (
            "res = cluster_bootstrap(df, 'cluster', "
            "lambda d: d['y'].mean(), n_boot=150, seed=0)\n"
            "print('###RESULTS_START###')\n"
            "print('n_clusters', res['n_clusters'])\n"
            "print('pa', type(paired_ability).__name__, "
            "type(rank_uncertainty).__name__)\n"
            "print('###RESULTS_END###')\n"
        )
        stdout, error, _plots = k.execute(
            code, analysis_dir="/tmp/toolkit_kernel_test/01")
        check("kernel_no_error", not error, str(error))
        check("kernel_bootstrap_ran", "n_clusters 12" in stdout, stdout)
        check("kernel_all_three_loaded",
              "pa function function" in stdout, stdout)
        reg = k.describe_namespace()
        check("kernel_registry_clean",
              "cluster_bootstrap" not in reg and "paired_ability" not in reg,
              "toolkit names should not clutter the registry")
    finally:
        k.cleanup()


if __name__ == "__main__":
    test_margin_exact()
    test_margin_noisy_ci()
    test_bradley_terry()
    test_bt_perfect_record_finite()
    test_bt_ties()
    test_connectivity()
    test_errors_instructive()
    test_cluster_bootstrap_scalar()
    test_cluster_bootstrap_small_cluster_warning()
    test_cluster_bootstrap_series_and_compose()
    test_cluster_bootstrap_errors()
    test_rank_uncertainty_normal_mode()
    test_rank_uncertainty_errors()
    test_rank_pool_dispersion_warning()
    test_reference_anchor_se_zero()
    test_ability_rank_composition_no_imputation()
    test_prompt_schema_agreement()
    test_kernel_preload()
    print(f"OK: {len(PASS)} checks passed")
