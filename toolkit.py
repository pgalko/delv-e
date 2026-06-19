"""
Vetted statistical estimators preloaded into the execution kernel.

WHY THIS EXISTS. The measured failure across benchmark runs is that a weak
Investigator names the decisive method (a paired-comparison model, a clustered
uncertainty estimate) and then defers it, because describing the algorithm to
a cheap executor costs a long error-prone spec. These functions collapse that
cost to one call. The Investigator still chooses the estimator from the data's
structure (METHOD ADEQUACY); the executor only transcribes the call; the
implementation lives here, once, under tests.

ADMISSION RULE. A function enters this module only after a logged run in which
a brain named its method class and then deferred it, and only when no commonly
used library (scipy, statsmodels) offers a turnkey route. Hard cap: five
functions; a sixth must merge with or replace an existing one. Current tickets:
  paired_ability    - named-then-deferred three times on the F1 benchmark
  cluster_bootstrap - pseudo-replication unaddressed on both benchmarks
  rank_uncertainty  - false-outlier verdicts issued twice without it

DESIGN CONTRACT. Functions print nothing and return tidy frames or dicts; the
spec decides what to print. Input validation raises one-line instructive
errors, because the caller is a cheap model on a retry budget and the error
message is part of the interface.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["paired_ability", "cluster_bootstrap", "rank_uncertainty"]

_Z95 = 1.959963984540054  # two-sided 95% normal quantile


# --------------------------------------------------------------------------
# shared validation helpers
# --------------------------------------------------------------------------

def _require_dataframe(obj, name):
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame, got {type(obj).__name__}")
    if len(obj) == 0:
        raise ValueError(f"{name} is empty")


def _require_columns(df, cols, df_name="df"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"column(s) {missing} not in {df_name}; available columns are: "
            f"{list(df.columns)}"
        )


def _largest_component(pairs):
    """Union-find over (a, b) pairs; returns the set of nodes in the largest
    connected component."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    comps = {}
    for node in parent:
        comps.setdefault(find(node), set()).add(node)
    return max(comps.values(), key=len) if comps else set()


# --------------------------------------------------------------------------
# 1. paired_ability
# --------------------------------------------------------------------------

def paired_ability(df, a_col, b_col, margin_col=None, win_col=None,
                   weight_col=None, ref=None, max_iter=1000, tol=1e-9):
    """Ability model over rows of A-vs-B contests.

    Exactly one of:
      win_col    : 1 if A won, 0 if B won, 0.5 a tie -> Bradley-Terry fit (MM)
      margin_col : continuous A-minus-B margin -> network-adjusted linear fit

    Returns a DataFrame with one row per entity: entity, ability, se, ci_low,
    ci_high, n_contests. Abilities are relative to the reference entity, the
    anchor (ability 0, se 0; all uncertainty is expressed relative to it);
    ref defaults to the entity with the most contests.
    Disconnected entities (no contest path to the reference) are dropped and
    listed in result.attrs["dropped_disconnected"].
    """
    _require_dataframe(df, "df")
    _require_columns(df, [a_col, b_col], "df")
    if (margin_col is None) == (win_col is None):
        raise ValueError("give exactly one of margin_col (continuous A-minus-B "
                         "margin) or win_col (1 if A won, 0 if B won)")
    value_col = margin_col if margin_col is not None else win_col
    _require_columns(df, [value_col], "df")
    if weight_col is not None:
        _require_columns(df, [weight_col], "df")

    use_cols = [a_col, b_col, value_col] + ([weight_col] if weight_col else [])
    d = df[use_cols].dropna()
    d = d[d[a_col] != d[b_col]]
    if len(d) == 0:
        raise ValueError("no usable rows after dropping NaNs and self-pairs")

    if win_col is not None:
        wins = d[win_col]
        bad = ~wins.isin([0, 0.5, 1])
        if bad.any():
            raise ValueError(f"win_col '{win_col}' must contain only 0, 1, or "
                             f"0.5 (a tie, counted as half a win for each "
                             f"side); found values like "
                             f"{sorted(wins[bad].unique()[:5])}")

    # connectivity: fit only the largest connected component
    keep = _largest_component(zip(d[a_col].tolist(), d[b_col].tolist()))
    all_entities = set(d[a_col]) | set(d[b_col])
    dropped = sorted(all_entities - keep, key=str)
    d = d[d[a_col].isin(keep) & d[b_col].isin(keep)]
    entities = sorted(keep, key=str)
    if len(entities) < 2:
        raise ValueError("fewer than 2 connected entities; an ability model "
                         "needs a connected comparison graph")

    idx = {e: i for i, e in enumerate(entities)}
    ai = d[a_col].map(idx).to_numpy()
    bi = d[b_col].map(idx).to_numpy()
    w = (d[weight_col].to_numpy(dtype=float) if weight_col
         else np.ones(len(d)))
    if np.any(w < 0):
        raise ValueError(f"weight_col '{weight_col}' contains negative weights")
    n_e = len(entities)
    n_contests = np.zeros(n_e)
    np.add.at(n_contests, ai, 1)
    np.add.at(n_contests, bi, 1)

    if ref is None:
        ref = entities[int(np.argmax(n_contests))]
    elif ref not in idx:
        raise ValueError(f"ref entity {ref!r} not among the connected entities; "
                         f"examples: {entities[:5]}")
    r = idx[ref]

    if margin_col is not None:
        ability, se, model = _fit_margin_network(
            ai, bi, d[value_col].to_numpy(dtype=float), w, n_e, r)
    else:
        ability, se, model = _fit_bradley_terry(
            ai, bi, d[value_col].to_numpy(dtype=float), w, n_e, r,
            max_iter=max_iter, tol=tol)

    # The reference is the anchor: its ability is 0 by construction and all
    # uncertainty is expressed relative to it, so its own se is 0, not NaN.
    # (A NaN here silently drops the reference from rank_uncertainty, which
    # pushed a live run into hand-imputing an SE for it.)
    se[r] = 0.0

    out = pd.DataFrame({
        "entity": entities,
        "ability": ability,
        "se": se,
        "ci_low": ability - _Z95 * se,
        "ci_high": ability + _Z95 * se,
        "n_contests": n_contests.astype(int),
    }).sort_values("ability", ascending=False).reset_index(drop=True)
    out.attrs["reference"] = ref
    out.attrs["model"] = model
    out.attrs["dropped_disconnected"] = dropped
    out.attrs["n_rows_fit"] = int(len(d))
    return out


def _fit_margin_network(ai, bi, y, w, n_e, r):
    """Weighted least squares on the signed-difference design (+1 for A, -1
    for B per row), reference column dropped, via the normal equations."""
    XtX = np.zeros((n_e, n_e))
    np.add.at(XtX, (ai, ai), w)
    np.add.at(XtX, (bi, bi), w)
    np.add.at(XtX, (ai, bi), -w)
    np.add.at(XtX, (bi, ai), -w)
    Xty = np.zeros(n_e)
    np.add.at(Xty, ai, w * y)
    np.add.at(Xty, bi, -w * y)

    free = [i for i in range(n_e) if i != r]
    A = XtX[np.ix_(free, free)]
    beta_free = np.linalg.solve(A, Xty[free])
    beta = np.zeros(n_e)
    beta[free] = beta_free

    resid = y - (beta[ai] - beta[bi])
    dof = max(len(y) - (n_e - 1), 1)
    sigma2 = float(np.sum(w * resid ** 2) / dof)
    cov = sigma2 * np.linalg.inv(A)
    se = np.full(n_e, np.nan)
    se[free] = np.sqrt(np.clip(np.diag(cov), 0, None))
    return beta, se, "margin_ols"


def _fit_bradley_terry(ai, bi, won_a, w, n_e, r, max_iter, tol, smooth=0.1):
    """Bradley-Terry via the MM algorithm on pair-aggregated counts, with a
    small symmetric smoothing count per observed pair so entities with a
    perfect record stay finite. SEs from the observed Fisher information."""
    # aggregate to per-ordered-pair win counts; a tie (0.5) is the standard
    # half-win convention, half a win credited in each direction
    wins = {}
    for a, b, win, wt in zip(ai, bi, won_a, w):
        if win == 0.5:
            wins[(a, b)] = wins.get((a, b), 0.0) + 0.5 * wt
            wins[(b, a)] = wins.get((b, a), 0.0) + 0.5 * wt
        else:
            key = (a, b) if win == 1 else (b, a)
            wins[key] = wins.get(key, 0.0) + wt
    pair_keys = {tuple(sorted(k)) for k in wins}
    for i, j in pair_keys:  # smoothing: both directions of every observed pair
        wins[(i, j)] = wins.get((i, j), 0.0) + smooth
        wins[(j, i)] = wins.get((j, i), 0.0) + smooth

    win_tot = np.zeros(n_e)
    for (i, _j), c in wins.items():
        win_tot[i] += c
    n_pair = {}
    for (i, j), c in wins.items():
        key = (min(i, j), max(i, j))
        n_pair[key] = n_pair.get(key, 0.0) + c

    p = np.ones(n_e)
    for _ in range(max_iter):
        denom = np.zeros(n_e)
        for (i, j), n_ij in n_pair.items():
            s = p[i] + p[j]
            denom[i] += n_ij / s
            denom[j] += n_ij / s
        p_new = win_tot / np.where(denom > 0, denom, 1.0)
        p_new = p_new / np.exp(np.mean(np.log(np.clip(p_new, 1e-300, None))))
        if np.max(np.abs(np.log(p_new) - np.log(p))) < tol:
            p = p_new
            break
        p = p_new

    beta = np.log(p) - np.log(p[r])

    # observed Fisher information in the log-ability parametrization
    info = np.zeros((n_e, n_e))
    for (i, j), n_ij in n_pair.items():
        pij = p[i] / (p[i] + p[j])
        v = n_ij * pij * (1 - pij)
        info[i, i] += v
        info[j, j] += v
        info[i, j] -= v
        info[j, i] -= v
    free = [i for i in range(n_e) if i != r]
    cov = np.linalg.inv(info[np.ix_(free, free)])
    se = np.full(n_e, np.nan)
    se[free] = np.sqrt(np.clip(np.diag(cov), 0, None))
    return beta, se, "bradley_terry"


# --------------------------------------------------------------------------
# 2. cluster_bootstrap
# --------------------------------------------------------------------------

def cluster_bootstrap(df, cluster_col, stat_fn, n_boot=2000, ci=0.95, seed=0):
    """Cluster bootstrap: resample whole clusters with replacement, apply
    stat_fn to each resampled frame, return percentile intervals.

    stat_fn maps a DataFrame to a scalar or to a pandas Series (e.g. one value
    per entity). Returns a dict with: estimate (full-sample stat), ci_low,
    ci_high, ci_level, n_clusters, n_boot_used, n_failed, draws, warning. For a
    Series statistic, estimate/ci_low/ci_high are Series and draws is a DataFrame
    (one row per replicate) suitable for rank_uncertainty(draws=...).

    Call with the dataframe positional and the rest by keyword, e.g.
        cluster_bootstrap(df, cluster_col="group",
                          stat_fn=lambda d: d["value"].mean(), n_boot=1000)
    """
    _require_dataframe(df, "df")
    _require_columns(df, [cluster_col], "df")
    if not callable(stat_fn):
        raise TypeError("stat_fn must be a callable mapping a DataFrame to a "
                        "scalar or a pandas Series")
    if n_boot < 100:
        raise ValueError(f"n_boot={n_boot} is too small; use at least 100")
    if not (0 < ci < 1):
        raise ValueError(f"ci must be between 0 and 1, got {ci}")

    clusters = pd.unique(df[cluster_col].dropna())
    n_clusters = len(clusters)
    if n_clusters < 2:
        raise ValueError(f"only {n_clusters} distinct cluster(s) in "
                         f"'{cluster_col}'; need at least 2 to resample")

    try:
        estimate = stat_fn(df)
    except Exception as e:
        raise RuntimeError(
            f"stat_fn failed on the full sample before any resampling; fix the "
            f"statistic first; error: {type(e).__name__}: {e}") from e
    is_series = isinstance(estimate, pd.Series)
    if not is_series:
        estimate = float(estimate)

    groups = {c: g for c, g in df.groupby(cluster_col, observed=True)}
    rng = np.random.default_rng(seed)
    draws, n_failed, last_err = [], 0, None
    for _ in range(n_boot):
        picked = rng.choice(clusters, size=n_clusters, replace=True)
        sample = pd.concat([groups[c] for c in picked], ignore_index=True)
        try:
            s = stat_fn(sample)
            draws.append(s if is_series else float(s))
        except Exception as e:  # a replicate may legitimately fail
            n_failed += 1
            last_err = e
    n_used = len(draws)
    if n_used < max(100, n_boot // 2):
        raise RuntimeError(
            f"stat_fn failed on {n_failed} of {n_boot} replicates (only "
            f"{n_used} usable); last error: {type(last_err).__name__}: {last_err}")

    alpha = (1.0 - ci) / 2.0
    if is_series:
        draws_df = pd.DataFrame(draws)  # rows = replicates, cols = entities
        ci_low = draws_df.quantile(alpha)
        ci_high = draws_df.quantile(1.0 - alpha)
        draws_out = draws_df
    else:
        arr = np.asarray(draws, dtype=float)
        ci_low = float(np.quantile(arr, alpha))
        ci_high = float(np.quantile(arr, 1.0 - alpha))
        draws_out = arr

    warning = None
    if n_clusters < 10:
        warning = (f"only {n_clusters} clusters; bootstrap intervals are "
                   f"unreliable below roughly 10 clusters, treat as approximate")

    return {
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_level": ci,
        "n_clusters": int(n_clusters),
        "n_boot_used": int(n_used),
        "n_failed": int(n_failed),
        "draws": draws_out,
        "warning": warning,
    }


# --------------------------------------------------------------------------
# 3. rank_uncertainty
# --------------------------------------------------------------------------

def rank_uncertainty(estimates=None, est_col=None, se_col=None,
                     entity_col=None, draws=None, higher_is_better=True,
                     n_sim=10000, seed=0):
    """Convert estimates with uncertainty, or bootstrap draws, into P(rank 1)
    and rank intervals per entity. Run this before any "best", "top", or
    "outlier" claim.

    P(rank 1) is relative to the pool passed in: entities with few
    observations and large se dominate the rank-1 draws and mask the real
    leaders. Filter to comparably-evidenced entities first unless ranking a
    mixed-evidence pool is deliberate. In estimates mode, an extreme se
    spread (max se at least 10x the median, 5+ entities) prints a warning
    and stores it in result.attrs['warning'].

    Mode A: estimates= DataFrame with est_col and se_col (entities from
            entity_col, or the index when entity_col is None); simulates
            n_sim independent normal draws per entity.
    Mode B: draws= DataFrame with one row per replicate and one column per
            entity (cluster_bootstrap's Series-statistic "draws" fits as-is).

    Returns a DataFrame sorted by p_rank1: entity, estimate, p_rank1,
    rank_median, rank_ci_low, rank_ci_high, n_draws.
    """
    if (estimates is None) == (draws is None):
        raise ValueError("give exactly one of estimates= (with est_col and "
                         "se_col) or draws= (replicates x entities DataFrame)")

    warning = None
    if draws is not None:
        _require_dataframe(draws, "draws")
        if draws.shape[1] < 2:
            raise ValueError("draws needs at least 2 entity columns to rank")
        mat = draws.to_numpy(dtype=float)
        names = list(draws.columns)
        point = np.nanmean(mat, axis=0)
        mode = "draws"
    else:
        _require_dataframe(estimates, "estimates")
        if est_col is None or se_col is None:
            raise ValueError("mode estimates= needs both est_col and se_col")
        _require_columns(estimates, [est_col, se_col], "estimates")
        if entity_col is not None:
            _require_columns(estimates, [entity_col], "estimates")
            names = estimates[entity_col].tolist()
        else:
            names = estimates.index.tolist()
        if len(set(names)) != len(names):
            raise ValueError("entity names are not unique; give entity_col or "
                             "a unique index")
        est = estimates[est_col].to_numpy(dtype=float)
        se = estimates[se_col].to_numpy(dtype=float)
        usable = ~(np.isnan(est) | np.isnan(se))
        if usable.sum() < 2:
            raise ValueError("fewer than 2 entities with non-NaN estimate and "
                             "se; cannot rank")
        if np.any(se[usable] < 0):
            raise ValueError(f"se_col '{se_col}' contains negative values")
        # Pool-hygiene warning: a pool mixing precisions lets low-evidence
        # entities dominate the rank-1 draws. Printed (stdout is the evidence
        # channel) and stored in attrs; mechanical, data-driven, advisory only.
        use_se = se[usable]
        med_se = float(np.median(use_se))
        if len(use_se) >= 5 and med_se > 0 and float(np.max(use_se)) >= 10 * med_se:
            warning = (f"rank pool mixes precisions: max se {float(np.max(use_se)):.3g} "
                       f"is >=10x the median se {med_se:.3g}; low-evidence entities "
                       f"will dominate rank-1 draws. Consider filtering the pool to "
                       f"comparably-evidenced entities before ranking.")
            print("WARNING [rank_uncertainty]: " + warning)
        rng = np.random.default_rng(seed)
        mat = np.full((n_sim, len(names)), np.nan)
        mat[:, usable] = rng.normal(est[usable], se[usable],
                                    size=(n_sim, int(usable.sum())))
        point = est
        mode = "normal"

    score = mat if higher_is_better else -mat
    n_draws_per = np.sum(~np.isnan(score), axis=0)

    # per-replicate ranks (1 = best), NaN entities excluded from that replicate
    order_desc = np.where(np.isnan(score), -np.inf, score)
    ranks = (-order_desc).argsort(axis=1).argsort(axis=1).astype(float) + 1
    ranks[np.isnan(score)] = np.nan

    with np.errstate(invalid="ignore"):
        p1 = np.nansum(ranks == 1, axis=0) / np.maximum(n_draws_per, 1)
        rank_med = np.nanmedian(ranks, axis=0)
        rank_lo = np.nanpercentile(ranks, 2.5, axis=0)
        rank_hi = np.nanpercentile(ranks, 97.5, axis=0)

    out = pd.DataFrame({
        "entity": names,
        "estimate": point,
        "p_rank1": p1,
        "rank_median": rank_med,
        "rank_ci_low": rank_lo,
        "rank_ci_high": rank_hi,
        "n_draws": n_draws_per.astype(int),
    }).sort_values("p_rank1", ascending=False).reset_index(drop=True)
    out.attrs["higher_is_better"] = bool(higher_is_better)
    out.attrs["mode"] = mode
    out.attrs["warning"] = warning
    return out