"""Statistical checks for the Criteo experiment."""
import numpy as np
import pandas as pd
from scipy import stats


def srm_test(n_treatment: int, n_control: int, expected_treatment_share: float = 0.85,
             alpha: float = 0.001) -> dict:
    """Sample Ratio Mismatch check with a chi-square goodness-of-fit test.

    Expected split is the *designed* ratio (85:15 here), not 50:50.
    A strict alpha (0.001) is the industry convention for SRM, because an SRM
    means the assignment pipeline is broken and every downstream result is suspect.
    """
    n = n_treatment + n_control
    observed = np.array([n_treatment, n_control])
    expected = np.array([expected_treatment_share, 1 - expected_treatment_share]) * n
    chi2, p = stats.chisquare(observed, expected)
    return {
        "n": n,
        "observed_treatment_share": n_treatment / n,
        "expected_treatment_share": expected_treatment_share,
        "chi2": chi2,
        "p_value": p,
        "srm_detected": p < alpha,
    }


def standardized_mean_difference(x_t: np.ndarray, x_c: np.ndarray) -> float:
    """SMD = (mean_t - mean_c) / sqrt((var_t + var_c) / 2).

    Unit-free effect size: does not grow with sample size, unlike a t-statistic.
    |SMD| < 0.1 is the usual threshold for 'balanced'.
    """
    x_t = np.asarray(x_t, dtype="float64")
    x_c = np.asarray(x_c, dtype="float64")
    pooled_sd = np.sqrt((x_t.var(ddof=1) + x_c.var(ddof=1)) / 2)
    if pooled_sd == 0:
        return 0.0
    return (x_t.mean() - x_c.mean()) / pooled_sd


def balance_table(df: pd.DataFrame, features: list[str], treatment_col: str = "treatment",
                  threshold: float = 0.1) -> pd.DataFrame:
    """Per-feature means, SMD and (for contrast) Welch t-test p-value."""
    t = df[df[treatment_col] == 1]
    c = df[df[treatment_col] == 0]
    rows = []
    for f in features:
        smd = standardized_mean_difference(t[f], c[f])
        p = stats.ttest_ind(t[f].astype("float64"), c[f].astype("float64"), equal_var=False).pvalue
        rows.append({"feature": f, "mean_treatment": t[f].mean(), "mean_control": c[f].mean(),
                     "smd": smd, "abs_smd": abs(smd), "t_test_p": p,
                     "balanced": abs(smd) < threshold})
    return pd.DataFrame(rows).set_index("feature").sort_values("abs_smd", ascending=False)


# ---------------------------------------------------------------------------
# Step 4: ITT, compliance and CACE (Wald / IV estimator)
# ---------------------------------------------------------------------------

def itt_effect(y_t: np.ndarray, y_c: np.ndarray, alpha: float = 0.05) -> dict:
    """Intent-to-Treat effect = mean(Y | assigned T) - mean(Y | assigned C).

    Unbiased because assignment is random. Unpooled (Welch-style) SE.
    """
    y_t = np.asarray(y_t, dtype="float64")
    y_c = np.asarray(y_c, dtype="float64")
    m_t, m_c = y_t.mean(), y_c.mean()
    se = np.sqrt(y_t.var(ddof=1) / len(y_t) + y_c.var(ddof=1) / len(y_c))
    z = stats.norm.ppf(1 - alpha / 2)
    diff = m_t - m_c
    return {"rate_t": m_t, "rate_c": m_c, "effect": diff, "se": se,
            "ci_low": diff - z * se, "ci_high": diff + z * se,
            "rel_lift": diff / m_c if m_c > 0 else np.nan,
            "p_value": 2 * stats.norm.sf(abs(diff / se))}


def wald_cace(y: np.ndarray, d: np.ndarray, z: np.ndarray, alpha: float = 0.05) -> dict:
    """Complier Average Causal Effect via the Wald / IV estimator.

    CACE = ITT_Y / ITT_D, where ITT_D = P(exposed | T) - P(exposed | C) is the
    'first stage' (compliance rate). SE from the delta method, including the
    covariance between ITT_Y and ITT_D.

    y: outcome, d: actual exposure (treatment received), z: random assignment (instrument).
    """
    y, d, z = (np.asarray(a, dtype="float64") for a in (y, d, z))
    t, c = z == 1, z == 0
    n_t, n_c = t.sum(), c.sum()

    itt_y = y[t].mean() - y[c].mean()
    itt_d = d[t].mean() - d[c].mean()
    var_itt_y = y[t].var(ddof=1) / n_t + y[c].var(ddof=1) / n_c
    var_itt_d = d[t].var(ddof=1) / n_t + d[c].var(ddof=1) / n_c
    cov = (np.cov(y[t], d[t])[0, 1] / n_t) + (np.cov(y[c], d[c])[0, 1] / n_c)

    cace = itt_y / itt_d
    var = (var_itt_y / itt_d**2 + itt_y**2 * var_itt_d / itt_d**4
           - 2 * itt_y * cov / itt_d**3)
    se = np.sqrt(var)
    zc = stats.norm.ppf(1 - alpha / 2)
    return {"itt_y": itt_y, "itt_d": itt_d, "cace": cace, "se": se,
            "ci_low": cace - zc * se, "ci_high": cace + zc * se,
            "first_stage_F": itt_d**2 / var_itt_d}


def bootstrap_cace(y: np.ndarray, d: np.ndarray, z: np.ndarray, n_boot: int = 2000,
                   seed: int = 42, alpha: float = 0.05) -> dict:
    """Fast stratified bootstrap of ITT and CACE.

    Because Y, D, Z are all binary, resampling rows is equivalent to drawing
    multinomial counts over the (D, Y) cells within each arm, so 2000 reps on
    1.4M rows take well under a second.
    """
    rng = np.random.default_rng(seed)
    y, d, z = (np.asarray(a).astype(int) for a in (y, d, z))
    out = {}
    for arm in (0, 1):
        m = z == arm
        cell = d[m] * 2 + y[m]                       # 0:(D0,Y0) 1:(D0,Y1) 2:(D1,Y0) 3:(D1,Y1)
        probs = np.bincount(cell, minlength=4) / m.sum()
        counts = rng.multinomial(m.sum(), probs, size=n_boot)
        n = m.sum()
        out[arm] = {"y": (counts[:, 1] + counts[:, 3]) / n, "d": (counts[:, 2] + counts[:, 3]) / n}
    itt_y = out[1]["y"] - out[0]["y"]
    itt_d = out[1]["d"] - out[0]["d"]
    cace = itt_y / itt_d
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return {"itt_ci": np.percentile(itt_y, q), "cace_ci": np.percentile(cace, q),
            "itt_draws": itt_y, "cace_draws": cace}


def complier_outcome_means(y: np.ndarray, d: np.ndarray, z: np.ndarray) -> dict:
    """Decompose outcomes by principal strata under one-sided non-compliance.

    With nobody exposed in control, assigned-treatment users are either
    compliers (exposed) or never-takers (not exposed). Randomization makes the
    never-taker share and their outcome the same in both arms, so:

        mean_C = p * Y0_compliers + (1 - p) * Y_never_takers
        => Y0_compliers = (mean_C - (1 - p) * Y_never_takers) / p

    Y0_compliers = visit rate the exposed users WOULD have had without the ad.
    """
    y, d, z = (np.asarray(a, dtype="float64") for a in (y, d, z))
    t, c = z == 1, z == 0
    p = d[t].mean()
    y_never = y[t & (d == 0)].mean()
    y1_compliers = y[t & (d == 1)].mean()
    y0_compliers = (y[c].mean() - (1 - p) * y_never) / p
    return {"complier_share": p, "y_never_takers": y_never,
            "y1_compliers": y1_compliers, "y0_compliers": y0_compliers,
            "cace": y1_compliers - y0_compliers}


# ---------------------------------------------------------------------------
# Step 5: Power analysis (unequal 85:15 allocation)
# ---------------------------------------------------------------------------

def _h_to_rate(p_base: float, h: float) -> float:
    """Invert Cohen's h: rate p_t such that 2*asin(sqrt(p_t)) - 2*asin(sqrt(p_base)) = h."""
    return float(np.sin(np.arcsin(np.sqrt(p_base)) + h / 2) ** 2)


def mde_proportion(p_base: float, n_treatment: int, n_control: int,
                   alpha: float = 0.05, power: float = 0.80) -> dict:
    """Minimum Detectable Effect for a two-proportion test with unequal groups.

    statsmodels' `ratio` = nobs2 / nobs1 = n_control / n_treatment (0.15/0.85 ≈ 0.176 here).
    """
    from statsmodels.stats.power import NormalIndPower
    h = NormalIndPower().solve_power(effect_size=None, nobs1=n_treatment, alpha=alpha,
                                     power=power, ratio=n_control / n_treatment,
                                     alternative="two-sided")
    p_t = _h_to_rate(p_base, h)
    return {"p_base": p_base, "n_total": n_treatment + n_control, "cohens_h": h,
            "mde_abs": p_t - p_base, "mde_rel": (p_t - p_base) / p_base}


def sample_size_proportion(p_base: float, rel_lift: float, treatment_share: float = 0.85,
                           alpha: float = 0.05, power: float = 0.80) -> dict:
    """Total users needed to detect a given relative lift with a given allocation."""
    from statsmodels.stats.power import NormalIndPower
    from statsmodels.stats.proportion import proportion_effectsize
    h = proportion_effectsize(p_base * (1 + rel_lift), p_base)
    ratio = (1 - treatment_share) / treatment_share
    n_t = NormalIndPower().solve_power(effect_size=h, nobs1=None, alpha=alpha, power=power,
                                       ratio=ratio, alternative="two-sided")
    n_c = n_t * ratio
    return {"p_base": p_base, "rel_lift": rel_lift, "abs_lift": p_base * rel_lift,
            "n_treatment": int(np.ceil(n_t)), "n_control": int(np.ceil(n_c)),
            "n_total": int(np.ceil(n_t + n_c))}


# ---------------------------------------------------------------------------
# Step 6: A/B test analysis
# ---------------------------------------------------------------------------

def two_proportion_test(y_t: np.ndarray, y_c: np.ndarray, alpha: float = 0.05) -> dict:
    """Two-proportion z-test with absolute and relative lift + 95% CIs.

    - p-value: statsmodels pooled z-test (H0: p_t == p_c)
    - absolute CI: unpooled Wald interval
    - relative CI: delta method on log(p_t / p_c), back-transformed (asymmetric, stays > -100%)
    """
    from statsmodels.stats.proportion import proportions_ztest
    y_t = np.asarray(y_t)
    y_c = np.asarray(y_c)
    x_t, x_c, n_t, n_c = int(y_t.sum()), int(y_c.sum()), len(y_t), len(y_c)
    p_t, p_c = x_t / n_t, x_c / n_c
    z_stat, p_val = proportions_ztest([x_t, x_c], [n_t, n_c], alternative="two-sided")

    zc = stats.norm.ppf(1 - alpha / 2)
    diff = p_t - p_c
    se = np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)

    log_rr = np.log(p_t / p_c)
    se_log = np.sqrt((1 - p_t) / x_t + (1 - p_c) / x_c)
    return {"n_t": n_t, "n_c": n_c, "rate_t": p_t, "rate_c": p_c,
            "abs_lift": diff, "abs_ci_low": diff - zc * se, "abs_ci_high": diff + zc * se,
            "rel_lift": p_t / p_c - 1,
            "rel_ci_low": np.exp(log_rr - zc * se_log) - 1,
            "rel_ci_high": np.exp(log_rr + zc * se_log) - 1,
            "z": z_stat, "p_value": p_val}


def bootstrap_binary_lift(y_t: np.ndarray, y_c: np.ndarray, n_boot: int = 10_000,
                          seed: int = 42, alpha: float = 0.05) -> dict:
    """Percentile bootstrap for absolute and relative lift of a binary metric.

    Resampling n rows with replacement from a 0/1 column = drawing Binomial(n, p_hat),
    so this is an exact (and very fast) row bootstrap.
    """
    rng = np.random.default_rng(seed)
    y_t = np.asarray(y_t)
    y_c = np.asarray(y_c)
    n_t, n_c = len(y_t), len(y_c)
    bt = rng.binomial(n_t, y_t.mean(), n_boot) / n_t
    bc = rng.binomial(n_c, y_c.mean(), n_boot) / n_c
    abs_draws, rel_draws = bt - bc, bt / bc - 1
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return {"abs_ci": np.percentile(abs_draws, q), "rel_ci": np.percentile(rel_draws, q),
            "abs_draws": abs_draws, "rel_draws": rel_draws,
            "p_le_zero": (abs_draws <= 0).mean()}


def regression_adjusted_effect(df: pd.DataFrame, outcome: str, features: list[str],
                               treatment_col: str = "treatment") -> dict:
    """Lin (2013) regression adjustment: y ~ T + Xc + T*Xc with HC2 robust SEs.

    Xc = features centred at their overall mean, so the T coefficient is the
    average treatment effect. Unbiased under randomization even if the linear
    model is wrong; it only uses X to soak up outcome variance (same idea as CUPED).
    """
    import statsmodels.api as sm
    X = df[features].astype("float64")
    Xc = X - X.mean()
    t = df[treatment_col].astype("float64").values
    design = pd.concat([pd.Series(t, index=df.index, name="T"), Xc,
                        Xc.mul(t, axis=0).add_prefix("T_x_")], axis=1)
    design = sm.add_constant(design)
    fit = sm.OLS(df[outcome].astype("float64").values, design).fit(cov_type="HC2")
    ci = fit.conf_int().loc["T"]
    return {"effect": fit.params["T"], "se": fit.bse["T"],
            "ci_low": ci[0], "ci_high": ci[1], "p_value": fit.pvalues["T"],
            "r2": fit.rsquared}


# ---------------------------------------------------------------------------
# Step 7: Segment analysis
# ---------------------------------------------------------------------------

def feature_segments(x: pd.Series, default_share_threshold: float = 0.5) -> pd.Series:
    """Bin an anonymized feature into readable segments.

    Many Criteo features sit on one 'default' value for most users, so plain
    quantiles collapse. If the mode covers >= 50% of users: 'default' vs
    non-default split at its median into 'low'/'high'. Otherwise tertiles.
    """
    mode = x.mode()[0]
    if (x == mode).mean() >= default_share_threshold:
        nd = x[x != mode]
        cut = nd.median()
        low = x <= cut
        if (low & (x != mode)).sum() == len(nd):   # ties at the median swallowed everything
            low = x < cut
        seg = np.where(x == mode, "default", np.where(low, "low", "high"))
        order = ["default", "low", "high"]
    else:
        seg = pd.qcut(x, 3, labels=["low", "medium", "high"], duplicates="drop").astype(str)
        order = ["low", "medium", "high"]
    return pd.Series(pd.Categorical(seg, categories=order, ordered=True), index=x.index)


def segment_effects(df: pd.DataFrame, segment_col: str, outcome: str = "visit",
                    treatment_col: str = "treatment", exposure_col: str = "exposure") -> pd.DataFrame:
    """ITT (and CACE) of `outcome` within each segment, with 95% CI and p-value."""
    rows = []
    for seg, g in df.groupby(segment_col, observed=True):
        t = g[treatment_col] == 1
        r = itt_effect(g.loc[t, outcome].values, g.loc[~t, outcome].values)
        expo = g.loc[t, exposure_col].mean()
        rows.append({"segment": seg, "n": len(g), "share": len(g) / len(df),
                     "rate_c": r["rate_c"], "rate_t": r["rate_t"],
                     "effect": r["effect"], "se": r["se"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                     "rel_lift": r["rel_lift"], "p_value": r["p_value"],
                     "exposure_rate": expo, "cace": r["effect"] / expo if expo > 0 else np.nan})
    return pd.DataFrame(rows).set_index("segment")


def heterogeneity_test(effects: np.ndarray, ses: np.ndarray) -> dict:
    """Cochran's Q: are the segment effects all equal? Q ~ chi2(k-1) under H0.

    One test per feature instead of many pairwise comparisons, so it avoids
    most of the multiple-testing problem.
    """
    effects, ses = np.asarray(effects), np.asarray(ses)
    w = 1 / ses**2
    pooled = (w * effects).sum() / w.sum()
    q = (w * (effects - pooled) ** 2).sum()
    k = len(effects)
    return {"Q": q, "df": k - 1, "p_value": stats.chi2.sf(q, k - 1),
            "I2": max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0}
