"""Uplift model evaluation helpers (wrapping sklift.metrics)."""
import numpy as np
import pandas as pd
from sklift.metrics import (qini_auc_score, uplift_auc_score, uplift_at_k,
                            qini_curve, uplift_curve)


def uplift_metrics(y, uplift, t, ks=(0.1, 0.2, 0.3)) -> dict:
    """Qini AUC, uplift AUC and uplift@k (actual treated-minus-control rate in the top k%)."""
    y, uplift, t = np.asarray(y), np.asarray(uplift), np.asarray(t)
    out = {"qini_auc": qini_auc_score(y, uplift, t),
           "uplift_auc": uplift_auc_score(y, uplift, t)}
    for k in ks:
        # strategy="overall": take the top k% of ALL users, then compare T vs C inside it
        out[f"uplift_at_{int(k * 100)}"] = uplift_at_k(y, uplift, t, strategy="overall", k=k)
    return out


def bootstrap_metric(y, uplift, t, metric=qini_auc_score, n_boot: int = 200, seed: int = 42) -> dict:
    """Stratified (by arm) bootstrap of a ranking metric on the test set, 95% percentile CI."""
    rng = np.random.default_rng(seed)
    y, uplift, t = np.asarray(y), np.asarray(uplift), np.asarray(t)
    idx_t, idx_c = np.where(t == 1)[0], np.where(t == 0)[0]
    vals = []
    for _ in range(n_boot):
        i = np.concatenate([rng.choice(idx_t, len(idx_t)), rng.choice(idx_c, len(idx_c))])
        vals.append(metric(y[i], uplift[i], t[i]))
    vals = np.array(vals)
    return {"mean": vals.mean(), "ci_low": np.percentile(vals, 2.5),
            "ci_high": np.percentile(vals, 97.5), "draws": vals}


def paired_bootstrap_diff(y, u_a, u_b, t, metric=qini_auc_score, n_boot: int = 200, seed: int = 42) -> dict:
    """Bootstrap CI for metric(model A) - metric(model B) on the SAME resamples."""
    rng = np.random.default_rng(seed)
    y, u_a, u_b, t = (np.asarray(a) for a in (y, u_a, u_b, t))
    idx_t, idx_c = np.where(t == 1)[0], np.where(t == 0)[0]
    diffs = []
    for _ in range(n_boot):
        i = np.concatenate([rng.choice(idx_t, len(idx_t)), rng.choice(idx_c, len(idx_c))])
        diffs.append(metric(y[i], u_a[i], t[i]) - metric(y[i], u_b[i], t[i]))
    diffs = np.array(diffs)
    return {"diff": diffs.mean(), "ci_low": np.percentile(diffs, 2.5),
            "ci_high": np.percentile(diffs, 97.5), "p_a_better": (diffs > 0).mean()}


def uplift_by_bin(y, uplift, t, n_bins: int = 10) -> pd.DataFrame:
    """Actual uplift (treated rate - control rate) per predicted-uplift decile.

    Bin 1 = highest predicted uplift. A good model shows a falling staircase.
    """
    d = pd.DataFrame({"y": np.asarray(y), "u": np.asarray(uplift), "t": np.asarray(t)})
    d["bin"] = pd.qcut(d.u.rank(method="first", ascending=False), n_bins,
                       labels=list(range(1, n_bins + 1)))
    rows = []
    for b, g in d.groupby("bin", observed=True):
        yt, yc = g.y[g.t == 1], g.y[g.t == 0]
        rt, rc = yt.mean(), yc.mean()
        rows.append({"bin": b, "n": len(g), "pred_uplift": g.u.mean(), "rate_t": rt, "rate_c": rc,
                     "actual_uplift": rt - rc,
                     "se": np.sqrt(rt * (1 - rt) / len(yt) + rc * (1 - rc) / len(yc))})
    return pd.DataFrame(rows).set_index("bin")


# ---------------------------------------------------------------------------
# Step 9: targeting curves and profit
# ---------------------------------------------------------------------------

def targeting_curve(score, y, t, exposure, grid=None) -> pd.DataFrame:
    """For each targeting share X, what happens if we target the top X% by `score`?

    Within the top X% of users (ranked by score) we measure, from the experiment:
      itt       = visit rate(treated) - visit rate(control)   -> incremental visits per targeted user
      expo_rate = share of treated users who were actually exposed -> impressions per targeted user
    Everything is per targeted user, so it can be scaled to any population size.
    """
    score, y, t, exposure = (np.asarray(a) for a in (score, y, t, exposure))
    order = np.argsort(-score, kind="stable")
    y, t, exposure = y[order], t[order], exposure[order]
    n = len(y)
    grid = np.round(np.arange(0.01, 1.0001, 0.01), 2) if grid is None else np.asarray(grid)

    cum_t = np.cumsum(t)
    cum_c = np.cumsum(1 - t)
    cum_yt = np.cumsum(y * t)
    cum_yc = np.cumsum(y * (1 - t))
    cum_et = np.cumsum(exposure * t)
    rows = []
    for x in grid:
        k = max(int(round(x * n)), 1) - 1
        rt = cum_yt[k] / cum_t[k]
        rc = cum_yc[k] / max(cum_c[k], 1)
        rows.append({"share": x, "n_users": k + 1, "itt": rt - rc,
                     "expo_rate": cum_et[k] / cum_t[k], "rate_t": rt, "rate_c": rc})
    return pd.DataFrame(rows).set_index("share")


def add_profit(curve: pd.DataFrame, population: int, ad_cost: float, visit_value: float,
               cost_model: str = "impression", effect_scale: float = 1.0) -> pd.DataFrame:
    """Add incremental visits, cost and profit for a population of `population` users.

    cost_model = "impression": pay ad_cost per ad actually shown (CPM-style buying).
    cost_model = "user":       pay ad_cost per user targeted, shown or not.
    effect_scale lets us shrink the lift (e.g. 0.76 = covariate-adjusted scenario from Step 6).
    """
    out = curve.copy()
    targeted = out.index.values * population
    out["targeted_users"] = targeted
    out["incr_visits"] = out.itt * effect_scale * targeted
    out["impressions"] = out.expo_rate * targeted
    paid_units = out.impressions if cost_model == "impression" else targeted
    out["ad_cost"] = paid_units * ad_cost
    out["value"] = out.incr_visits * visit_value
    out["profit"] = out.value - out.ad_cost
    out["roi"] = out.profit / out.ad_cost
    return out
