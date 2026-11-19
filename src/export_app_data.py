"""Export every number the web app shows into app/public/data/app_data.json.

Nothing in the app is hard-coded: re-run this script after re-training and the
dashboard updates. Usage (from the project root):

    python -m src.export_app_data
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data import AD_COST, FEATURES, PROJECT_ROOT, VISIT_VALUE, load_criteo
from src.evaluation import bootstrap_metric, qini_auc_score, uplift_metrics
from src.stats_tests import (balance_table, itt_effect, regression_adjusted_effect,
                             srm_test, two_proportion_test, wald_cace)
from src.uplift_models import prognostic_score

OUT = PROJECT_ROOT / "app" / "public" / "data" / "app_data.json"
MODELS = PROJECT_ROOT / "models"


def r(x, nd=8):
    """Round numpy/python numbers for compact JSON.

    8 decimals keeps rates exact enough that the app's 2-decimal display matches
    the notebooks (e.g. 0.04885008 shows as 4.89%, not 4.88%).
    """
    return None if x is None else round(float(x), nd)


def experiment_summary(df: pd.DataFrame) -> dict:
    t = df.treatment.values
    n_t, n_c = int(t.sum()), int((t == 0).sum())
    srm = srm_test(n_t, n_c, 0.85)
    visit = two_proportion_test(df.visit[t == 1].values, df.visit[t == 0].values)
    conv = two_proportion_test(df.conversion[t == 1].values, df.conversion[t == 0].values)
    adj_visit = regression_adjusted_effect(df, "visit", FEATURES)
    adj_expo = regression_adjusted_effect(df, "exposure", FEATURES)
    cace = wald_cace(df.visit.values, df.exposure.values, t)
    tg = df[df.treatment == 1]
    naive = tg.visit[tg.exposure == 1].mean() - tg.visit[tg.exposure == 0].mean()
    smd = balance_table(df, FEATURES).abs_smd.max()
    prog = prognostic_score(df, FEATURES, "visit")
    prog_diff = itt_effect(prog[t == 1], prog[t == 0])
    return {
        "n": int(len(df)), "n_treatment": n_t, "n_control": n_c,
        "treatment_share": r(n_t / len(df)), "srm_chi2": r(srm["chi2"], 3), "srm_p": r(srm["p_value"], 4),
        "max_abs_smd": r(smd, 4),
        "prognostic_gap": r(prog_diff["effect"]), "prognostic_z": r(prog_diff["effect"] / prog_diff["se"], 2),
        "visit": {"rate_t": r(visit["rate_t"]), "rate_c": r(visit["rate_c"]), "itt": r(visit["abs_lift"]),
                  "ci": [r(visit["abs_ci_low"]), r(visit["abs_ci_high"])], "rel": r(visit["rel_lift"], 4),
                  "z": r(visit["z"], 2), "p": float(visit["p_value"]),
                  "adjusted": r(adj_visit["effect"]), "adjusted_ci": [r(adj_visit["ci_low"]), r(adj_visit["ci_high"])]},
        "conversion": {"rate_t": r(conv["rate_t"]), "rate_c": r(conv["rate_c"]), "itt": r(conv["abs_lift"]),
                       "ci": [r(conv["abs_ci_low"]), r(conv["abs_ci_high"])], "rel": r(conv["rel_lift"], 4)},
        "exposure_rate": r(tg.exposure.mean()),
        "cace": {"estimate": r(cace["cace"]), "ci": [r(cace["ci_low"]), r(cace["ci_high"])],
                 "first_stage_f": r(cace["first_stage_F"], 0),
                 "adjusted": r(adj_visit["effect"] / adj_expo["effect"])},
        "naive": r(naive), "selection_bias": r(naive - cace["cace"]),
    }


def model_summary(scores: pd.DataFrame) -> dict:
    y, t = scores.visit.values, scores.treatment.values
    out = {}
    for key, col in [("s_learner", "score_s_learner"), ("activity", "score_activity")]:
        m = uplift_metrics(y, scores[col].values, t)
        # same settings as notebook 07 (200 resamples, seed 42) so the app and README agree
        b = bootstrap_metric(y, scores[col].values, t, qini_auc_score, n_boot=200)
        out[key] = {"qini": r(m["qini_auc"], 4), "qini_ci": [r(b["ci_low"], 4), r(b["ci_high"], 4)],
                    "uplift_at_10": r(m["uplift_at_10"]), "uplift_at_20": r(m["uplift_at_20"])}
    return out


def targeting_curves() -> dict:
    """Per targeting share: incremental visits and impressions per targeted user."""
    cur = pd.read_parquet(MODELS / "targeting_curves.parquet")
    names = {"C": "s_learner", "C′": "activity", "B": "random"}
    out = {}
    for code, name in names.items():
        c = cur[cur.strategy == code].sort_values("share")
        out[name] = {"share": [r(v, 2) for v in c.share], "itt": [r(v) for v in c.itt],
                     "expo_rate": [r(v) for v in c.expo_rate]}
    return out


def scorer_data(df: pd.DataFrame, scores: pd.DataFrame) -> dict:
    """What the in-browser scorer needs: feature percentiles, sample users, score distribution."""
    pct = np.linspace(0, 100, 101)
    feat_q = {f: [r(v, 5) for v in np.percentile(df[f], pct)] for f in FEATURES}
    s = scores.score_s_learner.values
    score_q = [r(v, 7) for v in np.percentile(s, pct)]
    rng = np.random.default_rng(7)
    # sample users across the whole score range so "random person" shows variety
    order = np.argsort(s)
    picks = order[np.linspace(0, len(s) - 1, 400).astype(int)]
    picks = rng.permutation(picks)
    sample = [[r(v, 5) for v in scores.iloc[i][FEATURES].values] for i in picks]
    return {"features": FEATURES, "feature_percentiles": feat_q, "score_percentiles": score_q,
            "sample_users": sample}


def main():
    df = load_criteo()
    scores = pd.read_parquet(MODELS / "test_scores.parquet")
    meta = joblib.load(MODELS / "best_uplift_model.joblib")
    data = {
        "assumptions": {"ad_cost": AD_COST, "visit_value": VISIT_VALUE, "population": 1_000_000,
                        "price_options": [0.5, 1.0, 1.5, 2.0]},
        "experiment": experiment_summary(df),
        "models": {"best": meta["name"], "config": meta["config"], **model_summary(scores)},
        "curves": targeting_curves(),
        "scorer": scorer_data(df, scores),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
