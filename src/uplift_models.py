"""Model helpers for the Criteo uplift project.

causalml does not build on Python 3.13 / Windows, so the S/T/X meta-learners
are implemented here on top of LightGBM.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold


def prognostic_score(df: pd.DataFrame, features: list[str], outcome: str = "visit",
                     treatment_col: str = "treatment", n_splits: int = 5,
                     seed: int = 0, **lgb_params) -> np.ndarray:
    """Predicted outcome *without* treatment, from a model trained on control users only.

    Control users get out-of-fold (cross-fitted) predictions so their score is not
    overfit; treatment users are scored by a model trained on all control users.
    """
    params = {"n_estimators": 200, "learning_rate": 0.05, "verbose": -1, **lgb_params}
    T = df[treatment_col].values
    y = df[outcome].values
    X = df[features]
    score = np.zeros(len(df))
    ctrl, treat = np.where(T == 0)[0], np.where(T == 1)[0]
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(ctrl):
        m = lgb.LGBMClassifier(**params).fit(X.iloc[ctrl[tr]], y[ctrl[tr]])
        score[ctrl[te]] = m.predict_proba(X.iloc[ctrl[te]])[:, 1]
    m = lgb.LGBMClassifier(**params).fit(X.iloc[ctrl], y[ctrl])
    score[treat] = m.predict_proba(X.iloc[treat])[:, 1]
    return score


# ---------------------------------------------------------------------------
# Step 8: Meta-learners (LightGBM base models)
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31,
                  "min_child_samples": 200, "subsample": 0.8, "subsample_freq": 1,
                  "colsample_bytree": 0.8, "verbose": -1}


class SLearner:
    """One model mu(x, t) with treatment as a feature. Uplift = mu(x, 1) - mu(x, 0).

    Simple, but trees may rarely split on t when the effect is small, which
    shrinks predicted uplift towards zero.
    """
    name = "S-Learner"

    def __init__(self, **params):
        self.params = {**DEFAULT_PARAMS, **params}

    def fit(self, X, t, y):
        self.model = lgb.LGBMClassifier(**self.params).fit(X.assign(treatment=np.asarray(t)), y)
        return self

    def predict_uplift(self, X):
        p1 = self.model.predict_proba(X.assign(treatment=1))[:, 1]
        p0 = self.model.predict_proba(X.assign(treatment=0))[:, 1]
        return p1 - p0


class TLearner:
    """Two models: mu1 on treated users, mu0 on control users. Uplift = mu1(x) - mu0(x).

    With an 85:15 split the control model sees far less data, and its noise
    flows straight into the uplift estimate.
    """
    name = "T-Learner"

    def __init__(self, **params):
        self.params = {**DEFAULT_PARAMS, **params}

    def fit(self, X, t, y):
        t, y = np.asarray(t), np.asarray(y)
        self.m1 = lgb.LGBMClassifier(**self.params).fit(X[t == 1], y[t == 1])
        self.m0 = lgb.LGBMClassifier(**self.params).fit(X[t == 0], y[t == 0])
        return self

    def predict_uplift(self, X):
        return self.m1.predict_proba(X)[:, 1] - self.m0.predict_proba(X)[:, 1]


class XLearner:
    """X-Learner (Kunzel et al., 2019).

    Stage 1: outcome models mu0 (control) and mu1 (treated), as in the T-learner.
    Stage 2: imputed individual effects
        treated users:  D1 = y - mu0(x)   (what happened minus what the control model expects)
        control users:  D0 = mu1(x) - y   (what the treated model expects minus what happened)
      then regress tau1 on D1 (treated rows) and tau0 on D0 (control rows).
    Stage 3: tau(x) = g * tau0(x) + (1 - g) * tau1(x), with g = P(T = 1).

    With 85% treated, g = 0.85, so the estimate leans on tau0. tau0 is fit on the
    small control group, but its targets use mu1, learned from the big treated
    group. That borrowing is why the X-learner suits imbalanced arms.
    """
    name = "X-Learner"

    def __init__(self, propensity=None, clf_params=None, reg_params=None):
        self.propensity = propensity
        self.clf_params = {**DEFAULT_PARAMS, **(clf_params or {})}
        self.reg_params = {**DEFAULT_PARAMS, **(reg_params or {})}

    def fit(self, X, t, y):
        t, y = np.asarray(t), np.asarray(y).astype(float)
        self.m1 = lgb.LGBMClassifier(**self.clf_params).fit(X[t == 1], y[t == 1])
        self.m0 = lgb.LGBMClassifier(**self.clf_params).fit(X[t == 0], y[t == 0])
        d1 = y[t == 1] - self.m0.predict_proba(X[t == 1])[:, 1]
        d0 = self.m1.predict_proba(X[t == 0])[:, 1] - y[t == 0]
        self.tau1 = lgb.LGBMRegressor(**self.reg_params).fit(X[t == 1], d1)
        self.tau0 = lgb.LGBMRegressor(**self.reg_params).fit(X[t == 0], d0)
        self.g = t.mean() if self.propensity is None else self.propensity
        return self

    def predict_uplift(self, X):
        return self.g * self.tau0.predict(X) + (1 - self.g) * self.tau1.predict(X)


class ActivityBaseline:
    """Not an uplift model: ranks users by predicted visit WITHOUT ads.

    The 'obvious' targeting rule from Step 7 (target the most active users).
    Uplift models have to beat this to be worth it.
    """
    name = "Activity score (baseline)"

    def __init__(self, **params):
        self.params = {**DEFAULT_PARAMS, **params}

    def fit(self, X, t, y):
        t = np.asarray(t)
        self.m0 = lgb.LGBMClassifier(**self.params).fit(X[t == 0], np.asarray(y)[t == 0])
        return self

    def predict_uplift(self, X):
        return self.m0.predict_proba(X)[:, 1]
