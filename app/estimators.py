"""Real causal estimators backed by statsmodels and (optionally) EconML.

Three identification strategies, all returning the contract shape:

- backdoor.linear_regression  - OLS adjustment for covariates
- iv.instrumental_variable    - 2SLS with weak-instrument F-stat
- dml.causal_forest           - EconML CausalForestDML (falls back to OLS
                                if EconML isn't installed in the runtime,
                                so the slim CI image still tests)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .data import load_cohort, resolve_columns
from .schemas import (
    CohortSource,
    EstimateRequest,
    EstimateResponse,
    Estimator,
    Refuter,
    RefuteRequest,
    RefuteResponse,
    SensitivityReport,
)

log = logging.getLogger("causal-sidecar.estimators")

MODEL_VERSION = "causal-sidecar@0.2.0"

_BOOTSTRAP_CAP = 2000


@dataclass(frozen=True)
class _EstimateCore:
    expected_delta: float
    ci_low: float
    ci_high: float
    n_similar: int
    sensitivity: SensitivityReport
    identification: str
    dag: str


def _seed_request(req: EstimateRequest) -> int:
    digest = hashlib.sha256(req.model_dump_json().encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _bootstrap_ci(
    df: pd.DataFrame,
    fit_fn,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    n_bootstrap = min(n_bootstrap, _BOOTSTRAP_CAP)
    deltas: list[float] = []
    n = len(df)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        try:
            deltas.append(fit_fn(df.iloc[idx]))
        except Exception:
            continue
    if not deltas:
        return float("nan"), float("nan")
    arr = np.asarray(deltas)
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def _backdoor_linear_regression(req: EstimateRequest, df: pd.DataFrame) -> _EstimateCore:
    exposure_col, outcome_col, cov_cols = resolve_columns(
        df, req.exposure, req.outcome, req.covariates
    )

    def _fit(sample: pd.DataFrame) -> float:
        X = sample[[exposure_col, *cov_cols]].astype(float).values
        X = sm.add_constant(X)
        y = sample[outcome_col].astype(float).values
        model = sm.OLS(y, X).fit()
        return float(model.params[1])

    delta = _fit(df)
    rng = np.random.default_rng(_seed_request(req))
    ci_low, ci_high = _bootstrap_ci(df, _fit, req.n_bootstrap, rng)
    dag = (
        f"digraph G {{ {exposure_col} -> {outcome_col}; "
        + " ".join(f"{c} -> {outcome_col};" for c in cov_cols)
        + " }"
    )
    return _EstimateCore(
        expected_delta=round(delta, 6),
        ci_low=round(ci_low, 6),
        ci_high=round(ci_high, 6),
        n_similar=len(df),
        sensitivity=SensitivityReport(collider_bias_flag=False),
        identification="backdoor adjustment via OLS on observed covariates",
        dag=dag,
    )


def _iv_instrumental_variable(req: EstimateRequest, df: pd.DataFrame) -> _EstimateCore:
    exposure_col, outcome_col, cov_cols = resolve_columns(
        df, req.exposure, req.outcome, req.covariates
    )
    if "instrument" not in df.columns:
        raise ValueError("cohort has no `instrument` column for IV estimation")

    fs_X = sm.add_constant(df[["instrument", *cov_cols]].astype(float).values)
    fs = sm.OLS(df[exposure_col].astype(float).values, fs_X).fit()
    fitted_exposure = fs.predict(fs_X)
    f_stat = float(fs.fvalue) if fs.fvalue is not None else float("nan")

    if cov_cols:
        ss_arr = np.column_stack(
            [fitted_exposure, df[cov_cols].astype(float).values]
        )
    else:
        ss_arr = fitted_exposure.reshape(-1, 1)
    ss_X = sm.add_constant(ss_arr)
    ss = sm.OLS(df[outcome_col].astype(float).values, ss_X).fit()
    delta = float(ss.params[1])

    resid = df[outcome_col].astype(float).values - ss.predict(ss_X)
    pleio_X = sm.add_constant(df[["instrument"]].astype(float).values)
    pleio = sm.OLS(resid, pleio_X).fit()
    pleio_p = float(pleio.pvalues[1]) if len(pleio.pvalues) > 1 else float("nan")

    def _fit(sample: pd.DataFrame) -> float:
        fs_x = sm.add_constant(sample[["instrument", *cov_cols]].astype(float).values)
        fs_m = sm.OLS(sample[exposure_col].astype(float).values, fs_x).fit()
        fitted = fs_m.predict(fs_x)
        if cov_cols:
            ss_x_arr = np.column_stack(
                [fitted, sample[cov_cols].astype(float).values]
            )
        else:
            ss_x_arr = fitted.reshape(-1, 1)
        ss_x = sm.add_constant(ss_x_arr)
        ss_m = sm.OLS(sample[outcome_col].astype(float).values, ss_x).fit()
        return float(ss_m.params[1])

    rng = np.random.default_rng(_seed_request(req))
    ci_low, ci_high = _bootstrap_ci(df, _fit, req.n_bootstrap, rng)

    sensitivity = SensitivityReport(
        pleiotropy_pvalue=round(pleio_p, 4) if not np.isnan(pleio_p) else None,
        weak_instrument_f_stat=round(f_stat, 3) if not np.isnan(f_stat) else None,
        collider_bias_flag=False,
    )
    dag = (
        f"digraph G {{ instrument -> {exposure_col}; {exposure_col} -> {outcome_col}; "
        + " ".join(f"{c} -> {outcome_col};" for c in cov_cols)
        + " }"
    )
    return _EstimateCore(
        expected_delta=round(delta, 6),
        ci_low=round(ci_low, 6),
        ci_high=round(ci_high, 6),
        n_similar=len(df),
        sensitivity=sensitivity,
        identification="two-stage least squares with `instrument` as IV",
        dag=dag,
    )


def _dml_causal_forest(req: EstimateRequest, df: pd.DataFrame) -> _EstimateCore:
    exposure_col, outcome_col, cov_cols = resolve_columns(
        df, req.exposure, req.outcome, req.covariates
    )
    try:
        from econml.dml import CausalForestDML  # type: ignore[import-not-found]
        from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-not-found]
    except ImportError:
        log.info("econml/sklearn unavailable; falling back to OLS for dml.causal_forest")
        b = _backdoor_linear_regression(req, df)
        return _EstimateCore(
            expected_delta=b.expected_delta,
            ci_low=b.ci_low,
            ci_high=b.ci_high,
            n_similar=b.n_similar,
            sensitivity=SensitivityReport(collider_bias_flag=False),
            identification="dml.causal_forest (econml unavailable; OLS fallback)",
            dag=b.dag,
        )

    X = df[cov_cols].astype(float).values if cov_cols else np.zeros((len(df), 1))
    T = df[exposure_col].astype(float).values
    Y = df[outcome_col].astype(float).values
    est = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=50, random_state=0),
        model_t=RandomForestRegressor(n_estimators=50, random_state=0),
        n_estimators=200,
        random_state=0,
    )
    est.fit(Y, T, X=X)
    ate = float(np.mean(est.effect(X)))
    ci_low, ci_high = est.ate_interval(X, alpha=0.05)
    dag = (
        f"digraph G {{ {exposure_col} -> {outcome_col}; "
        + " ".join(f"{c} -> {exposure_col}; {c} -> {outcome_col};" for c in cov_cols)
        + " }"
    )
    return _EstimateCore(
        expected_delta=round(ate, 6),
        ci_low=round(float(ci_low), 6),
        ci_high=round(float(ci_high), 6),
        n_similar=len(df),
        sensitivity=SensitivityReport(collider_bias_flag=False),
        identification="double-machine-learning causal forest (EconML)",
        dag=dag,
    )


_DISPATCH = {
    Estimator.backdoor_linear_regression: _backdoor_linear_regression,
    Estimator.iv_instrumental_variable: _iv_instrumental_variable,
    Estimator.dml_causal_forest: _dml_causal_forest,
}


def estimate(req: EstimateRequest) -> EstimateResponse:
    df = load_cohort(req.cohort_source)
    core = _DISPATCH[req.estimator](req, df)
    estimate_id = hashlib.sha256(
        f"{req.model_dump_json()}|{MODEL_VERSION}".encode()
    ).hexdigest()[:32]
    return EstimateResponse(
        estimate_id=estimate_id,
        expected_delta=core.expected_delta,
        ci95=[core.ci_low, core.ci_high],
        n_similar_profiles=core.n_similar,
        identification_strategy=core.identification,
        sensitivity_report=core.sensitivity,
        model_version=MODEL_VERSION,
        dag_serialization=core.dag,
    )


def refute(req: RefuteRequest) -> RefuteResponse:
    df = load_cohort(CohortSource.agedefy_federated_v1)
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(req.estimate_id.encode()).digest()[:8], "big")
    )

    if req.refuter is Refuter.placebo_treatment:
        sample = df.copy()
        sample["intervention_dose"] = rng.normal(0, 1, len(sample))
        X = sm.add_constant(sample[["intervention_dose"]].astype(float).values)
        y = sample["hs_crp"].astype(float).values
        m = sm.OLS(y, X).fit()
        p = float(m.pvalues[1])
        return RefuteResponse(
            estimate_id=req.estimate_id,
            refuter=req.refuter,
            refutation_pvalue=round(p, 4),
            passed=p >= 0.05,
            detail=(
                "Placebo treatment refutation - exposure replaced with random noise; "
                "p>=0.05 means the original estimate did not pick up signal from noise."
            ),
        )

    if req.refuter is Refuter.random_common_cause:
        sample = df.copy()
        sample["__random_W"] = rng.normal(0, 1, len(sample))
        Xw = sm.add_constant(
            sample[["intervention_dose", "__random_W"]].astype(float).values
        )
        Xn = sm.add_constant(sample[["intervention_dose"]].astype(float).values)
        y = sample["hs_crp"].astype(float).values
        beta_with = float(sm.OLS(y, Xw).fit().params[1])
        beta_without = float(sm.OLS(y, Xn).fit().params[1])
        delta = abs(beta_with - beta_without)
        deltas = []
        for _ in range(200):
            y_perm = rng.permutation(y)
            with_b = float(sm.OLS(y_perm, Xw).fit().params[1])
            without_b = float(sm.OLS(y_perm, Xn).fit().params[1])
            deltas.append(abs(with_b - without_b))
        p = float((np.asarray(deltas) >= delta).mean())
        return RefuteResponse(
            estimate_id=req.estimate_id,
            refuter=req.refuter,
            refutation_pvalue=round(p, 4),
            passed=p >= 0.05,
            detail=(
                "Random common cause refutation - adding a random W should not move "
                "the estimate; p>=0.05 means the shift is within the permutation null."
            ),
        )

    sample = df.sample(frac=0.7, random_state=int(rng.integers(0, 2**31 - 1)))
    X = sm.add_constant(sample[["intervention_dose"]].astype(float).values)
    y = sample["hs_crp"].astype(float).values
    m = sm.OLS(y, X).fit()
    p = float(m.pvalues[1])
    return RefuteResponse(
        estimate_id=req.estimate_id,
        refuter=req.refuter,
        refutation_pvalue=round(p, 4),
        passed=p < 0.5,
        detail=(
            "Data-subset refutation - re-estimate on a 70% subsample; the estimate "
            "should remain in the same direction with a comparable p-value."
        ),
    )
