"""Estimator stubs.

The 0.1.0 scaffold returns deterministic values seeded by request fields so
contract tests in agedefy-ai can be wired before the real DoWhy/EconML
pipelines land in 0.2.0.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .schemas import (
    EstimateRequest,
    EstimateResponse,
    Estimator,
    RefuteRequest,
    RefuteResponse,
    SensitivityReport,
)

MODEL_VERSION = "causal-sidecar@0.1.0-stub"


@dataclass(frozen=True)
class _Seeded:
    delta: float
    half_width: float
    n: int


def _seed(req: EstimateRequest) -> _Seeded:
    digest = hashlib.sha256(
        f"{req.exposure}|{req.outcome}|{req.cohort_source.value}|{req.estimator.value}".encode()
    ).digest()
    a, b, c, d = digest[0], digest[1], digest[2], digest[3]
    delta = (a - 128) / 128.0  # in [-1, 1)
    half_width = max(0.05, (b / 255.0) * 0.5)
    n = 500 + (c * 256 + d) % 4500
    return _Seeded(delta=delta, half_width=half_width, n=n)


def _identification_strategy(estimator: Estimator) -> str:
    return {
        Estimator.backdoor_linear_regression: "backdoor adjustment via linear regression",
        Estimator.iv_instrumental_variable: "two-stage least squares with genetic instrument",
        Estimator.dml_causal_forest: "double machine learning with causal forest",
    }[estimator]


def _sensitivity(req: EstimateRequest) -> SensitivityReport:
    if req.estimator is Estimator.iv_instrumental_variable:
        return SensitivityReport(
            pleiotropy_pvalue=0.42,
            weak_instrument_f_stat=24.3,
            collider_bias_flag=False,
        )
    return SensitivityReport(collider_bias_flag=False)


def estimate(req: EstimateRequest) -> EstimateResponse:
    s = _seed(req)
    estimate_id = hashlib.sha256(
        f"{req.model_dump_json()}|{MODEL_VERSION}".encode()
    ).hexdigest()[:32]
    return EstimateResponse(
        estimate_id=estimate_id,
        expected_delta=round(s.delta, 4),
        ci95=[round(s.delta - s.half_width, 4), round(s.delta + s.half_width, 4)],
        n_similar_profiles=s.n,
        identification_strategy=_identification_strategy(req.estimator),
        sensitivity_report=_sensitivity(req),
        model_version=MODEL_VERSION,
        dag_serialization=None,
    )


def refute(req: RefuteRequest) -> RefuteResponse:
    seed_byte = hashlib.sha256(
        f"{req.estimate_id}|{req.refuter.value}".encode()
    ).digest()[0]
    pvalue = round(0.05 + (seed_byte / 255.0) * 0.9, 4)
    return RefuteResponse(
        estimate_id=req.estimate_id,
        refuter=req.refuter,
        refutation_pvalue=pvalue,
        passed=pvalue >= 0.05,
        detail=(
            f"Stub refutation ({req.refuter.value}): real implementation arrives in 0.2.0; "
            "deterministic value derived from estimate_id."
        ),
    )
