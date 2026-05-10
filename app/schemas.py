"""Pydantic models matching `agents/sidecars/causal-sidecar.contract.yml`."""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, conlist


class CohortSource(str, Enum):
    uk_biobank = "uk_biobank"
    all_of_us = "all_of_us"
    agedefy_federated_v1 = "agedefy_federated_v1"


class Estimator(str, Enum):
    backdoor_linear_regression = "backdoor.linear_regression"
    iv_instrumental_variable = "iv.instrumental_variable"
    dml_causal_forest = "dml.causal_forest"


class Refuter(str, Enum):
    placebo_treatment = "placebo_treatment"
    random_common_cause = "random_common_cause"
    data_subset_refuter = "data_subset_refuter"


class EstimateRequest(BaseModel):
    exposure: str = Field(..., examples=["rapamycin_6mg_weekly"])
    outcome: str = Field(..., examples=["hs_crp"])
    covariates: list[str] = Field(default_factory=list)
    cohort_source: CohortSource
    estimator: Estimator = Estimator.backdoor_linear_regression
    n_bootstrap: int = Field(default=200, ge=1, le=5000)
    user_profile_hash: str | None = Field(
        default=None,
        description="SHA-256 of the requesting user's biomarker vector (no PII).",
    )


class SensitivityReport(BaseModel):
    pleiotropy_pvalue: float | None = None
    weak_instrument_f_stat: float | None = None
    collider_bias_flag: bool = False


CI95 = Annotated[list[float], conlist(float, min_length=2, max_length=2)]  # type: ignore[type-arg]


class EstimateResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    estimate_id: str
    expected_delta: float
    ci95: list[float] = Field(..., min_length=2, max_length=2)
    n_similar_profiles: int
    identification_strategy: str
    sensitivity_report: SensitivityReport = Field(default_factory=SensitivityReport)
    model_version: str
    dag_serialization: str | None = None


class RefuteRequest(BaseModel):
    estimate_id: str
    refuter: Refuter


class RefuteResponse(BaseModel):
    estimate_id: str
    refuter: Refuter
    refutation_pvalue: float
    passed: bool
    detail: str
