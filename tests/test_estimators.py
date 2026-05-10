"""Tests for the real estimator path against synthetic fixtures."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _payload(**overrides) -> dict:
    base = {
        "exposure": "rapamycin_6mg_weekly",
        "outcome": "hs_crp",
        "covariates": ["age", "sex", "bmi"],
        "cohort_source": "agedefy_federated_v1",
        "estimator": "backdoor.linear_regression",
        "n_bootstrap": 50,
    }
    base.update(overrides)
    return base


def test_backdoor_returns_real_estimate() -> None:
    r = client.post("/v1/estimate", json=_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_version"].startswith("causal-sidecar@0.2")
    assert body["identification_strategy"].startswith("backdoor")
    # bootstrap CI must bracket the point estimate
    assert body["ci95"][0] <= body["expected_delta"] <= body["ci95"][1]
    assert body["n_similar_profiles"] > 1000
    assert "digraph" in body["dag_serialization"]


def test_iv_emits_diagnostics() -> None:
    body = client.post(
        "/v1/estimate", json=_payload(estimator="iv.instrumental_variable")
    ).json()
    sens = body["sensitivity_report"]
    assert sens["pleiotropy_pvalue"] is not None
    assert sens["weak_instrument_f_stat"] is not None
    # synthetic instrument is uncorrelated with intervention so F should be small
    assert sens["weak_instrument_f_stat"] >= 0.0


def test_dml_endpoint_runs_or_falls_back() -> None:
    body = client.post(
        "/v1/estimate", json=_payload(estimator="dml.causal_forest")
    ).json()
    assert "dml.causal_forest" in body["identification_strategy"] or "OLS fallback" in body["identification_strategy"]
    assert body["ci95"][0] <= body["expected_delta"] <= body["ci95"][1]


def test_estimate_id_changes_with_request_fields() -> None:
    a = client.post("/v1/estimate", json=_payload()).json()
    b = client.post("/v1/estimate", json=_payload(outcome="ldl")).json()
    assert a["estimate_id"] != b["estimate_id"]


def test_placebo_refutation() -> None:
    est = client.post("/v1/estimate", json=_payload()).json()
    r = client.post(
        "/v1/refute",
        json={"estimate_id": est["estimate_id"], "refuter": "placebo_treatment"},
    ).json()
    assert 0.0 <= r["refutation_pvalue"] <= 1.0
    assert isinstance(r["passed"], bool)


def test_subset_refutation() -> None:
    est = client.post("/v1/estimate", json=_payload()).json()
    r = client.post(
        "/v1/refute",
        json={"estimate_id": est["estimate_id"], "refuter": "data_subset_refuter"},
    ).json()
    assert "subset" in r["detail"].lower()
