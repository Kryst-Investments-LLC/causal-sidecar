from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_healthz() -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert "model_version" in body


def _estimate_payload() -> dict:
    return {
        "exposure": "rapamycin_6mg_weekly",
        "outcome": "hs_crp",
        "covariates": ["age", "sex", "bmi"],
        "cohort_source": "agedefy_federated_v1",
        "estimator": "backdoor.linear_regression",
        "n_bootstrap": 200,
    }


def test_estimate_returns_contract_shape() -> None:
    r = client.post("/v1/estimate", json=_estimate_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    for key in (
        "estimate_id",
        "expected_delta",
        "ci95",
        "n_similar_profiles",
        "identification_strategy",
        "sensitivity_report",
        "model_version",
    ):
        assert key in body
    assert len(body["ci95"]) == 2
    assert body["ci95"][0] <= body["expected_delta"] <= body["ci95"][1]
    assert body["n_similar_profiles"] >= 1


def test_estimate_is_deterministic() -> None:
    r1 = client.post("/v1/estimate", json=_estimate_payload()).json()
    r2 = client.post("/v1/estimate", json=_estimate_payload()).json()
    assert r1["estimate_id"] == r2["estimate_id"]
    assert r1["expected_delta"] == r2["expected_delta"]
    assert r1["ci95"] == r2["ci95"]


def test_iv_estimator_emits_iv_sensitivity_fields() -> None:
    payload = _estimate_payload() | {"estimator": "iv.instrumental_variable"}
    body = client.post("/v1/estimate", json=payload).json()
    sens = body["sensitivity_report"]
    assert sens["pleiotropy_pvalue"] is not None
    assert sens["weak_instrument_f_stat"] is not None


def test_refute_endpoint() -> None:
    est = client.post("/v1/estimate", json=_estimate_payload()).json()
    r = client.post(
        "/v1/refute",
        json={"estimate_id": est["estimate_id"], "refuter": "placebo_treatment"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["estimate_id"] == est["estimate_id"]
    assert body["refuter"] == "placebo_treatment"
    assert 0.0 <= body["refutation_pvalue"] <= 1.0
    assert isinstance(body["passed"], bool)


def test_invalid_cohort_source_rejected() -> None:
    bad = _estimate_payload() | {"cohort_source": "not_a_real_cohort"}
    assert client.post("/v1/estimate", json=bad).status_code == 422


def test_trace_header_propagation() -> None:
    tp = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    r = client.get("/healthz", headers={"traceparent": tp})
    assert r.headers.get("traceparent") == tp
