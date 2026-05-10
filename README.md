# causal-sidecar

Python sidecar implementing the [`causal-sidecar` contract](https://github.com/Kryst-Investments-LLC/agedefy-ai/blob/main/agents/sidecars/causal-sidecar.contract.yml)
consumed by `agedefy-ai`'s `causal-inference-agent`.

Stateless FastAPI service hosting [DoWhy](https://github.com/py-why/dowhy),
[EconML](https://github.com/py-why/econml), and
[statsmodels](https://www.statsmodels.org/) estimators. Cohort data is mounted
read-only at `/data`; egress is forbidden by the deployment manifest.

## Endpoints

| Method | Path           | Purpose                                            |
| ------ | -------------- | -------------------------------------------------- |
| POST   | `/v1/estimate` | Identify + estimate causal effect of an exposure   |
| POST   | `/v1/refute`   | Run refutation / sensitivity tests on an estimate  |
| GET    | `/healthz`     | Liveness                                           |

Estimators (selected per request):

- `backdoor.linear_regression` (default) — DoWhy backdoor adjustment
- `iv.instrumental_variable` — DoWhy / statsmodels 2SLS with weak-instrument F
- `dml.causal_forest` — EconML CausalForestDML

## Local dev

```bash
poetry install
poetry run uvicorn app.main:app --reload --port 8080
curl http://localhost:8080/healthz
```

## Container

```bash
docker build -t ghcr.io/kryst-investments-llc/causal-sidecar:0.1.0 .
docker run --rm -p 8080:8080 \
  -v "$(pwd)/data:/data:ro" \
  ghcr.io/kryst-investments-llc/causal-sidecar:0.1.0
```

## Security

- mTLS terminated at the platform mesh (Linkerd / Istio); the app trusts the
  `x-forwarded-client-cert` header injected by the sidecar.
- W3C trace context propagated via `traceparent` / `tracestate`.
- No outbound network egress from the container.

## Status

`v0.1.0` — scaffold. Estimator implementations are stubs that return
deterministic values for contract testing. Real DoWhy/EconML pipelines land
in `v0.2.0`.
