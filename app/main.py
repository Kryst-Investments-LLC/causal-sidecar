"""FastAPI application for the causal sidecar."""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse

from . import __version__
from .estimators import MODEL_VERSION, estimate, refute
from .schemas import EstimateRequest, EstimateResponse, RefuteRequest, RefuteResponse

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger("causal-sidecar")

app = FastAPI(
    title="causal-sidecar",
    version=__version__,
    description="DoWhy/EconML/statsmodels sidecar for the AgeDefy causal-inference-agent",
    docs_url=None if os.environ.get("DISABLE_DOCS") == "1" else "/docs",
    redoc_url=None,
)


@app.middleware("http")
async def propagate_trace(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Forward W3C trace headers to logs and the response."""
    traceparent = request.headers.get("traceparent")
    response: Response = await call_next(request)
    if traceparent:
        response.headers["traceparent"] = traceparent
    return response


@app.get("/healthz", status_code=status.HTTP_200_OK)
def healthz() -> dict[str, str]:
    return {"status": "ok", "version": __version__, "model_version": MODEL_VERSION}


@app.post(
    "/v1/estimate",
    response_model=EstimateResponse,
    response_model_exclude_none=False,
)
def post_estimate(req: EstimateRequest) -> EstimateResponse:
    log.info(
        "estimate exposure=%s outcome=%s cohort=%s estimator=%s",
        req.exposure,
        req.outcome,
        req.cohort_source.value,
        req.estimator.value,
    )
    return estimate(req)


@app.post(
    "/v1/refute",
    response_model=RefuteResponse,
)
def post_refute(req: RefuteRequest) -> RefuteResponse:
    log.info("refute estimate_id=%s refuter=%s", req.estimate_id, req.refuter.value)
    return refute(req)


@app.exception_handler(ValueError)
async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )
