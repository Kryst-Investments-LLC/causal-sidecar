"""Cohort data loading.

Strategy: cohort data is mounted read-only at `/data/<cohort_source>.parquet`.
If absent, fall back to a deterministic synthetic dataset shaped like the
columns the estimators expect. The synthetic generator is seeded by
cohort_source so results are reproducible across calls.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .schemas import CohortSource

log = logging.getLogger("causal-sidecar.data")

DATA_ROOT = Path(os.environ.get("CAUSAL_DATA_ROOT", "/data"))
SYNTHETIC_N = int(os.environ.get("CAUSAL_SYNTHETIC_N", "5000"))


# Synthetic biomarker columns we generate for any cohort fallback.
# Real cohorts are expected to expose at least these columns.
BIOMARKERS = (
    "age",
    "sex",          # 0/1
    "bmi",
    "hs_crp",
    "ldl",
    "hdl",
    "hba1c",
    "epi_age",
    "vo2_max",
)

EXPOSURES_CONTINUOUS = {
    "rapamycin_6mg_weekly",
    "metformin_1500mg_daily",
    "nmn_1g_daily",
    "rapamycin",
    "metformin",
}


def _synthetic(cohort: CohortSource, n: int = SYNTHETIC_N) -> pd.DataFrame:
    """Deterministic synthetic dataset.

    Uses a seeded RNG so two calls with the same cohort key produce the same
    frame — important so estimate_id is stable across requests.
    """
    seed = int.from_bytes(cohort.value.encode(), "big") % (2**32 - 1)
    rng = np.random.default_rng(seed)
    age = rng.normal(55, 12, n).clip(18, 95)
    sex = rng.integers(0, 2, n)
    bmi = rng.normal(26 + 0.05 * (age - 55), 4, n).clip(15, 60)
    # simple structural model so estimators have something real to find
    base_hs_crp = 1.2 + 0.02 * (bmi - 25) + 0.01 * (age - 55) + rng.normal(0, 0.5, n)
    return pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "hs_crp": np.exp(base_hs_crp).clip(0.05, 50),
            "ldl": rng.normal(120, 30, n).clip(40, 300),
            "hdl": rng.normal(55, 15, n).clip(15, 120),
            "hba1c": rng.normal(5.5, 0.6, n).clip(4.0, 12.0),
            "epi_age": age + rng.normal(0, 4, n),
            "vo2_max": rng.normal(35 - 0.2 * (age - 55), 6, n).clip(10, 70),
            # canned exposures: a continuous "intervention dose" and a binary indicator
            "intervention_dose": rng.normal(0, 1, n),
            "intervention": rng.integers(0, 2, n),
            # an instrument correlated with intervention but not the outcome directly
            "instrument": rng.normal(0, 1, n),
        }
    )


@lru_cache(maxsize=8)
def load_cohort(cohort: CohortSource) -> pd.DataFrame:
    """Load a cohort frame.

    Looks for `/data/<cohort>.parquet` first, then `/data/<cohort>.csv`,
    and finally falls back to a deterministic synthetic dataset.
    """
    parquet = DATA_ROOT / f"{cohort.value}.parquet"
    csv = DATA_ROOT / f"{cohort.value}.csv"
    if parquet.exists():
        log.info("loading parquet cohort=%s path=%s", cohort.value, parquet)
        return pd.read_parquet(parquet)
    if csv.exists():
        log.info("loading csv cohort=%s path=%s", cohort.value, csv)
        return pd.read_csv(csv)
    log.warning(
        "no data file for cohort=%s at %s; using synthetic fixture (n=%d)",
        cohort.value,
        DATA_ROOT,
        SYNTHETIC_N,
    )
    return _synthetic(cohort)


def resolve_columns(
    df: pd.DataFrame,
    exposure: str,
    outcome: str,
    covariates: list[str],
) -> tuple[str, str, list[str]]:
    """Map request fields to dataframe columns.

    Real cohorts will have a column literally named like the exposure (e.g.
    `rapamycin_6mg_weekly`). For synthetic / fallback frames we don't have
    that, so we map any unknown exposure to the canned `intervention` /
    `intervention_dose` column. Same for outcomes.
    """
    if exposure in df.columns:
        exposure_col = exposure
    elif exposure in EXPOSURES_CONTINUOUS and "intervention_dose" in df.columns:
        exposure_col = "intervention_dose"
    elif "intervention" in df.columns:
        exposure_col = "intervention"
    else:
        raise ValueError(
            f"cohort has no column for exposure={exposure!r} and no fallback `intervention` column"
        )

    if outcome in df.columns:
        outcome_col = outcome
    else:
        raise ValueError(f"cohort has no column for outcome={outcome!r}")

    resolved_covs = [c for c in covariates if c in df.columns]
    return exposure_col, outcome_col, resolved_covs
