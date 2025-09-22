from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

from portfolio.src.risk.engines import covariance


def test_sample_covariance_positive_semidefinite(unit_returns_df) -> None:
    sample_matrix = covariance.compute_sample_covariance(unit_returns_df)
    expected = np.cov(unit_returns_df.to_numpy(), rowvar=False, ddof=1)
    np.testing.assert_allclose(sample_matrix, expected, rtol=1e-7, atol=1e-9)
    assert covariance.is_positive_semidefinite(sample_matrix)


def test_ledoit_wolf_matches_reference(unit_returns_df) -> None:
    lw_matrix = covariance.compute_ledoit_wolf_covariance(unit_returns_df)
    from sklearn.covariance import LedoitWolf

    reference = LedoitWolf().fit(unit_returns_df.to_numpy()).covariance_
    np.testing.assert_allclose(lw_matrix, reference, rtol=1e-6)
    assert covariance.is_positive_semidefinite(lw_matrix)


def test_ewma_enforces_decay_bounds(unit_returns_df) -> None:
    with pytest.raises(ValueError):
        covariance.compute_ewma_covariance(unit_returns_df, decay_lambda=1.2)

    ewma_matrix = covariance.compute_ewma_covariance(unit_returns_df, decay_lambda=0.94)
    assert covariance.is_positive_semidefinite(ewma_matrix)
    assert np.all(np.diag(ewma_matrix) >= 0)
