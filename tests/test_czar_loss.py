"""Regression tests for the public CZAR loss and LightGBM objective."""

import numpy as np
import pytest

from allora_forge_builder_kit import (
    czar_gradient,
    czar_hessian,
    czar_loss,
    make_czar_objective,
)


def test_three_regions_match_canonical_reference_values():
    """Pin the loss and intentional pseudo grad/hess in all three regions."""
    y_true = np.array([1.0, 1.0, 1.0])
    y_pred = np.array([-0.5, 0.5, 1.5])
    std = np.ones(3)

    np.testing.assert_allclose(
        czar_loss(y_true, y_pred, std),
        [1.277257108913288, 0.270314256882699, 0.260455524262833],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        czar_gradient(y_true, y_pred, std),
        [-1.433012701892219, -0.617088652765469, 0.172864372318235],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        czar_hessian(y_true, y_pred, std),
        [0.866025403784439, 0.997230939256113, 0.866025403784439],
        rtol=1e-12,
    )


@pytest.mark.parametrize("y_pred", [-0.5, 0.5])
def test_gradient_matches_loss_finite_difference_in_analytical_regions(y_pred):
    """Regions 1/2 are analytical; region 3 is intentionally a pseudo-gradient."""
    step = 1e-6
    upper = czar_loss(1.0, y_pred + step, 1.0, epsilon=0)
    lower = czar_loss(1.0, y_pred - step, 1.0, epsilon=0)
    finite_difference = (upper - lower) / (2 * step)

    assert czar_gradient(1.0, y_pred, 1.0) == pytest.approx(
        finite_difference, rel=1e-6, abs=1e-8
    )


def test_scalar_and_row_specific_std_are_supported():
    y_true = np.array([0.2, -0.3, 0.4])
    y_pred = np.array([0.1, -0.1, 0.6])

    scalar_grad = czar_gradient(y_true, y_pred, 0.5)
    array_grad = czar_gradient(y_true, y_pred, np.full(3, 0.5))
    scalar_hess = czar_hessian(y_true, y_pred, 0.5)
    array_hess = czar_hessian(y_true, y_pred, np.full(3, 0.5))

    np.testing.assert_allclose(scalar_grad, array_grad)
    np.testing.assert_allclose(scalar_hess, array_hess)


def test_invalid_std_values_are_sanitized_to_finite_outputs():
    std = np.array([0.0, -1.0, np.nan, np.inf])
    y_true = np.zeros(4)
    y_pred = np.zeros(4)

    assert np.isfinite(czar_loss(y_true, y_pred, std)).all()
    assert np.isfinite(czar_gradient(y_true, y_pred, std)).all()
    assert np.isfinite(czar_hessian(y_true, y_pred, std)).all()


def test_zero_target_boundary_is_regression_pinned():
    assert czar_gradient(0.0, -0.25, 1.0) == pytest.approx(-0.21650635094610965)
    assert czar_hessian(0.0, -0.25, 1.0) == pytest.approx(0.8660254037844387)


@pytest.mark.parametrize("alpha", [-0.01, 1.01])
def test_loss_rejects_alpha_outside_unit_interval(alpha):
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        czar_loss(0.1, 0.2, 1.0, alpha=alpha)


class _DatasetLike:
    def __init__(self, labels):
        self._labels = labels

    def get_label(self):
        return self._labels


def test_objective_factory_accepts_arrays_and_dataset_like_labels():
    y_true = np.array([0.1, -0.2, 0.3])
    y_pred = np.array([0.05, -0.1, 0.4])
    objective = make_czar_objective(std=np.array([0.2, 0.3, 0.4]))

    array_grad, array_hess = objective(y_true, y_pred)
    dataset_grad, dataset_hess = objective(_DatasetLike(y_true), y_pred)

    np.testing.assert_allclose(array_grad, dataset_grad)
    np.testing.assert_allclose(array_hess, dataset_hess)
    assert array_grad.shape == y_true.shape
    assert array_hess.shape == y_true.shape
    assert np.isfinite(array_grad).all()
    assert np.isfinite(array_hess).all()
    assert (array_hess >= 1e-6).all()
