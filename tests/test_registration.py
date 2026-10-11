"""Tests for Sim(3) coordinate alignment and Umeyama algorithm."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from roomshift.alignment.registration import AlignmentManager


def test_umeyama_perfect_transform():
    np.random.seed(42)
    # Generate 3D source points
    source_pts = np.random.uniform(-2, 2, size=(30, 3))

    # Apply known Sim(3) transform: scale=1.45, rotation=35 deg around Z, translation=[0.8, -1.2, 0.5]
    true_scale = 1.45
    true_rot = Rotation.from_euler("z", 35, degrees=True).as_matrix()
    true_t = np.array([[0.8], [-1.2], [0.5]])

    target_pts = (true_scale * (true_rot @ source_pts.T) + true_t).T

    est_scale, est_R, est_t = AlignmentManager.umeyama_alignment(source_pts, target_pts, with_scale=True)

    assert est_scale == pytest.approx(true_scale, rel=1e-3)
    np.testing.assert_allclose(est_R, true_rot, atol=1e-3)
    np.testing.assert_allclose(est_t, true_t, atol=1e-3)


def test_umeyama_insufficient_points_raises():
    pts = np.array([[0, 0, 0], [1, 1, 1]])
    with pytest.raises(ValueError):
        AlignmentManager.umeyama_alignment(pts, pts)
