"""Tests for camera frustum visibility and unobserved region integrity (Requirement 7)."""

import numpy as np
import pytest

from roomshift.analysis.visibility import VisibilityChecker
from roomshift.config import VisibilityConfig
from roomshift.sfm.base import CameraPose


def test_visibility_frustum_in_and_out():
    K = np.array([
        [500.0, 0.0, 320.0],
        [0.0, 500.0, 240.0],
        [0.0, 0.0, 1.0]
    ])
    # Camera at world origin looking down +Z axis
    R = np.eye(3)
    t = np.zeros((3, 1))
    cam = CameraPose(frame_idx=0, R=R, t=t, K=K, image_width=640, image_height=480)

    checker = VisibilityChecker(VisibilityConfig(min_depth=0.1, max_depth_ratio=2.0))

    # Point directly in center of field of view at depth 2.0
    pt_visible = np.array([0.0, 0.0, 2.0])
    count_vis = checker.check_point_visibility(pt_visible, [cam], max_scene_depth=5.0)
    assert count_vis == 1

    # Point BEHIND camera (Z = -1.0)
    pt_behind = np.array([0.0, 0.0, -1.0])
    count_behind = checker.check_point_visibility(pt_behind, [cam], max_scene_depth=5.0)
    assert count_behind == 0

    # Point far to the right outside horizontal FOV
    pt_outside = np.array([10.0, 0.0, 2.0])
    count_outside = checker.check_point_visibility(pt_outside, [cam], max_scene_depth=5.0)
    assert count_outside == 0


def test_unobserved_cluster_triggers_uncertainty():
    K = np.array([
        [500.0, 0.0, 320.0],
        [0.0, 500.0, 240.0],
        [0.0, 0.0, 1.0]
    ])
    # Camera looking forward
    cam = CameraPose(frame_idx=0, R=np.eye(3), t=np.zeros((3, 1)), K=K, image_width=640, image_height=480)
    checker = VisibilityChecker()

    # Cluster located behind the camera
    cluster_behind = np.array([
        [0.1, 0.1, -2.0],
        [0.2, 0.1, -2.0],
        [0.1, 0.2, -2.0]
    ])

    fraction, med_cams = checker.compute_cluster_visibility(cluster_behind, [cam])
    assert fraction == 0.0
    assert med_cams == 0
