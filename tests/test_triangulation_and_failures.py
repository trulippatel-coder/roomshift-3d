"""Comprehensive unit tests for triangulation geometry, feature matching, and failure handling."""

import tempfile
from pathlib import Path
import cv2
import numpy as np
import pytest

from roomshift.config import SfMConfig, VideoConfig
from roomshift.data.video_loader import Keyframe, VideoLoader
from roomshift.sfm.base import CameraPose, ReconstructionResult
from roomshift.sfm.opencv_sfm import OpenCVSfMReconstructor


def test_triangulation_valid_cheirality():
    """Tests that triangulated points with positive depth in front of both cameras pass cheirality."""
    sfm = OpenCVSfMReconstructor()
    K = np.array([
        [500.0, 0.0, 320.0],
        [0.0, 500.0, 240.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # Camera 1 at origin
    R1 = np.eye(3, dtype=np.float64)
    t1 = np.zeros((3, 1), dtype=np.float64)

    # Camera 2 displaced horizontally along X by 0.5 units
    R2 = np.eye(3, dtype=np.float64)
    t2 = np.array([[-0.5], [0.0], [0.0]], dtype=np.float64)

    # Genuine 3D points in front of both cameras at depth Z = 2.0
    true_pts3d = np.array([
        [0.0, 0.0, 2.0],
        [0.2, -0.1, 2.5],
        [-0.1, 0.3, 1.8]
    ], dtype=np.float64)

    # Project into camera 1
    p1_cam = (R1 @ true_pts3d.T + t1).T
    pts1 = (K @ p1_cam.T).T
    pts1 = pts1[:, :2] / pts1[:, 2:]

    # Project into camera 2
    p2_cam = (R2 @ true_pts3d.T + t2).T
    pts2 = (K @ p2_cam.T).T
    pts2 = pts2[:, :2] / pts2[:, 2:]

    tri_pts, valid = sfm._triangulate_points_with_poses(
        K, R1, t1, R2, t2, pts1, pts2
    )

    assert np.all(valid)
    assert len(tri_pts) == len(true_pts3d)
    np.testing.assert_allclose(tri_pts, true_pts3d, atol=1e-3)


def test_triangulation_rejects_negative_depth():
    """Tests that points behind either camera are rejected by cheirality."""
    sfm = OpenCVSfMReconstructor()
    K = np.array([
        [500.0, 0.0, 320.0],
        [0.0, 500.0, 240.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    R1 = np.eye(3, dtype=np.float64)
    t1 = np.zeros((3, 1), dtype=np.float64)
    R2 = np.eye(3, dtype=np.float64)
    t2 = np.array([[-0.5], [0.0], [0.0]], dtype=np.float64)

    # Point behind camera 1 (Z = -2.0)
    behind_pts3d = np.array([[0.0, 0.0, -2.0]], dtype=np.float64)

    p1_cam = (R1 @ behind_pts3d.T + t1).T
    pts1 = (K @ p1_cam.T).T
    pts1 = pts1[:, :2] / pts1[:, 2:]

    p2_cam = (R2 @ behind_pts3d.T + t2).T
    pts2 = (K @ p2_cam.T).T
    pts2 = pts2[:, :2] / pts2[:, 2:]

    tri_pts, valid = sfm._triangulate_points_with_poses(
        K, R1, t1, R2, t2, pts1, pts2
    )

    assert not np.any(valid)
    assert len(tri_pts) == 0


def test_triangulation_rejects_zero_parallax_collinear_rays():
    """Tests that points with zero baseline / collinear rays are rejected."""
    sfm = OpenCVSfMReconstructor()
    K = np.eye(3, dtype=np.float64) * 500.0
    K[2, 2] = 1.0

    # Two identical cameras with zero baseline
    R1 = np.eye(3, dtype=np.float64)
    t1 = np.zeros((3, 1), dtype=np.float64)
    R2 = np.eye(3, dtype=np.float64)
    t2 = np.zeros((3, 1), dtype=np.float64)

    pts1 = np.array([[100.0, 100.0]], dtype=np.float32)
    pts2 = np.array([[100.0, 100.0]], dtype=np.float32)

    tri_pts, valid = sfm._triangulate_points_with_poses(
        K, R1, t1, R2, t2, pts1, pts2
    )
    # Rays are completely collinear; parallax check must reject
    assert not np.any(valid)


def test_feature_matching_mutual_consistency():
    """Tests that feature matcher enforces mutual nearest neighbor consistency."""
    sfm = OpenCVSfMReconstructor()
    # Create distinct synthetic descriptor matrices
    descs1 = np.eye(32, dtype=np.float32)
    descs2 = np.eye(32, dtype=np.float32)

    matches = sfm._match_pairs(descs1, descs2)
    assert len(matches) == 32
    # Check 1-to-1 mapping
    for q, t in matches:
        assert q == t


def test_video_loader_nonexistent_file_raises():
    """Tests that VideoLoader raises FileNotFoundError for missing paths."""
    loader = VideoLoader()
    with pytest.raises(FileNotFoundError):
        loader.extract_keyframes("nonexistent_video_path.mp4")


def test_video_loader_empty_video_raises(tmp_path):
    """Tests that VideoLoader raises ValueError for unreadable or 0-byte video files."""
    empty_file = tmp_path / "corrupt.mp4"
    empty_file.write_bytes(b"")

    loader = VideoLoader()
    with pytest.raises(ValueError):
        loader.extract_keyframes(empty_file)


def test_sfm_insufficient_keyframes_raises():
    """Tests that SfM reconstructor rejects single-view input."""
    sfm = OpenCVSfMReconstructor()
    single_kf = [Keyframe(0, 0.0, np.zeros((200, 200, 3), dtype=np.uint8), 50.0)]
    with pytest.raises(ValueError):
        sfm.reconstruct(single_kf)
