"""Tests for change detection, spatial clustering, and multi-modal classification."""

import numpy as np
import pytest

from roomshift.analysis.change_detector import ChangeDetector
from roomshift.analysis.classifier import ChangeClassifier, ClassifiedObject
from roomshift.analysis.visibility import VisibilityChecker
from roomshift.config import ChangeDetectionConfig
from roomshift.sfm.base import CameraPose


def test_distance_computation_and_clustering():
    np.random.seed(0)
    detector = ChangeDetector(ChangeDetectionConfig(distance_threshold=0.1, dbscan_eps=0.2, dbscan_min_points=5))

    # Base points
    pts_base = np.zeros((100, 3))
    cols_base = np.ones((100, 3))

    # Target points: 80 points identical, 20 points shifted away to (3, 3, 3)
    pts_target = np.zeros((100, 3))
    pts_target[80:] = np.random.uniform(2.8, 3.2, size=(20, 3))
    cols_target = np.ones((100, 3))

    dists = detector.compute_distances(pts_target, pts_base)
    assert np.all(dists[:80] < 0.01)
    assert np.all(dists[80:] > 2.5)

    clusters = detector.cluster_residual_points(pts_target, cols_target, dists, source_label="after")
    assert len(clusters) == 1
    assert clusters[0].num_points == 20
    assert clusters[0].centroid[0] > 2.5


def test_classifier_identifies_moved_pair():
    # Setup mock clusters for a moved object
    cb_pts = np.random.uniform(-0.1, 0.1, size=(30, 3)) + np.array([-1.0, 0.0, 0.0])
    ca_pts = np.random.uniform(-0.1, 0.1, size=(30, 3)) + np.array([1.0, 0.0, 0.0])

    detector = ChangeDetector()
    eig_b, el_b, pl_b, sp_b = detector.extract_shape_descriptors(cb_pts)
    eig_a, el_a, pl_a, sp_a = detector.extract_shape_descriptors(ca_pts)

    from roomshift.analysis.change_detector import SpatialCluster

    cluster_b = SpatialCluster(
        cluster_id=1, source_sequence="before", points=cb_pts, colors=np.ones_like(cb_pts),
        centroid=np.array([-1.0, 0.0, 0.0]), bbox_min=np.min(cb_pts, axis=0), bbox_max=np.max(cb_pts, axis=0),
        dimensions=np.ptp(cb_pts, axis=0), estimated_volume=0.05, eigenvalues=eig_b, elongation=el_b,
        planarity=pl_b, sphericity=sp_b, mean_color_rgb=np.array([0.5, 0.5, 0.5]),
        mean_distance_to_counterpart=2.0, visibility_in_other=1.0, median_cameras_observed=4
    )

    cluster_a = SpatialCluster(
        cluster_id=2, source_sequence="after", points=ca_pts, colors=np.ones_like(ca_pts),
        centroid=np.array([1.0, 0.0, 0.0]), bbox_min=np.min(ca_pts, axis=0), bbox_max=np.max(ca_pts, axis=0),
        dimensions=np.ptp(ca_pts, axis=0), estimated_volume=0.05, eigenvalues=eig_a, elongation=el_a,
        planarity=pl_a, sphericity=sp_a, mean_color_rgb=np.array([0.5, 0.5, 0.5]),
        mean_distance_to_counterpart=2.0, visibility_in_other=1.0, median_cameras_observed=4
    )

    classifier = ChangeClassifier()
    sim = classifier._compute_cluster_similarity(cluster_b, cluster_a)
    assert sim > 0.70  # High shape and volume similarity

    # Mock cameras
    K = np.eye(3) * 500
    cam = CameraPose(frame_idx=0, R=np.eye(3), t=np.zeros((3, 1)), K=K, image_width=640, image_height=480)

    classified = classifier.classify(
        [cluster_b], [cluster_a],
        [cam], [cam],
        cb_pts, ca_pts,
        np.array([2.0] * len(cb_pts)), np.array([2.0] * len(ca_pts))
    )

    labels = [o.label for o in classified]
    assert "moved" in labels
    moved_obj = next(o for o in classified if o.label == "moved")
    assert moved_obj.displacement_magnitude == pytest.approx(2.0, rel=0.05)
