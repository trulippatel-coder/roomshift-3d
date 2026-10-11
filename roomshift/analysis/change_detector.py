"""Point cloud geometric distance computation, DBSCAN clustering, and shape descriptor extraction."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

from roomshift.config import ChangeDetectionConfig

logger = logging.getLogger(__name__)


@dataclass
class SpatialCluster:
    """Represents a spatially coherent 3D cluster of changed or displaced points."""
    cluster_id: int
    source_sequence: str  # "before" or "after"
    points: np.ndarray    # (K, 3)
    colors: np.ndarray    # (K, 3) in [0, 1]
    centroid: np.ndarray  # (3,)
    bbox_min: np.ndarray  # (3,)
    bbox_max: np.ndarray  # (3,)
    dimensions: np.ndarray  # (3,) [dx, dy, dz]
    estimated_volume: float
    eigenvalues: np.ndarray  # (3,) sorted descending
    elongation: float
    planarity: float
    sphericity: float
    mean_color_rgb: np.ndarray  # (3,)
    mean_distance_to_counterpart: float
    visibility_in_other: float = 1.0  # Fraction observed by opposite cameras
    median_cameras_observed: int = 2

    @property
    def num_points(self) -> int:
        return len(self.points)


class ChangeDetector:
    """Analyzes geometric discrepancies between aligned point clouds to isolate candidate change clusters."""

    def __init__(self, config: Optional[ChangeDetectionConfig] = None):
        self.config = config or ChangeDetectionConfig()

    def compute_distances(
        self,
        pts_source: np.ndarray,
        pts_target: np.ndarray
    ) -> np.ndarray:
        """Computes Euclidean nearest-neighbor distance from each source point to the target cloud."""
        if len(pts_source) == 0 or len(pts_target) == 0:
            return np.zeros(len(pts_source))

        tree = KDTree(pts_target)
        dists, _ = tree.query(pts_source, k=1, workers=4)
        return dists.astype(np.float64)

    def extract_shape_descriptors(
        self,
        cluster_pts: np.ndarray
    ) -> Tuple[np.ndarray, float, float, float]:
        """Computes PCA eigenvalues and geometric invariants (elongation, planarity, sphericity)."""
        if len(cluster_pts) < 4:
            return np.ones(3), 0.0, 0.0, 1.0

        cov = np.cov(cluster_pts.T)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(np.maximum(eigvals, 1e-8))[::-1]

        total = np.sum(eigvals)
        l1, l2, l3 = eigvals / total

        elongation = float((l1 - l2) / l1) if l1 > 0 else 0.0
        planarity = float((l2 - l3) / l1) if l1 > 0 else 0.0
        sphericity = float(l3 / l1) if l1 > 0 else 0.0

        return eigvals, elongation, planarity, sphericity

    def cluster_residual_points(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        distances: np.ndarray,
        source_label: str
    ) -> List[SpatialCluster]:
        """Clusters residual changed points into spatial object candidates using DBSCAN."""
        high_dist_mask = distances > self.config.distance_threshold
        cand_pts = points[high_dist_mask]
        cand_cols = colors[high_dist_mask]
        cand_dists = distances[high_dist_mask]

        if len(cand_pts) < self.config.dbscan_min_points:
            return []

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cand_pts)

        labels = np.array(
            pcd.cluster_dbscan(
                eps=self.config.dbscan_eps,
                min_points=self.config.dbscan_min_points,
                print_progress=False
            )
        )

        clusters: List[SpatialCluster] = []
        unique_labels = [l for l in np.unique(labels) if l >= 0]

        for cid in unique_labels:
            mask = labels == cid
            c_pts = cand_pts[mask]
            c_cols = cand_cols[mask]
            c_dists = cand_dists[mask]

            centroid = np.mean(c_pts, axis=0)
            b_min = np.min(c_pts, axis=0)
            b_max = np.max(c_pts, axis=0)
            dims = b_max - b_min
            volume = float(np.prod(np.maximum(dims, 0.02)))

            eigvals, elongation, planarity, sphericity = self.extract_shape_descriptors(c_pts)
            mean_rgb = np.mean(c_cols, axis=0) if len(c_cols) > 0 else np.array([0.5, 0.5, 0.5])

            cluster = SpatialCluster(
                cluster_id=int(cid),
                source_sequence=source_label,
                points=c_pts,
                colors=c_cols,
                centroid=centroid,
                bbox_min=b_min,
                bbox_max=b_max,
                dimensions=dims,
                estimated_volume=volume,
                eigenvalues=eigvals,
                elongation=elongation,
                planarity=planarity,
                sphericity=sphericity,
                mean_color_rgb=mean_rgb,
                mean_distance_to_counterpart=float(np.mean(c_dists))
            )
            clusters.append(cluster)

        return clusters

    def detect_changes(
        self,
        pts_before_aligned: np.ndarray,
        cols_before: np.ndarray,
        pts_after: np.ndarray,
        cols_after: np.ndarray
    ) -> Tuple[List[SpatialCluster], List[SpatialCluster], np.ndarray, np.ndarray]:
        """Performs two-way distance querying and clusters discrepancy regions for both sequences."""
        # Before -> After distances
        dists_b2a = self.compute_distances(pts_before_aligned, pts_after)
        # After -> Before distances
        dists_a2b = self.compute_distances(pts_after, pts_before_aligned)

        clusters_before = self.cluster_residual_points(
            pts_before_aligned, cols_before, dists_b2a, source_label="before"
        )
        clusters_after = self.cluster_residual_points(
            pts_after, cols_after, dists_a2b, source_label="after"
        )

        return clusters_before, clusters_after, dists_b2a, dists_a2b
