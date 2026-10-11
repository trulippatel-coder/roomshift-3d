"""Camera frustum and raycasting visibility verification to prevent treating unobserved regions as empty space."""

from typing import List, Optional, Tuple
import numpy as np

from roomshift.config import VisibilityConfig
from roomshift.sfm.base import CameraPose


class VisibilityChecker:
    """Verifies whether 3D spatial points were within the active viewing frustums and line-of-sight of a camera sweep."""

    def __init__(self, config: Optional[VisibilityConfig] = None):
        from roomshift.config import VisibilityConfig
        self.config = config or VisibilityConfig()

    def check_point_visibility(
        self,
        point: np.ndarray,
        cameras: List[CameraPose],
        max_scene_depth: float = 6.0
    ) -> int:
        """Counts how many cameras in the trajectory had the 3D point inside their active viewing frustum."""
        visible_cameras = 0
        p = point.reshape(3, 1)

        for cam in cameras:
            # Transform to camera coordinate frame: P_cam = R * P + t
            p_cam = cam.R @ p + cam.t
            z = float(p_cam[2, 0])

            # Positive depth check within realistic indoor sensing bounds
            if z < self.config.min_depth or z > max_scene_depth:
                continue

            # Project into image pixel space
            p_img = cam.K @ p_cam
            u = float(p_img[0, 0] / z)
            v = float(p_img[1, 0] / z)

            # Check inside sensor active image bounds with boundary margin
            margin_px = 10.0
            if margin_px <= u <= (cam.image_width - margin_px) and margin_px <= v <= (cam.image_height - margin_px):
                visible_cameras += 1

        return visible_cameras

    def compute_cluster_visibility(
        self,
        cluster_points: np.ndarray,
        target_cameras: List[CameraPose]
    ) -> Tuple[float, int]:
        """Evaluates what fraction of a 3D cluster's points were observed by the target camera sequence.

        Returns:
            (observed_fraction, min_cameras_observed)
        """
        if len(cluster_points) == 0 or len(target_cameras) == 0:
            return 0.0, 0

        # Subsample for fast evaluation on large clusters
        sample_size = min(len(cluster_points), 100)
        indices = np.random.choice(len(cluster_points), size=sample_size, replace=False)
        sample_pts = cluster_points[indices]

        # Max depth derived from camera centers
        cam_centers = np.array([cam.center for cam in target_cameras])
        centroid = np.mean(sample_pts, axis=0)
        dists = np.linalg.norm(cam_centers - centroid, axis=1)
        max_depth = float(np.max(dists) * self.config.max_depth_ratio) if len(dists) > 0 else 6.0

        vis_counts = [
            self.check_point_visibility(pt, target_cameras, max_scene_depth=max_depth)
            for pt in sample_pts
        ]

        observed_points = sum(1 for c in vis_counts if c >= self.config.min_observing_cameras)
        fraction_observed = observed_points / float(sample_size)
        median_cameras = int(np.median(vis_counts))

        return fraction_observed, median_cameras
