"""Abstract base class and data structures for 3D reconstruction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import open3d as o3d

from roomshift.data.video_loader import Keyframe


@dataclass
class CameraPose:
    """Represents an estimated camera pose in world coordinates."""
    frame_idx: int
    R: np.ndarray  # (3, 3) rotation matrix
    t: np.ndarray  # (3, 1) translation vector
    K: np.ndarray  # (3, 3) intrinsic matrix
    image_width: int
    image_height: int

    @property
    def center(self) -> np.ndarray:
        """Camera optical center in world coordinates: C = -R^T * t."""
        return (-self.R.T @ self.t).ravel()

    @property
    def viewing_direction(self) -> np.ndarray:
        """Principal optical axis unit vector in world coordinates: R^T * [0, 0, 1]^T."""
        z_cam = np.array([0.0, 0.0, 1.0])
        return (self.R.T @ z_cam).ravel()

    def project_point(self, point_3d: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Projects a 3D world point into image coordinates (u, v, depth)."""
        p_cam = self.R @ point_3d.reshape(3, 1) + self.t
        depth = float(p_cam[2, 0])
        if depth <= 0.01:
            return None
        p_img = self.K @ p_cam
        u = float(p_img[0, 0] / p_img[2, 0])
        v = float(p_img[1, 0] / p_img[2, 0])
        return u, v, depth


@dataclass
class ReconstructionResult:
    """Holds the 3D reconstruction outputs, camera trajectories, and metadata."""
    points_3d: np.ndarray  # (N, 3) float64
    colors_rgb: np.ndarray  # (N, 3) float64 in [0.0, 1.0]
    camera_poses: List[CameraPose]
    keypoints_2d: Dict[int, np.ndarray] = field(default_factory=dict)
    descriptors: Dict[int, np.ndarray] = field(default_factory=dict)
    point_visibility: Dict[int, List[int]] = field(default_factory=dict)  # point_idx -> list of cam indices
    kp_to_3d: Dict[Tuple[int, int], int] = field(default_factory=dict)  # (frame_idx, kp_idx) -> point_3d_idx
    backend_name: str = "opencv"
    reprojection_rmse: float = 0.0
    scale_is_metric: bool = False  # Monocular SfM is unscaled scene units unless calibrated

    @property
    def num_points(self) -> int:
        return len(self.points_3d)

    @property
    def num_cameras(self) -> int:
        return len(self.camera_poses)

    def to_open3d(self) -> o3d.geometry.PointCloud:
        """Converts point cloud into an Open3D PointCloud object."""
        pcd = o3d.geometry.PointCloud()
        if len(self.points_3d) > 0:
            pcd.points = o3d.utility.Vector3dVector(self.points_3d)
            pcd.colors = o3d.utility.Vector3dVector(self.colors_rgb)
        return pcd

    def save_ply(self, output_path: Union[str, Path]) -> Path:
        """Saves the reconstructed 3D point cloud to a standard PLY file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pcd = self.to_open3d()
        o3d.io.write_point_cloud(str(output_path), pcd)
        return output_path


class BaseReconstructor(ABC):
    """Abstract interface for 3D reconstruction engines."""

    @abstractmethod
    def reconstruct(self, keyframes: List[Keyframe]) -> ReconstructionResult:
        """Executes Structure from Motion reconstruction on the extracted keyframes."""
        pass
