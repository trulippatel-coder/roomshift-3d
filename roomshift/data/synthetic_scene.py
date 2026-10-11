"""Synthetic 3D indoor scene and video generator with ground-truth spatial changes."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np


@dataclass
class GroundTruthObject:
    """Represents an object in the synthetic scene."""
    name: str
    center: np.ndarray  # (3,)
    size: np.ndarray    # (3,) dx, dy, dz
    color_bgr: Tuple[int, int, int]
    change_type: str    # "unchanged", "added", "removed", "moved", "uncertain"
    target_center: np.ndarray = field(default_factory=lambda: np.zeros(3))


class SyntheticSceneGenerator:
    """Generates synthetic 3D indoor environments and renders realistic camera video sweeps."""

    def __init__(self, width: int = 640, height: int = 480, fov_deg: float = 60.0):
        self.width = width
        self.height = height
        focal = (width / 2.0) / np.tan(np.deg2rad(fov_deg / 2.0))
        self.K = np.array([
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    def _sample_box_points(
        self,
        center: np.ndarray,
        size: np.ndarray,
        density: int = 250,
        color_bgr: Tuple[int, int, int] = (200, 200, 200)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generates 3D surface points with texture/color on a rectangular cuboid."""
        dx, dy, dz = size / 2.0
        cx, cy, cz = center

        pts = []
        # Sample on faces
        n_side = int(np.sqrt(density))
        u = np.linspace(-1, 1, n_side)
        v = np.linspace(-1, 1, n_side)
        uu, vv = np.meshgrid(u, v)
        ones = np.ones_like(uu)

        # Top & Bottom (Z)
        pts.append(np.column_stack([cx + dx * uu.ravel(), cy + dy * vv.ravel(), cz + dz * ones.ravel()]))
        pts.append(np.column_stack([cx + dx * uu.ravel(), cy + dy * vv.ravel(), cz - dz * ones.ravel()]))
        # Front & Back (Y)
        pts.append(np.column_stack([cx + dx * uu.ravel(), cy + dy * ones.ravel(), cz + dz * vv.ravel()]))
        pts.append(np.column_stack([cx + dx * uu.ravel(), cy - dy * ones.ravel(), cz + dz * vv.ravel()]))
        # Left & Right (X)
        pts.append(np.column_stack([cx + dx * ones.ravel(), cy + dy * uu.ravel(), cz + dz * vv.ravel()]))
        pts.append(np.column_stack([cx - dx * ones.ravel(), cy + dy * uu.ravel(), cz + dz * vv.ravel()]))

        all_pts = np.vstack(pts)
        colors = np.tile(np.array(color_bgr, dtype=np.uint8), (len(all_pts), 1))
        # Add slight texture variation
        noise = np.random.randint(-15, 15, size=colors.shape)
        colors = np.clip(colors.astype(int) + noise, 0, 255).astype(np.uint8)
        return all_pts, colors

    def get_ground_truth_objects(self) -> Tuple[List[GroundTruthObject], List[GroundTruthObject]]:
        """Returns the ground truth object definitions for before and after states."""
        # Common static environment
        table = GroundTruthObject("Table", np.array([0.0, 0.0, -0.2]), np.array([1.2, 0.8, 0.5]), (120, 150, 180), "unchanged")
        lamp = GroundTruthObject("Desk Lamp", np.array([0.4, -0.25, 0.25]), np.array([0.15, 0.15, 0.4]), (60, 180, 240), "unchanged")
        floor = GroundTruthObject("Floor", np.array([0.0, 0.0, -0.6]), np.array([3.0, 3.0, 0.05]), (200, 200, 200), "unchanged")
        wall = GroundTruthObject("Back Wall", np.array([0.0, 1.2, 0.2]), np.array([3.0, 0.1, 1.6]), (170, 170, 170), "unchanged")

        # Dynamic / Changing objects
        chair_before = GroundTruthObject("Office Chair", np.array([-0.5, -0.4, -0.1]), np.array([0.45, 0.45, 0.7]), (210, 80, 50), "moved", np.array([0.45, -0.4, -0.1]))
        chair_after = GroundTruthObject("Office Chair", np.array([0.45, -0.4, -0.1]), np.array([0.45, 0.45, 0.7]), (210, 80, 50), "moved", np.array([-0.5, -0.4, -0.1]))

        laptop_before = GroundTruthObject("Laptop", np.array([-0.2, 0.05, 0.12]), np.array([0.3, 0.22, 0.15]), (50, 50, 50), "removed")
        cup_after = GroundTruthObject("Coffee Cup", np.array([0.1, 0.15, 0.14]), np.array([0.12, 0.12, 0.18]), (20, 140, 60), "added")

        # Unobserved object behind a divider
        secret_box = GroundTruthObject("Hidden Corner Box", np.array([1.3, 1.0, -0.3]), np.array([0.3, 0.3, 0.3]), (100, 100, 220), "uncertain")

        before_objs = [table, lamp, floor, wall, chair_before, laptop_before, secret_box]
        after_objs = [table, lamp, floor, wall, chair_after, cup_after]

        return before_objs, after_objs

    def build_point_cloud(self, objects: List[GroundTruthObject]) -> Tuple[np.ndarray, np.ndarray]:
        """Constructs a dense 3D point cloud from object bounding cuboids."""
        all_pts, all_cols = [], []
        for obj in objects:
            pts, cols = self._sample_box_points(obj.center, obj.size, density=200, color_bgr=obj.color_bgr)
            all_pts.append(pts)
            all_cols.append(cols)
        return np.vstack(all_pts), np.vstack(all_cols)

    def render_frame(
        self,
        pts_3d: np.ndarray,
        cols_bgr: np.ndarray,
        cam_pos: np.ndarray,
        look_at: np.ndarray = np.array([0.0, 0.0, 0.0])
    ) -> np.ndarray:
        """Projects textured 3D points onto a camera view with z-buffering and ambient shading."""
        # Camera extrinsic matrix [R | t]
        forward = look_at - cam_pos
        forward = forward / np.linalg.norm(forward)
        up_approx = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up_approx)
        if np.linalg.norm(right) < 1e-4:
            right = np.array([1.0, 0.0, 0.0])
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        # OpenCV camera frame: X right, Y down, Z forward
        R_cam = np.array([right, -up, forward])
        t_cam = -R_cam @ cam_pos

        # Transform points to camera coordinates
        pts_cam = (R_cam @ pts_3d.T + t_cam[:, None]).T

        # Positive depth filter
        valid = pts_cam[:, 2] > 0.1
        pts_cam = pts_cam[valid]
        cols_bgr = cols_bgr[valid]

        # Project via K
        pts_proj = (self.K @ pts_cam.T).T
        u = (pts_proj[:, 0] / pts_proj[:, 2]).astype(int)
        v = (pts_proj[:, 1] / pts_proj[:, 2]).astype(int)
        z = pts_cam[:, 2]

        in_bounds = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        u, v, z, cols_bgr = u[in_bounds], v[in_bounds], z[in_bounds], cols_bgr[in_bounds]

        # Render frame with z-buffer
        frame = np.full((self.height, self.width, 3), 235, dtype=np.uint8)
        z_buf = np.full((self.height, self.width), np.inf, dtype=np.float32)

        # Sort far-to-near for clean splatting
        order = np.argsort(-z)
        u, v, z, cols_bgr = u[order], v[order], z[order], cols_bgr[order]

        for ui, vi, zi, ci in zip(u, v, z, cols_bgr):
            for du in (-1, 0, 1):
                for dv in (-1, 0, 1):
                    uu, vv = ui + du, vi + dv
                    if 0 <= uu < self.width and 0 <= vv < self.height:
                        if zi < z_buf[vv, uu]:
                            z_buf[vv, uu] = zi
                            frame[vv, uu] = ci

        # Add camera sensor grain/texture for realistic feature detection
        noise = np.random.normal(0, 3, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return frame

    def generate_video(
        self,
        objects: List[GroundTruthObject],
        output_path: Union[str, Path],
        num_frames: int = 36,
        radius: float = 2.1,
        height: float = 0.9,
        arc_deg: float = 120.0
    ) -> Path:
        """Generates an MP4 video sweep around the scene."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        pts_3d, cols = self.build_point_cloud(objects)
        angles = np.linspace(-np.deg2rad(arc_deg / 2), np.deg2rad(arc_deg / 2), num_frames)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, 15.0, (self.width, self.height))

        for angle in angles:
            cam_x = radius * np.sin(angle)
            cam_y = -radius * np.cos(angle)
            cam_z = height
            cam_pos = np.array([cam_x, cam_y, cam_z])
            frame = self.render_frame(pts_3d, cols, cam_pos)
            writer.write(frame)

        writer.release()
        return output_path
