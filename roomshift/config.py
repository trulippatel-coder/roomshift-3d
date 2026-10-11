"""Configuration dataclasses and default parameters for RoomShift 3D."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


@dataclass
class VideoConfig:
    """Settings for video decoding and keyframe extraction."""
    target_width: int = 640  # Resize frame width for fast, reliable processing
    min_laplacian_var: float = 30.0  # Sharpness filter (reject motion blurred frames)
    min_motion_feature_dist: float = 12.0  # Min average feature movement to select new keyframe
    max_keyframes: int = 30  # Cap number of keyframes per video
    sample_stride: int = 2  # Evaluate every Nth frame from video
    min_keyframes: int = 5


@dataclass
class SfMConfig:
    """Settings for Structure from Motion reconstruction."""
    backend: Literal["opencv", "colmap"] = "opencv"
    colmap_binary: Optional[str] = None  # Auto-detected if installed
    feature_type: Literal["sift", "orb"] = "sift"
    max_features: int = 2500
    ratio_test: float = 0.75
    ransac_prob: float = 0.999
    ransac_threshold: float = 1.5  # Epipolar reprojection error in pixels
    min_triangulation_angle_deg: float = 2.0
    focal_length_factor: float = 1.2  # Pinhole focal length estimate = factor * max(W, H)
    outlier_nb_neighbors: int = 20
    outlier_std_ratio: float = 2.0


@dataclass
class AlignmentConfig:
    """Settings for Sim(3) co-registration between before and after reconstructions."""
    ransac_threshold: float = 0.08  # Max distance for Umeyama Sim(3) RANSAC inliers
    ransac_iterations: int = 2000
    icp_max_correspondence_distance: float = 0.06
    icp_max_iterations: int = 60
    allow_scaling: bool = True  # Monocular SfM requires scale estimation


@dataclass
class VisibilityConfig:
    """Settings for frustum & raycasting visibility verification."""
    min_observing_cameras: int = 2  # Min cameras that must have seen the region
    frustum_fov_margin_deg: float = 2.0  # Margin around camera FOV
    max_depth_ratio: float = 1.5  # Max depth relative to average camera distance
    min_depth: float = 0.05  # Near clipping plane


@dataclass
class ChangeDetectionConfig:
    """Settings for 3D geometric change analysis and classification."""
    distance_threshold: float = 0.06  # Distance above which a point is a candidate change (unscaled scene units)
    dbscan_eps: float = 0.08  # Spatial clustering radius
    dbscan_min_points: int = 6  # Minimum points to form an object cluster
    similarity_threshold: float = 0.45  # Threshold for associating moved clusters (shape/volume correlation)
    uncertain_min_visibility: int = 2  # Clusters with < this visibility in other capture are classified as uncertain


@dataclass
class RoomShiftConfig:
    """Master pipeline configuration."""
    video: VideoConfig = field(default_factory=VideoConfig)
    sfm: SfMConfig = field(default_factory=SfMConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    visibility: VisibilityConfig = field(default_factory=VisibilityConfig)
    change: ChangeDetectionConfig = field(default_factory=ChangeDetectionConfig)
    output_dir: Path = field(default_factory=lambda: Path("output"))
    cache_dir: Path = field(default_factory=lambda: Path("cache"))
    cache_enabled: bool = True
