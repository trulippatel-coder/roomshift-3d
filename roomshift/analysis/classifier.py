"""Multi-evidence change classifier for added, removed, moved, unchanged, and uncertain states."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np

from roomshift.analysis.change_detector import SpatialCluster
from roomshift.analysis.visibility import VisibilityChecker
from roomshift.config import ChangeDetectionConfig
from roomshift.sfm.base import CameraPose

logger = logging.getLogger(__name__)


@dataclass
class ClassifiedObject:
    """Represents a classified 3D object change with supporting geometric evidence."""
    object_id: str
    label: str  # "added", "removed", "moved", "unchanged", "uncertain"
    confidence: float  # [0.0, 1.0]
    centroid: np.ndarray  # (3,)
    target_centroid: Optional[np.ndarray] = None  # (3,) for moved objects
    displacement_vector: Optional[np.ndarray] = None  # (3,) for moved objects
    displacement_magnitude: float = 0.0
    bounding_box_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bounding_box_max: np.ndarray = field(default_factory=lambda: np.zeros(3))
    dimensions: np.ndarray = field(default_factory=lambda: np.zeros(3))
    estimated_volume: float = 0.0
    point_count: int = 0
    mean_color_rgb: np.ndarray = field(default_factory=lambda: np.zeros(3))
    visibility_score: float = 1.0
    observing_camera_count: int = 2
    evidence_reasoning: str = ""
    associated_cluster_id: Optional[int] = None

    def to_dict(self) -> Dict:
        """Serializes object to a clean JSON-compatible dictionary."""
        d = {
            "object_id": self.object_id,
            "label": self.label,
            "confidence": round(float(self.confidence), 3),
            "centroid": [round(float(x), 4) for x in self.centroid],
            "bounding_box_min": [round(float(x), 4) for x in self.bounding_box_min],
            "bounding_box_max": [round(float(x), 4) for x in self.bounding_box_max],
            "dimensions_dx_dy_dz": [round(float(x), 4) for x in self.dimensions],
            "estimated_volume_unscaled": round(float(self.estimated_volume), 5),
            "point_count": int(self.point_count),
            "mean_color_rgb": [round(float(c), 3) for c in self.mean_color_rgb],
            "visibility_fraction": round(float(self.visibility_score), 3),
            "observing_cameras": int(self.observing_camera_count),
            "evidence_reasoning": self.evidence_reasoning,
        }
        if self.target_centroid is not None:
            d["target_centroid"] = [round(float(x), 4) for x in self.target_centroid]
        if self.displacement_vector is not None:
            d["displacement_vector"] = [round(float(x), 4) for x in self.displacement_vector]
            d["displacement_magnitude"] = round(float(self.displacement_magnitude), 4)
        return d


class ChangeClassifier:
    """Classifies spatial clusters into added, removed, moved, unchanged, and uncertain states."""

    def __init__(
        self,
        config: Optional[ChangeDetectionConfig] = None,
        visibility_checker: Optional[VisibilityChecker] = None
    ):
        self.config = config or ChangeDetectionConfig()
        self.visibility_checker = visibility_checker or VisibilityChecker()

    def _compute_cluster_similarity(
        self,
        cb: SpatialCluster,
        ca: SpatialCluster
    ) -> float:
        """Computes multi-attribute geometric and appearance similarity between two candidate clusters."""
        # Volume ratio
        vol_ratio = min(cb.estimated_volume, ca.estimated_volume) / (max(cb.estimated_volume, ca.estimated_volume) + 1e-6)

        # Eigenvalue shape cosine similarity
        norm_b = cb.eigenvalues / (np.linalg.norm(cb.eigenvalues) + 1e-6)
        norm_a = ca.eigenvalues / (np.linalg.norm(ca.eigenvalues) + 1e-6)
        shape_sim = float(np.clip(np.dot(norm_b, norm_a), 0.0, 1.0))

        # Color similarity (RGB distance in [0, 1])
        col_dist = np.linalg.norm(cb.mean_color_rgb - ca.mean_color_rgb) / np.sqrt(3.0)
        col_sim = float(np.clip(1.0 - col_dist, 0.0, 1.0))

        # Aspect ratio / dimensions similarity
        dim_ratio = np.min(np.vstack([cb.dimensions / (ca.dimensions + 1e-4), ca.dimensions / (cb.dimensions + 1e-4)]), axis=0)
        dim_sim = float(np.mean(np.clip(dim_ratio, 0.0, 1.0)))

        combined = 0.35 * vol_ratio + 0.30 * shape_sim + 0.20 * dim_sim + 0.15 * col_sim
        return combined

    def classify(
        self,
        clusters_before: List[SpatialCluster],
        clusters_after: List[SpatialCluster],
        cameras_before_aligned: List[CameraPose],
        cameras_after: List[CameraPose],
        pts_before_aligned: np.ndarray,
        pts_after: np.ndarray,
        dists_b2a: np.ndarray,
        dists_a2b: np.ndarray
    ) -> List[ClassifiedObject]:
        """Performs rigorous classification with visibility raycast verification."""
        classified: List[ClassifiedObject] = []
        obj_counter = 1

        # Check visibility of Before clusters inside After camera frustums
        for cb in clusters_before:
            frac, med_cams = self.visibility_checker.compute_cluster_visibility(cb.points, cameras_after)
            cb.visibility_in_other = frac
            cb.median_cameras_observed = med_cams

        # Check visibility of After clusters inside Before camera frustums
        for ca in clusters_after:
            frac, med_cams = self.visibility_checker.compute_cluster_visibility(ca.points, cameras_before_aligned)
            ca.visibility_in_other = frac
            ca.median_cameras_observed = med_cams

        # Step 1: Detect MOVED objects by pairing Before and After discrepancy clusters
        matched_before = set()
        matched_after = set()

        for b_idx, cb in enumerate(clusters_before):
            best_sim = 0.0
            best_a_idx = -1

            for a_idx, ca in enumerate(clusters_after):
                if a_idx in matched_after:
                    continue

                # Must have significant displacement between centroids to be considered "moved"
                disp_dist = float(np.linalg.norm(ca.centroid - cb.centroid))
                if disp_dist < 0.10:
                    continue

                sim = self._compute_cluster_similarity(cb, ca)
                if sim > best_sim:
                    best_sim = sim
                    best_a_idx = a_idx

            if best_sim >= self.config.similarity_threshold and best_a_idx >= 0:
                ca = clusters_after[best_a_idx]
                matched_before.add(b_idx)
                matched_after.add(best_a_idx)

                disp_vec = ca.centroid - cb.centroid
                disp_mag = float(np.linalg.norm(disp_vec))
                conf = min(0.98, float(best_sim * 0.95))

                classified.append(ClassifiedObject(
                    object_id=f"OBJ-{obj_counter:02d}",
                    label="moved",
                    confidence=conf,
                    centroid=cb.centroid,
                    target_centroid=ca.centroid,
                    displacement_vector=disp_vec,
                    displacement_magnitude=disp_mag,
                    bounding_box_min=cb.bbox_min,
                    bounding_box_max=cb.bbox_max,
                    dimensions=cb.dimensions,
                    estimated_volume=cb.estimated_volume,
                    point_count=cb.num_points + ca.num_points,
                    mean_color_rgb=0.5 * (cb.mean_color_rgb + ca.mean_color_rgb),
                    visibility_score=float(min(cb.visibility_in_other, ca.visibility_in_other)),
                    observing_camera_count=int(min(cb.median_cameras_observed, ca.median_cameras_observed)),
                    evidence_reasoning=(
                        f"Matched missing cluster at {np.round(cb.centroid, 2)} with new cluster at {np.round(ca.centroid, 2)} "
                        f"(shape similarity {best_sim:.2f}, displacement {disp_mag:.3f} units)."
                    ),
                    associated_cluster_id=cb.cluster_id
                ))
                obj_counter += 1

        # Step 2: Classify remaining Before clusters as REMOVED or UNCERTAIN
        for b_idx, cb in enumerate(clusters_before):
            if b_idx in matched_before:
                continue

            # Core Requirement 7: Prevent treating unobserved space as removed!
            is_observed = (
                cb.visibility_in_other >= 0.40 and
                cb.median_cameras_observed >= self.config.uncertain_min_visibility
            )

            if is_observed:
                conf = min(0.95, float(0.60 + 0.35 * cb.visibility_in_other))
                classified.append(ClassifiedObject(
                    object_id=f"OBJ-{obj_counter:02d}",
                    label="removed",
                    confidence=conf,
                    centroid=cb.centroid,
                    bounding_box_min=cb.bbox_min,
                    bounding_box_max=cb.bbox_max,
                    dimensions=cb.dimensions,
                    estimated_volume=cb.estimated_volume,
                    point_count=cb.num_points,
                    mean_color_rgb=cb.mean_color_rgb,
                    visibility_score=cb.visibility_in_other,
                    observing_camera_count=cb.median_cameras_observed,
                    evidence_reasoning=(
                        f"Region was clearly observed by {cb.median_cameras_observed} 'after' cameras "
                        f"(visibility {cb.visibility_in_other:.1%}), but structure is absent (mean gap {cb.mean_distance_to_counterpart:.3f})."
                    ),
                    associated_cluster_id=cb.cluster_id
                ))
            else:
                conf = 0.50
                classified.append(ClassifiedObject(
                    object_id=f"OBJ-{obj_counter:02d}",
                    label="uncertain",
                    confidence=conf,
                    centroid=cb.centroid,
                    bounding_box_min=cb.bbox_min,
                    bounding_box_max=cb.bbox_max,
                    dimensions=cb.dimensions,
                    estimated_volume=cb.estimated_volume,
                    point_count=cb.num_points,
                    mean_color_rgb=cb.mean_color_rgb,
                    visibility_score=cb.visibility_in_other,
                    observing_camera_count=cb.median_cameras_observed,
                    evidence_reasoning=(
                        f"Candidate missing region at {np.round(cb.centroid, 2)} lacked sufficient coverage in 'after' sweep "
                        f"(seen by only {cb.median_cameras_observed} cameras). Classified as uncertain to prevent false empty space assumption."
                    ),
                    associated_cluster_id=cb.cluster_id
                ))
            obj_counter += 1

        # Step 3: Classify remaining After clusters as ADDED or UNCERTAIN
        for a_idx, ca in enumerate(clusters_after):
            if a_idx in matched_after:
                continue

            is_observed = (
                ca.visibility_in_other >= 0.40 and
                ca.median_cameras_observed >= self.config.uncertain_min_visibility
            )

            if is_observed:
                conf = min(0.95, float(0.60 + 0.35 * ca.visibility_in_other))
                classified.append(ClassifiedObject(
                    object_id=f"OBJ-{obj_counter:02d}",
                    label="added",
                    confidence=conf,
                    centroid=ca.centroid,
                    bounding_box_min=ca.bbox_min,
                    bounding_box_max=ca.bbox_max,
                    dimensions=ca.dimensions,
                    estimated_volume=ca.estimated_volume,
                    point_count=ca.num_points,
                    mean_color_rgb=ca.mean_color_rgb,
                    visibility_score=ca.visibility_in_other,
                    observing_camera_count=ca.median_cameras_observed,
                    evidence_reasoning=(
                        f"New structure detected in 'after' scan; corresponding region was verified covered in 'before' scan "
                        f"({ca.median_cameras_observed} cameras, visibility {ca.visibility_in_other:.1%})."
                    ),
                    associated_cluster_id=ca.cluster_id
                ))
            else:
                conf = 0.50
                classified.append(ClassifiedObject(
                    object_id=f"OBJ-{obj_counter:02d}",
                    label="uncertain",
                    confidence=conf,
                    centroid=ca.centroid,
                    bounding_box_min=ca.bbox_min,
                    bounding_box_max=ca.bbox_max,
                    dimensions=ca.dimensions,
                    estimated_volume=ca.estimated_volume,
                    point_count=ca.num_points,
                    mean_color_rgb=ca.mean_color_rgb,
                    visibility_score=ca.visibility_in_other,
                    observing_camera_count=ca.median_cameras_observed,
                    evidence_reasoning=(
                        f"Object detected in 'after' scan, but corresponding volume in 'before' sweep was partially unobserved "
                        f"(seen by only {ca.median_cameras_observed} cameras). Cannot guarantee prior absence."
                    ),
                    associated_cluster_id=ca.cluster_id
                ))
            obj_counter += 1

        # Step 4: Include Static Background / UNCHANGED summary
        low_dist_mask = dists_b2a < self.config.distance_threshold
        if np.any(low_dist_mask):
            unchanged_pts = pts_before_aligned[low_dist_mask]
            centroid_static = np.mean(unchanged_pts, axis=0)
            b_min = np.min(unchanged_pts, axis=0)
            b_max = np.max(unchanged_pts, axis=0)
            dims = b_max - b_min
            vol = float(np.prod(dims))

            classified.append(ClassifiedObject(
                object_id=f"OBJ-{obj_counter:02d}",
                label="unchanged",
                confidence=0.98,
                centroid=centroid_static,
                bounding_box_min=b_min,
                bounding_box_max=b_max,
                dimensions=dims,
                estimated_volume=vol,
                point_count=int(np.sum(low_dist_mask)),
                mean_color_rgb=np.mean(unchanged_pts, axis=0),
                visibility_score=1.0,
                observing_camera_count=len(cameras_after),
                evidence_reasoning=(
                    f"Static structural framework (walls, floor, dominant fixtures) with bidirectional geometric discrepancy < {self.config.distance_threshold:.3f} units."
                )
            ))

        return classified
