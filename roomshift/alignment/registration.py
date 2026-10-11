"""Sim(3) cross-sequence alignment and ICP co-registration."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import logging
import cv2
import numpy as np
import open3d as o3d

from roomshift.config import AlignmentConfig
from roomshift.sfm.base import CameraPose, ReconstructionResult

logger = logging.getLogger(__name__)


@dataclass
class Sim3Result:
    """Holds the 7-DOF Similarity transformation: scale s, rotation R, translation t."""
    scale: float
    R: np.ndarray  # (3, 3)
    t: np.ndarray  # (3, 1)
    transformation_4x4: np.ndarray  # (4, 4) homogeneous matrix
    rmse: float
    fitness: float
    inlier_count: int

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Applies s * R * p + t to a set of (N, 3) 3D points."""
        if len(points) == 0:
            return points
        return (self.scale * (self.R @ points.T) + self.t).T

    def transform_camera(self, cam: CameraPose) -> CameraPose:
        """Transforms camera extrinsic pose to the aligned coordinate frame."""
        # Under similarity transform p' = s R_sim p + t_sim
        # Camera center C' = s R_sim C + t_sim
        # Rotation R_cam' = R_cam @ R_sim^T
        # Translation t_cam' = -R_cam' @ C'
        new_R = cam.R @ self.R.T
        new_C = self.scale * (self.R @ cam.center) + self.t.ravel()
        new_t = (-new_R @ new_C).reshape(3, 1)
        return CameraPose(
            frame_idx=cam.frame_idx,
            R=new_R,
            t=new_t,
            K=cam.K,
            image_width=cam.image_width,
            image_height=cam.image_height
        )


class AlignmentManager:
    """Aligns two independent SfM reconstructions into a unified coordinate system."""

    def __init__(self, config: Optional[AlignmentConfig] = None):
        self.config = config or AlignmentConfig()

    @staticmethod
    def umeyama_alignment(
        source_pts: np.ndarray,
        target_pts: np.ndarray,
        with_scale: bool = True
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Computes optimal similarity transformation (scale, R, t) using the closed-form Umeyama algorithm."""
        n = source_pts.shape[0]
        if n < 3:
            raise ValueError("Umeyama alignment requires at least 3 point correspondences.")

        mu_src = np.mean(source_pts, axis=0)
        mu_tgt = np.mean(target_pts, axis=0)

        src_centered = source_pts - mu_src
        tgt_centered = target_pts - mu_tgt

        var_src = np.mean(np.sum(src_centered**2, axis=1))

        # Cross-covariance matrix
        cov = (tgt_centered.T @ src_centered) / n
        U, D, Vt = np.linalg.svd(cov)

        S = np.eye(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S[2, 2] = -1.0

        R = U @ S @ Vt

        if with_scale:
            scale = float(np.sum(D * np.diag(S)) / (var_src + 1e-12))
        else:
            scale = 1.0

        t = (mu_tgt - scale * (R @ mu_src)).reshape(3, 1)
        return scale, R, t

    def _ransac_umeyama(
        self,
        src_corr: np.ndarray,
        tgt_corr: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Robustly estimates Sim(3) parameters via RANSAC with Umeyama hypotheses."""
        n = len(src_corr)
        if n < 4:
            scale, R, t = self.umeyama_alignment(src_corr, tgt_corr, with_scale=self.config.allow_scaling)
            return scale, R, t, np.ones(n, dtype=bool)

        best_inliers = np.zeros(n, dtype=bool)
        best_count = 0
        best_scale, best_R, best_t = 1.0, np.eye(3), np.zeros((3, 1))

        for _ in range(self.config.ransac_iterations):
            idx = np.random.choice(n, size=4, replace=False)
            try:
                s, R, t = self.umeyama_alignment(src_corr[idx], tgt_corr[idx], with_scale=self.config.allow_scaling)
                if s <= 0.02 or s > 50.0:  # Physically implausible scale ratio
                    continue
            except Exception:
                continue

            pred_tgt = (s * (R @ src_corr.T) + t).T
            errors = np.linalg.norm(pred_tgt - tgt_corr, axis=1)
            inliers = errors < self.config.ransac_threshold
            count = np.sum(inliers)

            if count > best_count:
                best_count = count
                best_inliers = inliers
                best_scale, best_R, best_t = s, R, t

        if best_count >= 4:
            # Recompute on all inliers
            best_scale, best_R, best_t = self.umeyama_alignment(
                src_corr[best_inliers],
                tgt_corr[best_inliers],
                with_scale=self.config.allow_scaling
            )
            pred_tgt = (best_scale * (best_R @ src_corr.T) + best_t).T
            errors = np.linalg.norm(pred_tgt - tgt_corr, axis=1)
            best_inliers = errors < self.config.ransac_threshold

        return best_scale, best_R, best_t, best_inliers

    def find_cross_correspondences(
        self,
        res_before: ReconstructionResult,
        res_after: ReconstructionResult
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Establishes 3D-3D point matches across before and after sequences using 2D SIFT descriptor matching."""
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        corr_before = []
        corr_after = []
        seen_pairs: Set[Tuple[int, int]] = set()

        has_tracks = bool(res_before.kp_to_3d and res_after.kp_to_3d)
        pair_votes: Dict[Tuple[int, int], int] = {}

        for kf_b_idx, descs_b in res_before.descriptors.items():
            if descs_b is None or len(descs_b) < 10:
                continue
            for kf_a_idx, descs_a in res_after.descriptors.items():
                if descs_a is None or len(descs_a) < 10:
                    continue

                m_b2a = matcher.knnMatch(descs_b, descs_a, k=2)
                m_a2b = matcher.knnMatch(descs_a, descs_b, k=2)

                good_b2a = {
                    m[0].queryIdx: m[0].trainIdx
                    for m in m_b2a if len(m) == 2 and m[0].distance < 0.78 * m[1].distance
                }
                good_a2b = {
                    m[0].queryIdx: m[0].trainIdx
                    for m in m_a2b if len(m) == 2 and m[0].distance < 0.78 * m[1].distance
                }

                mutual = [(q, t) for q, t in good_b2a.items() if good_a2b.get(t) == q]

                for b_kp_idx, a_kp_idx in mutual:
                    if has_tracks:
                        pair_b = (kf_b_idx, b_kp_idx)
                        pair_a = (kf_a_idx, a_kp_idx)
                        if pair_b in res_before.kp_to_3d and pair_a in res_after.kp_to_3d:
                            id_b = res_before.kp_to_3d[pair_b]
                            id_a = res_after.kp_to_3d[pair_a]
                            pair_votes[(id_b, id_a)] = pair_votes.get((id_b, id_a), 0) + 1
                    else:
                        # Fallback for when kp_to_3d is not available: projection lookup
                        cam_b = res_before.camera_poses[min(kf_b_idx, len(res_before.camera_poses) - 1)]
                        cam_a = res_after.camera_poses[min(kf_a_idx, len(res_after.camera_poses) - 1)]
                        p_b = res_before.keypoints_2d[kf_b_idx][b_kp_idx]
                        p_a = res_after.keypoints_2d[kf_a_idx][a_kp_idx]
                        pt3d_b = self._find_projected_3d_point(p_b, res_before.points_3d, cam_b)
                        pt3d_a = self._find_projected_3d_point(p_a, res_after.points_3d, cam_a)
                        if pt3d_b is not None and pt3d_a is not None:
                            corr_before.append(pt3d_b)
                            corr_after.append(pt3d_a)

        if has_tracks and pair_votes:
            sorted_pairs = sorted(pair_votes.items(), key=lambda x: x[1], reverse=True)
            seen_b: Set[int] = set()
            seen_a: Set[int] = set()
            for (id_b, id_a), _ in sorted_pairs:
                if id_b not in seen_b and id_a not in seen_a:
                    seen_b.add(id_b)
                    seen_a.add(id_a)
                    corr_before.append(res_before.points_3d[id_b])
                    corr_after.append(res_after.points_3d[id_a])

        if len(corr_before) < 4:
            return np.empty((0, 3)), np.empty((0, 3))

        return np.array(corr_before, dtype=np.float64), np.array(corr_after, dtype=np.float64)

    @staticmethod
    def _find_projected_3d_point(
        pt_2d: np.ndarray,
        pts_3d: np.ndarray,
        cam: CameraPose,
        pixel_thresh: float = 3.0
    ) -> Optional[np.ndarray]:
        """Finds closest 3D point whose image projection matches pt_2d."""
        if len(pts_3d) == 0:
            return None
        p_cam = (cam.R @ pts_3d.T + cam.t).T
        pos_z = p_cam[:, 2] > 0.05
        if not np.any(pos_z):
            return None

        p_cam = p_cam[pos_z]
        pts_3d_valid = pts_3d[pos_z]

        proj = (cam.K @ p_cam.T).T
        u = proj[:, 0] / proj[:, 2]
        v = proj[:, 1] / proj[:, 2]

        dists = np.sqrt((u - pt_2d[0])**2 + (v - pt_2d[1])**2)
        min_idx = np.argmin(dists)
        if dists[min_idx] <= pixel_thresh:
            return pts_3d_valid[min_idx]
        return None

    def align(
        self,
        res_before: ReconstructionResult,
        res_after: ReconstructionResult
    ) -> Sim3Result:
        """Executes full alignment: cross-sequence feature matching, Umeyama RANSAC, and ICP refinement."""
        if res_before.num_points < 10 or res_after.num_points < 10:
            raise ValueError("Cannot align reconstructions: insufficient 3D points.")

        # Step 1: Find 2D-3D cross correspondences
        src_corr, tgt_corr = self.find_cross_correspondences(res_before, res_after)

        scale, R, t = 1.0, np.eye(3), np.zeros((3, 1))
        inliers_count = 0
        inlier_mask = np.zeros(len(src_corr), dtype=bool)

        if len(src_corr) >= 4:
            scale, R, t, inlier_mask = self._ransac_umeyama(src_corr, tgt_corr)
            inliers_count = int(np.sum(inlier_mask))
            logger.info(f"Cross-sequence Sim(3) estimated from {inliers_count} inlier matches (scale={scale:.4f})")
        else:
            # Fallback: Open3D FPFH Global Registration on downsampled geometry
            logger.info("Insufficient 2D-3D cross matches; computing FPFH geometric global registration.")
            scale, R, t = self._fpfh_global_registration(res_before, res_after)

        # Build initial 4x4 similarity transformation
        T_init = np.eye(4)
        T_init[:3, :3] = scale * R
        T_init[:3, 3:] = t

        # Calculate initial RMSE on inliers
        if inliers_count >= 3:
            pred_tgt = (scale * (R @ src_corr[inlier_mask].T) + t).T
            umeyama_rmse = float(np.sqrt(np.mean((pred_tgt - tgt_corr[inlier_mask])**2)))
        else:
            umeyama_rmse = 0.10

        # Step 2: Open3D ICP Refinement on persistent static geometry
        final_scale = scale
        final_R = R
        final_t = t
        final_T = T_init
        final_rmse = umeyama_rmse
        final_fitness = float(inliers_count / max(1, len(src_corr)))

        try:
            pcd_before = res_before.to_open3d()
            pcd_after = res_after.to_open3d()

            pcd_transformed = pcd_before.transform(T_init)
            pcd_transformed.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            pcd_after.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

            icp_dist = max(self.config.icp_max_correspondence_distance, self.config.ransac_threshold * 1.5)
            icp_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.config.icp_max_iterations
            )
            icp_result = o3d.pipelines.registration.registration_icp(
                pcd_transformed,
                pcd_after,
                max_correspondence_distance=icp_dist,
                init=np.eye(4),
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=icp_criteria
            )

            # Accept ICP only if it achieved positive fitness without corrupting transformation
            if icp_result.fitness > 0.05 and icp_result.inlier_rmse < (umeyama_rmse * 2.0 + 0.1):
                T_refine = icp_result.transformation
                T_candidate = T_refine @ T_init
                c_scale = float(np.cbrt(np.linalg.det(T_candidate[:3, :3])))
                if 0.05 * scale < c_scale < 20.0 * scale:
                    final_T = T_candidate
                    final_scale = c_scale
                    final_R = T_candidate[:3, :3] / (final_scale if final_scale > 1e-6 else 1.0)
                    u, _, vt = np.linalg.svd(final_R)
                    final_R = u @ vt
                    final_t = T_candidate[:3, 3:4]
                    final_rmse = float(icp_result.inlier_rmse)
                    final_fitness = float(icp_result.fitness)
        except Exception as e:
            logger.warning(f"ICP refinement skipped: {e}")

        final_sim3 = Sim3Result(
            scale=final_scale,
            R=final_R,
            t=final_t,
            transformation_4x4=final_T,
            rmse=final_rmse,
            fitness=final_fitness,
            inlier_count=inliers_count or int(final_fitness * res_before.num_points)
        )
        return final_sim3

    def _fpfh_global_registration(
        self,
        res_before: ReconstructionResult,
        res_after: ReconstructionResult
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Estimates coarse alignment using Fast Point Feature Histograms (FPFH) and RANSAC."""
        pcd_b = res_before.to_open3d()
        pcd_a = res_after.to_open3d()

        voxel_size = 0.05
        pcd_b_down = pcd_b.voxel_down_sample(voxel_size)
        pcd_a_down = pcd_a.voxel_down_sample(voxel_size)

        pcd_b_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        pcd_a_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

        fpfh_b = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_b_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
        )
        fpfh_a = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_a_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
        )

        distance_threshold = voxel_size * 1.5
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            pcd_b_down, pcd_a_down, fpfh_b, fpfh_a,
            mutual_filter=True,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )

        T = result.transformation
        scale = float(np.cbrt(np.linalg.det(T[:3, :3])))
        R = T[:3, :3] / (scale if scale > 1e-6 else 1.0)
        t = T[:3, 3:4]
        return scale, R, t
