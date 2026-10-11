"""Native OpenCV Structure-from-Motion (SfM) 3D reconstruction engine."""

from typing import Dict, List, Optional, Set, Tuple
import logging
import cv2
import numpy as np
import open3d as o3d

from roomshift.config import SfMConfig
from roomshift.data.video_loader import Keyframe
from roomshift.sfm.base import BaseReconstructor, CameraPose, ReconstructionResult

logger = logging.getLogger(__name__)


class OpenCVSfMReconstructor(BaseReconstructor):
    """Incremental Structure-from-Motion reconstructor using OpenCV and Open3D."""

    def __init__(self, config: Optional[SfMConfig] = None):
        self.config = config or SfMConfig()
        if self.config.feature_type == "sift":
            # Using contrastThreshold=0.02 extracts rich stable keypoints in indoor environments
            self.detector = cv2.SIFT_create(nfeatures=self.config.max_features, contrastThreshold=0.02)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            self.detector = cv2.ORB_create(nfeatures=self.config.max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def _estimate_intrinsics(self, width: int, height: int) -> np.ndarray:
        """Estimates standard pinhole camera intrinsic matrix K."""
        focal = self.config.focal_length_factor * max(width, height)
        cx, cy = width / 2.0, height / 2.0
        return np.array([
            [focal, 0.0, cx],
            [0.0, focal, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    def _extract_features(self, keyframe: Keyframe) -> Tuple[np.ndarray, np.ndarray]:
        """Detects 2D keypoints and extracts descriptors."""
        kps, descs = self.detector.detectAndCompute(keyframe.gray, None)
        if descs is None or len(kps) == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 128), dtype=np.float32)

        pts = np.float32([kp.pt for kp in kps])
        if self.config.feature_type == "sift" and descs.dtype != np.float32:
            descs = descs.astype(np.float32)
        return pts, descs

    def _match_pairs(
        self,
        descs1: Optional[np.ndarray],
        descs2: Optional[np.ndarray]
    ) -> List[Tuple[int, int]]:
        """Matches descriptors between two views using Lowe's ratio test and mutual consistency."""
        if descs1 is None or descs2 is None or len(descs1) < 4 or len(descs2) < 4:
            return []

        knn_matches = self.matcher.knnMatch(descs1, descs2, k=2)
        good = []
        for m in knn_matches:
            if len(m) == 2 and m[0].distance < self.config.ratio_test * m[1].distance:
                good.append((m[0].queryIdx, m[0].trainIdx))

        # Mutual consistency verification
        knn_reverse = self.matcher.knnMatch(descs2, descs1, k=2)
        rev_dict = {}
        for m in knn_reverse:
            if len(m) == 2 and m[0].distance < self.config.ratio_test * m[1].distance:
                rev_dict[m[0].queryIdx] = m[0].trainIdx

        mutual = [m for m in good if rev_dict.get(m[1]) == m[0]]
        candidates = mutual if len(mutual) >= 8 else good

        # Strictly enforce 1-to-1 correspondence (no duplicate query or train features)
        unique_matches = []
        seen_train = set()
        seen_query = set()
        for q_idx, t_idx in candidates:
            if t_idx not in seen_train and q_idx not in seen_query:
                seen_train.add(t_idx)
                seen_query.add(q_idx)
                unique_matches.append((q_idx, t_idx))

        return unique_matches

    def _triangulate_points_with_poses(
        self,
        K: np.ndarray,
        R1: np.ndarray,
        t1: np.ndarray,
        R2: np.ndarray,
        t2: np.ndarray,
        pts1: np.ndarray,
        pts2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Triangulates 3D points from two views and enforces cheirality and parallax angle."""
        if len(pts1) == 0 or len(pts2) == 0:
            return np.empty((0, 3), dtype=np.float64), np.zeros(0, dtype=bool)

        P1 = K @ np.hstack([R1, t1])
        P2 = K @ np.hstack([R2, t2])

        pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        pts4d /= (pts4d[3] + 1e-12)
        pts3d = pts4d[:3].T

        # True cheirality check: depths in both camera coordinate frames must be positive
        d1 = (R1 @ pts3d.T + t1)[2]
        d2 = (R2 @ pts3d.T + t2)[2]

        valid = (d1 > 0.05) & (d2 > 0.05) & (d1 < 100.0) & (d2 < 100.0)

        # Parallax angle check to eliminate unstable / degenerate collinear rays
        if np.any(valid):
            C1 = (-R1.T @ t1).ravel()
            C2 = (-R2.T @ t2).ravel()
            v1 = pts3d - C1
            v2 = pts3d - C2
            norm1 = np.linalg.norm(v1, axis=1, keepdims=True) + 1e-9
            norm2 = np.linalg.norm(v2, axis=1, keepdims=True) + 1e-9
            cos_parallax = np.sum((v1 / norm1) * (v2 / norm2), axis=1)
            # Minimum parallax angle ~ 0.5 degrees: cos(0.5 deg) ~ 0.99996
            valid_parallax = cos_parallax < 0.99996
            valid = valid & valid_parallax

        return pts3d[valid], valid

    def reconstruct(self, keyframes: List[Keyframe]) -> ReconstructionResult:
        """Executes sequential Structure-from-Motion pipeline over extracted keyframes."""
        if len(keyframes) < 2:
            raise ValueError(f"SfM requires at least 2 keyframes, received {len(keyframes)}")

        w, h = keyframes[0].image.shape[1], keyframes[0].image.shape[0]
        K = self._estimate_intrinsics(w, h)

        # Step 1: Feature detection across all keyframes
        all_pts: Dict[int, np.ndarray] = {}
        all_descs: Dict[int, np.ndarray] = {}
        for i, kf in enumerate(keyframes):
            pts, descs = self._extract_features(kf)
            all_pts[i] = pts
            all_descs[i] = descs

        # Step 2: Select optimal initial two-view pair based on Essential matrix inliers & baseline
        best_pair = (0, 1)
        best_matches: List[Tuple[int, int]] = []
        max_inliers = 0
        best_E = None
        best_mask = None

        # Check (0, 1) first as natural sequential baseline, then nearby pairs if needed
        candidate_pairs = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
        for i, j in candidate_pairs:
            if j >= len(keyframes):
                continue
            matches = self._match_pairs(all_descs[i], all_descs[j])
            if len(matches) >= 8:
                pts0 = all_pts[i][[m[0] for m in matches]]
                pts1 = all_pts[j][[m[1] for m in matches]]
                E, mask = cv2.findEssentialMat(
                    pts0, pts1, K,
                    method=cv2.RANSAC,
                    prob=self.config.ransac_prob,
                    threshold=self.config.ransac_threshold
                )
                inliers = int(mask.sum()) if mask is not None else 0
                if inliers > max_inliers:
                    max_inliers = inliers
                    best_pair = (i, j)
                    best_matches = matches
                    best_E = E
                    best_mask = mask
                if i == 0 and j == 1 and inliers >= 10:
                    # (0, 1) provides ideal sequential foundation
                    break

        if max_inliers < 6 or best_E is None or best_mask is None:
            raise RuntimeError(
                f"Insufficient matching feature points between initial keyframes ({max_inliers} found). "
                "Ensure sufficient scene texture and camera baseline overlap."
            )

        i0, i1 = best_pair
        inlier_indices = np.where(best_mask.ravel() == 1)[0]
        m_inliers = [best_matches[idx] for idx in inlier_indices]
        pts0_in = all_pts[i0][[m[0] for m in m_inliers]]
        pts1_in = all_pts[i1][[m[1] for m in m_inliers]]

        _, R_rel, t_rel, pose_mask = cv2.recoverPose(best_E, pts0_in, pts1_in, K)

        # Initial camera poses (unit baseline between initial pair)
        R0 = np.eye(3, dtype=np.float64)
        t0 = np.zeros((3, 1), dtype=np.float64)
        R1 = R_rel.astype(np.float64)
        t1 = t_rel.astype(np.float64)

        registered_poses: Dict[int, CameraPose] = {
            i0: CameraPose(frame_idx=keyframes[i0].frame_idx, R=R0, t=t0, K=K, image_width=w, image_height=h),
            i1: CameraPose(frame_idx=keyframes[i1].frame_idx, R=R1, t=t1, K=K, image_width=w, image_height=h)
        }

        # Step 3: Triangulate initial point cloud
        tri_pts, valid_mask = self._triangulate_points_with_poses(K, R0, t0, R1, t1, pts0_in, pts1_in)

        points_3d: List[np.ndarray] = list(tri_pts)
        colors_rgb: List[np.ndarray] = []
        visibility_dict: Dict[int, List[int]] = {}
        track_map: Dict[Tuple[int, int], int] = {}

        for pt_idx, (m_pair, is_valid) in enumerate(zip(m_inliers, valid_mask)):
            if is_valid:
                new_id = len(colors_rgb)
                px = int(np.clip(all_pts[i0][m_pair[0]][0], 0, w - 1))
                py = int(np.clip(all_pts[i0][m_pair[0]][1], 0, h - 1))
                colors_rgb.append(keyframes[i0].rgb[py, px] / 255.0)
                visibility_dict[new_id] = [i0, i1]
                track_map[(i0, m_pair[0])] = new_id
                track_map[(i1, m_pair[1])] = new_id

        # Step 4: Incremental registration of remaining keyframes in sequential video order
        remaining_order = []
        for idx in range(i0 - 1, -1, -1):
            remaining_order.append(idx)
        for idx in range(i0 + 1, i1):
            if idx not in registered_poses:
                remaining_order.append(idx)
        for idx in range(i1 + 1, len(keyframes)):
            remaining_order.append(idx)

        for curr_idx in remaining_order:
            # Find 2D-3D correspondences from all registered views
            pts_3d_corr = []
            pts_2d_corr = []
            seen_3d: Set[int] = set()
            seen_2d: Set[int] = set()

            for reg_idx in sorted(registered_poses.keys(), key=lambda r: abs(r - curr_idx)):
                matches = self._match_pairs(all_descs[reg_idx], all_descs[curr_idx])
                for m_reg, m_curr in matches:
                    if (reg_idx, m_reg) in track_map:
                        pt_id = track_map[(reg_idx, m_reg)]
                        if pt_id not in seen_3d and m_curr not in seen_2d:
                            seen_3d.add(pt_id)
                            seen_2d.add(m_curr)
                            pts_3d_corr.append(points_3d[pt_id])
                            pts_2d_corr.append(all_pts[curr_idx][m_curr])

            pnp_success = False
            if len(pts_3d_corr) >= 4:
                pts_3d_arr = np.array(pts_3d_corr, dtype=np.float64)
                pts_2d_arr = np.array(pts_2d_corr, dtype=np.float64)

                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    pts_3d_arr, pts_2d_arr, K, None,
                    flags=cv2.SOLVEPNP_EPNP,
                    reprojectionError=4.0,
                    confidence=0.99
                )

                if success and inliers is not None and len(inliers) >= 4:
                    R_curr, _ = cv2.Rodrigues(rvec)
                    t_curr = tvec.reshape(3, 1)
                    pnp_success = True

            if not pnp_success:
                # Scale-consistent relative pose fallback from nearest registered keyframe
                closest_reg = min(registered_poses.keys(), key=lambda r: abs(r - curr_idx))
                matches_adj = self._match_pairs(all_descs[closest_reg], all_descs[curr_idx])
                if len(matches_adj) >= 6:
                    p_adj = all_pts[closest_reg][[m[0] for m in matches_adj]]
                    p_curr = all_pts[curr_idx][[m[1] for m in matches_adj]]
                    E_rel, mask_rel = cv2.findEssentialMat(
                        p_adj, p_curr, K,
                        method=cv2.RANSAC,
                        prob=0.99,
                        threshold=self.config.ransac_threshold
                    )
                    if E_rel is not None and mask_rel is not None:
                        in_rel = mask_rel.ravel() == 1
                        if in_rel.sum() >= 5:
                            _, R_step, t_step, _ = cv2.recoverPose(E_rel, p_adj[in_rel], p_curr[in_rel], K)
                            R_adj = registered_poses[closest_reg].R
                            t_adj = registered_poses[closest_reg].t

                            # Scale propagation from depth of shared 3D points
                            shared_depths = []
                            for m_a, m_c in matches_adj:
                                if (closest_reg, m_a) in track_map:
                                    pt_id = track_map[(closest_reg, m_a)]
                                    p3 = points_3d[pt_id]
                                    d = float((R_adj @ p3.reshape(3, 1) + t_adj)[2, 0])
                                    if d > 0.1:
                                        shared_depths.append(d)

                            if len(shared_depths) >= 3:
                                step_scale = float(np.median(shared_depths)) * 0.12 * max(1, abs(curr_idx - closest_reg))
                            else:
                                step_scale = 0.5 * max(1, abs(curr_idx - closest_reg))

                            R_curr = R_step.astype(float) @ R_adj
                            t_curr = R_step.astype(float) @ t_adj + t_step.astype(float) * step_scale
                            pnp_success = True

            if pnp_success:
                registered_poses[curr_idx] = CameraPose(
                    frame_idx=keyframes[curr_idx].frame_idx,
                    R=R_curr,
                    t=t_curr,
                    K=K,
                    image_width=w,
                    image_height=h
                )

                # Link existing 3D point observations and triangulate new points
                for reg_idx in list(registered_poses.keys()):
                    if reg_idx == curr_idx or abs(reg_idx - curr_idx) > 2:
                        continue
                    pose_reg = registered_poses[reg_idx]
                    matches_new = self._match_pairs(all_descs[reg_idx], all_descs[curr_idx])

                    new_pts_reg = []
                    new_pts_curr = []
                    pairs_to_register = []

                    for ma, mc in matches_new:
                        in_reg = (reg_idx, ma) in track_map
                        in_curr = (curr_idx, mc) in track_map
                        if in_reg and not in_curr:
                            pt_id = track_map[(reg_idx, ma)]
                            track_map[(curr_idx, mc)] = pt_id
                            if curr_idx not in visibility_dict[pt_id]:
                                visibility_dict[pt_id].append(curr_idx)
                        elif in_curr and not in_reg:
                            pt_id = track_map[(curr_idx, mc)]
                            track_map[(reg_idx, ma)] = pt_id
                            if reg_idx not in visibility_dict[pt_id]:
                                visibility_dict[pt_id].append(reg_idx)
                        elif not in_reg and not in_curr:
                            new_pts_reg.append(all_pts[reg_idx][ma])
                            new_pts_curr.append(all_pts[curr_idx][mc])
                            pairs_to_register.append((ma, mc))

                    if len(new_pts_reg) >= 4:
                        new_tri, new_val = self._triangulate_points_with_poses(
                            K, pose_reg.R, pose_reg.t, R_curr, t_curr,
                            np.array(new_pts_reg, dtype=np.float32),
                            np.array(new_pts_curr, dtype=np.float32)
                        )
                        for n_i, (pt_3d, is_valid) in enumerate(zip(new_tri, new_val)):
                            if is_valid:
                                pt_id = len(points_3d)
                                points_3d.append(pt_3d)
                                p_img = new_pts_curr[n_i]
                                px = int(np.clip(p_img[0], 0, w - 1))
                                py = int(np.clip(p_img[1], 0, h - 1))
                                colors_rgb.append(keyframes[curr_idx].rgb[py, px] / 255.0)
                                visibility_dict[pt_id] = [reg_idx, curr_idx]
                                ma, mc = pairs_to_register[n_i]
                                track_map[(reg_idx, ma)] = pt_id
                                track_map[(curr_idx, mc)] = pt_id

        # Convert to numpy arrays
        points_np = np.array(points_3d, dtype=np.float64)
        colors_np = np.array(colors_rgb, dtype=np.float64)

        # Statistical outlier removal using Open3D
        if len(points_np) > 30:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_np)
            pcd.colors = o3d.utility.Vector3dVector(colors_np)
            nb_nn = min(self.config.outlier_nb_neighbors, max(4, len(points_np) // 8))
            cl, ind = pcd.remove_statistical_outlier(
                nb_neighbors=nb_nn,
                std_ratio=max(2.5, self.config.outlier_std_ratio)
            )
            filtered_indices = set(ind)
            if len(filtered_indices) >= 15:
                # Remap indices in track_map and visibility_dict
                old_to_new = {old_i: new_i for new_i, old_i in enumerate(sorted(filtered_indices))}
                points_np = np.asarray(cl.points)
                colors_np = np.asarray(cl.colors)
                visibility_dict = {
                    old_to_new[old_i]: visibility_dict[old_i]
                    for old_i in filtered_indices
                    if old_i in visibility_dict
                }
                track_map = {
                    k: old_to_new[v]
                    for k, v in track_map.items()
                    if v in old_to_new
                }

        poses_list = [registered_poses[k] for k in sorted(registered_poses.keys())]

        logger.info(
            f"SfM reconstruction completed: {len(points_np)} 3D points, "
            f"{len(poses_list)} registered cameras"
        )

        return ReconstructionResult(
            points_3d=points_np,
            colors_rgb=colors_np,
            camera_poses=poses_list,
            keypoints_2d=all_pts,
            descriptors=all_descs,
            point_visibility=visibility_dict,
            kp_to_3d=track_map,
            backend_name="opencv",
            scale_is_metric=False
        )
