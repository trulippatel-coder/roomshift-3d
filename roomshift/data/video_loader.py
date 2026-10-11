"""Video decoding and intelligent keyframe extraction for 3D reconstruction."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
import logging

import cv2
import numpy as np

from roomshift.config import VideoConfig

logger = logging.getLogger(__name__)


@dataclass
class Keyframe:
    """Represents a selected keyframe from a video sequence."""
    frame_idx: int
    timestamp_sec: float
    image: np.ndarray  # BGR format uint8 (H, W, 3)
    sharpness: float
    relative_motion: float = 0.0

    @property
    def gray(self) -> np.ndarray:
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    @property
    def rgb(self) -> np.ndarray:
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)


class VideoLoader:
    """Extracts high-quality, geometrically distributed keyframes from input videos."""

    def __init__(self, config: Optional[VideoConfig] = None):
        self.config = config or VideoConfig()

    @staticmethod
    def compute_sharpness(gray_frame: np.ndarray) -> float:
        """Computes blur metric using the variance of the Laplacian operator."""
        laplacian = cv2.Laplacian(gray_frame, cv2.CV_64F)
        return float(laplacian.var())

    @staticmethod
    def estimate_motion(
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        max_corners: int = 150
    ) -> float:
        """Estimates pixel displacement between consecutive candidate frames using Lucas-Kanade optical flow."""
        pts_prev = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=max_corners,
            qualityLevel=0.01,
            minDistance=10
        )
        if pts_prev is None or len(pts_prev) < 8:
            return 0.0

        pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            pts_prev,
            None,
            winSize=(21, 21),
            maxLevel=3
        )

        valid = status.ravel() == 1
        if np.sum(valid) < 5:
            return 0.0

        displacements = np.linalg.norm(pts_curr[valid] - pts_prev[valid], axis=-1)
        return float(np.median(displacements))

    def extract_keyframes(self, video_path: Union[str, Path]) -> List[Keyframe]:
        """Extracts keyframes from a video file, filtering out blurry frames and near-duplicate views."""
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file does not exist: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file with OpenCV: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30.0

        raw_candidates: List[Tuple[int, float, np.ndarray, float]] = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.config.sample_stride == 0:
                h, w = frame.shape[:2]
                if w > self.config.target_width:
                    scale = self.config.target_width / float(w)
                    target_h = int(h * scale)
                    resized = cv2.resize(frame, (self.config.target_width, target_h), interpolation=cv2.INTER_AREA)
                else:
                    resized = frame

                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                sharpness = self.compute_sharpness(gray)
                time_sec = frame_idx / fps
                raw_candidates.append((frame_idx, time_sec, resized, sharpness))

            frame_idx += 1

        cap.release()

        if not raw_candidates:
            raise ValueError(f"Video {video_path.name} contains zero decodable frames.")

        # Baseline filtering: reject blurry frames unless too few pass
        filtered = [
            c for c in raw_candidates if c[3] >= self.config.min_laplacian_var
        ]
        if len(filtered) < self.config.min_keyframes:
            # Fall back to the sharpest available frames
            raw_candidates.sort(key=lambda x: x[3], reverse=True)
            filtered = raw_candidates[: max(self.config.min_keyframes, min(len(raw_candidates), 10))]
            filtered.sort(key=lambda x: x[0])  # Restore chronological order

        # Geometric motion filtering: select frames that provide baseline for SfM triangulation
        keyframes: List[Keyframe] = []
        first_idx, first_t, first_img, first_sharp = filtered[0]
        keyframes.append(Keyframe(
            frame_idx=first_idx,
            timestamp_sec=first_t,
            image=first_img,
            sharpness=first_sharp,
            relative_motion=0.0
        ))

        last_selected_gray = cv2.cvtColor(first_img, cv2.COLOR_BGR2GRAY)

        for idx, t_sec, img, sharp in filtered[1:]:
            curr_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            disp = self.estimate_motion(last_selected_gray, curr_gray)

            # Accept if sufficient motion baseline or if we are running low on frames
            if disp >= self.config.min_motion_feature_dist or len(keyframes) < self.config.min_keyframes:
                keyframes.append(Keyframe(
                    frame_idx=idx,
                    timestamp_sec=t_sec,
                    image=img,
                    sharpness=sharp,
                    relative_motion=disp
                ))
                last_selected_gray = curr_gray

                if len(keyframes) >= self.config.max_keyframes:
                    break

        logger.info(
            f"Extracted {len(keyframes)} keyframes from {video_path.name} "
            f"(total {total_frames} frames, FPS {fps:.1f})"
        )
        return keyframes
