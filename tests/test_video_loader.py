"""Tests for video frame decoding and keyframe selection."""

from pathlib import Path
import numpy as np
import pytest

from roomshift.config import VideoConfig
from roomshift.data.video_loader import VideoLoader


def test_sharpness_computation():
    loader = VideoLoader()
    # Sharp texture image
    sharp_img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    # Blurry image
    blur_img = np.full((200, 200), 128, dtype=np.uint8)

    sharp_score = loader.compute_sharpness(sharp_img)
    blur_score = loader.compute_sharpness(blur_img)

    assert sharp_score > blur_score
    assert blur_score == pytest.approx(0.0, abs=1e-3)


def test_keyframe_extraction_from_video():
    video_path = Path("data/samples/before_room.mp4")
    assert video_path.exists(), "Sample video before_room.mp4 must exist for test"

    config = VideoConfig(min_keyframes=4, max_keyframes=15, target_width=480)
    loader = VideoLoader(config)
    keyframes = loader.extract_keyframes(video_path)

    assert len(keyframes) >= 4
    assert len(keyframes) <= 15
    for kf in keyframes:
        assert kf.image.shape[1] == 480
        assert kf.sharpness > 0
        assert kf.image.shape[2] == 3
