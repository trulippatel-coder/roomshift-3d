"""End-to-end integration test running the RoomShift pipeline on synthetic benchmark videos."""

from pathlib import Path
import pytest

from roomshift.config import RoomShiftConfig
from roomshift.pipeline import RoomShiftPipeline


def test_full_pipeline_run():
    before_video = Path("data/samples/before_room.mp4")
    after_video = Path("data/samples/after_room.mp4")

    assert before_video.exists(), "before_room.mp4 must exist"
    assert after_video.exists(), "after_room.mp4 must exist"

    config = RoomShiftConfig()
    config.video.min_keyframes = 4
    config.video.max_keyframes = 10
    config.video.target_width = 480

    pipeline = RoomShiftPipeline(config)
    progress_records = []

    def progress_cb(msg, pct):
        progress_records.append((msg, pct))

    result = pipeline.run(before_video, after_video, progress_callback=progress_cb)

    # Assert valid execution
    assert len(result.keyframes_before) >= 4
    assert len(result.keyframes_after) >= 4
    assert result.reconstruction_before.num_points > 0
    assert result.reconstruction_after.num_points > 0
    assert len(result.cameras_before_aligned) > 0
    assert result.sim3_alignment.scale > 0
    assert len(result.classified_objects) > 0
    assert "schema_version" in result.report
    assert result.figure_3d is not None
    assert len(progress_records) > 0
    assert progress_records[-1][1] == 1.0
