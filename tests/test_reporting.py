"""Tests for structured JSON report export and narrative formatting."""

import json
from pathlib import Path
import numpy as np
import pytest

from roomshift.alignment.registration import Sim3Result
from roomshift.analysis.classifier import ClassifiedObject
from roomshift.reporting.exporter import ReportExporter
from roomshift.sfm.base import CameraPose, ReconstructionResult


def test_report_generation_and_export(tmp_path):
    # Setup mock inputs
    res_b = ReconstructionResult(
        points_3d=np.zeros((100, 3)),
        colors_rgb=np.zeros((100, 3)),
        camera_poses=[CameraPose(0, np.eye(3), np.zeros((3, 1)), np.eye(3), 640, 480)],
        backend_name="opencv",
        scale_is_metric=False
    )
    res_a = ReconstructionResult(
        points_3d=np.zeros((120, 3)),
        colors_rgb=np.zeros((120, 3)),
        camera_poses=[CameraPose(0, np.eye(3), np.zeros((3, 1)), np.eye(3), 640, 480)],
        backend_name="opencv",
        scale_is_metric=False
    )
    sim3 = Sim3Result(
        scale=1.05,
        R=np.eye(3),
        t=np.zeros((3, 1)),
        transformation_4x4=np.eye(4),
        rmse=0.012,
        fitness=0.95,
        inlier_count=85
    )
    objs = [
        ClassifiedObject(
            object_id="OBJ-01",
            label="added",
            confidence=0.92,
            centroid=np.array([0.5, 0.2, 0.1]),
            dimensions=np.array([0.2, 0.2, 0.3]),
            estimated_volume=0.012,
            point_count=45,
            observing_camera_count=3,
            evidence_reasoning="Confirmed new object with 3 observing cameras."
        ),
        ClassifiedObject(
            object_id="OBJ-02",
            label="uncertain",
            confidence=0.50,
            centroid=np.array([1.5, 1.2, 0.0]),
            observing_camera_count=0,
            evidence_reasoning="Unobserved region in opposing sweep."
        )
    ]

    report = ReportExporter.generate_report(res_b, res_a, sim3, objs, execution_time_sec=2.5)

    assert report["schema_version"] == "1.0.0"
    assert report["change_summary"]["counts_by_category"]["added"] == 1
    assert report["change_summary"]["counts_by_category"]["uncertain"] == 1
    assert "Uncertain / Unobserved Regions" in report["natural_language_narrative"]

    out_file = tmp_path / "test_report.json"
    ReportExporter.save_json(report, out_file)
    assert out_file.exists()

    with open(out_file) as f:
        loaded = json.load(f)
    assert loaded["metadata"]["system"] == "RoomShift 3D"
