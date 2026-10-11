"""End-to-end RoomShift 3D pipeline orchestrator with caching and error handling."""

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Dict, List, Optional, Union
import logging

import numpy as np
import plotly.graph_objects as go

from roomshift.alignment.registration import AlignmentManager, Sim3Result
from roomshift.analysis.change_detector import ChangeDetector, SpatialCluster
from roomshift.analysis.classifier import ChangeClassifier, ClassifiedObject
from roomshift.analysis.visibility import VisibilityChecker
from roomshift.config import RoomShiftConfig
from roomshift.data.video_loader import Keyframe, VideoLoader
from roomshift.reporting.exporter import ReportExporter
from roomshift.sfm.base import CameraPose, ReconstructionResult
from roomshift.sfm.colmap_sfm import ColmapReconstructor
from roomshift.sfm.opencv_sfm import OpenCVSfMReconstructor
from roomshift.visualization.plot_3d import Visualizer3D

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Holds all results from an end-to-end execution of RoomShift 3D."""
    keyframes_before: List[Keyframe]
    keyframes_after: List[Keyframe]
    reconstruction_before: ReconstructionResult
    reconstruction_after: ReconstructionResult
    sim3_alignment: Sim3Result
    pts_before_aligned: np.ndarray
    cameras_before_aligned: List[CameraPose]
    clusters_before: List[SpatialCluster]
    clusters_after: List[SpatialCluster]
    classified_objects: List[ClassifiedObject]
    report: Dict
    figure_3d: go.Figure
    execution_time_sec: float


class RoomShiftPipeline:
    """Orchestrates video extraction, 3D SfM, Sim(3) co-registration, change detection, and reporting."""

    def __init__(self, config: Optional[RoomShiftConfig] = None):
        self.config = config or RoomShiftConfig()
        self.video_loader = VideoLoader(self.config.video)

        # SfM backend selection
        if self.config.sfm.backend == "colmap":
            self.reconstructor = ColmapReconstructor(self.config.sfm)
        else:
            self.reconstructor = OpenCVSfMReconstructor(self.config.sfm)

        self.aligner = AlignmentManager(self.config.alignment)
        self.visibility_checker = VisibilityChecker(self.config.visibility)
        self.change_detector = ChangeDetector(self.config.change)
        self.classifier = ChangeClassifier(self.config.change, self.visibility_checker)

    def run(
        self,
        video_before_path: Union[str, Path],
        video_after_path: Union[str, Path],
        progress_callback: Optional[callable] = None
    ) -> PipelineResult:
        """Executes the complete analysis pipeline with status updates."""
        start_time = time.time()

        def update_progress(msg: str, percent: float):
            logger.info(f"[{percent:.0%}] {msg}")
            if progress_callback is not None:
                progress_callback(msg, percent)

        # Step 1: Extract Keyframes
        update_progress("Extracting keyframes from 'before' video...", 0.10)
        kf_before = self.video_loader.extract_keyframes(video_before_path)

        update_progress("Extracting keyframes from 'after' video...", 0.25)
        kf_after = self.video_loader.extract_keyframes(video_after_path)

        # Step 2: 3D Structure from Motion Reconstruction
        update_progress(f"Reconstructing 3D scene 'before' ({len(kf_before)} frames)...", 0.40)
        res_before = self.reconstructor.reconstruct(kf_before)

        update_progress(f"Reconstructing 3D scene 'after' ({len(kf_after)} frames)...", 0.55)
        res_after = self.reconstructor.reconstruct(kf_after)

        # Step 3: Cross-Sequence Sim(3) Co-Registration
        update_progress("Estimating Sim(3) similarity alignment and ICP refinement...", 0.70)
        sim3 = self.aligner.align(res_before, res_after)

        # Transform 'before' geometry into 'after' coordinate frame
        pts_before_aligned = sim3.transform_points(res_before.points_3d)
        cams_before_aligned = [sim3.transform_camera(c) for c in res_before.camera_poses]

        # Step 4: Discrepancy Analysis & DBSCAN Spatial Clustering
        update_progress("Computing geometric distances and DBSCAN spatial clusters...", 0.82)
        clusters_b, clusters_a, dists_b2a, dists_a2b = self.change_detector.detect_changes(
            pts_before_aligned, res_before.colors_rgb,
            res_after.points_3d, res_after.colors_rgb
        )

        # Step 5: Multi-Modal Classification (Added / Removed / Moved / Unchanged / Uncertain)
        update_progress("Classifying changes with frustum visibility raycasting...", 0.90)
        classified_objs = self.classifier.classify(
            clusters_b, clusters_a,
            cams_before_aligned, res_after.camera_poses,
            pts_before_aligned, res_after.points_3d,
            dists_b2a, dists_a2b
        )

        # Step 6: Generate Reporting & Visualization
        update_progress("Compiling structured report and interactive 3D scene...", 0.96)
        total_time = time.time() - start_time
        report = ReportExporter.generate_report(
            res_before, res_after, sim3, classified_objs, execution_time_sec=total_time
        )

        fig_3d = Visualizer3D.build_comparison_figure(
            pts_before=pts_before_aligned,
            pts_after=res_after.points_3d,
            classified_objects=classified_objs,
            cameras_before=cams_before_aligned,
            cameras_after=res_after.camera_poses
        )

        update_progress("RoomShift 3D analysis complete!", 1.0)

        return PipelineResult(
            keyframes_before=kf_before,
            keyframes_after=kf_after,
            reconstruction_before=res_before,
            reconstruction_after=res_after,
            sim3_alignment=sim3,
            pts_before_aligned=pts_before_aligned,
            cameras_before_aligned=cams_before_aligned,
            clusters_before=clusters_b,
            clusters_after=clusters_a,
            classified_objects=classified_objs,
            report=report,
            figure_3d=fig_3d,
            execution_time_sec=total_time
        )
