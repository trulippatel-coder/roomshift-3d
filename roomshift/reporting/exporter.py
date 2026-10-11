"""Structured JSON export and natural-language narrative generator."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

from roomshift.alignment.registration import Sim3Result
from roomshift.analysis.classifier import ClassifiedObject
from roomshift.sfm.base import ReconstructionResult


class ReportExporter:
    """Produces verified structured JSON reports and readable natural-language narratives."""

    @staticmethod
    def generate_report(
        res_before: ReconstructionResult,
        res_after: ReconstructionResult,
        sim3: Sim3Result,
        classified_objects: List[ClassifiedObject],
        execution_time_sec: float = 0.0
    ) -> Dict:
        """Constructs the comprehensive structured dictionary."""
        counts = {
            "added": sum(1 for o in classified_objects if o.label == "added"),
            "removed": sum(1 for o in classified_objects if o.label == "removed"),
            "moved": sum(1 for o in classified_objects if o.label == "moved"),
            "uncertain": sum(1 for o in classified_objects if o.label == "uncertain"),
            "unchanged": sum(1 for o in classified_objects if o.label == "unchanged"),
        }

        report = {
            "schema_version": "1.0.0",
            "metadata": {
                "system": "RoomShift 3D",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "sfm_backend": res_before.backend_name,
                "scale_status": "Monocular relative coordinates (uncalibrated scene units; metric scale not asserted)",
                "total_processing_time_seconds": round(execution_time_sec, 2)
            },
            "reconstruction": {
                "before_sequence": {
                    "num_points": res_before.num_points,
                    "num_cameras": res_before.num_cameras,
                },
                "after_sequence": {
                    "num_points": res_after.num_points,
                    "num_cameras": res_after.num_cameras,
                }
            },
            "co_registration": {
                "method": "Sim(3) Umeyama RANSAC with Point-to-Plane ICP Refinement",
                "estimated_relative_scale": round(float(sim3.scale), 5),
                "rotation_matrix_3x3": [[round(float(v), 5) for v in row] for row in sim3.R],
                "translation_vector_3x1": [round(float(v), 5) for v in sim3.t.ravel()],
                "transformation_matrix_4x4": [[round(float(v), 5) for v in row] for row in sim3.transformation_4x4],
                "alignment_rmse": round(float(sim3.rmse), 5),
                "alignment_fitness": round(float(sim3.fitness), 4),
                "inlier_correspondences": int(sim3.inlier_count)
            },
            "change_summary": {
                "total_spatial_objects_analyzed": len(classified_objects),
                "counts_by_category": counts
            },
            "detected_objects": [obj.to_dict() for obj in classified_objects],
        }

        # Build natural language narrative
        narrative = ReportExporter.generate_narrative(report)
        report["natural_language_narrative"] = narrative

        return report

    @staticmethod
    def generate_narrative(report: Dict) -> str:
        """Generates clear, executive-grade natural language explanations."""
        counts = report["change_summary"]["counts_by_category"]
        reg = report["co_registration"]
        objects = report["detected_objects"]

        lines = [
            "# RoomShift 3D: Spatial Change Analysis Report",
            "",
            "## Executive Summary",
            f"RoomShift 3D completed automated structure-from-motion reconstruction and Sim(3) co-registration "
            f"with an alignment RMSE of **{reg['alignment_rmse']:.4f}** (fitness score: {reg['alignment_fitness']:.1%}).",
            "",
            f"- **Moved Objects**: {counts.get('moved', 0)} detected",
            f"- **Added Objects**: {counts.get('added', 0)} confirmed",
            f"- **Removed Objects**: {counts.get('removed', 0)} confirmed",
            f"- **Uncertain / Unobserved Regions**: {counts.get('uncertain', 0)} flagged",
            "",
            "## Detailed Spatial Findings",
            ""
        ]

        # Specific objects
        for obj in objects:
            if obj["label"] == "unchanged":
                continue

            lbl = obj["label"].upper()
            oid = obj["object_id"]
            conf = obj["confidence"] * 100.0
            centroid = obj["centroid"]

            if obj["label"] == "moved":
                tgt = obj.get("target_centroid", [0, 0, 0])
                mag = obj.get("displacement_magnitude", 0.0)
                lines.append(
                    f"### {oid}: {lbl} (Confidence: {conf:.1f}%)\n"
                    f"- **Initial Position**: `({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})`\n"
                    f"- **Target Position**: `({tgt[0]:.2f}, {tgt[1]:.2f}, {tgt[2]:.2f})`\n"
                    f"- **Spatial Displacement**: `{mag:.3f}` unscaled units\n"
                    f"- **Volume**: `{obj['estimated_volume_unscaled']:.4f}` units³ ({obj['point_count']} points)\n"
                    f"- **Evidence**: {obj['evidence_reasoning']}\n"
                )
            elif obj["label"] in ("added", "removed"):
                lines.append(
                    f"### {oid}: {lbl} (Confidence: {conf:.1f}%)\n"
                    f"- **Location**: `({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})`\n"
                    f"- **Volume**: `{obj['estimated_volume_unscaled']:.4f}` units³ ({obj['point_count']} points)\n"
                    f"- **Camera Coverage**: Observed in {obj['observing_cameras']} opposing views ({obj['visibility_fraction']:.1%})\n"
                    f"- **Evidence**: {obj['evidence_reasoning']}\n"
                )
            elif obj["label"] == "uncertain":
                lines.append(
                    f"### {oid}: UNCERTAIN / UNOBSERVED REGION\n"
                    f"- **Location**: `({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})`\n"
                    f"- **Integrity Notice**: Region was only observed by {obj['observing_cameras']} camera views. "
                    f"In adherence to conservative computer vision standards, this region is designated *Uncertain* "
                    f"rather than presumed absent, avoiding false positive removal claims.\n"
                )

        lines.extend([
            "## Scale & Metric Notation",
            "Measurements are expressed in relative scene units established during structure-from-motion recovery. "
            "Metric units (meters) are deliberately not asserted because monocular video capture has scale gauge ambiguity "
            "without external calibration markers or metric depth sensors."
        ])

        return "\n".join(lines)

    @classmethod
    def save_json(cls, report: Dict, output_path: Union[str, Path]) -> Path:
        """Saves report to a formatted JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return output_path
