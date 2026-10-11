"""Interactive Plotly 3D visualization for point clouds, camera frustums, and spatial changes."""

from typing import List, Optional, Tuple
import numpy as np
import plotly.graph_objects as go

from roomshift.analysis.classifier import ClassifiedObject
from roomshift.sfm.base import CameraPose


class Visualizer3D:
    """Builds interactive WebGL 3D visualizations with custom themes and spatial layers."""

    # Curated modern color palette
    COLOR_MAP = {
        "added": "#00E676",      # Bright Emerald Green
        "removed": "#FF1744",    # Crimson Red
        "moved": "#00B0FF",      # Vivid Sky Blue
        "uncertain": "#FF9100",  # Vivid Amber / Orange
        "unchanged": "#78909C",  # Slate Blue-Gray
    }

    @staticmethod
    def _create_box_lines(min_pt: np.ndarray, max_pt: np.ndarray) -> Tuple[List[float], List[float], List[float]]:
        """Generates line coordinates for a 3D bounding box wireframe."""
        x0, y0, z0 = min_pt
        x1, y1, z1 = max_pt

        # 12 edges of a box with None separators for single trace
        xs = [x0, x1, x1, x0, x0, None, x0, x1, x1, x0, x0, None, x0, x0, None, x1, x1, None, x1, x1, None, x0, x0]
        ys = [y0, y0, y1, y1, y0, None, y0, y0, y1, y1, y0, None, y0, y0, None, y0, y0, None, y1, y1, None, y1, y1]
        zs = [z0, z0, z0, z0, z0, None, z1, z1, z1, z1, z1, None, z0, z1, None, z0, z1, None, z0, z1, None, z0, z1]
        return xs, ys, zs

    @classmethod
    def build_comparison_figure(
        cls,
        pts_before: np.ndarray,
        pts_after: np.ndarray,
        classified_objects: List[ClassifiedObject],
        cameras_before: Optional[List[CameraPose]] = None,
        cameras_after: Optional[List[CameraPose]] = None,
        subsample_max: int = 12000
    ) -> go.Figure:
        """Constructs a multi-layer interactive Plotly 3D scene."""
        fig = go.Figure()

        # Step 1: Add Before Cloud (subsampled if large)
        if len(pts_before) > 0:
            step_b = max(1, len(pts_before) // subsample_max)
            sub_b = pts_before[::step_b]
            fig.add_trace(go.Scatter3d(
                x=sub_b[:, 0],
                y=sub_b[:, 1],
                z=sub_b[:, 2],
                mode="markers",
                marker=dict(size=2.2, color="#90CAF9", opacity=0.45),
                name="Before Scan (Aligned)",
                hoverinfo="skip"
            ))

        # Step 2: Add After Cloud (subsampled if large)
        if len(pts_after) > 0:
            step_a = max(1, len(pts_after) // subsample_max)
            sub_a = pts_after[::step_a]
            fig.add_trace(go.Scatter3d(
                x=sub_a[:, 0],
                y=sub_a[:, 1],
                z=sub_a[:, 2],
                mode="markers",
                marker=dict(size=2.2, color="#B0BEC5", opacity=0.45),
                name="After Scan",
                hoverinfo="skip"
            ))

        # Step 3: Add Classified Objects & Displacements
        for obj in classified_objects:
            if obj.label == "unchanged":
                continue  # Background points already shown

            color = cls.COLOR_MAP.get(obj.label, "#FFFFFF")
            label_title = f"{obj.object_id} ({obj.label.upper()})"

            # Centroid Marker
            fig.add_trace(go.Scatter3d(
                x=[obj.centroid[0]],
                y=[obj.centroid[1]],
                z=[obj.centroid[2]],
                mode="markers+text",
                marker=dict(size=8, color=color, symbol="diamond", line=dict(color="#111", width=1.5)),
                text=[label_title],
                textposition="top center",
                name=label_title,
                hovertemplate=(
                    f"<b>{label_title}</b><br>"
                    f"Confidence: {obj.confidence:.1%}<br>"
                    f"Centroid: (%{{x:.2f}}, %{{y:.2f}}, %{{z:.2f}})<br>"
                    f"Volume: {obj.estimated_volume:.4f}<br>"
                    f"Observations: {obj.observing_camera_count} cameras<br>"
                    f"<i>{obj.evidence_reasoning}</i><extra></extra>"
                )
            ))

            # Bounding Box Wireframe
            bx, by, bz = cls._create_box_lines(obj.bounding_box_min, obj.bounding_box_max)
            fig.add_trace(go.Scatter3d(
                x=bx, y=by, z=bz,
                mode="lines",
                line=dict(color=color, width=3.5),
                showlegend=False,
                hoverinfo="skip"
            ))

            # If Moved, draw displacement line vector from initial to target centroid
            if obj.label == "moved" and obj.target_centroid is not None:
                fig.add_trace(go.Scatter3d(
                    x=[obj.centroid[0], obj.target_centroid[0]],
                    y=[obj.centroid[1], obj.target_centroid[1]],
                    z=[obj.centroid[2], obj.target_centroid[2]],
                    mode="lines+markers",
                    line=dict(color="#00E5FF", width=5, dash="dash"),
                    marker=dict(size=[4, 9], color=["#00E5FF", "#FFFF00"], symbol=["circle", "diamond"]),
                    name=f"{obj.object_id} Trajectory",
                    hovertemplate=f"Displacement: {obj.displacement_magnitude:.3f} units<extra></extra>"
                ))

        # Step 4: Add Camera Trajectories
        if cameras_before is not None and len(cameras_before) > 0:
            c_b = np.array([cam.center for cam in cameras_before])
            fig.add_trace(go.Scatter3d(
                x=c_b[:, 0], y=c_b[:, 1], z=c_b[:, 2],
                mode="lines+markers",
                marker=dict(size=3.5, color="#29B6F6"),
                line=dict(color="#0288D1", width=2),
                name="Before Camera Path"
            ))

        if cameras_after is not None and len(cameras_after) > 0:
            c_a = np.array([cam.center for cam in cameras_after])
            fig.add_trace(go.Scatter3d(
                x=c_a[:, 0], y=c_a[:, 1], z=c_a[:, 2],
                mode="lines+markers",
                marker=dict(size=3.5, color="#AB47BC"),
                line=dict(color="#7B1FA2", width=2),
                name="After Camera Path"
            ))

        # Modern Dark-Mode Layout
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            scene=dict(
                xaxis=dict(title="X", backgroundcolor="#161B22", gridcolor="#30363D", showbackground=True),
                yaxis=dict(title="Y", backgroundcolor="#161B22", gridcolor="#30363D", showbackground=True),
                zaxis=dict(title="Z", backgroundcolor="#161B22", gridcolor="#30363D", showbackground=True),
                aspectmode="data"
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            legend=dict(
                x=0.02, y=0.98,
                bgcolor="rgba(22, 27, 34, 0.85)",
                bordercolor="#30363D",
                borderwidth=1
            )
        )

        return fig
