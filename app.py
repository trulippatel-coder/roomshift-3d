"""RoomShift 3D - Interactive Web Dashboard & CLI Application.

Provides a Streamlit interface and CLI entrypoint for 3D spatial change detection
in indoor environments from monocular before/after video sweeps.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("roomshift.app")


def run_cli(args: argparse.Namespace):
    """Executes the RoomShift 3D pipeline from the command line."""
    from roomshift.config import RoomShiftConfig
    from roomshift.pipeline import RoomShiftPipeline
    from roomshift.reporting.exporter import ReportExporter

    before_path = Path(args.before)
    after_path = Path(args.after)

    if not before_path.exists():
        logger.error(f"'Before' video not found: {before_path}")
        sys.exit(1)
    if not after_path.exists():
        logger.error(f"'After' video not found: {after_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir or "output")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = RoomShiftConfig()
    config.sfm.backend = args.backend
    config.video.target_width = args.target_width

    logger.info("Initializing RoomShift 3D Pipeline...")
    pipeline = RoomShiftPipeline(config)

    def progress_callback(msg: str, pct: float):
        print(f"[{pct:4.0%}] {msg}", flush=True)

    result = pipeline.run(before_path, after_path, progress_callback=progress_callback)

    # Save structured JSON
    report_path = output_dir / "roomshift_report.json"
    ReportExporter.save_json(result.report, report_path)
    logger.info(f"Report exported to: {report_path}")

    # Save 3D Plotly visualization
    html_path = output_dir / "roomshift_3d_scene.html"
    result.figure_3d.write_html(str(html_path))
    logger.info(f"Interactive 3D visualization exported to: {html_path}")

    # Print executive narrative
    print("\n" + "=" * 60)
    print(result.report.get("natural_language_narrative", ""))
    print("=" * 60 + "\n")


def run_streamlit_dashboard():
    """Renders the interactive Streamlit Web Dashboard."""
    import streamlit as st
    import plotly.graph_objects as go
    from PIL import Image

    from roomshift.config import RoomShiftConfig
    from roomshift.pipeline import RoomShiftPipeline, PipelineResult
    from roomshift.reporting.exporter import ReportExporter
    from roomshift.visualization.plot_3d import Visualizer3D

    # Set page configuration
    st.set_page_config(
        page_title="RoomShift 3D | Spatial Change Intelligence",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom Glassmorphic Dark UI Styling
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        code, pre {
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* Top Hero Header */
        .hero-container {
            background: linear-gradient(135deg, rgba(22, 27, 34, 0.95) 0%, rgba(13, 17, 23, 0.98) 100%);
            border: 1px solid rgba(48, 54, 61, 0.8);
            border-radius: 16px;
            padding: 28px 32px;
            margin-bottom: 24px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.36);
            position: relative;
            overflow: hidden;
        }
        .hero-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #00E676, #00B0FF, #AB47BC, #FF9100);
        }
        .hero-title {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(90deg, #FFFFFF, #E0E0E0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .hero-subtitle {
            color: #8B949E;
            font-size: 1.05rem;
            margin-top: 8px;
            margin-bottom: 0;
        }

        /* Metric Cards */
        .metric-card {
            background: rgba(22, 27, 34, 0.8);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(48, 54, 61, 0.6);
            border-radius: 14px;
            padding: 18px 20px;
            text-align: center;
            transition: all 0.25s ease;
        }
        .metric-card:hover {
            border-color: rgba(56, 139, 253, 0.5);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        }
        .metric-val {
            font-size: 1.8rem;
            font-weight: 700;
            margin-top: 4px;
            color: #58A6FF;
        }
        .metric-lbl {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #8B949E;
            font-weight: 500;
        }

        /* Object Change Cards */
        .obj-card {
            background: rgba(22, 27, 34, 0.85);
            border: 1px solid rgba(48, 54, 61, 0.7);
            border-radius: 12px;
            padding: 18px 22px;
            margin-bottom: 14px;
            transition: all 0.2s ease;
        }
        .obj-card:hover {
            border-color: rgba(110, 118, 129, 0.8);
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
        }
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .badge-added { background: rgba(0, 230, 118, 0.15); color: #00E676; border: 1px solid rgba(0, 230, 118, 0.3); }
        .badge-removed { background: rgba(255, 23, 68, 0.15); color: #FF5252; border: 1px solid rgba(255, 23, 68, 0.3); }
        .badge-moved { background: rgba(0, 176, 255, 0.15); color: #40C4FF; border: 1px solid rgba(0, 176, 255, 0.3); }
        .badge-uncertain { background: rgba(255, 145, 0, 0.15); color: #FFAB40; border: 1px solid rgba(255, 145, 0, 0.3); }
        .badge-unchanged { background: rgba(120, 144, 156, 0.15); color: #B0BEC5; border: 1px solid rgba(120, 144, 156, 0.3); }

        /* Status badge */
        .tag-pill {
            display: inline-block;
            background: rgba(56, 139, 253, 0.15);
            color: #58A6FF;
            border: 1px solid rgba(56, 139, 253, 0.3);
            border-radius: 20px;
            padding: 2px 10px;
            font-size: 0.75rem;
            font-weight: 500;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #0E1117;
            border-right: 1px solid #21262D;
        }
    </style>
    """, unsafe_allow_html=True)

    # Hero Header Section
    st.markdown("""
    <div class="hero-container">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
            <div>
                <h1 class="hero-title">RoomShift 3D <span class="tag-pill">v0.1.0</span></h1>
                <p class="hero-subtitle">Automated 3D Spatial Change Detection & Object Discrepancy Intelligence</p>
            </div>
            <div style="margin-top: 8px;">
                <span class="tag-pill">Monocular SfM</span> &nbsp;
                <span class="tag-pill">Sim(3) Umeyama</span> &nbsp;
                <span class="tag-pill">Frustum Raycasting</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar: Input Selection and Configuration
    with st.sidebar:
        st.markdown("### ⚙️ Analysis Setup")
        input_mode = st.radio(
            "Video Input Source",
            ["Built-in Benchmark Dataset", "Upload Custom Videos"],
            index=0
        )

        before_video_path: Optional[Path] = None
        after_video_path: Optional[Path] = None

        if input_mode == "Built-in Benchmark Dataset":
            sample_before = Path("data/samples/before_room.mp4")
            sample_after = Path("data/samples/after_room.mp4")
            gt_json = Path("data/samples/ground_truth.json")

            if sample_before.exists() and sample_after.exists():
                st.success("✅ Synthetic benchmark dataset loaded (30-frame sweeps with ground truth).")
                before_video_path = sample_before
                after_video_path = sample_after

                if gt_json.exists():
                    with open(gt_json) as f:
                        gt_data = json.load(f)
                    with st.expander("Ground Truth Changes"):
                        for cat, items in gt_data.get("ground_truth_changes", {}).items():
                            st.markdown(f"**{cat.capitalize()}**: {', '.join(items)}")
            else:
                st.warning("⚠️ Benchmark videos missing. Generating now...")
                from examples.generate_sample_data import generate_benchmark_dataset
                before_video_path, after_video_path, _ = generate_benchmark_dataset()
                st.success("Generated benchmark videos!")
        else:
            col_u1, col_u2 = st.columns(2)
            with col_u1:
                uploaded_b = st.file_uploader("Before Video (MP4/MOV)", type=["mp4", "mov", "avi"])
            with col_u2:
                uploaded_a = st.file_uploader("After Video (MP4/MOV)", type=["mp4", "mov", "avi"])

            if uploaded_b and uploaded_a:
                tmp_dir = Path(tempfile.mkdtemp())
                before_video_path = tmp_dir / uploaded_b.name
                after_video_path = tmp_dir / uploaded_a.name
                with open(before_video_path, "wb") as f:
                    f.write(uploaded_b.read())
                with open(after_video_path, "wb") as f:
                    f.write(uploaded_a.read())
                st.success("Custom videos uploaded successfully.")

        st.markdown("---")
        st.markdown("### 🔧 Pipeline Parameters")
        with st.expander("Structure-from-Motion (SfM)"):
            sfm_backend = st.selectbox("Backend", ["opencv", "colmap"], index=0)
            target_width = st.select_slider("Target Frame Width (px)", options=[360, 480, 640, 960], value=640)
            ratio_test = st.slider("Lowe's Ratio Test", min_value=0.60, max_value=0.90, value=0.78, step=0.02)
            max_features = st.slider("Max SIFT Features", min_value=1000, max_value=5000, value=3000, step=500)

        with st.expander("Sim(3) Alignment & ICP"):
            ransac_thresh = st.slider("Umeyama Inlier Threshold", min_value=0.02, max_value=0.20, value=0.08, step=0.01)
            icp_max_iter = st.slider("ICP Iterations", min_value=20, max_value=100, value=60, step=10)

        with st.expander("Spatial Clustering & Verification"):
            dist_thresh = st.slider("Distance Threshold", min_value=0.02, max_value=0.15, value=0.06, step=0.01)
            dbscan_eps = st.slider("DBSCAN Epsilon", min_value=0.03, max_value=0.20, value=0.08, step=0.01)
            dbscan_min_pts = st.slider("DBSCAN Min Points", min_value=4, max_value=30, value=6, step=1)
            min_vis_cams = st.slider("Uncertain Visibility Min Cameras", min_value=1, max_value=5, value=2, step=1)

        run_btn = st.button("🚀 Run RoomShift 3D Pipeline", use_container_width=True, type="primary")

    # Session State to hold analysis results
    if "pipeline_result" not in st.session_state:
        st.session_state.pipeline_result = None

    # Run Analysis Trigger
    if run_btn:
        if before_video_path is None or after_video_path is None:
            st.error("Please provide valid 'Before' and 'After' videos before running analysis.")
        else:
            config = RoomShiftConfig()
            config.video.target_width = target_width
            config.video.min_keyframes = 4
            config.video.max_keyframes = 15
            config.sfm.backend = sfm_backend
            config.sfm.max_features = max_features
            config.sfm.ratio_test = ratio_test
            config.alignment.ransac_threshold = ransac_thresh
            config.alignment.icp_max_iterations = icp_max_iter
            config.change.distance_threshold = dist_thresh
            config.change.dbscan_eps = dbscan_eps
            config.change.dbscan_min_points = dbscan_min_pts
            config.change.uncertain_min_visibility = min_vis_cams

            pipeline = RoomShiftPipeline(config)

            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def progress_cb(msg: str, pct: float):
                progress_bar.progress(pct)
                status_text.markdown(f"**Current Operation**: `{msg}`")

            try:
                with st.spinner("Processing video sweeps with 3D Structure-from-Motion..."):
                    result = pipeline.run(before_video_path, after_video_path, progress_callback=progress_cb)
                st.session_state.pipeline_result = result
                progress_bar.progress(1.0)
                status_text.success(f"Analysis completed in {result.execution_time_sec:.2f} seconds!")
                time.sleep(0.5)
            except Exception as e:
                st.error(f"Pipeline error: {str(e)}")
                logger.exception("Pipeline execution failed")

    result: Optional[PipelineResult] = st.session_state.pipeline_result

    if result is not None:
        # Top KPI Summary Bar
        kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5 = st.columns(5)
        counts = result.report["change_summary"]["counts_by_category"]

        with kpi_c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Moved Objects</div>
                <div class="metric-val" style="color: #40C4FF;">{counts.get('moved', 0)}</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Added Objects</div>
                <div class="metric-val" style="color: #00E676;">{counts.get('added', 0)}</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Removed Objects</div>
                <div class="metric-val" style="color: #FF5252;">{counts.get('removed', 0)}</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Uncertain / Unobserved</div>
                <div class="metric-val" style="color: #FFAB40;">{counts.get('uncertain', 0)}</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_c5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Sim(3) Scale Ratio</div>
                <div class="metric-val" style="color: #E0E0E0;">{result.sim3_alignment.scale:.3f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabbed Navigation
        tab_3d, tab_changes, tab_metrics, tab_keyframes, tab_report = st.tabs([
            "🌐 Interactive 3D Scene",
            "🔍 Detected Spatial Changes",
            "📐 Sim(3) Co-Registration",
            "🎞️ Keyframe Alignment",
            "📄 Narrative & Export"
        ])

        # TAB 1: Interactive 3D WebGL Visualization
        with tab_3d:
            st.markdown("#### Multi-Layer Interactive 3D Scene Viewer")
            st.caption("Rotate, pan, and zoom to inspect aligned 3D point clouds, detected objects, bounding boxes, and camera trajectories.")

            # Layer display controls
            ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns(4)
            with ctrl_c1:
                show_before = st.checkbox("Show 'Before' Points", value=True)
            with ctrl_c2:
                show_after = st.checkbox("Show 'After' Points", value=True)
            with ctrl_c3:
                show_cameras = st.checkbox("Show Camera Trajectories", value=True)
            with ctrl_c4:
                show_boxes = st.checkbox("Show Bounding Wireframes", value=True)

            # Build customized figure based on toggles
            pts_b = result.pts_before_aligned if show_before else np.empty((0, 3))
            pts_a = result.reconstruction_after.points_3d if show_after else np.empty((0, 3))
            cams_b = result.cameras_before_aligned if show_cameras else []
            cams_a = result.reconstruction_after.camera_poses if show_cameras else []

            objs_to_show = result.classified_objects if show_boxes else []

            fig = Visualizer3D.build_comparison_figure(
                pts_before=pts_b,
                pts_after=pts_a,
                classified_objects=objs_to_show,
                cameras_before=cams_b,
                cameras_after=cams_a
            )
            fig.update_layout(height=650)
            st.plotly_chart(fig, use_container_width=True)

            # Color Map Legend
            st.markdown("""
            <div style="display: flex; gap: 18px; justify-content: center; flex-wrap: wrap; margin-top: -10px; font-size: 0.85rem;">
                <span><span style="color: #00E676; font-size: 1.1rem;">■</span> Added</span>
                <span><span style="color: #FF1744; font-size: 1.1rem;">■</span> Removed</span>
                <span><span style="color: #00B0FF; font-size: 1.1rem;">■</span> Moved (with Trajectory)</span>
                <span><span style="color: #FF9100; font-size: 1.1rem;">■</span> Uncertain / Unobserved</span>
                <span><span style="color: #90CAF9; font-size: 1.1rem;">■</span> Before Cloud</span>
                <span><span style="color: #B0BEC5; font-size: 1.1rem;">■</span> After Cloud</span>
            </div>
            """, unsafe_allow_html=True)

        # TAB 2: Detected Spatial Changes Cards
        with tab_changes:
            st.markdown("#### Verified Spatial Discrepancies")
            st.caption("Each cluster is classified using multi-view geometric evidence and camera frustum raycasting.")

            filter_type = st.selectbox(
                "Filter by Change State",
                ["All Discrepancies", "Moved", "Added", "Removed", "Uncertain"]
            )

            filtered_objs = [
                obj for obj in result.classified_objects
                if obj.label != "unchanged" and (filter_type == "All Discrepancies" or obj.label.lower() == filter_type.lower())
            ]

            if not filtered_objs:
                st.info("No spatial changes matching the selected filter.")
            else:
                for obj in filtered_objs:
                    badge_class = f"badge-{obj.label}"
                    centroid_str = f"({obj.centroid[0]:.2f}, {obj.centroid[1]:.2f}, {obj.centroid[2]:.2f})"

                    st.markdown(f"""
                    <div class="obj-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span class="badge {badge_class}">{obj.label}</span> &nbsp;
                                <strong style="font-size: 1.1rem; color: #FFF;">{obj.object_id}</strong>
                            </div>
                            <div style="color: #8B949E; font-size: 0.85rem;">
                                Confidence: <strong>{obj.confidence:.1%}</strong>
                            </div>
                        </div>
                        <div style="margin-top: 10px; font-size: 0.92rem; color: #C9D1D9;">
                            <strong>Centroid Location:</strong> <code>{centroid_str}</code> &nbsp;|&nbsp;
                            <strong>Volume:</strong> <code>{obj.estimated_volume:.4f} units³</code> &nbsp;|&nbsp;
                            <strong>Points:</strong> <code>{obj.point_count}</code> &nbsp;|&nbsp;
                            <strong>Cameras:</strong> <code>{obj.observing_camera_count}</code> ({obj.visibility_score:.1%} coverage)
                        </div>
                    """, unsafe_allow_html=True)

                    if obj.label == "moved" and obj.target_centroid is not None:
                        tgt_str = f"({obj.target_centroid[0]:.2f}, {obj.target_centroid[1]:.2f}, {obj.target_centroid[2]:.2f})"
                        st.markdown(f"""
                        <div style="margin-top: 6px; font-size: 0.90rem; color: #40C4FF;">
                            ➜ <strong>Shifted to:</strong> <code>{tgt_str}</code> &nbsp;|&nbsp;
                            <strong>Displacement Magnitude:</strong> <code>{obj.displacement_magnitude:.3f} units</code>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown(f"""
                        <div style="margin-top: 10px; padding: 10px 14px; background: rgba(13, 17, 23, 0.7); border-radius: 8px; font-size: 0.86rem; color: #8B949E; border-left: 3px solid rgba(88, 166, 255, 0.6);">
                            <em>Evidence: {obj.evidence_reasoning}</em>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # TAB 3: Sim(3) Co-Registration Details
        with tab_metrics:
            st.markdown("#### Sim(3) Similarity Transformation & Scale Alignment")
            st.caption("Cross-sequence feature matching with Umeyama RANSAC and Point-to-Plane ICP refinement.")

            reg = result.report["co_registration"]
            m_c1, m_c2, m_c3, m_c4 = st.columns(4)
            with m_c1:
                st.metric("Scale Ratio (s)", f"{reg['estimated_relative_scale']:.4f}")
            with m_c2:
                st.metric("Alignment RMSE", f"{reg['alignment_rmse']:.4f}")
            with m_c3:
                st.metric("ICP Fitness", f"{reg['alignment_fitness']:.1%}")
            with m_c4:
                st.metric("Inlier Matches", f"{reg['inlier_correspondences']}")

            st.markdown("##### 4×4 Similarity Transformation Matrix")
            st.code(
                "\n".join(["[" + ", ".join([f"{v:10.5f}" for v in row]) + "]" for row in reg["transformation_matrix_4x4"]]),
                language="python"
            )

            st.info(
                "💡 **Scale Notice**: In accordance with monocular computer vision theory, relative units are unscaled. "
                "Absolute metric scale (meters) is not asserted without calibrated physical markers or external depth sensors."
            )

        # TAB 4: Keyframe Gallery
        with tab_keyframes:
            st.markdown("#### Extracted Keyframes & Sharpness Scores")
            st.markdown("##### 'Before' Scene Sweep")
            cols_b = st.columns(min(5, len(result.keyframes_before)))
            for i, kf in enumerate(result.keyframes_before[:10]):
                with cols_b[i % len(cols_b)]:
                    st.image(kf.rgb, caption=f"Frame #{kf.frame_idx} (var: {kf.sharpness:.1f})", use_container_width=True)

            st.markdown("##### 'After' Scene Sweep")
            cols_a = st.columns(min(5, len(result.keyframes_after)))
            for i, kf in enumerate(result.keyframes_after[:10]):
                with cols_a[i % len(cols_a)]:
                    st.image(kf.rgb, caption=f"Frame #{kf.frame_idx} (var: {kf.sharpness:.1f})", use_container_width=True)

        # TAB 5: Narrative Report & Structured JSON Export
        with tab_report:
            st.markdown("#### Executive Report Narrative")
            st.markdown(result.report.get("natural_language_narrative", ""))

            st.markdown("---")
            st.markdown("#### Export Artifacts")
            exp_c1, exp_c2 = st.columns(2)

            json_str = json.dumps(result.report, indent=2)
            with exp_c1:
                st.download_button(
                    label="📥 Download Structured JSON Report",
                    data=json_str,
                    file_name="roomshift_report.json",
                    mime="application/json",
                    use_container_width=True
                )

            with exp_c2:
                html_bytes = result.figure_3d.to_html().encode("utf-8")
                st.download_button(
                    label="🌐 Download Standalone 3D HTML Scene",
                    data=html_bytes,
                    file_name="roomshift_3d_scene.html",
                    mime="text/html",
                    use_container_width=True
                )

            with st.expander("View Raw JSON Schema"):
                st.json(result.report)

    else:
        # Default Welcome State
        st.info("👈 Select your video inputs in the sidebar and click **Run RoomShift 3D Pipeline** to begin spatial analysis.")

        demo_c1, demo_c2 = st.columns(2)
        with demo_c1:
            st.markdown("""
            ### 📌 Core Capabilities
            - **End-to-End Monocular Pipeline**: Takes two casual video sweeps of a room and extracts 3D change intelligence.
            - **Sim(3) 7-DoF Co-Registration**: Solves unknown relative scale, 3D rotation, and 3D translation via Umeyama algorithm & Point-to-Plane ICP.
            - **Multi-Modal Change Classification**: Accurately distinguishes **Added**, **Removed**, **Moved**, and **Unchanged** objects.
            - **Anti-False-Removal (Uncertain Space)**: Verifies opposing camera frustum visibility to ensure unobserved corners are never falsely reported as removed.
            """)
        with demo_c2:
            st.markdown("""
            ### 🛠️ Architecture Workflow
            1. **Adaptive Keyframe Selection**: Optical flow and Laplacian variance motion-aware sampling.
            2. **Native SfM Reconstruction**: High-precision SIFT / ORB feature tracking & Essential matrix recovery.
            3. **Cross-Sequence Matching**: Bi-directional descriptor alignment.
            4. **DBSCAN Residual Clustering**: 3D spatial grouping of geometric discrepancies.
            5. **Interactive 3D WebGL Visualization**: Plotly 3D visualizer with bounding boxes and trajectory lines.
            """)


def main():
    """Main CLI and Web dashboard router."""
    parser = argparse.ArgumentParser(
        description="RoomShift 3D - 3D Spatial Change Detection for Indoor Environments"
    )
    parser.add_argument("--web", action="store_true", help="Launch the interactive Streamlit dashboard")
    parser.add_argument("--before", type=str, help="Path to 'before' room sweep video")
    parser.add_argument("--after", type=str, help="Path to 'after' room sweep video")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save report and visualization")
    parser.add_argument("--backend", type=str, default="opencv", choices=["opencv", "colmap"], help="SfM backend")
    parser.add_argument("--target-width", type=int, default=640, help="Resize target width for frame extraction")

    # If invoked directly via `streamlit run app.py`
    try:
        import streamlit as st
        if hasattr(st, "runtime") and st.runtime.exists():
            run_streamlit_dashboard()
            return
    except ImportError:
        pass

    # If invoked as a CLI executable or with arguments
    args, unknown = parser.parse_known_args()

    if args.before and args.after:
        run_cli(args)
    else:
        # Launch Streamlit app
        import subprocess
        logger.info("Launching RoomShift 3D Streamlit Web Dashboard...")
        cmd = [sys.executable, "-m", "streamlit", "run", __file__] + unknown
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
