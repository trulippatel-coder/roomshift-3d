"""Generates benchmark sample videos with ground truth 3D spatial changes."""

from typing import Tuple
from pathlib import Path
import json
import logging
import numpy as np

from roomshift.data.synthetic_scene import SyntheticSceneGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_benchmark_dataset(output_dir: Path = Path("data/samples")) -> Tuple[Path, Path, Path]:
    """Generates synthetic before/after MP4 video sweeps and ground truth metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    gen = SyntheticSceneGenerator(width=640, height=480, fov_deg=60.0)
    before_objs, after_objs = gen.get_ground_truth_objects()

    before_video_path = output_dir / "before_room.mp4"
    after_video_path = output_dir / "after_room.mp4"
    gt_json_path = output_dir / "ground_truth.json"

    logger.info("Generating 'before' scene video...")
    gen.generate_video(before_objs, before_video_path, num_frames=30, radius=2.2, height=0.8, arc_deg=100.0)

    logger.info("Generating 'after' scene video...")
    gen.generate_video(after_objs, after_video_path, num_frames=30, radius=2.2, height=0.8, arc_deg=100.0)

    # Save ground truth annotations
    gt_data = {
        "description": "RoomShift 3D Synthetic Benchmark Indoor Environment",
        "objects_before": [
            {
                "name": obj.name,
                "center": obj.center.tolist(),
                "size": obj.size.tolist(),
                "change_type": obj.change_type,
            }
            for obj in before_objs
        ],
        "objects_after": [
            {
                "name": obj.name,
                "center": obj.center.tolist(),
                "size": obj.size.tolist(),
                "change_type": obj.change_type,
            }
            for obj in after_objs
        ],
        "ground_truth_changes": {
            "moved": ["Office Chair (shifted across table)"],
            "removed": ["Laptop (missing from table in after video)"],
            "added": ["Coffee Cup (new on table in after video)"],
            "unchanged": ["Table", "Floor", "Back Wall", "Desk Lamp"],
            "uncertain_unobserved": ["Hidden Corner Box (obstructed behind divider)"]
        }
    }

    with open(gt_json_path, "w", encoding="utf-8") as f:
        json.dump(gt_data, f, indent=2)

    logger.info(f"Benchmark videos generated at: {output_dir}")
    return before_video_path, after_video_path, gt_json_path


if __name__ == "__main__":
    generate_benchmark_dataset()
