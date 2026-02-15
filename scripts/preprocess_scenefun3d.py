"""
Render all scenes in a root directory structure.

Iterates through {root_dir}/{visit_id}/{video_id} and renders annotations for each scene.

Before running this example, ensure the dataset is organized as:
root_dir/
  {visit_id}/
    {video_id}/
      hires_wide/
      hires_depth/
      hires_poses.traj
      ...

SceneFun3D Toolkit
"""

import tyro
from typing import Annotated, Optional
from pathlib import Path
import subprocess
import sys


def find_scenes(root_dir: Path):
    """Find all visit_id/video_id combinations in root_dir

    Args:
        root_dir: Root directory containing visit_id subdirectories

    Returns:
        List of (visit_id, video_id) tuples
    """
    scenes = []

    # Iterate through visit_id directories
    for visit_path in sorted(root_dir.iterdir()):
        if not visit_path.is_dir():
            continue

        visit_id = visit_path.name

        # Skip if not a valid visit_id (should be numeric or specific format)
        # Iterate through video_id directories
        for video_path in sorted(visit_path.iterdir()):
            if not video_path.is_dir():
                continue

            video_id = video_path.name

            # Skip special directories
            if video_id.endswith(".json") or video_id.endswith(".ply") or video_id.endswith(".npy"):
                continue

            # Check if this looks like a valid video directory
            # (contains hires_wide or other expected subdirectories)
            if (video_path / "hires_wide").exists() or (video_path / "lowres_wide").exists():
                scenes.append((visit_id, video_id))

    return scenes


def main(
    root_dir: Annotated[
        str, tyro.conf.arg(help="Root directory containing dataset ({visit_id}/{video_id} structure)")
    ],
    output_dir: Annotated[str, tyro.conf.arg(help="Output directory for rendered results")],
    visit_id: Annotated[
        Optional[str],
        tyro.conf.arg(help="Single visit_id to process (optional, processes all if not specified)"),
    ] = None,
    video_id: Annotated[
        Optional[str],
        tyro.conf.arg(help="Single video_id to process (optional, processes all if not specified)"),
    ] = None,
    visibility_threshold: Annotated[
        float, tyro.conf.arg(help="Occlusion visibility threshold (0.05=strict, 0.1=default, 0.25=lenient)")
    ] = 0.05,
    edge_margin_ratio: Annotated[
        float, tyro.conf.arg(help="Reject frames where mask centroid is within this fraction of image edge (0.0-0.5)")
    ] = 0.1,
    max_depth: Annotated[
        float, tyro.conf.arg(help="Maximum distance (meters) from camera to annotation. 0=disabled")
    ] = 3.0,
    mask_alpha: Annotated[float, tyro.conf.arg(help="Mask opacity for visualization (0.0-1.0)")] = 0.5,
    max_views: Annotated[
        Optional[int], tyro.conf.arg(help="Max viewpoints per description. None=all valid frames")
    ] = None,
    min_view_gap: Annotated[int, tyro.conf.arg(help="Minimum frame index gap between selected viewpoints")] = 5,
):
    # Validate root_dir
    root_path = Path(root_dir).expanduser()
    if not root_path.exists():
        print(f"Error: root_dir '{root_dir}' does not exist")
        sys.exit(1)

    # Determine which scenes to process
    if visit_id is not None and video_id is not None:
        # Process single scene
        scenes = [(visit_id, video_id)]
        print(f"Processing single scene: {visit_id}/{video_id}")
    else:
        # Find all scenes in root_dir
        scenes = find_scenes(root_path)
        print(f"Found {len(scenes)} scenes in {root_dir}")

        if len(scenes) == 0:
            print("No valid scenes found. Directory structure should be {root_dir}/{visit_id}/{video_id}/")
            sys.exit(1)

    # Process each scene
    for idx, (vid_id, video_id) in enumerate(scenes):
        print(f"\n{'=' * 80}")
        print(f"Processing scene {idx + 1}/{len(scenes)}: {vid_id}/{video_id}")
        print(f"{'=' * 80}\n")

        # Build command to call render_annotations.py
        script_path = str(Path(__file__).parent / "render_annotations.py")
        cmd = [
            sys.executable,
            script_path,
            "--data-dir",
            root_dir,
            "--output-dir",
            output_dir,
            "--visit-id",
            vid_id,
            "--video-id",
            video_id,
            "--visibility-threshold",
            str(visibility_threshold),
            "--edge-margin-ratio",
            str(edge_margin_ratio),
            "--max-depth",
            str(max_depth),
            "--mask-alpha",
            str(mask_alpha),
            "--min-view-gap",
            str(min_view_gap),
        ]
        if max_views is not None:
            cmd.extend(["--max-views", str(max_views)])

        # Run the command
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"Warning: Scene {vid_id}/{video_id} failed, skipping...")
            continue

    print(f"\n{'=' * 80}")
    print(f"Completed processing {len(scenes)} scenes")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    tyro.cli(main)
