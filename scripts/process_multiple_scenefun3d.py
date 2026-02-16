"""
Render all scenes in a root directory structure.

Iterates through {root_dir}/{visit_id}/{video_id} and renders annotations for each scene.
Supports parallel processing with --n-workers.

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

import json
import random
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Annotated, Optional
from pathlib import Path
import subprocess
import sys

import tyro


def find_scenes(root_dir: Path):
    """Find all visit_id/video_id combinations in root_dir."""
    scenes = []
    for visit_path in sorted(root_dir.iterdir()):
        if not visit_path.is_dir():
            continue
        visit_id = visit_path.name
        for video_path in sorted(visit_path.iterdir()):
            if not video_path.is_dir():
                continue
            video_id = video_path.name
            if video_id.endswith(".json") or video_id.endswith(".ply") or video_id.endswith(".npy"):
                continue
            if (video_path / "hires_wide").exists() or (video_path / "lowres_wide").exists():
                scenes.append((visit_id, video_id))
    return scenes


def process_scene(args):
    """Process a single scene in a subprocess, writing to a temp directory."""
    vid_id, v_id, script_path, root_dir, tmp_dir, common_args = args

    cmd = [
        sys.executable,
        script_path,
        "--data-dir", root_dir,
        "--output-dir", tmp_dir,
        "--visit-id", vid_id,
        "--video-id", v_id,
        *common_args,
    ]
    result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        return vid_id, v_id, False, result.stdout[-500:] if result.stdout else ""
    # Print summary line from output (last non-empty line)
    lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
    summary = lines[-1] if lines else ""
    return vid_id, v_id, True, summary


def merge_scene_result(tmp_dir: str, out_path: Path, all_records: list):
    """Merge a single scene's results into the output directory immediately.

    Called from the main thread only (as_completed loop), so no lock needed.
    """
    tmp_path = Path(tmp_dir)
    ds_file = tmp_path / "dataset.json"
    if not ds_file.exists():
        return 0

    with open(ds_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    # Copy files (images, masks, depth, vis)
    for subdir in ("images", "masks", "depths", "vis"):
        src = tmp_path / subdir
        dst = out_path / subdir
        if not src.exists():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for fpath in src.iterdir():
            shutil.copy2(fpath, dst / fpath.name)

    # Append and write merged dataset.json
    all_records.extend(records)
    dataset_path = out_path / "dataset.json"
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    # Cleanup this scene's temp dir
    shutil.rmtree(tmp_path, ignore_errors=True)

    return len(records)


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
    ] = 2.0,
    mask_alpha: Annotated[float, tyro.conf.arg(help="Mask opacity for visualization (0.0-1.0)")] = 0.5,
    max_views: Annotated[
        Optional[int], tyro.conf.arg(help="Max viewpoints per description. None=all valid frames")
    ] = 1,
    min_view_gap: Annotated[int, tyro.conf.arg(help="Minimum frame index gap between selected viewpoints")] = 5,
    output_size: Annotated[int, tyro.conf.arg(help="Output image size NxN (e.g., 840)")] = 840,
    adaptive_crop_scale: Annotated[Optional[float], tyro.conf.arg(help="Image-to-bbox scale for adaptive crop (e.g., 2.5 = output image is 2.5x the annotation bbox). None=fixed crop")] = None,
    bbox_margin: Annotated[float, tyro.conf.arg(help="Margin ratio added around each bbox (e.g., 0.1 = 10%% of bbox size per side)")] = 0.2,
    n_workers: Annotated[int, tyro.conf.arg(help="Number of parallel workers (1=sequential)")] = 16,
    val_ratio: Annotated[float, tyro.conf.arg(help="Fraction of scenes for validation split (0.0=no split, 0.2=20%% val)")] = 0.1,
    split_seed: Annotated[int, tyro.conf.arg(help="Random seed for train/val split")] = 42,
):
    root_path = Path(root_dir).expanduser()
    if not root_path.exists():
        print(f"Error: root_dir '{root_dir}' does not exist")
        sys.exit(1)

    if visit_id is not None and video_id is not None:
        scenes = [(visit_id, video_id)]
        print(f"Processing single scene: {visit_id}/{video_id}")
    else:
        scenes = find_scenes(root_path)
        print(f"Found {len(scenes)} scenes in {root_dir}")
        if len(scenes) == 0:
            print("No valid scenes found.")
            sys.exit(1)

    # Split scenes into train/val if requested
    if val_ratio > 0.0:
        rng = random.Random(split_seed)
        shuffled = list(scenes)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_ratio))
        val_scenes = set(map(tuple, shuffled[:n_val]))
        print(f"Train/val split: {len(scenes) - n_val} train, {n_val} val (seed={split_seed})")
    else:
        val_scenes = set()

    script_path = str(Path(__file__).parent / "process_scenefun3d.py")
    base_out_path = Path(output_dir)

    # Build common args shared by all scenes
    common_args = [
        "--visibility-threshold", str(visibility_threshold),
        "--edge-margin-ratio", str(edge_margin_ratio),
        "--max-depth", str(max_depth),
        "--mask-alpha", str(mask_alpha),
        "--min-view-gap", str(min_view_gap),
    ]
    common_args.extend(["--output-size", str(output_size)])
    common_args.extend(["--bbox-margin", str(bbox_margin)])
    if max_views is not None:
        common_args.extend(["--max-views", str(max_views)])
    if adaptive_crop_scale is not None:
        common_args.extend(["--adaptive-crop-scale", str(adaptive_crop_scale)])

    # Create per-scene temp directories so subprocesses don't conflict on dataset.json
    tmp_base = tempfile.mkdtemp(prefix="scenefun3d_")
    task_args = []
    for vid_id, v_id in scenes:
        tmp_dir = str(Path(tmp_base) / f"{vid_id}_{v_id}")
        task_args.append((vid_id, v_id, script_path, root_dir, tmp_dir, common_args))

    # Determine output path for each scene based on split
    def get_out_path(vid_id, v_id):
        if not val_scenes:
            return base_out_path
        if (vid_id, v_id) in val_scenes:
            return base_out_path / "val"
        return base_out_path / "train"

    # Process scenes — merge results immediately as each scene completes
    print(f"Processing {len(scenes)} scenes with {n_workers} workers...")
    succeeded = 0
    failed = 0
    train_records = []
    val_records = []

    def handle_result(vid_id, v_id, ok, msg, tmp_dir, idx):
        nonlocal succeeded, failed
        if ok:
            succeeded += 1
            out_path = get_out_path(vid_id, v_id)
            out_path.mkdir(parents=True, exist_ok=True)
            is_val = (vid_id, v_id) in val_scenes
            records = val_records if is_val else train_records
            n_rec = merge_scene_result(tmp_dir, out_path, records)
            split_tag = " [val]" if is_val else ""
            print(f"[{idx}/{len(scenes)}] {vid_id}/{v_id}: OK ({n_rec} records, {len(records)} total){split_tag} | {msg}")
        else:
            failed += 1
            print(f"[{idx}/{len(scenes)}] {vid_id}/{v_id}: FAILED | {msg[:200]}")

    if n_workers <= 1:
        # Sequential
        for i, args in enumerate(task_args):
            print(f"[{i+1}/{len(scenes)}] {args[0]}/{args[1]}...")
            vid_id, v_id, ok, msg = process_scene(args)
            handle_result(vid_id, v_id, ok, msg, args[4], i + 1)
    else:
        # Parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_scene, a): a for a in task_args}
            for i, future in enumerate(as_completed(futures)):
                a = futures[future]
                vid_id, v_id, ok, msg = future.result()
                handle_result(vid_id, v_id, ok, msg, a[4], i + 1)

    # Cleanup temp base dir
    shutil.rmtree(tmp_base, ignore_errors=True)

    print(f"\n{'=' * 80}")
    if val_scenes:
        print(f"Done! train: {len(train_records)} records, val: {len(val_records)} records from {succeeded} scenes ({failed} failed)")
        print(f"Output: {base_out_path / 'train' / 'dataset.json'}")
        print(f"        {base_out_path / 'val' / 'dataset.json'}")
    else:
        total = len(train_records)
        print(f"Done! {total} records from {succeeded} scenes ({failed} failed)")
        print(f"Output: {base_out_path / 'dataset.json'}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    tyro.cli(main)
