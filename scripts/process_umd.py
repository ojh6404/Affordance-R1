"""
Convert UMD Part-Affordance Dataset to SceneFun3D-like JSON format.

Reads RGB-D images with pixel-level affordance labels, detects ground plane via RANSAC,
and generates pick-up trajectories for grasp/wrap-grasp/support affordances.

Usage:
    python scripts/process_umd.py --root-dir ~/alderamin/datasets/umd/ --output-dir /tmp/umd_test
"""

import json
import random
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated, Optional

import cv2
import numpy as np
import tyro
from scipy.io import loadmat


# UMD affordance label IDs
AFFORDANCE_NAMES = {
    1: "grasp",
    2: "cut",
    3: "scoop",
    4: "contain",
    5: "pound",
    6: "support",
    7: "grasp",  # wrap-grasp -> grasp
}

# Affordances that get pick-up trajectories
PICKUP_AFFORDANCES = {"grasp", "support"}

# Kinect v1 default intrinsics
KINECT_FX = 570.3
KINECT_FY = 570.3
KINECT_CX = 320.0
KINECT_CY = 240.0

# Problem templates per affordance
PROBLEM_TEMPLATES = {
    "grasp": "To pick up the {tool}, where should I grasp?",
    "cut": "To cut with the {tool}, which part should I use?",
    "scoop": "To scoop with the {tool}, which part should I use?",
    "contain": "Where is the containing part of the {tool}?",
    "pound": "To pound with the {tool}, which part should I use?",
    "support": "To support something with the {tool}, which part should I use?",
}


def find_manual_samples(root_dir: Path, include_clutter: bool = True):
    """Find all manual-annotation frames (every 3rd frame, starting at index 1).

    Returns list of dicts with keys: tool_name, frame_index, rgb_path, depth_path, label_path.
    """
    samples = []

    # Part 1: tools
    tools_dir = root_dir / "part-affordance-dataset" / "tools"
    if tools_dir.exists():
        for tool_dir in sorted(tools_dir.iterdir()):
            if not tool_dir.is_dir():
                continue
            tool_name = tool_dir.name
            # Find all rgb files to discover frames
            rgb_files = sorted(tool_dir.glob(f"{tool_name}_*_rgb.jpg"))
            for rgb_path in rgb_files:
                # Extract frame index from filename: {tool}_{frame:08d}_rgb.jpg
                match = re.match(rf"^{re.escape(tool_name)}_(\d+)_rgb\.jpg$", rgb_path.name)
                if not match:
                    continue
                frame_idx = int(match.group(1))
                # Manual annotations are every 3rd frame (1, 4, 7, ...)
                if (frame_idx - 1) % 3 != 0:
                    continue

                depth_path = rgb_path.parent / f"{tool_name}_{frame_idx:08d}_depth.png"
                label_path = rgb_path.parent / f"{tool_name}_{frame_idx:08d}_label.mat"
                if depth_path.exists() and label_path.exists():
                    samples.append(
                        {
                            "tool_name": tool_name,
                            "frame_index": frame_idx,
                            "rgb_path": str(rgb_path),
                            "depth_path": str(depth_path),
                            "label_path": str(label_path),
                            "source": "tools",
                        }
                    )

    # Part 2: clutter scenes
    if include_clutter:
        clutter_dir = root_dir / "part-affordance-clutter" / "clutter"
        if clutter_dir.exists():
            for scene_dir in sorted(clutter_dir.iterdir()):
                if not scene_dir.is_dir():
                    continue
                scene_name = scene_dir.name
                rgb_files = sorted(scene_dir.glob(f"{scene_name}_*_rgb.jpg"))
                for rgb_path in rgb_files:
                    match = re.match(rf"^{re.escape(scene_name)}_(\d+)_rgb\.jpg$", rgb_path.name)
                    if not match:
                        continue
                    frame_idx = int(match.group(1))
                    if (frame_idx - 1) % 3 != 0:
                        continue

                    depth_path = rgb_path.parent / f"{scene_name}_{frame_idx:08d}_depth.png"
                    label_path = rgb_path.parent / f"{scene_name}_{frame_idx:08d}_label.mat"
                    if depth_path.exists() and label_path.exists():
                        samples.append(
                            {
                                "tool_name": scene_name,
                                "frame_index": frame_idx,
                                "rgb_path": str(rgb_path),
                                "depth_path": str(depth_path),
                                "label_path": str(label_path),
                                "source": "clutter",
                            }
                        )

    return samples


def depth_to_pointcloud(depth_mm, fx, fy, cx, cy):
    """Convert uint16 depth (mm) to 3D point cloud in camera coordinates.

    Returns:
        points_3d: (H, W, 3) array in meters, invalid points are NaN.
        valid_mask: (H, W) boolean mask of valid depth pixels.
    """
    h, w = depth_mm.shape
    z = depth_mm.astype(np.float64) / 1000.0  # mm -> meters

    valid_mask = z > 0
    z[~valid_mask] = np.nan

    u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))
    x = (u_coords - cx) * z / fx
    y = (v_coords - cy) * z / fy

    points_3d = np.stack([x, y, z], axis=-1)
    return points_3d, valid_mask


def detect_ground_plane(points_3d_flat, threshold=0.01, n_iterations=1000, rng=None):
    """RANSAC ground plane detection on 3D points.

    Args:
        points_3d_flat: (N, 3) valid 3D points.
        threshold: inlier distance threshold in meters.
        n_iterations: number of RANSAC iterations.

    Returns:
        normal: (3,) unit normal vector of the plane.
        d: plane distance (ax + by + cz + d = 0).
        inlier_ratio: fraction of inlier points.
        Returns (None, None, 0) if detection fails.
    """
    if len(points_3d_flat) < 3:
        return None, None, 0

    if rng is None:
        rng = np.random.default_rng(42)

    n_pts = len(points_3d_flat)
    best_inliers = 0
    best_normal = None
    best_d = None

    for _ in range(n_iterations):
        # Sample 3 random points
        idx = rng.choice(n_pts, size=3, replace=False)
        p0, p1, p2 = points_3d_flat[idx]

        # Compute plane normal
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal = normal / norm
        d = -np.dot(normal, p0)

        # Count inliers
        distances = np.abs(points_3d_flat @ normal + d)
        n_inliers = np.sum(distances < threshold)

        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_normal = normal
            best_d = d

    if best_normal is None:
        return None, None, 0

    # Ensure normal points upward (away from ground, i.e. Y component is negative in Kinect coords)
    # Kinect: Y axis points down, so "up" direction has negative Y
    if best_normal[1] > 0:
        best_normal = -best_normal
        best_d = -best_d  # type: ignore[operator]

    return best_normal, best_d, best_inliers / n_pts


def create_pickup_trajectory(plane_normal, n_points=5, length=0.1):
    """Generate pick-up trajectory points along plane normal from origin.

    Returns:
        trajectory: list of n_points [x, y, z] lists (relative to origin).
    """
    trajectory = []
    for i in range(n_points):
        t = i * length / (n_points - 1) if n_points > 1 else 0
        point = plane_normal * t
        trajectory.append([round(float(c), 4) for c in point])
    return trajectory


def project_3d_to_2d(point_3d, fx, fy, cx, cy):
    """Project a 3D camera-coordinate point to 2D pixel coordinates."""
    if point_3d[2] <= 0:
        return None
    u = round(float(point_3d[0] * fx / point_3d[2] + cx))
    v = round(float(point_3d[1] * fy / point_3d[2] + cy))
    return [u, v]


def process_frame(
    sample,
    intrinsics,
    ransac_threshold,
    ransac_iterations,
    traj_length,
    traj_points,
    bbox_margin,
    output_size,
    adaptive_crop_scale,
    affordances=None,
):
    """Process a single frame: extract affordance masks, detect plane, generate records.

    Returns:
        list of records for this frame, or empty list on failure.
        Also returns (rgb_cropped, masks_dict, depth_cropped) for saving.
    """
    fx, fy, cx, cy = intrinsics
    tool_name = sample["tool_name"]
    frame_idx = sample["frame_index"]

    # Load data
    rgb = cv2.imread(sample["rgb_path"])
    if rgb is None:
        return [], None
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = rgb.shape[:2]

    depth_mm = cv2.imread(sample["depth_path"], cv2.IMREAD_UNCHANGED)
    if depth_mm is None:
        return [], None
    if depth_mm.dtype != np.uint16:
        depth_mm = depth_mm.astype(np.uint16)

    mat_data = loadmat(sample["label_path"])
    gt_label = mat_data["gt_label"]  # (480, 640) uint8

    # Build point cloud for ground plane detection
    points_3d, valid_mask = depth_to_pointcloud(depth_mm, fx, fy, cx, cy)
    valid_points = points_3d[valid_mask]

    # Subsample for faster RANSAC
    if len(valid_points) > 50000:
        rng = np.random.default_rng(frame_idx)
        subsample_idx = rng.choice(len(valid_points), size=50000, replace=False)
        plane_points = valid_points[subsample_idx]
    else:
        plane_points = valid_points
        rng = np.random.default_rng(frame_idx)

    plane_normal, plane_d, _inlier_ratio = detect_ground_plane(
        plane_points, threshold=ransac_threshold, n_iterations=ransac_iterations, rng=rng
    )

    # Extract per-affordance masks and connected components
    affordance_items = []
    for label_id, aff_name in AFFORDANCE_NAMES.items():
        if affordances is not None and aff_name not in affordances:
            continue
        binary = (gt_label == label_id).astype(np.uint8)
        if binary.sum() == 0:
            continue

        # Connected components to separate distinct regions
        n_labels, labels_cc, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        for cc_idx in range(1, n_labels):  # skip background (0)
            cc_mask = (labels_cc == cc_idx).astype(np.uint8) * 255
            area = stats[cc_idx, cv2.CC_STAT_AREA]
            if area < 50:  # skip tiny components
                continue

            # Bbox from connected component stats
            x0 = stats[cc_idx, cv2.CC_STAT_LEFT]
            y0 = stats[cc_idx, cv2.CC_STAT_TOP]
            w_cc = stats[cc_idx, cv2.CC_STAT_WIDTH]
            h_cc = stats[cc_idx, cv2.CC_STAT_HEIGHT]
            x1 = x0 + w_cc
            y1 = y0 + h_cc

            # Centroid
            cx_pt = int(round(centroids[cc_idx][0]))
            cy_pt = int(round(centroids[cc_idx][1]))

            affordance_items.append(
                {
                    "aff_name": aff_name,
                    "mask": cc_mask,
                    "bbox": [x0, y0, x1, y1],
                    "point": [cx_pt, cy_pt],
                    "area": area,
                    "cc_idx": cc_idx,
                }
            )

    if len(affordance_items) == 0:
        return [], None

    # Group all affordances for this frame into a single record
    solution = []
    masks_to_save = {}  # aff_name -> combined mask for that affordance

    for item in affordance_items:
        aff_name = item["aff_name"]
        bbox = item["bbox"]
        point = item["point"]

        sol_item = {
            "bbox_2d": bbox,
            "point_2d": point,
            "affordance": aff_name,
        }

        # Generate trajectory for pick-up affordances
        if aff_name in PICKUP_AFFORDANCES and plane_normal is not None:
            sol_item["motion_type"] = "trans"
            sol_item["motion_axis"] = [round(float(x), 4) for x in plane_normal]

            # Get 3D origin from mask centroid depth
            cu, cv_pt = point
            cu = np.clip(cu, 0, w_orig - 1)
            cv_pt = np.clip(cv_pt, 0, h_orig - 1)
            z_origin = depth_mm[cv_pt, cu] / 1000.0
            if z_origin > 0:
                ox = (cu - cx) * z_origin / fx
                oy = (cv_pt - cy) * z_origin / fy
                oz = z_origin
                origin_cam = [round(float(ox), 4), round(float(oy), 4), round(float(oz), 4)]
                sol_item["motion_origin_cam"] = origin_cam
                sol_item["motion_origin_2d"] = [cu, cv_pt]

                # 2D motion axis
                origin_3d = np.array([ox, oy, oz])
                axis_end_3d = origin_3d + plane_normal * 0.15
                axis_end_2d = project_3d_to_2d(axis_end_3d, fx, fy, cx, cy)
                if axis_end_2d is not None:
                    dx = axis_end_2d[0] - cu
                    dy = axis_end_2d[1] - cv_pt
                    norm_2d = (dx**2 + dy**2) ** 0.5
                    if norm_2d > 1e-6:
                        sol_item["motion_axis_2d"] = [round(dx / norm_2d, 4), round(dy / norm_2d, 4)]

                # Generate trajectory
                traj = create_pickup_trajectory(plane_normal, n_points=traj_points, length=traj_length)
                sol_item["trajectory"] = traj

                # Project trajectory to 2D
                traj_2d = []
                for pt in traj:
                    p_cam = origin_3d + np.array(pt)
                    p2d = project_3d_to_2d(p_cam, fx, fy, cx, cy)
                    traj_2d.append(p2d)
                sol_item["trajectory_2d"] = traj_2d

        solution.append(sol_item)

        # Accumulate masks per affordance
        if aff_name not in masks_to_save:
            masks_to_save[aff_name] = item["mask"].copy()
        else:
            masks_to_save[aff_name] = np.maximum(masks_to_save[aff_name], item["mask"])

    # Combined mask for all affordances
    combined_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
    for m in masks_to_save.values():
        combined_mask = np.maximum(combined_mask, m)

    # Crop and resize
    ys_mask, xs_mask = np.where(combined_mask > 0)
    if len(xs_mask) > 0:
        cx_crop = int(np.mean(xs_mask))
        cy_crop = int(np.mean(ys_mask))
    else:
        cx_crop, cy_crop = w_orig // 2, h_orig // 2

    img_min_side = min(w_orig, h_orig)
    if adaptive_crop_scale is not None and len(xs_mask) > 0:
        bbox_w = int(xs_mask.max()) - int(xs_mask.min())
        bbox_h = int(ys_mask.max()) - int(ys_mask.min())
        actual_crop = min(max(int(max(bbox_w, bbox_h) * adaptive_crop_scale), output_size), img_min_side)
    else:
        actual_crop = min(output_size, img_min_side)

    half = actual_crop // 2
    crop_x0 = max(0, min(cx_crop - half, w_orig - actual_crop))
    crop_y0 = max(0, min(cy_crop - half, h_orig - actual_crop))
    crop_x1, crop_y1 = crop_x0 + actual_crop, crop_y0 + actual_crop

    rgb_cropped = rgb[crop_y0:crop_y1, crop_x0:crop_x1]
    depth_cropped = depth_mm[crop_y0:crop_y1, crop_x0:crop_x1]
    combined_mask_cropped = combined_mask[crop_y0:crop_y1, crop_x0:crop_x1]

    # Crop individual affordance masks
    masks_cropped = {}
    for aff_name, m in masks_to_save.items():
        masks_cropped[aff_name] = m[crop_y0:crop_y1, crop_x0:crop_x1]

    # Resize if needed
    if actual_crop != output_size:
        resize_scale = output_size / actual_crop
        rgb_cropped = cv2.resize(rgb_cropped, (output_size, output_size), interpolation=cv2.INTER_AREA)
        depth_cropped = cv2.resize(depth_cropped, (output_size, output_size), interpolation=cv2.INTER_NEAREST)
        combined_mask_cropped = cv2.resize(
            combined_mask_cropped, (output_size, output_size), interpolation=cv2.INTER_NEAREST
        )
        for aff_name in masks_cropped:
            masks_cropped[aff_name] = cv2.resize(
                masks_cropped[aff_name], (output_size, output_size), interpolation=cv2.INTER_NEAREST
            )
    else:
        resize_scale = 1.0

    out_h = out_w = output_size if actual_crop != output_size else actual_crop

    # Adjust 2D coordinates in solution
    out_intrinsics = (
        fx * resize_scale,
        fy * resize_scale,
        (cx - crop_x0) * resize_scale,
        (cy - crop_y0) * resize_scale,
    )

    for sol_item in solution:
        b = sol_item["bbox_2d"]
        bx0 = int((b[0] - crop_x0) * resize_scale)
        by0 = int((b[1] - crop_y0) * resize_scale)
        bx1 = int((b[2] - crop_x0) * resize_scale)
        by1 = int((b[3] - crop_y0) * resize_scale)
        mw = int((bx1 - bx0) * bbox_margin)
        mh = int((by1 - by0) * bbox_margin)
        sol_item["bbox_2d"] = [
            max(0, bx0 - mw),
            max(0, by0 - mh),
            min(out_w, bx1 + mw),
            min(out_h, by1 + mh),
        ]
        p = sol_item["point_2d"]
        sol_item["point_2d"] = [
            int((p[0] - crop_x0) * resize_scale),
            int((p[1] - crop_y0) * resize_scale),
        ]
        if "motion_origin_2d" in sol_item:
            mo = sol_item["motion_origin_2d"]
            sol_item["motion_origin_2d"] = [
                int((mo[0] - crop_x0) * resize_scale),
                int((mo[1] - crop_y0) * resize_scale),
            ]

        # Re-project trajectory_2d with cropped intrinsics
        if "trajectory" in sol_item and "motion_origin_cam" in sol_item:
            o_cam = np.array(sol_item["motion_origin_cam"])
            fx_o, fy_o, cx_o, cy_o = out_intrinsics
            traj_2d = []
            for pt in sol_item["trajectory"]:
                p_cam = o_cam + np.array(pt)
                if p_cam[2] > 0:
                    u = round(float(p_cam[0] * fx_o / p_cam[2] + cx_o))
                    v = round(float(p_cam[1] * fy_o / p_cam[2] + cy_o))
                    traj_2d.append([u, v])
                else:
                    traj_2d.append(None)
            sol_item["trajectory_2d"] = traj_2d

    # Build file names
    base_name = f"{tool_name}_{frame_idx:08d}"

    # Build records: one per affordance type present
    records = []
    aff_names_present = list(masks_cropped.keys())

    for aff_name in aff_names_present:
        sample_id = f"{base_name}_{aff_name}"
        aff_solution = [s for s in solution if s["affordance"] == aff_name]
        if not aff_solution:
            continue

        problem_text = PROBLEM_TEMPLATES.get(aff_name, f"Where is the {aff_name} part of the {tool_name}?")
        # Clean tool name for problem text (e.g. "knife_01" -> "knife")
        tool_display = tool_name.rsplit("_", 1)[0] if "_" in tool_name else tool_name
        problem_text = problem_text.format(tool=tool_display)

        record = {
            "id": sample_id,
            "problem": problem_text,
            "solution": aff_solution,
            "image": f"images/{base_name}.jpg",
            "mask": f"masks/{sample_id}.png",
            "depth": f"depths/{base_name}.png",
            "img_height": out_h,
            "img_width": out_w,
            "aff_name": aff_name,
            "part_name": None,
            "metadata": {
                "dataset": "UMD",
                "tool_name": tool_name,
                "frame_index": frame_idx,
                "source": sample["source"],
                "intrinsics": [round(float(x), 2) for x in out_intrinsics],
            },
        }

        if plane_normal is not None and plane_d is not None:
            record["metadata"]["plane_normal"] = [round(float(x), 4) for x in plane_normal]
            record["metadata"]["plane_distance"] = round(float(plane_d), 4)

        records.append(record)

    return records, {
        "rgb": rgb_cropped,
        "depth": depth_cropped,
        "combined_mask": combined_mask_cropped,
        "masks": masks_cropped,
        "base_name": base_name,
    }


def process_frame_worker(args):
    """Worker function for parallel processing of a single frame."""
    sample, output_dir, intrinsics, params = args
    try:
        records, data = process_frame(
            sample,
            intrinsics,
            ransac_threshold=params["ransac_threshold"],
            ransac_iterations=params["ransac_iterations"],
            traj_length=params["traj_length"],
            traj_points=params["traj_points"],
            bbox_margin=params["bbox_margin"],
            output_size=params["output_size"],
            adaptive_crop_scale=params["adaptive_crop_scale"],
            affordances=params.get("affordances"),
        )
        if not records or data is None:
            return sample["tool_name"], sample["frame_index"], 0, "no annotations"

        out_path = Path(output_dir)
        base_name = data["base_name"]

        # Save image (shared across affordances)
        img_path = out_path / "images" / f"{base_name}.jpg"
        if not img_path.exists():
            cv2.imwrite(str(img_path), cv2.cvtColor(data["rgb"], cv2.COLOR_RGB2BGR))

        # Save depth (shared across affordances)
        depth_path = out_path / "depths" / f"{base_name}.png"
        if not depth_path.exists():
            cv2.imwrite(str(depth_path), data["depth"])

        # Save per-affordance masks
        for aff_name, mask in data["masks"].items():
            mask_path = out_path / "masks" / f"{base_name}_{aff_name}.png"
            cv2.imwrite(str(mask_path), mask)

        # Save visualization
        vis_img = data["rgb"].copy()
        mask_bool = data["combined_mask"] > 0
        overlay = np.zeros_like(vis_img)
        overlay[mask_bool] = (0, 255, 0)
        vis_img[mask_bool] = (0.5 * overlay[mask_bool] + 0.5 * vis_img[mask_bool]).astype(np.uint8)

        # Draw trajectories on visualization
        for rec in records:
            for sol in rec["solution"]:
                if "trajectory_2d" in sol:
                    pts = [p for p in sol["trajectory_2d"] if p is not None]
                    for j in range(len(pts) - 1):
                        cv2.line(vis_img, tuple(pts[j]), tuple(pts[j + 1]), (0, 100, 255), 2)
                    for p in pts:
                        cv2.circle(vis_img, tuple(p), 4, (0, 100, 255), -1)

        vis_path = out_path / "vis" / f"{base_name}.jpg"
        cv2.imwrite(str(vis_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

        return sample["tool_name"], sample["frame_index"], len(records), records

    except Exception as e:
        return sample["tool_name"], sample["frame_index"], 0, f"error: {e}"


def main(
    root_dir: Annotated[str, tyro.conf.arg(help="UMD dataset root directory")] = "~/alderamin/datasets/umd/",
    output_dir: Annotated[str, tyro.conf.arg(help="Output directory")] = "affordance_dataset/umd",
    output_size: Annotated[int, tyro.conf.arg(help="Output image size NxN")] = 480,
    n_workers: Annotated[int, tyro.conf.arg(help="Number of parallel workers (1=sequential)")] = 16,
    val_ratio: Annotated[float, tyro.conf.arg(help="Fraction of tools for validation split")] = 0.1,
    split_seed: Annotated[int, tyro.conf.arg(help="Random seed for train/val split")] = 42,
    include_clutter: Annotated[bool, tyro.conf.arg(help="Include clutter scenes")] = True,
    ransac_threshold: Annotated[float, tyro.conf.arg(help="RANSAC inlier distance threshold (meters)")] = 0.01,
    ransac_iterations: Annotated[int, tyro.conf.arg(help="RANSAC iterations")] = 1000,
    traj_length: Annotated[float, tyro.conf.arg(help="Trajectory length (meters)")] = 0.1,
    traj_points: Annotated[int, tyro.conf.arg(help="Number of trajectory points")] = 5,
    bbox_margin: Annotated[float, tyro.conf.arg(help="Bbox margin ratio")] = 0.2,
    adaptive_crop_scale: Annotated[
        Optional[float], tyro.conf.arg(help="Adaptive crop scale (None=fixed crop)")
    ] = None,
    affordances: Annotated[tuple[str, ...], tyro.conf.arg(help="Affordance types to include (default: grasp)")] = (
        "grasp",
    ),
):
    t_start = time.time()

    root_path = Path(root_dir).expanduser()
    if not root_path.exists():
        print(f"Error: root_dir '{root_dir}' does not exist")
        return

    # Find samples
    print("Finding manual annotation samples...")
    samples = find_manual_samples(root_path, include_clutter=include_clutter)
    print(f"Found {len(samples)} manual annotation frames")
    if len(samples) == 0:
        print("No samples found.")
        return

    # Get unique tool names for train/val split
    tool_names = sorted({s["tool_name"] for s in samples})
    print(f"Tools: {len(tool_names)} unique")

    # Train/val split by tool name
    if val_ratio > 0.0:
        rng = random.Random(split_seed)
        shuffled_tools = list(tool_names)
        rng.shuffle(shuffled_tools)
        n_val = max(1, int(len(shuffled_tools) * val_ratio))
        val_tools = set(shuffled_tools[:n_val])
        print(f"Train/val split: {len(tool_names) - n_val} train, {n_val} val tools (seed={split_seed})")
    else:
        val_tools = set()

    # Create output directories
    base_out = Path(output_dir)
    if val_tools:
        for split in ("train", "val"):
            for subdir in ("images", "masks", "depths", "vis"):
                (base_out / split / subdir).mkdir(parents=True, exist_ok=True)
    else:
        for subdir in ("images", "masks", "depths", "vis"):
            (base_out / subdir).mkdir(parents=True, exist_ok=True)

    aff_set = set(affordances) if affordances else None
    print(f"Affordances: {sorted(aff_set) if aff_set else 'all'}")

    intrinsics = (KINECT_FX, KINECT_FY, KINECT_CX, KINECT_CY)
    params = {
        "ransac_threshold": ransac_threshold,
        "ransac_iterations": ransac_iterations,
        "traj_length": traj_length,
        "traj_points": traj_points,
        "bbox_margin": bbox_margin,
        "output_size": output_size,
        "adaptive_crop_scale": adaptive_crop_scale,
        "affordances": aff_set,
    }

    # Prepare worker args with correct output dirs
    worker_args = []
    for sample in samples:
        is_val = sample["tool_name"] in val_tools
        if val_tools:
            out_dir = str(base_out / ("val" if is_val else "train"))
        else:
            out_dir = str(base_out)
        worker_args.append((sample, out_dir, intrinsics, params))

    # Process frames
    print(f"Processing {len(samples)} frames with {n_workers} workers...")
    train_records = []
    val_records = []
    succeeded = 0
    failed = 0

    def handle_result(tool_name, frame_idx, n_rec, result, is_val, idx):
        nonlocal succeeded, failed
        if isinstance(result, list) and n_rec > 0:
            succeeded += 1
            records_list = val_records if is_val else train_records
            records_list.extend(result)
            if idx % 200 == 0 or idx == len(samples):
                split_tag = " [val]" if is_val else ""
                print(
                    f"  [{idx}/{len(samples)}] {tool_name}_{frame_idx:08d}: "
                    f"{n_rec} records{split_tag} (total: {len(train_records)}+{len(val_records)})"
                )
        else:
            failed += 1
            if idx % 500 == 0:
                msg = result if isinstance(result, str) else "failed"
                print(f"  [{idx}/{len(samples)}] {tool_name}_{frame_idx:08d}: {msg}")

    if n_workers <= 1:
        for i, args in enumerate(worker_args):
            sample = args[0]
            result = process_frame_worker(args)
            tool_name, frame_idx, n_rec, records_or_msg = result
            is_val = sample["tool_name"] in val_tools
            handle_result(tool_name, frame_idx, n_rec, records_or_msg, is_val, i + 1)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_frame_worker, a): (i, a[0]) for i, a in enumerate(worker_args)}
            for future in as_completed(futures):
                idx_0, sample = futures[future]
                try:
                    tool_name, frame_idx, n_rec, records_or_msg = future.result()
                except Exception as e:
                    tool_name, frame_idx = sample["tool_name"], sample["frame_index"]
                    n_rec, records_or_msg = 0, f"exception: {e}"
                is_val = sample["tool_name"] in val_tools
                handle_result(tool_name, frame_idx, n_rec, records_or_msg, is_val, idx_0 + 1)

    # Write dataset.json
    if val_tools:
        for split, records in [("train", train_records), ("val", val_records)]:
            ds_path = base_out / split / "dataset.json"
            with open(ds_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            print(f"  {split}: {len(records)} records -> {ds_path}")
    else:
        ds_path = base_out / "dataset.json"
        with open(ds_path, "w", encoding="utf-8") as f:
            json.dump(train_records, f, indent=2, ensure_ascii=False)
        print(f"  {len(train_records)} records -> {ds_path}")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 80}")
    print(
        f"Done! {len(train_records)} train + {len(val_records)} val records "
        f"from {succeeded} frames ({failed} skipped) in {elapsed:.1f}s"
    )
    print(f"{'=' * 80}")


if __name__ == "__main__":
    tyro.cli(main)
