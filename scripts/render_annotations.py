"""
Render 3D annotations onto RGB images and save frames, masks, and visualizations.

Based on visualize_cameras.py, this script projects each annotation onto RGB images
and saves the results for each camera pose where the annotation is visible.

Before running this example, download the required assets:
python -m data_downloader.data_asset_download --split custom --download_dir data --visit_id 420683 --video_id 42445137 --dataset_assets hires_wide_intrinsics laser_scan_5mm hires_poses hires_wide annotations hires_depth

SceneFun3D Toolkit
"""

from typing import Annotated
import sys
from pathlib import Path
from tqdm import tqdm
import imageio.v2 as imageio
import cv2
import os
import json
import numpy as np
import tyro

SCENEFUN3D_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../third_party/scenefun3d"))
sys.path.insert(0, SCENEFUN3D_ROOT)
from utils.data_parser import DataParser


def project_annotation_to_image(
    points_3d,
    annotation_indices,
    camera_to_world,
    intrinsics,
    img_shape,
    depth_map=None,
    visibility_threshold=0.05,
    min_visible_ratio=0.15,
    edge_margin_ratio=0.1,
    max_depth=3.0,
    mask_type="polygon",
    point_radius=5,
):
    """Project a single annotation to image

    Args:
        points_3d: N x 3 array of 3D points in world coordinates
        annotation_indices: indices of points belonging to this annotation
        camera_to_world: 4x4 camera-to-world transformation matrix
        intrinsics: (fx, fy, cx, cy) tuple
        img_shape: (h, w, c) image shape
        depth_map: H x W depth map for occlusion checking (optional)
        visibility_threshold: threshold for occlusion checking (relative to depth)
        min_visible_ratio: minimum fraction of annotation points that must be visible (0.0-1.0)
        edge_margin_ratio: reject if mask centroid is within this fraction of image edge (0.0-0.5)
        max_depth: maximum distance (meters) from camera to annotation centroid
        mask_type: 'points' or 'polygon' - whether to draw points or convex hull polygon
        point_radius: radius of circles when mask_type='points'

    Returns:
        binary_mask: H x W binary mask (0 or 255)
        is_visible: boolean indicating if annotation is visible in this frame
        visible_ratio: fraction of annotation points visible (0.0-1.0)
    """
    h, w = img_shape[:2]
    binary_mask = np.zeros((h, w), dtype=np.uint8)

    masked_points = points_3d[annotation_indices]
    fx, fy, cx, cy = intrinsics

    # World to camera transform
    world_to_camera = np.linalg.inv(camera_to_world)

    # Convert to homogeneous coordinates
    points_homo = np.concatenate([masked_points, np.ones([masked_points.shape[0], 1])], axis=1).T

    # Transform to camera frame
    points_cam = world_to_camera @ points_homo

    # Filter points behind camera
    valid_depth = points_cam[2] > 0
    points_cam = points_cam[:, valid_depth]

    if points_cam.shape[1] == 0:
        return binary_mask, False, 0.0

    # Reject if annotation is too far from camera
    if max_depth > 0:
        median_depth = np.median(points_cam[2])
        if median_depth > max_depth:
            return binary_mask, False, 0.0

    # Project to image using intrinsics
    u = (points_cam[0] * fx) / points_cam[2] + cx
    v = (points_cam[1] * fy) / points_cam[2] + cy
    depths = points_cam[2]

    # Round to integer pixel coordinates
    u = np.round(u).astype(int)
    v = np.round(v).astype(int)

    # Filter points outside image
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)

    # Occlusion checking with depth map
    if depth_map is not None:
        # Get valid pixel coordinates and their depths
        u_valid = u[valid]
        v_valid = v[valid]
        depths_valid = depths[valid]

        # Get depth from depth map at these pixel locations
        depth_at_pixel = depth_map[v_valid, u_valid]

        # Point is visible if:
        # 1. depth_at_pixel > 0 (valid depth measurement)
        # 2. point depth is less than or close to depth map depth
        valid_depth_measurement = depth_at_pixel > 0
        is_visible = depths_valid <= depth_at_pixel + visibility_threshold * depth_at_pixel
        occlusion_mask = valid_depth_measurement & is_visible

        # Update valid mask
        temp_valid = np.zeros(len(valid), dtype=bool)
        temp_valid[valid] = occlusion_mask
        valid = temp_valid

    u, v = u[valid], v[valid]

    if len(u) == 0:
        return binary_mask, False, 0.0

    # Check minimum visibility ratio: reject frames where annotation is mostly occluded
    total_annotation_points = len(annotation_indices)
    visible_ratio = len(u) / total_annotation_points if total_annotation_points > 0 else 0
    if visible_ratio < min_visible_ratio:
        return binary_mask, False, 0.0

    # Reject if mask centroid is too close to image edge
    if edge_margin_ratio > 0:
        centroid_u = np.mean(u)
        centroid_v = np.mean(v)
        margin_x = w * edge_margin_ratio
        margin_y = h * edge_margin_ratio
        if centroid_u < margin_x or centroid_u > w - margin_x or centroid_v < margin_y or centroid_v > h - margin_y:
            return binary_mask, False, 0.0

    # Draw mask based on type
    if mask_type == "polygon":
        # Create convex hull of projected points
        points_2d = np.stack([u, v], axis=1)

        # Need at least 3 points for a polygon
        if len(points_2d) >= 3:
            hull = cv2.convexHull(points_2d.astype(np.int32))
            cv2.fillPoly(binary_mask, [hull], 255)
        else:
            # Fall back to points if too few points
            for ui, vi in zip(u, v):
                cv2.circle(binary_mask, (ui, vi), point_radius, 255, -1)
    else:  # mask_type == 'points'
        # Draw circles around each point
        for ui, vi in zip(u, v):
            cv2.circle(binary_mask, (ui, vi), point_radius, 255, -1)

    return binary_mask, True, visible_ratio


def create_visualization(
    rgb_image, mask, color=(0, 255, 0), alpha=0.5, label=None, descriptions=None, motions=None
):
    """Create visualization by overlaying mask on RGB image

    Args:
        rgb_image: H x W x 3 RGB image
        mask: H x W binary mask (0 or 255)
        color: RGB tuple for mask color
        alpha: opacity for blending
        label: annotation label (e.g., "door_handle")
        descriptions: list of description strings
        motions: list of motion types (e.g., "trans", "rot")

    Returns:
        vis_image: H x W x 3 visualization image
    """
    vis_image = rgb_image.copy()

    # Create colored overlay
    mask_bool = mask > 0
    overlay = np.zeros_like(rgb_image)
    overlay[mask_bool] = color

    # Blend with original image
    vis_image[mask_bool] = (alpha * overlay[mask_bool] + (1 - alpha) * rgb_image[mask_bool]).astype(np.uint8)

    # Add text annotations at top-left corner
    if label or descriptions or motions:
        # Position text at top-left corner
        text_x = 10
        text_y = 25

        # Build text lines
        text_lines = []
        if label:
            text_lines.append(f"Label: {label}")
        if motions:
            motion_str = ", ".join(motions)
            text_lines.append(f"Motion: {motion_str}")
        if descriptions:
            for i, desc in enumerate(descriptions):
                # Truncate long descriptions
                if len(desc) > 60:
                    desc = desc[:57] + "..."
                text_lines.append(f"Desc: {desc}")

        # Draw text with background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        line_height = 25

        for i, text_line in enumerate(text_lines):
            y_pos = text_y + i * line_height

            # Get text size for background
            (text_w, text_h), baseline = cv2.getTextSize(text_line, font, font_scale, thickness)

            # Draw semi-transparent background
            bg_y1 = y_pos - text_h - 5
            bg_y2 = y_pos + baseline + 5
            bg_x1 = text_x - 5
            bg_x2 = text_x + text_w + 5

            overlay_bg = vis_image[bg_y1:bg_y2, bg_x1:bg_x2].copy()
            cv2.rectangle(vis_image, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            vis_image[bg_y1:bg_y2, bg_x1:bg_x2] = cv2.addWeighted(
                overlay_bg, 0.3, vis_image[bg_y1:bg_y2, bg_x1:bg_x2], 0.7, 0
            )

            # Draw text
            cv2.putText(
                vis_image,
                text_line,
                (text_x, y_pos),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

    return vis_image


def main(
    data_dir: Annotated[str, tyro.conf.arg(help="Path to the dataset")],
    output_dir: Annotated[str, tyro.conf.arg(help="Output directory for rendered results")],
    visit_id: Annotated[str, tyro.conf.arg(help="Visit identifier")] = "420683",
    video_id: Annotated[str, tyro.conf.arg(help="Video sequence identifier")] = "42445137",
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
        int | None, tyro.conf.arg(help="Max viewpoints per description. None=all valid frames")
    ] = None,
    min_view_gap: Annotated[int, tyro.conf.arg(help="Minimum frame index gap between selected viewpoints")] = 5,
):
    # Read required data assets
    dataParser = DataParser(data_dir)

    # Load laser scan
    print("Loading laser scan...")
    try:
        laser_point_cloud = dataParser.get_laser_scan(visit_id)
        points_original = np.array(laser_point_cloud.points)
    except FileNotFoundError:
        print(f"Error: Laser scan not found for visit_id {visit_id}, skipping scene")
        return

    # Load crop mask to filter out noise points (glass, reflective surfaces)
    print("Loading crop mask...")
    crop_mask_path = Path(data_dir) / visit_id / f"{visit_id}_crop_mask.npy"
    if crop_mask_path.exists():
        crop_mask = np.load(crop_mask_path)
        print(f"Crop mask: {crop_mask.sum()}/{len(crop_mask)} valid points ({crop_mask.sum()/len(crop_mask)*100:.1f}%)")
    else:
        crop_mask = None
        print("Warning: Crop mask not found, using all points")

    # Load annotations
    print("Loading annotations...")
    try:
        annotations = dataParser.get_annotations(visit_id)
        print(f"Loaded {len(annotations)} annotations")
    except FileNotFoundError:
        print(f"Error: Annotations not found for visit_id {visit_id}, skipping scene")
        return

    # Load descriptions and motions
    print("Loading descriptions...")
    try:
        descriptions = dataParser.get_descriptions(visit_id)
    except FileNotFoundError:
        print(f"Error: Descriptions not found for visit_id {visit_id}, skipping scene")
        return

    print("Loading motions...")
    try:
        motions = dataParser.get_motions(visit_id)
    except FileNotFoundError:
        print(f"Warning: Motions not found for visit_id {visit_id}, continuing without motions")
        motions = []

    # Create lookup dictionaries
    # annotation_map: annot_id -> annotation data
    annotation_map = {annot["annot_id"]: annot for annot in annotations}

    # motion_map: annot_id -> full motion data
    motion_map = {}
    for motion in motions:
        annot_id = motion.get("annot_id")
        if annot_id is not None:
            motion_map[annot_id] = motion

    # Load camera poses and RGB frames
    print("Loading camera poses...")
    try:
        poses = dataParser.get_camera_trajectory(visit_id, video_id, pose_source="colmap")
    except FileNotFoundError:
        print(f"Error: Camera poses not found for {visit_id}/{video_id}, skipping scene")
        return

    print("Loading RGB frames...")
    try:
        rgb_frames = dataParser.get_rgb_frames(visit_id, video_id, data_asset_identifier="hires_wide")
    except FileNotFoundError:
        print(f"Error: RGB frames not found for {visit_id}/{video_id}, skipping scene")
        return

    print("Loading camera intrinsics...")
    try:
        intrinsics = dataParser.get_camera_intrinsics(
            visit_id, video_id, data_asset_identifier="hires_wide_intrinsics"
        )
    except FileNotFoundError:
        print(f"Error: Camera intrinsics not found for {visit_id}/{video_id}, skipping scene")
        return

    print("Loading depth frames...")
    try:
        depth_frames = dataParser.get_depth_frames(visit_id, video_id, data_asset_identifier="hires_depth")
    except FileNotFoundError:
        print(f"Warning: Depth frames not found for {visit_id}/{video_id}, skipping scene")
        return

    # Match frames with poses
    matched_data = []
    timestamps = sorted(rgb_frames.keys())
    for i, ts in enumerate(timestamps):
        nearest_pose = dataParser.get_nearest_pose(ts, poses, time_distance_threshold=0.1)
        if nearest_pose is not None and ts in intrinsics and ts in depth_frames:
            matched_data.append(
                {
                    "timestamp": ts,
                    "pose": nearest_pose,
                    "rgb_path": rgb_frames[ts],
                    "intrinsics_path": intrinsics[ts],
                    "depth_path": depth_frames[ts],
                    "frame_index": len(matched_data),
                }
            )

    print(f"Matched {len(matched_data)} frames with poses")

    def get_frame_intrinsics(data):
        """Load and scale intrinsics for a frame."""
        with open(data["intrinsics_path"], "r") as f:
            parts = f.readline().strip().split()
            w_i, h_i = int(parts[0]), int(parts[1])
            fx, fy = float(parts[2]), float(parts[3])
            cx, cy = float(parts[4]), float(parts[5])
        rgb_img = imageio.imread(data["rgb_path"])
        if rgb_img.shape[1] != w_i or rgb_img.shape[0] != h_i:
            sx, sy = rgb_img.shape[1] / w_i, rgb_img.shape[0] / h_i
            return rgb_img, (fx * sx, fy * sy, cx * sx, cy * sy)
        return rgb_img, (fx, fy, cx, cy)

    def project_point_3d_to_2d(point_3d, pose, intr, img_shape):
        """Project a single 3D point to 2D pixel coordinates."""
        world_to_cam = np.linalg.inv(pose)
        p_cam = world_to_cam @ np.append(point_3d, 1)
        if p_cam[2] <= 0:
            return None
        fx, fy, cx, cy = intr
        u = int(round(p_cam[0] * fx / p_cam[2] + cx))
        v = int(round(p_cam[1] * fy / p_cam[2] + cy))
        h, w = img_shape[:2]
        if 0 <= u < w and 0 <= v < h:
            return [u, v]
        return None

    # Collect dataset records
    records = []

    # Create output directories
    out_path = Path(output_dir)
    images_dir = out_path / "images"
    masks_dir = out_path / "masks"
    vis_dir = out_path / "vis"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Process each description (one record per description, all annotations together)
    for desc_idx, desc in enumerate(descriptions):
        desc_id = desc["desc_id"]
        description_text = desc["description"]
        annot_ids = desc.get("annot_id", [])

        if not isinstance(annot_ids, list):
            annot_ids = [annot_ids]

        # Filter valid annotations
        valid_annots = []
        for annot_id in annot_ids:
            annot = annotation_map.get(annot_id)
            if annot is None or annot["label"] == "exclude":
                continue
            indices = annot["indices"]
            if crop_mask is not None:
                indices = [i for i in indices if crop_mask[i]]
                if len(indices) == 0:
                    continue
            valid_annots.append({
                "annot_id": annot_id,
                "label": annot["label"].replace("_", " "),
                "indices": indices,
                "motion": motion_map.get(annot_id),
            })

        if len(valid_annots) == 0:
            continue

        print(f"\nProcessing description {desc_idx}/{len(descriptions)}: {description_text[:50]}...")
        print(f"  Annotations: {[a['label'] for a in valid_annots]}")

        # Find all frames where ALL annotations are visible, scored by min visibility
        candidates = []  # list of (score, frame_index, data, intr, frame_results)

        for data in tqdm(matched_data, desc="    Scanning frames", leave=False):
            rgb_img, intr = get_frame_intrinsics(data)
            original_shape = rgb_img.shape

            depth_map = dataParser.read_depth_frame(data["depth_path"])
            if depth_map.shape[0] != original_shape[0] or depth_map.shape[1] != original_shape[1]:
                depth_map = cv2.resize(
                    depth_map, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST
                )

            # Project all annotations in this frame
            frame_results = []
            all_visible = True
            for va in valid_annots:
                mask, is_visible, vis_ratio = project_annotation_to_image(
                    points_original, va["indices"], data["pose"], intr, original_shape,
                    depth_map=depth_map, visibility_threshold=visibility_threshold,
                    edge_margin_ratio=edge_margin_ratio, max_depth=max_depth,
                )
                if not is_visible:
                    all_visible = False
                    break
                frame_results.append((mask, vis_ratio))

            if not all_visible:
                continue

            score = min(r[1] for r in frame_results)
            candidates.append((score, data["frame_index"], data, intr, frame_results))

        if len(candidates) == 0:
            print(f"  No frame found where all annotations are visible, skipping")
            continue

        # Sort by score descending, then select diverse views with min_view_gap
        frame_indices = sorted(c[1] for c in candidates)
        print(f"  Valid frame indices: {frame_indices[0]}..{frame_indices[-1]} (span={frame_indices[-1]-frame_indices[0]})")
        candidates.sort(key=lambda x: x[0], reverse=True)
        selected = []
        for cand in candidates:
            _, fidx, _, _, _ = cand
            if all(abs(fidx - s[1]) >= min_view_gap for s in selected):
                selected.append(cand)
                if max_views is not None and len(selected) >= max_views:
                    break

        print(f"  Selected {len(selected)} viewpoints from {len(candidates)} valid frames")

        # Save each selected viewpoint
        for view_idx, (score, fidx, sel_data, sel_intr, sel_results) in enumerate(selected):
            rgb_img = imageio.imread(sel_data["rgb_path"])
            h_img, w_img = rgb_img.shape[:2]

            # World-to-camera rotation (for converting motion direction)
            world_to_cam = np.linalg.inv(sel_data["pose"])
            R_w2c = world_to_cam[:3, :3]

            # Build solution items and combined mask
            solution = []
            combined_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            motion_arrows = []  # for visualization: (origin_2d, direction_2d, motion_type)

            for va, (mask, vis_ratio) in zip(valid_annots, sel_results):
                combined_mask = np.maximum(combined_mask, mask)

                # Extract bbox and point from mask
                ys, xs = np.where(mask > 0)
                bbox_2d = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                point_2d = [int(np.mean(xs)), int(np.mean(ys))]

                item = {
                    "bbox_2d": bbox_2d,
                    "point_2d": point_2d,
                    "affordance": va["label"],
                }

                # Add motion (convert to camera coordinate system, OpenCV convention)
                md = va["motion"]
                if md is not None:
                    item["motion_type"] = md["motion_type"]

                    # Convert motion axis: world -> camera (rotation only)
                    motion_dir_world = np.array(md["motion_dir"], dtype=np.float64)
                    motion_dir_cam = R_w2c @ motion_dir_world
                    # Normalize to unit vector
                    norm = np.linalg.norm(motion_dir_cam)
                    if norm > 1e-6:
                        motion_dir_cam = motion_dir_cam / norm
                    item["motion_axis"] = [round(float(x), 4) for x in motion_dir_cam]

                    # Project motion origin to 2D and convert origin to camera coords
                    origin_idx = md.get("motion_origin_idx")
                    origin_2d = None
                    if origin_idx is not None and origin_idx < len(points_original):
                        origin_3d_world = points_original[origin_idx]
                        origin_2d = project_point_3d_to_2d(
                            origin_3d_world, sel_data["pose"], sel_intr, rgb_img.shape
                        )
                        # Also store origin in camera coordinates
                        origin_cam = world_to_cam @ np.append(origin_3d_world, 1.0)
                        item["motion_origin_cam"] = [round(float(x), 4) for x in origin_cam[:3]]

                    if origin_2d is not None:
                        item["motion_origin_2d"] = origin_2d

                        # Compute 2D arrow for visualization using camera-space direction
                        origin_3d_world = points_original[origin_idx]
                        axis_end_3d = origin_3d_world + np.array(md["motion_dir"]) * 0.15
                        axis_end_2d = project_point_3d_to_2d(
                            axis_end_3d, sel_data["pose"], sel_intr, rgb_img.shape
                        )
                        if axis_end_2d is not None:
                            motion_arrows.append((origin_2d, axis_end_2d, md["motion_type"]))

                    # Generate 5-point trajectory in motion-origin local frame
                    # (origin = motion_origin, rotation = camera rotation)
                    if "motion_origin_cam" in item:
                        o_cam = np.array(item["motion_origin_cam"])
                        axis = motion_dir_cam  # unit vector, camera coords

                        if md["motion_type"] == "rot":
                            # Rotation: arc of annotation centroid around axis
                            # Compute centroid in camera coords
                            annot_pts_world = points_original[va["indices"]]
                            centroid_world = annot_pts_world.mean(axis=0)
                            centroid_cam = (world_to_cam @ np.append(centroid_world, 1.0))[:3]
                            p_rel = centroid_cam - o_cam  # relative to origin

                            # 5 points along 45-degree arc using Rodrigues' formula
                            max_angle = np.pi / 4
                            traj = []
                            for ti in range(5):
                                theta = ti * max_angle / 4
                                p_rot = (
                                    p_rel * np.cos(theta)
                                    + np.cross(axis, p_rel) * np.sin(theta)
                                    + axis * np.dot(axis, p_rel) * (1 - np.cos(theta))
                                )
                                traj.append([round(float(c), 4) for c in p_rot])
                            item["trajectory"] = traj

                        elif md["motion_type"] == "trans":
                            # Translation: straight line from origin along axis
                            max_dist = 0.1  # 10cm
                            traj = []
                            for ti in range(5):
                                t = ti * max_dist / 4
                                point = axis * t
                                traj.append([round(float(c), 4) for c in point])
                            item["trajectory"] = traj

                solution.append(item)

            # Save image and mask
            sample_id = f"{visit_id}_{video_id}_{desc_id[:8]}_v{view_idx}"
            img_filename = f"{sample_id}.jpg"
            mask_filename = f"{sample_id}.png"

            imageio.imwrite(images_dir / img_filename, rgb_img)
            imageio.imwrite(masks_dir / mask_filename, combined_mask)

            # Save visualization with motion arrows
            vis_img = create_visualization(
                rgb_img, combined_mask, color=(0, 255, 0), alpha=mask_alpha,
                label=", ".join(a["label"] for a in valid_annots),
                descriptions=[description_text],
                motions=[a["motion"]["motion_type"] for a in valid_annots if a["motion"]],
            )
            for arrow_origin, arrow_end, m_type in motion_arrows:
                color = (255, 100, 0) if m_type == "rot" else (0, 100, 255)
                cv2.arrowedLine(vis_img, tuple(arrow_origin), tuple(arrow_end), color, 3, tipLength=0.3)
                cv2.circle(vis_img, tuple(arrow_origin), 6, color, -1)
            imageio.imwrite(vis_dir / img_filename, vis_img)

            # Build record
            aff_names = list(set(a["label"] for a in valid_annots))
            record = {
                "id": sample_id,
                "problem": description_text,
                "solution": solution,
                "image": f"images/{img_filename}",
                "mask": f"masks/{mask_filename}",
                "img_height": h_img,
                "img_width": w_img,
                "aff_name": aff_names[0] if len(aff_names) == 1 else aff_names,
                "part_name": None,
                "metadata": {
                    "dataset": "SceneFun3D",
                    "visit_id": visit_id,
                    "video_id": video_id,
                    "desc_id": desc_id,
                    "annot_ids": [a["annot_id"] for a in valid_annots],
                    "frame_index": fidx,
                    "view_index": view_idx,
                    "timestamp": sel_data["timestamp"],
                    "intrinsics": list(sel_intr),
                    "visibility_ratio": round(score, 3),
                },
            }
            records.append(record)

        labels_str = ", ".join(a["label"] for a in valid_annots)
        print(f"  Saved {len(selected)} views: {labels_str}")

    # Write dataset.json
    dataset_path = out_path / "dataset.json"

    # Append to existing dataset.json if it exists (for multi-scene processing)
    if dataset_path.exists():
        with open(dataset_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        records = existing + records

    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"\nDone! {len(records)} total records in {dataset_path}")


if __name__ == "__main__":
    tyro.cli(main)
