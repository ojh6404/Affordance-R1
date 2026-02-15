"""
3D + 2D visualization of processed SceneFun3D dataset using viser + plotly.

Loads dataset.json, reconstructs per-sample point clouds from RGB + depth,
and visualizes annotations (masks, bounding boxes, motion arrows) in 3D.
Also shows annotated 2D images via plotly panel.

Usage:
    python scripts/visualize_3d.py \
        --dataset-json reasonaff_json/scenefun3d/dataset.json \
        --data-dir ~/alderamin/datasets/scenefun3d
"""

import json
import os
import sys
import time
from typing import Annotated

import cv2
import imageio.v2 as imageio
import numpy as np
import plotly.express as px
import tyro
import viser

SCENEFUN3D_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../third_party/scenefun3d"))
sys.path.insert(0, SCENEFUN3D_ROOT)
from utils.data_parser import DataParser


# Distinct colors for multiple annotations (RGB 0-255)
ANNOT_COLORS = [
    (255, 50, 50),
    (50, 255, 50),
    (50, 50, 255),
    (255, 255, 50),
    (255, 50, 255),
    (50, 255, 255),
    (255, 150, 50),
    (150, 50, 255),
]


def backproject_depth(rgb, depth, intrinsics, downsample=2):
    """Back-project RGB-D image to colored 3D point cloud in camera coordinates."""
    h, w = depth.shape[:2]
    fx, fy, cx, cy = intrinsics

    u = np.arange(0, w, downsample)
    v = np.arange(0, h, downsample)
    u, v = np.meshgrid(u, v)

    d = depth[v, u]
    r = rgb[v, u]

    valid = d > 0
    u = u[valid].astype(np.float64)
    v = v[valid].astype(np.float64)
    d = d[valid].astype(np.float64)
    colors = r[valid]

    x = (u - cx) * d / fx
    y = (v - cy) * d / fy
    z = d

    points = np.stack([x, y, z], axis=1)
    return points, colors


def create_2d_vis(rgb, mask, solution, alpha=0.5):
    """Create annotated 2D visualization image.

    Draws mask overlay, bounding boxes, points, labels, and motion arrows.
    """
    vis = rgb.copy()
    h, w = vis.shape[:2]

    # Mask overlay
    mask_bool = mask > 0
    if mask_bool.any():
        overlay = np.zeros_like(vis)
        overlay[mask_bool] = (0, 255, 0)
        vis[mask_bool] = (alpha * overlay[mask_bool] + (1 - alpha) * vis[mask_bool]).astype(np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX

    for ann_idx, item in enumerate(solution):
        color = ANNOT_COLORS[ann_idx % len(ANNOT_COLORS)]
        # BGR for cv2
        color_bgr = color[::-1]
        label = item.get("affordance", "")

        # Bounding box
        bbox = item.get("bbox_2d")
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            # Label above bbox
            (tw, th), _ = cv2.getTextSize(label, font, 0.6, 1)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 4), font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        # Center point
        pt = item.get("point_2d")
        if pt is not None:
            cv2.circle(vis, tuple(pt), 5, color, -1)
            cv2.circle(vis, tuple(pt), 5, (255, 255, 255), 1)

        # Motion arrow
        origin_2d = item.get("motion_origin_2d")
        motion_type = item.get("motion_type")
        motion_axis = item.get("motion_axis")
        if origin_2d is not None and motion_axis is not None:
            m_color = (255, 100, 0) if motion_type == "rot" else (0, 100, 255)
            # Draw arrow using stored 2D origin; approximate end from axis direction
            # We just draw from origin in direction of first two axis components (screen direction)
            ox, oy = origin_2d
            # Use a fixed pixel length for the arrow
            arrow_len = 60
            # motion_axis is in camera coords: X=right, Y=down, Z=forward
            # For 2D arrow, use X and Y components
            ax = motion_axis[0]
            ay = motion_axis[1]
            norm = np.sqrt(ax ** 2 + ay ** 2)
            if norm > 1e-6:
                dx = int(ax / norm * arrow_len)
                dy = int(ay / norm * arrow_len)
                cv2.arrowedLine(vis, (ox, oy), (ox + dx, oy + dy), m_color, 3, tipLength=0.3)
                cv2.circle(vis, (ox, oy), 6, m_color, -1)

                # Label motion type
                cv2.putText(vis, motion_type, (ox + 8, oy - 8), font, 0.5, m_color, 1, cv2.LINE_AA)

        # Trajectory (project from motion-origin local frame to 2D)
        traj = item.get("trajectory")
        origin_cam = item.get("motion_origin_cam")
        intr = item.get("_intrinsics")  # injected by caller
        if traj is not None and origin_cam is not None and intr is not None and len(traj) >= 2:
            fx_i, fy_i, cx_i, cy_i = intr
            o = np.array(origin_cam)
            m_type = item.get("motion_type")
            t_color = (255, 180, 50) if m_type == "rot" else (50, 180, 255)

            # Convert trajectory from motion-origin local frame to camera frame, then project to 2D
            pts_2d = []
            for tp in traj:
                p_cam = np.array(tp) + o  # camera coords
                if p_cam[2] > 0:
                    pu = int(round(p_cam[0] * fx_i / p_cam[2] + cx_i))
                    pv = int(round(p_cam[1] * fy_i / p_cam[2] + cy_i))
                    pts_2d.append((pu, pv))

            if len(pts_2d) >= 2:
                n = len(pts_2d)
                fade = np.array([220, 220, 220], dtype=np.float64)
                base = np.array(t_color, dtype=np.float64)
                # Draw trajectory lines with gradient
                for i in range(n - 1):
                    t = i / max(n - 1, 1)
                    c = tuple(int(v) for v in base * (1 - t) + fade * t)
                    cv2.line(vis, pts_2d[i], pts_2d[i + 1], c, 2, cv2.LINE_AA)
                # Draw points: larger and more faded further from contact
                for i, p in enumerate(pts_2d):
                    t = i / max(n - 1, 1)
                    c = tuple(int(v) for v in base * (1 - t) + fade * t)
                    radius = 4 + i
                    cv2.circle(vis, p, radius, c, -1)
                    cv2.circle(vis, p, radius, (255, 255, 255), 1)

    return vis


def main(
    dataset_json: Annotated[str, tyro.conf.arg(help="Path to processed dataset.json")],
    data_dir: Annotated[str, tyro.conf.arg(help="Path to original SceneFun3D data directory")],
    downsample: Annotated[int, tyro.conf.arg(help="Pixel stride for point cloud subsampling")] = 4,
    point_size: Annotated[float, tyro.conf.arg(help="Point size for visualization")] = 0.005,
    port: Annotated[int, tyro.conf.arg(help="Viser server port")] = 8080,
):
    # Load dataset
    with open(dataset_json, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"Loaded {len(records)} records from {dataset_json}")

    if len(records) == 0:
        print("No records found, exiting")
        return

    base_dir = os.path.dirname(dataset_json)
    data_parser = DataParser(data_dir)

    # Create viser server
    server = viser.ViserServer(port=port)
    print(f"Viser server running at http://localhost:{port}")

    # --- GUI ---
    with server.gui.add_folder("Sample"):
        sample_slider = server.gui.add_slider(
            "Index",
            min=0,
            max=len(records) - 1,
            step=1,
            initial_value=0,
        )
        sample_text = server.gui.add_text("ID", initial_value="")
        desc_text = server.gui.add_text("Description", initial_value="")

    with server.gui.add_folder("2D View"):
        show_2d = server.gui.add_checkbox("Show 2D panel", initial_value=True)

    # Plotly panel right after 2D View folder
    placeholder = np.zeros((100, 100, 3), dtype=np.uint8)
    fig = px.imshow(placeholder)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    plotly_handle = server.gui.add_plotly(figure=fig, aspect=1.0)

    with server.gui.add_folder("Display"):
        show_mask = server.gui.add_checkbox("Highlight mask", initial_value=True)
        show_motion = server.gui.add_checkbox("Show motion arrows", initial_value=True)
        show_traj = server.gui.add_checkbox("Show trajectory", initial_value=True)
        show_bbox = server.gui.add_checkbox("Show bounding boxes", initial_value=True)
        pt_size = server.gui.add_slider("Point size", min=0.001, max=0.02, step=0.001, initial_value=point_size)
        ds_slider = server.gui.add_slider("Downsample", min=1, max=8, step=1, initial_value=downsample)

    # Cache depth frames per scene
    depth_cache = {}

    def update_scene(_=None):
        idx = sample_slider.value
        record = records[idx]
        meta = record["metadata"]
        solution = record.get("solution", [])

        sample_text.value = record["id"]
        desc_text.value = record["problem"]

        # Load RGB
        img_path = os.path.join(base_dir, record["image"])
        rgb = imageio.imread(img_path)
        h_img, w_img = rgb.shape[:2]

        # Load mask
        mask_path = os.path.join(base_dir, record["mask"])
        mask = imageio.imread(mask_path)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if mask.shape[0] != h_img or mask.shape[1] != w_img:
            mask = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)

        # --- 2D plotly panel ---
        if show_2d.value:
            # Inject intrinsics for 2D trajectory projection
            for item in solution:
                item["_intrinsics"] = meta["intrinsics"]
            vis_2d = create_2d_vis(rgb, mask, solution)
            fig2d = px.imshow(vis_2d)
            fig2d.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False),
            )
            plotly_handle.figure = fig2d
            plotly_handle.visible = True
        else:
            plotly_handle.visible = False

        # --- Load depth for 3D ---
        cache_key = (meta["visit_id"], meta["video_id"])
        if cache_key not in depth_cache:
            try:
                depth_frames = data_parser.get_depth_frames(
                    meta["visit_id"], meta["video_id"], data_asset_identifier="hires_depth"
                )
                depth_cache[cache_key] = depth_frames
            except FileNotFoundError:
                depth_cache[cache_key] = {}
        depth_frames = depth_cache[cache_key]

        timestamp = meta["timestamp"]
        if timestamp not in depth_frames:
            print(f"Warning: no depth for {record['id']}")
            return

        depth = data_parser.read_depth_frame(depth_frames[timestamp])
        if depth.shape[0] != h_img or depth.shape[1] != w_img:
            depth = cv2.resize(depth, (w_img, h_img), interpolation=cv2.INTER_NEAREST)

        # Intrinsics
        fx, fy, cx, cy = meta["intrinsics"]

        # --- 3D point cloud ---
        ds = int(ds_slider.value)
        points, colors = backproject_depth(rgb, depth, (fx, fy, cx, cy), downsample=ds)

        # Highlight mask region in 3D
        if show_mask.value:
            mask_ds = mask[::ds, ::ds]
            valid_depth = depth[::ds, ::ds] > 0
            mask_flat = mask_ds[valid_depth] > 0

            for ann_idx, item in enumerate(solution):
                ann_color = np.array(ANNOT_COLORS[ann_idx % len(ANNOT_COLORS)], dtype=np.uint8)
                colors[mask_flat] = (0.6 * ann_color + 0.4 * colors[mask_flat]).astype(np.uint8)

        if len(points) == 0:
            print(f"Warning: no valid points for {record['id']}")
            return
        center = np.mean(points, axis=0)
        points_centered = points - center

        # Clear old scene
        server.scene.remove_by_name("pointcloud")
        server.scene.remove_by_name("annotations")
        server.scene.remove_by_name("motions")

        # Add point cloud
        server.scene.add_point_cloud(
            "pointcloud",
            points=points_centered.astype(np.float32),
            colors=colors,
            point_size=pt_size.value,
        )

        # Per-annotation 3D elements
        for ann_idx, item in enumerate(solution):
            color_rgb = ANNOT_COLORS[ann_idx % len(ANNOT_COLORS)]
            label = item.get("affordance", "unknown")

            # 3D bounding box
            if show_bbox.value:
                bbox = item.get("bbox_2d")
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    bbox_depth = depth[max(0, y1):min(h_img, y2), max(0, x1):min(w_img, x2)]
                    valid_d = bbox_depth[bbox_depth > 0]
                    if len(valid_d) > 0:
                        d_mean = np.median(valid_d)
                        corners_2d = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                        corners_3d = np.array([
                            [(cu - cx) * d_mean / fx, (cv - cy) * d_mean / fy, d_mean]
                            for cu, cv in corners_2d
                        ]) - center

                        edges = np.array([
                            [corners_3d[0], corners_3d[1]],
                            [corners_3d[1], corners_3d[2]],
                            [corners_3d[2], corners_3d[3]],
                            [corners_3d[3], corners_3d[0]],
                        ])
                        server.scene.add_line_segments(
                            f"annotations/bbox_{ann_idx}",
                            points=edges.astype(np.float32),
                            colors=color_rgb,
                            line_width=3,
                        )
                        bbox_center = corners_3d.mean(axis=0)
                        server.scene.add_label(
                            f"annotations/label_{ann_idx}",
                            text=label,
                            position=bbox_center.astype(np.float32),
                        )

            # 3D motion arrow
            if show_motion.value:
                motion_origin_cam = item.get("motion_origin_cam")
                motion_axis = item.get("motion_axis")
                motion_type = item.get("motion_type")

                if motion_origin_cam is not None and motion_axis is not None:
                    origin = np.array(motion_origin_cam) - center
                    direction = np.array(motion_axis)
                    arrow_len = 0.15
                    end = origin + direction * arrow_len

                    m_color = (255, 100, 0) if motion_type == "rot" else (0, 100, 255)

                    server.scene.add_line_segments(
                        f"motions/arrow_{ann_idx}",
                        points=np.array([[origin, end]]).astype(np.float32),
                        colors=m_color,
                        line_width=5,
                    )

                    if motion_type == "rot":
                        end_neg = origin - direction * arrow_len
                        server.scene.add_line_segments(
                            f"motions/arrow_neg_{ann_idx}",
                            points=np.array([[origin, end_neg]]).astype(np.float32),
                            colors=m_color,
                            line_width=3,
                        )

                    server.scene.add_point_cloud(
                        f"motions/origin_{ann_idx}",
                        points=origin.reshape(1, 3).astype(np.float32),
                        colors=np.array([m_color], dtype=np.uint8),
                        point_size=pt_size.value * 4,
                    )

            # 3D trajectory with gradient
            if show_traj.value:
                traj = item.get("trajectory")
                motion_origin_cam = item.get("motion_origin_cam")
                if traj is not None and motion_origin_cam is not None and len(traj) >= 2:
                    o_cam = np.array(motion_origin_cam)
                    traj_cam = np.array(traj) + o_cam
                    traj_centered = traj_cam - center
                    n = len(traj_centered)

                    m_type = item.get("motion_type")
                    base = np.array([255, 180, 50] if m_type == "rot" else [50, 180, 255], dtype=np.float64)
                    fade = np.array([200, 200, 200], dtype=np.float64)

                    # Per-segment gradient colors: shape (N, 2, 3) for start/end of each segment
                    seg_colors = []
                    segments = []
                    for i in range(n - 1):
                        t0 = i / max(n - 1, 1)
                        t1 = (i + 1) / max(n - 1, 1)
                        c0 = (base * (1 - t0) + fade * t0).astype(np.uint8)
                        c1 = (base * (1 - t1) + fade * t1).astype(np.uint8)
                        seg_colors.append([c0, c1])
                        segments.append([traj_centered[i], traj_centered[i + 1]])
                    segments = np.array(segments)
                    seg_colors = np.array(seg_colors)

                    server.scene.add_line_segments(
                        f"motions/traj_line_{ann_idx}",
                        points=segments.astype(np.float32),
                        colors=seg_colors,
                        line_width=4,
                    )

                    # Per-point gradient colors + decreasing size
                    pt_colors = np.array([
                        (base * (1 - i / max(n - 1, 1)) + fade * (i / max(n - 1, 1))).astype(np.uint8)
                        for i in range(n)
                    ])
                    pt_sizes = np.array([pt_size.value * (3.5 - 2.0 * i / max(n - 1, 1)) for i in range(n)])

                    # Add each point separately for different sizes
                    for i in range(n):
                        server.scene.add_point_cloud(
                            f"motions/traj_pt_{ann_idx}_{i}",
                            points=traj_centered[i:i + 1].astype(np.float32),
                            colors=pt_colors[i:i + 1],
                            point_size=float(pt_sizes[i]),
                        )

    # Register callbacks
    sample_slider.on_update(update_scene)
    show_mask.on_update(update_scene)
    show_motion.on_update(update_scene)
    show_traj.on_update(update_scene)
    show_bbox.on_update(update_scene)
    show_2d.on_update(update_scene)
    pt_size.on_update(update_scene)
    ds_slider.on_update(update_scene)

    # Initial render
    update_scene()

    print("Ready. Use the GUI to navigate samples.")
    while True:
        time.sleep(0.2)


if __name__ == "__main__":
    tyro.cli(main)
