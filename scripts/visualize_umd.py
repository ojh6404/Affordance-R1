"""
3D + 2D visualization of processed UMD Part-Affordance dataset using viser + plotly.

Loads dataset.json, reconstructs per-sample point clouds from RGB + depth,
and visualizes affordance annotations (masks, bounding boxes, motion arrows,
pick-up trajectories, ground plane) in 3D + 2D.

Usage:
    python scripts/visualize_umd.py \
        --dataset-json affordance_dataset/umd/dataset.json

    python scripts/visualize_umd.py \
        --dataset-json affordance_dataset/umd/train/dataset.json \
        --filter-affordance grasp
"""

import json
import os
import time
from typing import Annotated, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import plotly.express as px
import tyro
import viser


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

# Affordance-specific colors
AFFORDANCE_COLORS = {
    "grasp": (50, 255, 50),
    "cut": (255, 50, 50),
    "scoop": (50, 50, 255),
    "contain": (255, 255, 50),
    "pound": (255, 50, 255),
    "support": (50, 255, 255),
}


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


def create_2d_vis(rgb, mask, solution, meta, alpha=0.5):
    """Create annotated 2D visualization image with UMD metadata overlay.

    Draws mask overlay, bounding boxes, points, labels, motion arrows, and trajectories.
    """
    vis = rgb.copy()

    # Mask overlay
    mask_bool = mask > 0
    if mask_bool.any():
        overlay = np.zeros_like(vis)
        overlay[mask_bool] = (0, 255, 0)
        vis[mask_bool] = (alpha * overlay[mask_bool] + (1 - alpha) * vis[mask_bool]).astype(np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    intr = meta.get("intrinsics")

    for ann_idx, item in enumerate(solution):
        aff_name = item.get("affordance", "")
        color = AFFORDANCE_COLORS.get(aff_name, ANNOT_COLORS[ann_idx % len(ANNOT_COLORS)])
        label = aff_name

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
        motion_axis_2d = item.get("motion_axis_2d")
        motion_axis = item.get("motion_axis")
        if origin_2d is not None and (motion_axis_2d is not None or motion_axis is not None):
            m_color = (255, 100, 0) if motion_type == "rot" else (0, 100, 255)
            ox, oy = origin_2d
            arrow_len = 60
            if motion_axis_2d is not None:
                ax, ay = motion_axis_2d
            else:
                ax, ay = motion_axis[0], motion_axis[1]
                norm = np.sqrt(ax**2 + ay**2)
                if norm > 1e-6:
                    ax, ay = ax / norm, ay / norm
                else:
                    ax, ay = 0, 0
            dx = int(ax * arrow_len)
            dy = int(ay * arrow_len)
            if abs(dx) + abs(dy) > 0:
                cv2.arrowedLine(vis, (ox, oy), (ox + dx, oy + dy), m_color, 3, tipLength=0.3)
                cv2.circle(vis, (ox, oy), 6, m_color, -1)
                cv2.putText(vis, motion_type, (ox + 8, oy - 8), font, 0.5, m_color, 1, cv2.LINE_AA)

        # Trajectory
        traj = item.get("trajectory")
        origin_cam = item.get("motion_origin_cam")
        if traj is not None and origin_cam is not None and intr is not None and len(traj) >= 2:
            fx_i, fy_i, cx_i, cy_i = intr
            o = np.array(origin_cam)
            t_color = (50, 180, 255)

            pts_2d = []
            for tp in traj:
                p_cam = np.array(tp) + o
                if p_cam[2] > 0:
                    pu = int(round(p_cam[0] * fx_i / p_cam[2] + cx_i))
                    pv = int(round(p_cam[1] * fy_i / p_cam[2] + cy_i))
                    pts_2d.append((pu, pv))

            if len(pts_2d) >= 2:
                n = len(pts_2d)
                fade = np.array([220, 220, 220], dtype=np.float64)
                base = np.array(t_color, dtype=np.float64)
                for i in range(n - 1):
                    t = i / max(n - 1, 1)
                    c = tuple(int(v) for v in base * (1 - t) + fade * t)
                    cv2.line(vis, pts_2d[i], pts_2d[i + 1], c, 2, cv2.LINE_AA)
                for i, p in enumerate(pts_2d):
                    t = i / max(n - 1, 1)
                    c = tuple(int(v) for v in base * (1 - t) + fade * t)
                    radius = 4 + i
                    cv2.circle(vis, p, radius, c, -1)
                    cv2.circle(vis, p, radius, (255, 255, 255), 1)

    # Metadata overlay at bottom
    tool_name = meta.get("tool_name", "")
    plane_normal = meta.get("plane_normal")
    info_lines = [f"Tool: {tool_name}"]
    if plane_normal is not None:
        info_lines.append(f"Plane normal: ({plane_normal[0]:.3f}, {plane_normal[1]:.3f}, {plane_normal[2]:.3f})")
    h_vis = vis.shape[0]
    for i, line in enumerate(info_lines):
        y_pos = h_vis - 10 - (len(info_lines) - 1 - i) * 20
        (tw, th), _ = cv2.getTextSize(line, font, 0.45, 1)
        cv2.rectangle(vis, (5, y_pos - th - 3), (10 + tw, y_pos + 3), (0, 0, 0), -1)
        cv2.putText(vis, line, (7, y_pos), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return vis


def main(
    dataset_json: Annotated[str, tyro.conf.arg(help="Path to processed UMD dataset.json")],
    downsample: Annotated[int, tyro.conf.arg(help="Pixel stride for point cloud subsampling")] = 4,
    point_size: Annotated[float, tyro.conf.arg(help="Point size for visualization")] = 0.005,
    port: Annotated[int, tyro.conf.arg(help="Viser server port")] = 8080,
    filter_affordance: Annotated[
        Optional[str], tyro.conf.arg(help="Show only records with this affordance (e.g. grasp, cut)")
    ] = None,
    filter_tool: Annotated[
        Optional[str], tyro.conf.arg(help="Show only records for this tool prefix (e.g. knife, hammer)")
    ] = None,
):
    # Load dataset
    with open(dataset_json, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"Loaded {len(records)} records from {dataset_json}")

    # Apply filters
    if filter_affordance:
        records = [r for r in records if r.get("aff_name") == filter_affordance]
        print(f"Filtered to {len(records)} records with affordance '{filter_affordance}'")
    if filter_tool:
        records = [r for r in records if r.get("metadata", {}).get("tool_name", "").startswith(filter_tool)]
        print(f"Filtered to {len(records)} records with tool prefix '{filter_tool}'")

    if len(records) == 0:
        print("No records found, exiting")
        return

    # Count affordance distribution
    aff_counts = {}
    for r in records:
        aff = r.get("aff_name", "unknown")
        aff_counts[aff] = aff_counts.get(aff, 0) + 1
    print(f"Affordance distribution: {aff_counts}")

    base_dir = os.path.dirname(dataset_json)

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
        tool_text = server.gui.add_text("Tool", initial_value="")
        aff_text = server.gui.add_text("Affordance", initial_value="")

    with server.gui.add_folder("2D View"):
        show_2d = server.gui.add_checkbox("Show 2D panel", initial_value=True)

    # Plotly panel
    placeholder = np.zeros((100, 100, 3), dtype=np.uint8)
    fig = px.imshow(placeholder)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    plotly_handle = server.gui.add_plotly(figure=fig, aspect=1.0)

    with server.gui.add_folder("Display"):
        show_mask = server.gui.add_checkbox("Highlight mask", initial_value=True)
        show_motion = server.gui.add_checkbox("Show motion arrows", initial_value=True)
        show_traj = server.gui.add_checkbox("Show trajectory", initial_value=True)
        show_bbox = server.gui.add_checkbox("Show bounding boxes", initial_value=True)
        show_plane = server.gui.add_checkbox("Show ground plane", initial_value=True)
        show_frustum = server.gui.add_checkbox("Show camera frustum", initial_value=True)
        frustum_scale = server.gui.add_slider("Frustum scale", min=0.01, max=0.5, step=0.01, initial_value=0.1)
        pt_size = server.gui.add_slider("Point size", min=0.001, max=0.02, step=0.001, initial_value=point_size)
        ds_slider = server.gui.add_slider("Downsample", min=1, max=8, step=1, initial_value=downsample)

    def update_scene(_=None):
        idx = sample_slider.value
        record = records[idx]
        meta = record["metadata"]
        solution = record.get("solution", [])

        sample_text.value = record["id"]
        desc_text.value = record["problem"]
        tool_text.value = meta.get("tool_name", "")
        aff_text.value = record.get("aff_name", "")

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
            vis_2d = create_2d_vis(rgb, mask, solution, meta)
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

        # --- Load depth ---
        depth = None
        if "depth" in record:
            depth_path = os.path.join(base_dir, record["depth"])
            if os.path.exists(depth_path):
                depth_mm = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                if depth_mm is not None:
                    depth = depth_mm.astype(np.float64) / 1000.0  # mm -> meters
        if depth is None:
            print(f"Warning: no depth for {record['id']}")
            return
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
                aff_name = item.get("affordance", "")
                ann_color = np.array(
                    AFFORDANCE_COLORS.get(aff_name, ANNOT_COLORS[ann_idx % len(ANNOT_COLORS)]),
                    dtype=np.uint8,
                )
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
        server.scene.remove_by_name("camera")
        server.scene.remove_by_name("plane")

        # Add point cloud
        server.scene.add_point_cloud(
            "pointcloud",
            points=points_centered.astype(np.float32),
            colors=colors,
            point_size=pt_size.value,
        )

        # --- Ground plane visualization ---
        if show_plane.value:
            plane_normal = meta.get("plane_normal")
            plane_d = meta.get("plane_distance")
            if plane_normal is not None and plane_d is not None:
                normal = np.array(plane_normal)
                # Draw plane as a grid of lines at the detected ground level
                # Find a point on the plane near the scene center
                plane_center = center - normal * (np.dot(normal, center) + plane_d)
                plane_center_c = plane_center - center

                # Create two tangent vectors for the plane
                if abs(normal[0]) < 0.9:
                    tangent1 = np.cross(normal, np.array([1, 0, 0]))
                else:
                    tangent1 = np.cross(normal, np.array([0, 1, 0]))
                tangent1 = tangent1 / np.linalg.norm(tangent1)
                tangent2 = np.cross(normal, tangent1)

                # Draw a small grid (0.2m x 0.2m)
                grid_size = 0.15
                grid_lines = []
                n_lines = 5
                for i in range(n_lines + 1):
                    t = -grid_size + 2 * grid_size * i / n_lines
                    p1 = plane_center_c + tangent1 * t - tangent2 * grid_size
                    p2 = plane_center_c + tangent1 * t + tangent2 * grid_size
                    grid_lines.append([p1, p2])
                    p3 = plane_center_c - tangent1 * grid_size + tangent2 * t
                    p4 = plane_center_c + tangent1 * grid_size + tangent2 * t
                    grid_lines.append([p3, p4])

                grid_lines = np.array(grid_lines)
                server.scene.add_line_segments(
                    "plane/grid",
                    points=grid_lines.astype(np.float32),
                    colors=(150, 150, 150),
                    line_width=1,
                )

                # Draw plane normal arrow from plane center
                normal_end = plane_center_c + normal * 0.1
                server.scene.add_line_segments(
                    "plane/normal",
                    points=np.array([[plane_center_c, normal_end]]).astype(np.float32),
                    colors=(255, 200, 0),
                    line_width=4,
                )

        # Per-annotation 3D elements
        for ann_idx, item in enumerate(solution):
            aff_name = item.get("affordance", "unknown")
            color_rgb = AFFORDANCE_COLORS.get(aff_name, ANNOT_COLORS[ann_idx % len(ANNOT_COLORS)])

            # 3D bounding box
            if show_bbox.value:
                bbox = item.get("bbox_2d")
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    bbox_depth = depth[max(0, y1) : min(h_img, y2), max(0, x1) : min(w_img, x2)]
                    valid_d = bbox_depth[bbox_depth > 0]
                    if len(valid_d) > 0:
                        d_mean = np.median(valid_d)
                        corners_2d = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                        corners_3d = (
                            np.array(
                                [[(cu - cx) * d_mean / fx, (cv - cy) * d_mean / fy, d_mean] for cu, cv in corners_2d]
                            )
                            - center
                        )

                        edges = np.array(
                            [
                                [corners_3d[0], corners_3d[1]],
                                [corners_3d[1], corners_3d[2]],
                                [corners_3d[2], corners_3d[3]],
                                [corners_3d[3], corners_3d[0]],
                            ]
                        )
                        server.scene.add_line_segments(
                            f"annotations/bbox_{ann_idx}",
                            points=edges.astype(np.float32),
                            colors=color_rgb,
                            line_width=3,
                        )
                        bbox_center = corners_3d.mean(axis=0)
                        server.scene.add_label(
                            f"annotations/label_{ann_idx}",
                            text=aff_name,
                            position=bbox_center.astype(np.float32),
                        )

            # 3D motion arrow
            if show_motion.value:
                motion_origin_cam = item.get("motion_origin_cam")
                motion_axis = item.get("motion_axis")

                if motion_origin_cam is not None and motion_axis is not None:
                    origin = np.array(motion_origin_cam) - center
                    direction = np.array(motion_axis)
                    arrow_len = 0.15
                    end = origin + direction * arrow_len

                    m_color = (0, 100, 255)  # UMD: always trans (pick-up)

                    server.scene.add_line_segments(
                        f"motions/arrow_{ann_idx}",
                        points=np.array([[origin, end]]).astype(np.float32),
                        colors=m_color,
                        line_width=5,
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

                    base_c = np.array([50, 180, 255], dtype=np.float64)
                    fade = np.array([200, 200, 200], dtype=np.float64)

                    # Per-segment gradient colors
                    seg_colors = []
                    segments = []
                    for i in range(n - 1):
                        t0 = i / max(n - 1, 1)
                        t1 = (i + 1) / max(n - 1, 1)
                        c0 = (base_c * (1 - t0) + fade * t0).astype(np.uint8)
                        c1 = (base_c * (1 - t1) + fade * t1).astype(np.uint8)
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
                    pt_colors = np.array(
                        [
                            (base_c * (1 - i / max(n - 1, 1)) + fade * (i / max(n - 1, 1))).astype(np.uint8)
                            for i in range(n)
                        ]
                    )
                    pt_sizes = np.array([pt_size.value * (3.5 - 2.0 * i / max(n - 1, 1)) for i in range(n)])

                    for i in range(n):
                        server.scene.add_point_cloud(
                            f"motions/traj_pt_{ann_idx}_{i}",
                            points=traj_centered[i : i + 1].astype(np.float32),
                            colors=pt_colors[i : i + 1],
                            point_size=float(pt_sizes[i]),
                            point_shape="rounded",
                        )

        # --- Camera frustum ---
        if show_frustum.value:
            cam_position = (-center).astype(np.float32)
            server.scene.add_frame(
                "camera/frame",
                position=cam_position,
                wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                show_axes=True,
                axes_length=0.1,
                axes_radius=0.005,
                origin_radius=0.008,
            )

            fov_y = 2.0 * np.arctan(h_img / (2.0 * fy))
            aspect = w_img / h_img
            server.scene.add_camera_frustum(
                "camera/frustum",
                fov=fov_y,
                aspect=aspect,
                scale=frustum_scale.value,
                position=cam_position,
                wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                image=rgb,
            )

    # Register callbacks
    sample_slider.on_update(update_scene)
    show_mask.on_update(update_scene)
    show_motion.on_update(update_scene)
    show_traj.on_update(update_scene)
    show_bbox.on_update(update_scene)
    show_plane.on_update(update_scene)
    show_frustum.on_update(update_scene)
    frustum_scale.on_update(update_scene)
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
