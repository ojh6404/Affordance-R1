import json
import re

import numpy as np
from scipy.optimize import linear_sum_assignment

from .aff_r1 import (
    aff_r1_score_accuracy_reward,
    aff_r1_score_non_repeat_reward,
    aff_reward_compute_score,
)


# ---------------------------------------------------------------------------
# Motion batch helpers
# ---------------------------------------------------------------------------


def batch_motion_type_match(pred_types, gt_types):
    """Binary match matrix for motion types.
    pred_types: list of M strings, gt_types: list of N strings
    Returns: (M, N) binary ndarray
    """
    M, N = len(pred_types), len(gt_types)
    result = np.zeros((M, N), dtype=float)
    for i in range(M):
        for j in range(N):
            if pred_types[i] == gt_types[j]:
                result[i, j] = 1.0
    return result


def points_to_direction(two_points):
    """Convert two-point axis representation to unit direction vectors.
    two_points: (M, 2, 2) — M pairs of 2D points [[x1,y1],[x2,y2]]
    Returns: (M, 2) unit direction vectors
    """
    direction = two_points[:, 1, :] - two_points[:, 0, :]  # (M, 2)
    norms = np.linalg.norm(direction, axis=1, keepdims=True)
    return direction / np.maximum(norms, 1e-8)


def batch_motion_axis_cosine(pred_axis_points, gt_axes, signed=False):
    """Cosine similarity between predicted two-point axes and GT unit vectors.
    pred_axis_points: (M, 2, 2) — predicted as two image points
    gt_axes: (N, 2) — GT unit direction vectors
    signed: if False, use abs() (rot: axis has no direction);
            if True, keep sign (trans: direction matters)
    Returns: (M, N) cosine similarity
    """
    pred_dirs = points_to_direction(pred_axis_points)  # (M, 2)
    gt_norms = np.linalg.norm(gt_axes, axis=1, keepdims=True)
    gt_dirs = gt_axes / np.maximum(gt_norms, 1e-8)
    cos = pred_dirs @ gt_dirs.T
    return cos if signed else np.abs(cos)


def batch_point_to_line_distance(pred_axis_points, gt_origins):
    """Distance from GT origin points to lines defined by predicted two-point axes.
    pred_axis_points: (M, 2, 2) — M pairs of 2D points [[x1,y1],[x2,y2]]
    gt_origins: (N, 2) — N origin points
    Returns: (M, N) distance matrix

    Formula: d = |cross(p2-p1, p1-origin)| / |p2-p1|
    """
    p1 = pred_axis_points[:, 0, :]  # (M, 2)
    p2 = pred_axis_points[:, 1, :]  # (M, 2)
    d = p2 - p1  # (M, 2)
    norms = np.linalg.norm(d, axis=1, keepdims=True)  # (M, 1)
    v = p1[:, None, :] - gt_origins[None, :, :]  # (M, N, 2)
    cross = d[:, None, 0] * v[:, :, 1] - d[:, None, 1] * v[:, :, 0]  # (M, N)
    return np.abs(cross) / np.maximum(norms, 1e-8)  # (M, N)


# ---------------------------------------------------------------------------
# Motion-specific reward components
# ---------------------------------------------------------------------------


def scenefun3d_format_reward(predict_str: str) -> float:
    """Unified format reward: thinking tags + 4 JSON fields. Max 5.0.

    Thinking format (max 1.0):
      <think>...<rethink>...<answer> structure, penalize empty think/rethink.

    JSON field format (max 4.0, each 1.0 per object, averaged):
      bbox_2d (list len 4), point_2d (list len 2), affordance (str),
      motion_axis_2d (list of 2 points, each len 2)
    """
    # --- Thinking format (max 1.0) ---
    pattern = r"<think>.*?</think>\s*<rethink>.*?</rethink>\s*<answer>.*?</answer>"
    match = re.fullmatch(pattern, predict_str, re.DOTALL)
    thinking_format_reward = 0.0
    if match:
        thinking_format_reward = 1.0
        think_match = re.search(r"<think>(.*?)</think>", predict_str, re.DOTALL)
        rethink_match = re.search(r"<rethink>(.*?)</rethink>", predict_str, re.DOTALL)
        if think_match and len(think_match.group(1).strip()) < 10:
            thinking_format_reward -= 0.5
        if rethink_match and len(rethink_match.group(1).strip()) < 10:
            thinking_format_reward -= 0.5

    # --- JSON field format (max 4.0) ---
    field_reward = 0.0
    try:
        json_match = re.search(r"<answer>\s*(.*?)\s*</answer>", predict_str, re.DOTALL)
        if not json_match:
            return thinking_format_reward
        data = json.loads(json_match.group(1))
        if not isinstance(data, list) or len(data) == 0:
            return thinking_format_reward
        data_cnt = len(data)
        for item in data:
            cur = 0.0
            # base fields (3)
            v = item.get("bbox_2d")
            if isinstance(v, list) and len(v) == 4:
                cur += 1.0
            v = item.get("point_2d")
            if isinstance(v, list) and len(v) == 2:
                cur += 1.0
            if "affordance" in item:
                cur += 1.0
            # motion field (1)
            v = item.get("motion_axis_2d")
            if isinstance(v, list) and len(v) == 2 and all(isinstance(p, list) and len(p) == 2 for p in v):
                cur += 1.0
            field_reward += cur / data_cnt
    except Exception:
        pass

    return thinking_format_reward + field_reward


def scenefun3d_motion_accuracy_reward(predict_str: str, ground_truth: str) -> float:
    """Motion accuracy via Hungarian matching. Max 3.0.

    3 binary components per (pred, gt) pair:
      motion_type exact match, axis |cosine| > 0.85,
      axis-origin distance < 50px (rot only, using GT motion_origin_2d)
    Normalized: total / max(M, N), max per pair = 3.0 (rot) or 2.0 (trans).
    """
    max_reward = 0.0
    MAX_OBJECTS = 120

    try:
        gt_data = json.loads(ground_truth)
        json_match = re.search(r"<answer>\s*(.*?)\s*</answer>", predict_str, re.DOTALL)
        if not json_match:
            return max_reward
        pred_data = json.loads(json_match.group(1))

        if len(pred_data) > MAX_OBJECTS:
            pred_data = pred_data[:MAX_OBJECTS]
        if len(gt_data) > MAX_OBJECTS:
            gt_data = gt_data[:MAX_OBJECTS]

        M, N = len(pred_data), len(gt_data)
        if M == 0 or N == 0:
            return max_reward

        # Check which GT objects have motion fields
        gt_has_motion = [
            all(k in item for k in ("motion_type", "motion_axis_2d"))
            for item in gt_data
        ]
        if not any(gt_has_motion):
            return max_reward  # no motion GT → skip, no penalty

        # Check which GT objects have an origin-on-line reference point:
        #   rot  → motion_origin_2d (rotation center)
        #   trans → point_2d (interaction point should lie on the axis)
        gt_has_origin_ref = [
            (item.get("motion_type") == "rot" and "motion_origin_2d" in item)
            or (item.get("motion_type") == "trans" and "point_2d" in item)
            for item in gt_data
        ]

        # --- Extract pred motion fields ---
        pred_motion_types = [item.get("motion_type", "") for item in pred_data]

        pred_axes_raw = []
        for item in pred_data:
            a = item.get("motion_axis_2d", [[0, 0], [1, 0]])
            if not isinstance(a, list) or len(a) != 2:
                a = [[0, 0], [1, 0]]
            pred_axes_raw.append(a)
        pred_axes = np.array(pred_axes_raw, dtype=float)  # (M, 2, 2)

        # --- Extract GT motion fields ---
        gt_motion_types = [item.get("motion_type", "") for item in gt_data]
        gt_axes = np.array([item.get("motion_axis_2d", [0, 0]) for item in gt_data], dtype=float)  # (N, 2) unit vec

        # Build origin reference: rot uses motion_origin_2d, trans uses point_2d
        gt_origins_list = []
        for item in gt_data:
            if item.get("motion_type") == "rot":
                gt_origins_list.append(item.get("motion_origin_2d", [0, 0]))
            else:  # trans or unknown
                gt_origins_list.append(item.get("point_2d", [0, 0]))
        gt_origins = np.array(gt_origins_list, dtype=float)  # (N, 2)

        # --- Compute motion reward matrices (M, N) ---
        type_reward = batch_motion_type_match(pred_motion_types, gt_motion_types)

        # Axis cosine: rot uses abs (undirected axis), trans uses signed (direction matters)
        axis_cos_abs = batch_motion_axis_cosine(pred_axes, gt_axes, signed=False)   # (M, N)
        axis_cos_signed = batch_motion_axis_cosine(pred_axes, gt_axes, signed=True)  # (M, N)
        axis_reward = np.zeros((M, N), dtype=float)
        for j in range(N):
            if gt_motion_types[j] == "trans":
                axis_reward[:, j] = (axis_cos_signed[:, j] > 0.85).astype(float)
            else:
                axis_reward[:, j] = (axis_cos_abs[:, j] > 0.85).astype(float)

        # Origin-on-line reward: rot (origin on axis) + trans (point_2d on axis)
        origin_reward = (batch_point_to_line_distance(pred_axes, gt_origins) < 30).astype(float)
        for j in range(N):
            if not gt_has_origin_ref[j]:
                origin_reward[:, j] = 0.0

        motion_reward = type_reward + axis_reward + origin_reward  # (M, N)

        # Zero out columns where GT has no motion
        for j in range(N):
            if not gt_has_motion[j]:
                motion_reward[:, j] = 0.0

        # Hungarian matching
        cost_matrix = 3.0 - motion_reward
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        total = motion_reward[row_idx, col_idx].sum()

        # Normalize: same style as aff_r1 accuracy (total / max_length)
        # Each matched pair contributes up to 3.0 (rot) or 2.0 (trans)
        n_motion_gt = sum(gt_has_motion)
        denominator = max(M, n_motion_gt)
        max_reward = total / denominator if denominator > 0 else 0.0

    except Exception:
        pass

    return max_reward


# ---------------------------------------------------------------------------
# Main score function
# ---------------------------------------------------------------------------


def scenefun3d_score(predict_str: str, ground_truth: str, affordance_truth: str, part_truth: str, sim_model):
    """Returns (total_reward, component_dict) when called normally."""
    # === Original base rewards (max 11.0) ===

    # Bbox count reward (max 1.0)
    bbox_count_reward = 0.0
    try:
        json_match = re.search(r"<answer>\s*(.*?)\s*</answer>", predict_str, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
            gt_data = json.loads(ground_truth)
            if len(data) == len(gt_data):
                bbox_count_reward = 1.0
    except Exception:
        pass

    format_reward = scenefun3d_format_reward(predict_str)                            # max 5.0
    accuracy_reward = aff_r1_score_accuracy_reward(predict_str, ground_truth)       # max 3.0
    non_repeat_reward = aff_r1_score_non_repeat_reward(predict_str)                # max 1.0
    aff_reward = aff_reward_compute_score(predict_str, affordance_truth, sim_model)  # max 1.0

    # === Motion rewards (max 3.0) ===

    motion_accuracy = scenefun3d_motion_accuracy_reward(predict_str, ground_truth)  # max 3.0

    reward = (
        # format + base accuracy + misc (max 11.0)
        format_reward + accuracy_reward + non_repeat_reward + aff_reward + bbox_count_reward
        # motion accuracy (max 3.0)
        + motion_accuracy
    )

    components = {
        "format_reward": format_reward,
        "accuracy_reward": accuracy_reward,
        "non_repeat_reward": non_repeat_reward,
        "aff_reward": aff_reward,
        "bbox_count_reward": bbox_count_reward,
        "motion_accuracy": motion_accuracy,
    }
    return reward, components


if __name__ == "__main__":
    predict_str = (
        "<think>The image shows a thermostat dial on the wall next to the door.</think>\n"
        "<rethink>The dial rotates around its center axis pointing left.</rethink>\n"
        '<answer>[{"bbox_2d": [315, 408, 339, 432], "point_2d": [327, 420], "affordance": "rotate", '
        '"motion_type": "rot", '
        '"motion_axis_2d": [[321, 429], [221, 432]]}]</answer>'
    )
    ground_truth = json.dumps([{
        "bbox_2d": [315, 408, 339, 432],
        "point_2d": [327, 420],
        "affordance": "rotate",
        "motion_type": "rot",
        "motion_origin_2d": [321, 429],
        "motion_axis_2d": [-0.9984, -0.0573],
    }])
    print("=== Individual Rewards ===")
    print("  format_reward:", scenefun3d_format_reward(predict_str), "/ 5.0")
    print("  accuracy_reward:", aff_r1_score_accuracy_reward(predict_str, ground_truth), "/ 3.0")
    print("  non_repeat_reward:", aff_r1_score_non_repeat_reward(predict_str), "/ 1.0")
    print("  motion_accuracy:", scenefun3d_motion_accuracy_reward(predict_str, ground_truth), "/ 3.0")

    print("\n=== scenefun3d_score (tuple return) ===")
    total, components = scenefun3d_score(predict_str, ground_truth, "rotate", "dial", sim_model=None)
    print(f"  total: {total}")
    for k, v in components.items():
        print(f"  {k}: {v}")

    # --- Trans example: point_2d should lie on axis ---
    print("\n=== Trans Example ===")
    predict_trans = (
        "<think>A drawer that slides open.</think>\n"
        "<rethink>It translates horizontally along the drawer rail.</rethink>\n"
        '<answer>[{"bbox_2d": [100, 200, 300, 400], "point_2d": [200, 300], "affordance": "pull", '
        '"motion_type": "trans", '
        '"motion_axis_2d": [[150, 300], [250, 300]]}]</answer>'
    )
    ground_truth_trans = json.dumps([{
        "bbox_2d": [100, 200, 300, 400],
        "point_2d": [200, 300],
        "affordance": "pull",
        "motion_type": "trans",
        "motion_axis_2d": [1.0, 0.0],
    }])
    print("  motion_accuracy:", scenefun3d_motion_accuracy_reward(predict_trans, ground_truth_trans), "/ 3.0")
