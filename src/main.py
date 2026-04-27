import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 1. 欺骗系统，让当前 Python 进程认为自己运行在原生的 X11 桌面下
os.environ["XDG_SESSION_TYPE"] = "x11"
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_LOGGING_RULES"] = "default.warning=false"

import time
import cv2
import numpy as np
import sapien.core as sapien
import random
import copy
from mplib import Pose

from src.robot.controller import Controller
from src.robot.motion_planning import MotionPlanning
# 👇 引入所有颜色的配置
from src.config.robot_varant import panda_config
from src.config.object_varant import block_red_config, block_green_config, block_yellow_config

ZONE_DROP_OFFSETS = {
    "goal": [(0.00, 0.00), (0.05, 0.00), (-0.05, 0.00), (0.00, 0.05), (0.00, -0.05)],
    "trash": [(0.00, 0.00), (0.06, 0.00), (-0.06, 0.00), (0.00, 0.06), (0.00, -0.06),
              (0.06, 0.06), (0.06, -0.06), (-0.06, 0.06), (-0.06, -0.06)],
}
ZONE_DROP_COUNTS = {zone_name: 0 for zone_name in ZONE_DROP_OFFSETS}

WRIST_CAMERA_WIDTH = 640
WRIST_CAMERA_HEIGHT = 480
WRIST_CAMERA_FOVY = 1.0
BLOCK_SIZE = 0.04
BOTTOM_LAYER_Z = 0.02
TOP_LAYER_Z = 0.065
IBVS_WINDOW_NAME = "IBVS Continuous View"
IBVS_WINDOW_READY = False

# ==========================================================
# 🧠 第一部分：视觉感知大脑 (Perception)
# ==========================================================
def check_occlusion(box_target, box_obstacle):
    """计算遮挡率：如果障碍物盖住了目标 >10% 的面积，判定为遮挡"""
    tx, ty, tw, th = box_target
    ox, oy, ow, oh = box_obstacle
    
    x_left = max(tx, ox)
    y_top = max(ty, oy)
    x_right = min(tx + tw, ox + ow)
    y_bottom = min(ty + th, oy + oh)
    
    if x_right < x_left or y_bottom < y_top:
        return False 
        
    overlap_area = (x_right - x_left) * (y_bottom - y_top)
    target_area = tw * th
    return (overlap_area / target_area) > 0.10


def get_color_mask(hsv, color):
    if color == "red":
        return cv2.inRange(hsv, np.array([0, 80, 40]), np.array([10, 255, 255])) + \
               cv2.inRange(hsv, np.array([160, 80, 40]), np.array([180, 255, 255]))
    if color == "green":
        return cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    if color == "yellow":
        return cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))
    return np.zeros(hsv.shape[:2], dtype=np.uint8)


def get_valid_contours(mask, min_area=100):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) > min_area]


def contour_center(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        return M["m10"] / M["m00"], M["m01"] / M["m00"]
    x, y, w, h = cv2.boundingRect(contour)
    return x + w / 2, y + h / 2


def contour_inplane_angle_deg(contour):
    (_, _), (width, height), angle = cv2.minAreaRect(contour)
    if width < 1e-6 or height < 1e-6:
        return 0.0
    if width < height:
        angle += 90.0
    while angle >= 45.0:
        angle -= 90.0
    while angle < -45.0:
        angle += 90.0
    return float(angle)


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=float)


def get_grasp_quaternion(yaw_deg):
    base_q = np.array([0.0, 1.0, 0.0, 0.0], dtype=float)
    yaw_rad = np.deg2rad(yaw_deg)
    yaw_q = np.array([np.cos(yaw_rad / 2.0), 0.0, 0.0, np.sin(yaw_rad / 2.0)], dtype=float)
    grasp_q = quaternion_multiply(yaw_q, base_q)
    return grasp_q / np.linalg.norm(grasp_q)


def build_grasp_yaw_candidates(estimated_yaw_deg, rotation_enabled):
    if not rotation_enabled:
        return [0.0]

    if estimated_yaw_deg is None:
        estimated_yaw_deg = 0.0
    centered_candidates = [estimated_yaw_deg, estimated_yaw_deg + 90.0, estimated_yaw_deg - 90.0]
    candidates = []
    for yaw_deg in centered_candidates:
        normalized_yaw = ((yaw_deg + 180.0) % 360.0) - 180.0
        if all(abs(normalized_yaw - existing_yaw) > 1e-3 for existing_yaw in candidates):
            candidates.append(normalized_yaw)
    return candidates


def capture_color_contours(camera, scene, color, min_area=100):
    scene.update_render()
    camera.take_picture()
    rgba = camera.get_picture('Color') if hasattr(camera, 'get_picture') else camera.get_color_rgba()
    bgr_img = cv2.cvtColor((rgba * 255).astype(np.uint8)[:, :, :3], cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    mask = get_color_mask(hsv, color)
    contours = get_valid_contours(mask, min_area=min_area)
    return bgr_img, contours


def project_world_point_to_image(camera_pose_world, point_world):
    cam_to_world = camera_pose_world.to_transformation_matrix()
    world_to_cam = np.linalg.inv(cam_to_world)
    point_cam = world_to_cam @ np.array([point_world[0], point_world[1], point_world[2], 1.0])
    depth = float(point_cam[0])
    if depth <= 1e-5:
        return None

    fy = (WRIST_CAMERA_HEIGHT / 2.0) / np.tan(WRIST_CAMERA_FOVY / 2.0)
    fx = fy
    cx = WRIST_CAMERA_WIDTH / 2.0
    cy = WRIST_CAMERA_HEIGHT / 2.0
    u = cx - fx * (point_cam[1] / depth)
    v = cy - fy * (point_cam[2] / depth)
    return float(u), float(v), depth


def predict_block_top_face_area(camera_pose_world, block_center_world):
    half = BLOCK_SIZE / 2.0
    top_z = block_center_world[2] + half
    corners_world = [
        [block_center_world[0] - half, block_center_world[1] - half, top_z],
        [block_center_world[0] - half, block_center_world[1] + half, top_z],
        [block_center_world[0] + half, block_center_world[1] + half, top_z],
        [block_center_world[0] + half, block_center_world[1] - half, top_z],
    ]
    projected = []
    for corner_world in corners_world:
        pixel = project_world_point_to_image(camera_pose_world, corner_world)
        if pixel is None:
            return None
        projected.append([pixel[0], pixel[1]])
    return float(abs(cv2.contourArea(np.array(projected, dtype=np.float32))))


def select_contour_near_hypotheses(contours, candidate_centers, reference_area=None):
    best_contour = None
    best_score = float("inf")
    for contour in contours:
        area = cv2.contourArea(contour)
        cx, cy = contour_center(contour)
        center_score = min((cx - px) ** 2 + (cy - py) ** 2 for px, py in candidate_centers)
        area_score = 0.0
        if reference_area is not None and reference_area > 0:
            area_score = abs(area / reference_area - 1.0) * 2500
        score = center_score + area_score
        if score < best_score:
            best_score = score
            best_contour = contour
    return best_contour


def collect_multiview_observations(camera, scene, robot, hand_link, cam_pose, mp, target_color, measure_pose, reference_center):
    view_offsets = [0.0, 0.05, -0.05]
    observations = []
    reference_area = None
    target_x = measure_pose.p[0] + 0.05
    target_y = measure_pose.p[1]

    for view_idx, offset_y in enumerate(view_offsets):
        observe_pose = Pose([measure_pose.p[0], measure_pose.p[1] + offset_y, measure_pose.p[2]], [0, 1, 0, 0])
        if view_idx > 0:
            print(f"📸 [主动视觉] 切换到观测位 {view_idx + 1}/{len(view_offsets)}，侧移 {offset_y:.3f}m")
            mp.move_to_pose(observe_pose)
            step_simulation(robot, scene, mp, 40)

        _, contours = capture_color_contours(camera, scene, target_color, min_area=80)
        if not contours:
            observations.append({"valid": False, "offset_y": offset_y})
            print(f"⚠️ [主动视觉] 观测位 {view_idx + 1} 丢失 [{target_color}] 轮廓。")
            continue

        hand_pose_world = hand_link.get_entity_pose()
        camera_pose_world = hand_pose_world * cam_pose

        if view_idx == 0:
            selected = select_reference_contour(contours, reference_center=reference_center)
        else:
            candidate_centers = []
            for layer_z in (BOTTOM_LAYER_Z, TOP_LAYER_Z):
                projected = project_world_point_to_image(camera_pose_world, [target_x, target_y, layer_z + BLOCK_SIZE / 2.0])
                if projected is not None:
                    candidate_centers.append((projected[0], projected[1]))
            if not candidate_centers:
                candidate_centers = [reference_center]
            selected = select_contour_near_hypotheses(contours, candidate_centers, reference_area=reference_area)

        if selected is None:
            observations.append({"valid": False, "offset_y": offset_y})
            print(f"⚠️ [主动视觉] 观测位 {view_idx + 1} 未找到稳定目标轮廓。")
            continue

        center_u, center_v = contour_center(selected)
        contour_area = cv2.contourArea(selected)
        reference_center = (center_u, center_v)
        if reference_area is None:
            reference_area = contour_area

        observations.append({
            "valid": True,
            "offset_y": offset_y,
            "center": (center_u, center_v),
            "area": contour_area,
            "camera_pose_world": camera_pose_world,
        })
        print(f"📍 [观测位 {view_idx + 1}] center=({center_u:.1f}, {center_v:.1f}) | area={contour_area:.0f}")

    mp.move_to_pose(Pose([measure_pose.p[0], measure_pose.p[1], measure_pose.p[2]], [0, 1, 0, 0]))
    step_simulation(robot, scene, mp, 40)
    return observations


def score_layer_hypothesis(observations, target_x, target_y, layer_name):
    layer_z = TOP_LAYER_Z if layer_name == "top" else BOTTOM_LAYER_Z
    total_score = 0.0
    used_views = 0

    for observation in observations:
        if not observation.get("valid"):
            total_score += 180.0
            continue

        camera_pose_world = observation["camera_pose_world"]
        projected_center = project_world_point_to_image(
            camera_pose_world, [target_x, target_y, layer_z + BLOCK_SIZE / 2.0]
        )
        predicted_area = predict_block_top_face_area(camera_pose_world, [target_x, target_y, layer_z])
        if projected_center is None or predicted_area is None or predicted_area <= 1.0:
            total_score += 220.0
            continue

        du = observation["center"][0] - projected_center[0]
        dv = observation["center"][1] - projected_center[1]
        center_error = np.hypot(du, dv)
        area_error = abs(np.log(max(observation["area"], 1.0) / max(predicted_area, 1.0)))
        total_score += center_error + area_error * 70.0
        used_views += 1

    if used_views == 0:
        return float("inf")
    return total_score / used_views


def estimate_layer_with_multiview(camera, scene, robot, hand_link, cam_pose, mp, target_color, measure_pose, reference_center, final_area, ibvs_hand_z):
    print("🧠 [多视角判层] 开始基于已知相机位姿做两层假设评分...")
    observations = collect_multiview_observations(
        camera, scene, robot, hand_link, cam_pose, mp, target_color, measure_pose, reference_center
    )
    target_x = measure_pose.p[0] + 0.05
    target_y = measure_pose.p[1]

    valid_observations = sum(1 for obs in observations if obs.get("valid"))
    if valid_observations < 2:
        fallback_layer, fallback_confidence = infer_layer_from_visual_cues(None, final_area, ibvs_hand_z)
        print(f"⚠️ [多视角判层] 有效观测不足 ({valid_observations})，回退到单帧视觉判定【{fallback_layer}层】。")
        return fallback_layer, fallback_confidence, {"top": None, "bottom": None}

    top_score = score_layer_hypothesis(observations, target_x, target_y, "top")
    bottom_score = score_layer_hypothesis(observations, target_x, target_y, "bottom")
    inferred_layer = "top" if top_score < bottom_score else "bottom"
    score_gap = abs(top_score - bottom_score)
    confidence = "high" if score_gap > 20.0 else "medium" if score_gap > 8.0 else "low"
    print(f"📊 [多视角判层] top_score={top_score:.2f} | bottom_score={bottom_score:.2f} | 判定={inferred_layer} | confidence={confidence}")
    return inferred_layer, confidence, {"top": top_score, "bottom": bottom_score}


def select_reference_contour(contours, reference_center=(320, 240), reference_area=None):
    best_contour = None
    best_score = float("inf")
    for contour in contours:
        area = cv2.contourArea(contour)
        cx, cy = contour_center(contour)
        distance_score = (cx - reference_center[0]) ** 2 + (cy - reference_center[1]) ** 2
        area_score = 0.0
        if reference_area is not None and reference_area > 0:
            area_score = abs(area / reference_area - 1.0) * 4000
        score = distance_score + area_score
        if score < best_score:
            best_score = score
            best_contour = contour
    return best_contour


def select_contour_near_reference(contours, reference_center, max_dist_px=85, reference_area=None):
    best_contour = None
    best_score = float("inf")
    for contour in contours:
        area = cv2.contourArea(contour)
        cx, cy = contour_center(contour)
        dist = np.hypot(cx - reference_center[0], cy - reference_center[1])
        if dist > max_dist_px:
            continue
        area_score = 0.0
        if reference_area is not None and reference_area > 0:
            area_score = abs(area / reference_area - 1.0) * 30.0
        score = dist + area_score
        if score < best_score:
            best_score = score
            best_contour = contour
    return best_contour


def estimate_object_height_from_parallax(camera, scene, robot, hand_link, mp, target_color, measure_pose, reference_center):
    COMMAND_BASELINE = 0.08
    FOCAL_LENGTH = 439.3
    HEIGHT_SPLIT_Z = 0.045

    _, base_contours = capture_color_contours(camera, scene, target_color, min_area=100)
    if not base_contours:
        print("⚠️ [主动视觉] 基准面丢失目标。")
        return None, "low"

    base_contour = select_reference_contour(base_contours, reference_center=reference_center)
    if base_contour is None:
        print("⚠️ [主动视觉] 基准面未找到稳定轮廓。")
        return None, "low"

    u1, v1 = contour_center(base_contour)
    base_area = cv2.contourArea(base_contour)
    print(f"🔍 [特征锁定] 质心亚像素坐标({u1:.1f}, {v1:.1f}), 真实轮廓面积={base_area:.0f}")

    estimates = []
    for direction in (1, -1):
        target_pose = Pose([measure_pose.p[0], measure_pose.p[1] + direction * COMMAND_BASELINE, measure_pose.p[2]], [0, 1, 0, 0])
        print(f"📸 正在向侧方平移期望值 {direction * COMMAND_BASELINE * 100:.1f}cm 以获取极线视差...")
        mp.move_to_pose(target_pose)

        for _ in range(50):
            robot.set_qf(robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))
            scene.step()
            if mp.controller.viewer:
                mp.controller.viewer.render()

        shifted_pose = hand_link.get_entity_pose()
        actual_baseline = np.linalg.norm(shifted_pose.p[:2] - measure_pose.p[:2])
        print(f"📊 [运动学真实值] 实际物理平移: {actual_baseline * 100:.2f}cm")

        _, shifted_contours = capture_color_contours(camera, scene, target_color, min_area=50)
        best_match = None
        best_score = float("inf")

        print("\n--- 🔬 [深度调试] 开始特征匹配评判 ---")
        for idx, contour in enumerate(shifted_contours):
            area2 = cv2.contourArea(contour)
            cx2, cy2 = contour_center(contour)
            dx = cx2 - u1
            dy = cy2 - v1
            area_ratio = area2 / base_area if base_area > 0 else 0.0
            score = abs(dy) * 6 + abs(area_ratio - 1.0) * 60
            print(f"  > 轮廓 {idx}: 视差 dx={abs(dx):.1f}px | 极线漂移 dy={dy:.1f}px | 面积比 {area_ratio:.2f} | 得分 {score:.1f}")
            if 20 < abs(dx) < 320 and score < best_score:
                best_score = score
                best_match = (abs(dx), shifted_pose.p[2] + 0.04)
        print("-" * 50)

        if best_match is not None and actual_baseline > 1e-4:
            disparity, cam_z_world = best_match
            depth_z_cam = (FOCAL_LENGTH * actual_baseline) / disparity
            obj_z_world = cam_z_world - depth_z_cam
            estimates.append((obj_z_world, best_score))
            print(f"✅ [视差锁定] 视差: {disparity:.1f}px | 估计 Obj_Z={obj_z_world:.3f}m")
        else:
            print("⚠️ [深度告警] 当前视角未得到有效视差。")

        mp.move_to_pose(Pose([measure_pose.p[0], measure_pose.p[1], measure_pose.p[2]], [0, 1, 0, 0]))
        for _ in range(40):
            robot.set_qf(robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))
            scene.step()
            if mp.controller.viewer:
                mp.controller.viewer.render()

    if not estimates:
        return None, "low"

    best_obj_z = float(np.median([item[0] for item in estimates]))
    confidence = "high" if abs(best_obj_z - HEIGHT_SPLIT_Z) >= 0.01 else "medium"
    print(f"🌍 [绝对高度] 综合估计 Obj_Z={best_obj_z:.3f}m | 置信度={confidence}")
    return best_obj_z, confidence


def step_simulation(robot, scene, mp, steps, render=True):
    for _ in range(steps):
        robot.set_qf(robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))
        scene.step()
        if render and mp.controller.viewer:
            mp.controller.viewer.render()


def drive_to_pose_ik(robot, scene, mp, pose, hold_steps=120):
    status, target_q = mp.planner.IK(pose, robot.get_qpos())
    if status != "Success":
        return False

    target_q_flat = np.array(target_q).flatten()
    for idx, arm_joint_idx in enumerate(mp.planner.move_group_joint_indices):
        robot.active_joints[arm_joint_idx].set_drive_target(float(target_q_flat[idx]))
    step_simulation(robot, scene, mp, hold_steps)
    return True


def ensure_ibvs_window():
    global IBVS_WINDOW_READY
    if IBVS_WINDOW_READY:
        return
    try:
        cv2.startWindowThread()
    except Exception:
        pass
    try:
        cv2.namedWindow(IBVS_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.waitKey(1)
        IBVS_WINDOW_READY = True
    except cv2.error:
        IBVS_WINDOW_READY = False


def show_ibvs_frame(frame):
    if not IBVS_WINDOW_READY:
        return
    try:
        cv2.imshow(IBVS_WINDOW_NAME, frame)
        cv2.waitKey(1)
    except cv2.error:
        pass


def get_zone_drop_target(zone_name, drop_zone):
    offsets = ZONE_DROP_OFFSETS.get(zone_name, [(0.0, 0.0)])
    drop_index = ZONE_DROP_COUNTS.get(zone_name, 0)
    offset_x, offset_y = offsets[drop_index % len(offsets)]
    ZONE_DROP_COUNTS[zone_name] = drop_index + 1
    return drop_zone[0] + offset_x, drop_zone[1] + offset_y


def place_held_object(robot, scene, mp, drop_x, drop_y, safe_height_z, target_color):
    BLOCK_Q = [0, 1, 0, 0]
    release_z = 0.13 if target_color == "red" else 0.145
    approach_z = max(release_z + 0.06, 0.22)

    mp.move_to_pose(Pose([drop_x, drop_y, safe_height_z], BLOCK_Q))
    step_simulation(robot, scene, mp, 20)
    mp.move_to_pose(Pose([drop_x, drop_y, approach_z], BLOCK_Q))
    step_simulation(robot, scene, mp, 30)

    for descend_z in (0.18, release_z):
        if not drive_to_pose_ik(robot, scene, mp, Pose([drop_x, drop_y, descend_z], BLOCK_Q), hold_steps=120):
            return False
        step_simulation(robot, scene, mp, 20)

    mp.open_gripper()
    step_simulation(robot, scene, mp, 60)

    if not drive_to_pose_ik(robot, scene, mp, Pose([drop_x, drop_y, approach_z], BLOCK_Q), hold_steps=80):
        return False
    mp.move_to_pose(Pose([drop_x, drop_y, safe_height_z], BLOCK_Q))
    step_simulation(robot, scene, mp, 20)
    return True


def infer_layer_from_visual_cues(obj_z_world, final_area, ibvs_hand_z):
    HEIGHT_SPLIT_Z = 0.045
    AREA_TOP_THRESHOLD = 12500
    HAND_Z_TOP_THRESHOLD = 0.24

    votes = []
    if obj_z_world is not None:
        votes.append("top" if obj_z_world > HEIGHT_SPLIT_Z else "bottom")
    if final_area is not None:
        votes.append("top" if final_area > AREA_TOP_THRESHOLD else "bottom")
    if ibvs_hand_z is not None:
        votes.append("top" if ibvs_hand_z > HAND_Z_TOP_THRESHOLD else "bottom")

    top_votes = votes.count("top")
    bottom_votes = votes.count("bottom")
    inferred_layer = "top" if top_votes > bottom_votes else "bottom"
    confidence = "high" if abs(top_votes - bottom_votes) >= 2 else "medium"
    return inferred_layer, confidence


def build_block_config(color_name, position):
    base_config_map = {
        "red": block_red_config,
        "green": block_green_config,
        "yellow": block_yellow_config,
    }
    config = copy.deepcopy(base_config_map[color_name])
    config["position"] = list(position)
    return config


def spawn_random_sorting_scene(controller):
    slot_y_values = [-0.18, -0.06, 0.06, 0.18]
    random.shuffle(slot_y_values)
    slot_x_values = [0.39, 0.41, 0.43, 0.45]
    used_slots = []
    spawned_layout = []

    def allocate_slot():
        index = len(used_slots)
        x = slot_x_values[index % len(slot_x_values)] + random.uniform(-0.01, 0.01)
        y = slot_y_values[index] + random.uniform(-0.01, 0.01)
        used_slots.append((x, y))
        return x, y

    def spawn_stack(bottom_color, top_color):
        base_x, base_y = allocate_slot()
        bottom_pos = [base_x, base_y, 0.02]
        top_pos = [base_x + random.uniform(-0.004, 0.004), base_y + random.uniform(-0.004, 0.004), 0.065]
        controller.add_object(build_block_config(bottom_color, bottom_pos))
        controller.add_object(build_block_config(top_color, top_pos))
        spawned_layout.append((bottom_color, "bottom", bottom_pos))
        spawned_layout.append((top_color, "top", top_pos))

    def spawn_single(color_name):
        base_x, base_y = allocate_slot()
        single_pos = [base_x, base_y, 0.02]
        controller.add_object(build_block_config(color_name, single_pos))
        spawned_layout.append((color_name, "single", single_pos))

    covered_top_color = random.choice(["yellow", "green"])
    spawn_stack("red", covered_top_color)

    second_stack_bottom = random.choice(["red", "yellow", "green"])
    second_stack_top_candidates = [color for color in ["red", "yellow", "green"] if color != second_stack_bottom]
    second_stack_top = random.choice(second_stack_top_candidates)
    spawn_stack(second_stack_bottom, second_stack_top)

    spawn_single("red")
    spawn_single(random.choice(["yellow", "green"]))

    print("🧪 随机场景布局如下：")
    for color_name, layer_name, position in spawned_layout:
        print(f"   - {color_name:>6} | {layer_name:<6} | pos=({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})")

    return spawned_layout


def gripper_has_object(robot, min_finger_gap=0.01):
    finger_qpos = np.array(robot.get_qpos())[-2:]
    mean_gap = float(np.mean(finger_qpos))
    print(f"🤏 [抓取校验] 当前夹爪开度={mean_gap:.4f}")
    return mean_gap > min_finger_gap


def run_grasp_attempt(robot, scene, mp, target_x, target_y, grasp_z, safe_height_z, grasp_yaw_deg):
    grasp_q = get_grasp_quaternion(grasp_yaw_deg)

    status, target_q = mp.planner.IK(Pose([target_x, target_y, grasp_z], grasp_q), robot.get_qpos())
    if status != "Success":
        print(f"❌ [抓取尝试] IK 失败，无法到达 Z={grasp_z:.3f}, yaw={grasp_yaw_deg:.1f}")
        return False

    target_q_flat = np.array(target_q).flatten()
    for idx, arm_joint_idx in enumerate(mp.planner.move_group_joint_indices):
        robot.active_joints[arm_joint_idx].set_drive_target(float(target_q_flat[idx]))
    step_simulation(robot, scene, mp, 200)

    mp.close_gripper()
    step_simulation(robot, scene, mp, 60)

    current_z = grasp_z
    while current_z < safe_height_z:
        current_z = min(current_z + 0.05, safe_height_z)
        status, target_q = mp.planner.IK(Pose([target_x, target_y, current_z], grasp_q), robot.get_qpos())
        if status == "Success":
            target_q_flat = np.array(target_q).flatten()
            for idx, arm_joint_idx in enumerate(mp.planner.move_group_joint_indices):
                robot.active_joints[arm_joint_idx].set_drive_target(float(target_q_flat[idx]))
        step_simulation(robot, scene, mp, 15)

    return gripper_has_object(robot)

def detect_all_objects(camera, scene, color="red"):
    """执行全局快照，返回该颜色所有物块的包围盒和面积 (按面积降序排序)"""
    bgr_img, valid_contours = capture_color_contours(camera, scene, color, min_area=100)
    if bgr_img is None:
        return [], []
    if not valid_contours:
        return [], []
        
    boxes = [cv2.boundingRect(c) for c in valid_contours]
    areas = [cv2.contourArea(c) for c in valid_contours]
    
    sorted_data = sorted(zip(boxes, areas), key=lambda x: x[1], reverse=True)
    return [x[0] for x in sorted_data], [x[1] for x in sorted_data]


def visual_servoing(camera, scene, robot, cam_pose, mp, color="red", target_box=None):
    """带有注意力锁定的 IBVS 视觉伺服"""
    print(f"🚀 开启 IBVS 模式！正在追踪 [{color}] 目标...")
    ensure_ibvs_window()
    
    lambda_xy = 0.0003  
    lambda_z = 0.025    
    target_u, target_v = 320, 240
    target_area = 10000 
    
    hand_link = [link for link in robot.get_links() if link.name == "panda_hand"][0]
    final_box = None
    final_area = 0   
    final_angle_deg = 0.0
    servo_step = 0
    stagnant_steps = 0
    last_state = None
    
    # 初始化注意力追踪中心
    if target_box is not None:
        tx, ty, tw, th = target_box
        tracking_center = (tx + tw//2, ty + th//2)
    else:
        tracking_center = None
    
    while True:
        servo_step += 1
        scene.update_render()
        camera.take_picture()
        rgba = camera.get_picture('Color') if hasattr(camera, 'get_picture') else camera.get_color_rgba()
        bgr_img = cv2.cvtColor((rgba * 255).astype(np.uint8)[:, :, :3], cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        
        mask = get_color_mask(hsv, color)
        valid_contours = get_valid_contours(mask, min_area=100)
        
        if not valid_contours:
            print(f"   [IBVS] 警告: 丢失 {color} 目标，伺服中止！")
            break
            
        # 注意力锁定机制
        if tracking_center is not None:
            best_c = None
            min_dist = float('inf')
            for c in valid_contours:
                x, y, w, h = cv2.boundingRect(c)
                cx, cy = x + w//2, y + h//2
                dist = (cx - tracking_center[0])**2 + (cy - tracking_center[1])**2
                if dist < min_dist:
                    min_dist = dist
                    best_c = c
            c = best_c
        else:
            c = max(valid_contours, key=cv2.contourArea)
            
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        cX, cY = x + w//2, y + h//2
        tracking_center = (cX, cY) # 动态更新追踪中心
        
        final_box = (x, y, w, h)
        final_area = area
        final_angle_deg = contour_inplane_angle_deg(c)
        
        # 绘制 UI
        cv2.rectangle(bgr_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.line(bgr_img, (target_u, target_v), (cX, cY), (0, 255, 255), 2)
        cv2.circle(bgr_img, (target_u, target_v), 5, (0, 0, 255), -1)
        cv2.putText(bgr_img, f"Error: ({cX-target_u}, {cY-target_v}) | Area: {area:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        show_ibvs_frame(bgr_img)
        
        e_u, e_v = cX - target_u, cY - target_v
        current_state = (round(cX / 4), round(cY / 4), round(area / 250))
        if current_state == last_state:
            stagnant_steps += 1
        else:
            stagnant_steps = 0
        last_state = current_state
        
        # 对齐判定
        if abs(e_u) < 15 and abs(e_v) < 15 and area > target_area:
            print("✅ [IBVS] 视觉对齐完成！")
            break
        if abs(e_u) < 18 and abs(e_v) < 18 and area > target_area * 0.7 and stagnant_steps > 25:
            print("⚠️ [IBVS] 已接近目标但收敛停滞，接受当前对齐结果。")
            break
        if servo_step >= 220:
            print("⚠️ [IBVS] 达到最大迭代次数，接受当前最佳对齐结果。")
            break

        v_cam_y = -lambda_xy * e_u
        v_cam_z = -lambda_xy * e_v
        v_cam_x = lambda_z if (abs(e_u) < 60 and abs(e_v) < 60) else 0.0 
        v_cam = np.array([v_cam_x, v_cam_y, v_cam_z])
        
        hand_pose_world = hand_link.get_entity_pose()
        cam_pose_world = hand_pose_world * cam_pose
        v_world = cam_pose_world.to_transformation_matrix()[:3, :3] @ v_cam
        
        base_link = [link for link in robot.get_links() if link.name == "panda_link0"][0]
        base_p = base_link.get_entity_pose().p
        target_p_base = hand_pose_world.p + v_world - base_p
        target_pose = Pose(target_p_base, hand_pose_world.q)
        
        res = mp.planner.plan_screw(target_pose, robot.get_qpos(), time_step=1/250)
        motion_success = False
        if res["status"] == "Success" and len(res["position"]) > 0:
            target_q = res["position"][-1]
            for j in range(len(mp.planner.move_group_joint_indices)):
                robot.active_joints[j].set_drive_target(target_q[j])
            motion_success = True
        else:
            try:
                status, target_q = mp.planner.IK(target_pose, robot.get_qpos())
                if status == "Success":
                    for j in range(len(mp.planner.move_group_joint_indices)):
                        robot.active_joints[j].set_drive_target(target_q[j])
                    motion_success = True
            except Exception as e:
                pass 
        if not motion_success:
            stagnant_steps += 3
        
        step_simulation(robot, scene, mp, 15)
            
    return final_box, final_area, final_angle_deg


# ==========================================================
# 🦾 第二部分：动作执行系统 (Action)
# ==========================================================
def execute_grasp(target_color, drop_zone, camera, scene, robot, cam_pose, mp, target_box=None, drop_zone_name="trash", layer_hint=None):
    """
    通用抓取封装（主动视觉多视角判层版）：
    1. IBVS 对齐 XY 坐标
    2. 拉升至观测基准面 (Z=0.25)
    3. 主动移动相机到多个已知观测位
    4. 对【顶层/底层】两种离散高度做重投影评分
    """
    BLOCK_Q = [0, 1, 0, 0]
    SAFE_HEIGHT_Z = 0.50
    MEASURE_HEIGHT_Z = 0.25 
    TOP_GRASP_Z = 0.180
    TOP_OBSTACLE_GRASP_Z_CANDIDATES = [0.188, 0.183, 0.180]
    BOTTOM_GRASP_Z = 0.080
    hand_link = [link for link in robot.get_links() if link.name == "panda_hand"][0]
    
    print(f"\n[动作系统] 开始执行 [{target_color}] 物块抓取序列...")
    mp.open_gripper()
    mp.move_to_pose(Pose([0.40, 0.10, SAFE_HEIGHT_Z], BLOCK_Q))
    for _ in range(30): scene.step()
    
    # 1. 启动 IBVS 伺服对齐
    final_box, final_area, final_angle_deg = visual_servoing(camera, scene, robot, cam_pose, mp, color=target_color, target_box=target_box)
    if final_box is None:
        print(f"❌ [动作系统] 伺服过程中丢失 [{target_color}] 目标。")
        return False
        
    ibvs_pose = hand_link.get_entity_pose()
    pose1 = hand_link.get_entity_pose()
    print(f"⬆️ [主动视觉] 伺服完毕(Z={ibvs_pose.p[2]:.3f})，拉回黄金基准面 Z={MEASURE_HEIGHT_Z} 进行严谨测距...")
    mp.move_to_pose(Pose([ibvs_pose.p[0], ibvs_pose.p[1], MEASURE_HEIGHT_Z], BLOCK_Q))
    step_simulation(robot, scene, mp, 50)
    pose1 = hand_link.get_entity_pose()

    reference_center = (320, 240)
    if final_box is not None:
        fx, fy, fw, fh = final_box
        reference_center = (fx + fw / 2, fy + fh / 2)

    inferred_layer, height_confidence, layer_scores = estimate_layer_with_multiview(
        camera, scene, robot, hand_link, cam_pose, mp, target_color, pose1, reference_center, final_area, ibvs_pose.p[2]
    )
    if layer_scores["top"] is not None and layer_scores["bottom"] is not None:
        print(f"-> [判定] 多视角评分 top={layer_scores['top']:.2f}, bottom={layer_scores['bottom']:.2f}，采用【{inferred_layer}层】")
    else:
        print(f"⚠️ [高度判定] 多视角观测不足，采用回退判定【{inferred_layer}层】。")
    if layer_hint is not None and inferred_layer != layer_hint:
        print(f"🧭 [任务先验] 当前目标是覆盖物，层级从【{inferred_layer}】修正为【{layer_hint}】。")
        inferred_layer = layer_hint
        if height_confidence == "low":
            height_confidence = "medium"

    # ==========================================================
    # 🦾 物理执行：偏置补偿与分拣
    # ==========================================================
    # 使用 pose1 作为目标对齐坐标（平移前准确对齐目标的XY）
    target_x = pose1.p[0] + 0.05
    target_y = pose1.p[1] + 0.00
    
    print("-> 测距完毕，回归目标位置并执行偏置补偿...")
    mp.move_to_pose(Pose([target_x, target_y, SAFE_HEIGHT_Z], BLOCK_Q))

    alternate_layer = "bottom" if inferred_layer == "top" else "top"
    strict_top_removal = layer_hint == "top" and target_color in ("yellow", "green")
    strict_reference_center = reference_center
    strict_reference_area = final_area
    if strict_top_removal:
        attempt_plans = [("top", grasp_z) for grasp_z in TOP_OBSTACLE_GRASP_Z_CANDIDATES]
    elif target_color in ("yellow", "green") and height_confidence == "low":
        attempt_layers = ["top", "bottom"]
        attempt_plans = [(layer_name, TOP_GRASP_Z if layer_name == "top" else BOTTOM_GRASP_Z) for layer_name in attempt_layers]
    elif target_color == "red" and height_confidence == "low":
        attempt_layers = [inferred_layer, alternate_layer]
        attempt_plans = [(layer_name, TOP_GRASP_Z if layer_name == "top" else BOTTOM_GRASP_Z) for layer_name in attempt_layers]
    else:
        attempt_layers = [inferred_layer, alternate_layer]
        attempt_plans = [(layer_name, TOP_GRASP_Z if layer_name == "top" else BOTTOM_GRASP_Z) for layer_name in attempt_layers]

    rotation_enabled = abs(final_angle_deg) >= 12.0
    grasp_yaw_candidates = build_grasp_yaw_candidates(final_angle_deg, rotation_enabled=rotation_enabled)
    if rotation_enabled:
        print(f"🔄 [姿态规划] 检测到目标旋转角={final_angle_deg:.1f}°, 启用 yaw 候选={grasp_yaw_candidates}")
    else:
        print(f"🔄 [姿态规划] 目标旋转角={final_angle_deg:.1f}°，未超过阈值，仅使用 yaw=0°")

    grasp_success = False
    for attempt_idx, (layer_name, grasp_z) in enumerate(attempt_plans):
        if strict_top_removal and attempt_idx > 0:
            _, strict_contours = capture_color_contours(camera, scene, target_color, min_area=80)
            strict_contour = select_contour_near_reference(
                strict_contours,
                strict_reference_center,
                max_dist_px=85,
                reference_area=strict_reference_area,
            )
            if strict_contour is None:
                print("🛑 [安全策略] 顶层覆盖物已偏离原位置或不可见，停止继续下探，等待重新观测。")
                break
        for grasp_yaw_deg in grasp_yaw_candidates:
            print(f"🦾 [抓取尝试] 按【{layer_name}层】执行，下潜 Z={grasp_z:.3f}, yaw={grasp_yaw_deg:.1f}")
            grasp_success = run_grasp_attempt(robot, scene, mp, target_x, target_y, grasp_z, SAFE_HEIGHT_Z, grasp_yaw_deg)
            if grasp_success:
                print(f"✅ [抓取成功] 已稳定夹住 [{target_color}] 物块。")
                break
            print(f"⚠️ [抓取失败] {layer_name}层 yaw={grasp_yaw_deg:.1f} 未夹住目标，准备重试。")
            mp.open_gripper()
            step_simulation(robot, scene, mp, 30)
            mp.move_to_pose(Pose([target_x, target_y, SAFE_HEIGHT_Z], BLOCK_Q))
        if grasp_success:
            break
        if strict_top_removal:
            if attempt_idx == len(attempt_plans) - 1:
                print("🛑 [安全策略] 覆盖物移除任务禁止回退到底层，结束本轮抓取并等待重新观测。")
                break

    if not grasp_success:
        if strict_top_removal:
            print(f"❌ [动作系统] [{target_color}] 顶层覆盖物抓取失败，已停止以避免误抓下层目标。")
        else:
            print(f"❌ [动作系统] [{target_color}] 目标两层高度均未抓取成功。")
        return False

    DROP_X, DROP_Y = get_zone_drop_target(drop_zone_name, drop_zone)
    print(f"📦 [投放] 目标区域={drop_zone_name} | 投放坐标=({DROP_X:.3f}, {DROP_Y:.3f})")
    if not place_held_object(robot, scene, mp, DROP_X, DROP_Y, SAFE_HEIGHT_Z, target_color):
        print(f"❌ [投放失败] 无法稳定完成 [{target_color}] 的放置。")
        return False
    return True
# ==========================================================
# 🚀 第三部分：全局主控入口 (Main Execution)
# ==========================================================
if __name__ == "__main__":
    controller = Controller()
    
    # 1. 机械臂底座归零及初始化防碰撞
    ROBOT_BASE_Z = 0.0
    panda_config["position"] = [0.0, 0.0, ROBOT_BASE_Z]
    robot = controller.add_robot(panda_config)
    
    safe_home_qpos = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]
    robot.set_qpos(safe_home_qpos)
    for i in range(len(robot.active_joints)):
        robot.active_joints[i].set_drive_target(safe_home_qpos[i])
    for _ in range(50):
        robot.set_qf(robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))
        controller.scene.step()

    # 安装手眼摄像头
    try:
        hand_link = [link for link in robot.get_links() if link.name == "panda_hand"][0]
        cam_pose = sapien.Pose([0.05, 0.0, 0.04], [0.7071, 0, -0.7071, 0])
        if hasattr(hand_link, "entity"): 
            import sapien.render
            camera = sapien.render.RenderCameraComponent(640, 480)
            camera.local_pose = cam_pose
            hand_link.entity.add_component(camera)
        else:
            camera = controller.scene.add_mounted_camera(
                name="wrist_camera", mount=hand_link, pose=cam_pose, 
                width=640, height=480, fovy=1.0, near=0.1, far=10.0
            )
    except Exception as e:
        camera = None

    mp = MotionPlanning(panda_config, controller)
    BLOCK_Q = [0, 1, 0, 0] 
    SAFE_HEIGHT_Z = 0.50   

    # ==========================================================
    # 2. 动态生成【平行多组杂乱环境】(真实流水线测试)
    # ==========================================================
    print("\n📦 正在生成随机叠层分拣测试环境...")
    spawn_random_sorting_scene(controller)

    # 自由落体稳定
    for _ in range(80): controller.scene.step()
    
    # 区域划分
    GOAL_ZONE = [-0.35, -0.55]  # 收集区放远一点，避免挡路
    TRASH_ZONE = [-0.35, 0.55]  # 垃圾区
    OBSTACLE_COLORS = ["yellow", "green"]

    print("\n" + "="*50)
    print("🌟 智能全局多目标分拣系统启动！")
    print("="*50)

    # ==========================================================
    # 3. 大脑决策主循环 (剥洋葱 + 找软柿子)
    # ==========================================================
    loop_count = 0
    while True:
        loop_count += 1
        print(f"\n[大脑状态机] >>> 第 {loop_count} 轮全局扫描评估 <<<")
        
        mp.move_to_pose(Pose([0.40, 0.10, SAFE_HEIGHT_Z], BLOCK_Q))
        for _ in range(30): controller.scene.step() 
        
        # 提取全场信息
        red_boxes, _ = detect_all_objects(camera, controller.scene, color="red")
        obstacles = {}
        for color in OBSTACLE_COLORS:
            boxes, _ = detect_all_objects(camera, controller.scene, color=color)
            obstacles[color] = boxes

        # 终止条件判定
        if not red_boxes:
            total_obstacles = sum(len(obstacles[c]) for c in OBSTACLE_COLORS)
            if total_obstacles == 0:
                print("🎉 桌面已完全清理，所有红色目标收集完毕！任务圆满结束！")
                break
            else:
                print("⚠️ 未发现红色！可能被 100% 遮挡。触发盲清机制...")
                for color in OBSTACLE_COLORS:
                    if obstacles[color]:
                        print(f"🧹 正在盲清 [{color}] 障碍物...")
                        execute_grasp(color, TRASH_ZONE, camera, controller.scene, robot, cam_pose, mp, target_box=obstacles[color][0], drop_zone_name="trash")
                        break
                continue 

        # 核心决策：在所有的红块中，找一个没被遮挡的“自由红块”
        free_red_box = None
        blocked_red_box = None
        blocking_obs_color = None
        blocking_obs_box = None

        for r_box in red_boxes:
            is_occluded = False
            for color in OBSTACLE_COLORS:
                for obs_box in obstacles[color]:
                    if check_occlusion(r_box, obs_box):
                        is_occluded = True
                        if blocked_red_box is None: # 记录第一个被挡住的
                            blocked_red_box = r_box
                            blocking_obs_color = color
                            blocking_obs_box = obs_box
                        break
                if is_occluded: break
            
            if not is_occluded:
                free_red_box = r_box # 找到自由红块
                break

        # 派发动作指令
        if free_red_box:
            print("✅ 发现无遮挡的红色目标！立刻执行抓取...")
            execute_grasp("red", GOAL_ZONE, camera, controller.scene, robot, cam_pose, mp, target_box=free_red_box, drop_zone_name="goal")
        else:
            print(f"⚠️ 所有红色目标均被遮挡！")
            print(f"🧹 决策：优先移除压住目标的 [{blocking_obs_color}] 障碍物...")
            execute_grasp(blocking_obs_color, TRASH_ZONE, camera, controller.scene, robot, cam_pose, mp, target_box=blocking_obs_box, drop_zone_name="trash", layer_hint="top")
            
    controller.out.release()
    controller.visualize(robot)
