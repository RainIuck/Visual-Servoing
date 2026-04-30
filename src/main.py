import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 1. 欺骗系统，让当前 Python 进程认为自己运行在原生的 X11 桌面下
os.environ["XDG_SESSION_TYPE"] = "x11"

# 2. 指定 OpenCV 使用 xcb 插件（因为我们只有这个插件）
os.environ["QT_QPA_PLATFORM"] = "xcb"

# 3. 屏蔽 Qt 底层的啰嗦警告日志
os.environ["QT_LOGGING_RULES"] = "default.warning=false"

import time
import argparse
import cv2
import numpy as np
import sapien.core as sapien
import random
from mplib import Pose

from src.robot.controller import Controller
from src.robot.motion_planning import MotionPlanning
from src.config.robot_varant import panda_config
from src.config.object_varant import block_red_config

def visual_servoing(camera, scene, robot, cam_pose, mp):
    print("🚀 开启基于图像的视觉伺服控制 (IBVS) 模式！")
    cv2.namedWindow("IBVS Continuous View", cv2.WINDOW_AUTOSIZE)
    
    lambda_xy = 0.0003  # 平移速度增益
    lambda_z = 0.025    # 深度逼近速度
    target_u, target_v = 320, 240
    # 💡 调小物块面积阈值。既然改成单物块了，我们只用 IBVS 做 XY 瞄准和初步下潜
    target_area = 10000 
    
    hand_link = [link for link in robot.get_links() if link.name == "panda_hand"][0]
    step_idx = 0
    
    while True:
        step_idx += 1
        scene.update_render()
        camera.take_picture()
        rgba = camera.get_picture('Color') if hasattr(camera, 'get_picture') else camera.get_color_rgba()
        bgr_img = cv2.cvtColor((rgba * 255).astype(np.uint8)[:, :, :3], cv2.COLOR_RGB2BGR)
        
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255])) + \
               cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("   [IBVS] 警告: 丢失目标，伺服停止！")
            break
            
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        cX, cY = x + w//2, y + h//2
        
        cv2.rectangle(bgr_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.line(bgr_img, (target_u, target_v), (cX, cY), (0, 255, 255), 2)
        cv2.circle(bgr_img, (target_u, target_v), 5, (0, 0, 255), -1)
        cv2.putText(bgr_img, f"Error: ({cX-target_u}, {cY-target_v})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(bgr_img, f"Area: {area:.0f}/{target_area}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("IBVS Continuous View", bgr_img)
        cv2.waitKey(20) 
        
        e_u, e_v = cX - target_u, cY - target_v
        
        # 抓取触发：XY 准心对齐，且进入预定高度(面积)
        if abs(e_u) < 15 and abs(e_v) < 15 and area > target_area:
            print("✅ [IBVS] 视觉对齐完成，准备执行精准下探！")
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
        if res["status"] == "Success" and len(res["position"]) > 0:
            target_q = res["position"][-1]
            for j in range(len(mp.planner.move_group_joint_indices)):
                robot.active_joints[j].set_drive_target(target_q[j])
        else:
            try:
                status, target_q = mp.planner.IK(target_pose, robot.get_qpos())
                if status == "Success":
                    for j in range(len(mp.planner.move_group_joint_indices)):
                        robot.active_joints[j].set_drive_target(target_q[j])
            except Exception as e:
                pass 
        
        for _ in range(15): 
            qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
            robot.set_qf(qf)
            scene.step()
            
        if mp.controller.viewer:
            mp.controller.viewer.render()
            
    cv2.destroyWindow("IBVS Continuous View")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Panda visual-servo pick-and-place demo.")
    parser.add_argument("--save-video", action="store_true", help="Save the global recording camera to output.avi.")
    args = parser.parse_args()

    controller = Controller(save_video=args.save_video)
    
    # 1. 机械臂底座归零
    ROBOT_BASE_Z = 0.0
    panda_config["position"] = [0.0, 0.0, ROBOT_BASE_Z]
    robot = controller.add_robot(panda_config)
    
    # ==========================================
    # 🛠️ 终极修复：彻底消灭 link5 初始碰撞报错
    # ==========================================
    # 定义安全伸展位 (最后两个 0.04 是夹爪完全张开)
    safe_home_qpos = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]
    
    # 第一步：瞬间掰到安全位置
    robot.set_qpos(safe_home_qpos)
    
    # 第二步：把底层的电机目标值也强行设为这个安全位置！防止夹爪乱缩
    for i in range(len(robot.active_joints)):
        robot.active_joints[i].set_drive_target(safe_home_qpos[i])
        
    # 第三步：让物理引擎空跑 50 步，彻底稳定下来
    for _ in range(50):
        robot.set_qf(robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))
        controller.scene.step()
    # ==========================================

    
    # 2. 安装手眼摄像头
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

    # 3. 动态随机生成待抓取物块
    # 在 X: [0.35, 0.45], Y: [0.05, 0.25] 的“黄金舒适区”内随机生成
    random_x = round(random.uniform(0.35, 0.45), 3)
    random_y = round(random.uniform(0.05, 0.25), 3)
    
    print(f"🎲 [测试] 本次物块随机坐标为: X={random_x}, Y={random_y}")
    
    block_red_config["position"] = [random_x, random_y, 0.02]
    controller.add_object(block_red_config)
    mp = MotionPlanning(panda_config, controller)
    
    BLOCK_Q = [0, 1, 0, 0] # 夹爪垂直朝下
    SAFE_HEIGHT_Z = 0.50   # 移动过程中的安全高度
    
    print("1. 准备夹爪并移动到物块上方的观察位...")
    mp.open_gripper()
    # 💡 观察位同步修改
    SCAN_X, SCAN_Y = 0.40, 0.10
    mp.move_to_pose(Pose([SCAN_X, SCAN_Y, SAFE_HEIGHT_Z], BLOCK_Q))
    
    print("2. 启动视觉伺服系统...")
    visual_servoing(camera, controller.scene, robot, cam_pose, mp)
    
    print("3. 执行坐标补偿与精确物理抓取...")
    current_p = hand_link.get_entity_pose().p
    
    COMPENSATE_X = 0.05  
    COMPENSATE_Y = 0.00  
    target_x = current_p[0] + COMPENSATE_X
    target_y = current_p[1] + COMPENSATE_Y
    
    # 移动到正上方并张开
    mp.move_to_pose(Pose([target_x, target_y, SAFE_HEIGHT_Z], BLOCK_Q))
    mp.open_gripper()
    # 提速：张开等待 20 -> 10
    for _ in range(10): 
        robot.set_qf(robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))
        controller.scene.step()

    GRASP_Z = 0.125 
    print(f"下潜至抓取高度: Z={GRASP_Z}")
    
    status, target_q = mp.planner.IK(Pose([target_x, target_y, GRASP_Z], BLOCK_Q), robot.get_qpos())
    if status == "Success":
        target_q_flat = np.array(target_q).flatten()
        for idx, arm_joint_idx in enumerate(mp.planner.move_group_joint_indices):
            robot.active_joints[arm_joint_idx].set_drive_target(float(target_q_flat[idx]))
            
     
        # 从 100 加大到 200 步 (约 1 秒)，确保电机100%走到 0.125 的位置
        for _ in range(200):
            robot.set_qf(robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))
            controller.scene.step()
            if mp.controller.viewer: mp.controller.viewer.render()
        
    mp.close_gripper()
    
    # 提速：摩擦力建立时间，200 -> 100 步 (约 0.4 秒，足够了)
    for _ in range(50): 
        robot.set_qf(robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))
        controller.scene.step()

    print("4. 提升物块...")
    current_z = GRASP_Z
    while current_z < SAFE_HEIGHT_Z:
        # 💡 核心修复 2：加大提速步长，0.02 -> 0.05 (每次提 5 厘米)
        current_z += 0.05  
        # 防止最后一步超过安全高度
        if current_z > SAFE_HEIGHT_Z: current_z = SAFE_HEIGHT_Z 
        
        status, target_q = mp.planner.IK(Pose([target_x, target_y, current_z], BLOCK_Q), robot.get_qpos())
        
        if status == "Success":
            target_q_flat = np.array(target_q).flatten()
            for idx, arm_joint_idx in enumerate(mp.planner.move_group_joint_indices):
                robot.active_joints[arm_joint_idx].set_drive_target(float(target_q_flat[idx]))
                
        # 提速：单次微步等待 30 -> 15 步
        for _ in range(15):
            robot.set_qf(robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))
            controller.scene.step()
            if mp.controller.viewer: mp.controller.viewer.render()

    print("✅ 成功拔起，开始平移...")
    
    print("5. 携带物块移动到指定的放置点...")
    DROP_X, DROP_Y = 0.4, -0.2
    target_drop_safe = Pose([DROP_X, DROP_Y, SAFE_HEIGHT_Z], BLOCK_Q)
    mp.move_to_pose(target_drop_safe) 
    
    print("6. 下降并松开夹爪...")
    mp.move_to_pose(Pose([DROP_X, DROP_Y, 0.13], BLOCK_Q))
    mp.open_gripper()
    
    # 提速：松开等待 30 -> 15
    for _ in range(15):
        robot.set_qf(robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))
        controller.scene.step()
        
    mp.move_to_pose(Pose([DROP_X, DROP_Y, SAFE_HEIGHT_Z], BLOCK_Q))
    
    if controller.out is not None:
        controller.out.release()
    controller.visualize(robot)
