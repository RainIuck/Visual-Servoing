import sapien
import cv2
import time
import numpy as np

from src.vpg_bridge.storage_bin import DEFAULT_STORAGE_BIN, add_storage_bin_to_scene

class Controller:
    def __init__(self, *, save_video: bool = False, video_path: str = "output.avi"):
        self.save_video = save_video
        self.scene = sapien.Scene()
        self.scene.add_ground(0)

        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

        self.viewer = self.scene.create_viewer()
        self.viewer.set_camera_xyz(x=0.4, y=1.5, z=0.5)
        self.viewer.set_camera_rpy(r=0, p=-0.2, y=-1.57)
        
        near, far = 0.1, 100
        width, height = 640, 480

        # Compute the recording camera pose by specifying forward(x), left(y) and up(z).
        # The view is a high oblique angle that covers the robot, VPG workspace,
        # and reachable discard row without looking through the arm.
        cam_pos = np.array([0.85, -1.65, 1.65], dtype=np.float32)
        cam_target = np.array([0.36, -0.08, 0.18], dtype=np.float32)
        forward = cam_target - cam_pos
        forward = forward / np.linalg.norm(forward)
        left = np.cross([0, 0, 1], forward)
        left = left / np.linalg.norm(left)
        up = np.cross(forward, left)
        mat44 = np.eye(4)
        mat44[:3, :3] = np.stack([forward, left, up], axis=1)
        mat44[:3, 3] = cam_pos

        self.camera = self.scene.add_camera(
            name="camera",
            width=width,
            height=height,
            fovy=np.deg2rad(55),
            near=near,
            far=far,
        )
        self.camera.entity.set_pose(sapien.Pose(mat44))
        
        self.out = None
        if self.save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
        
    def add_entity(self, config, fix_root_link):
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = fix_root_link
        entities = loader.load_multiple(config['urdf_path'])
        
       
        #entities = loader.load_multiple(config['urdf_path'])
        
        # 设置姿态
        pose = sapien.Pose(config['position'], config['orientation'])
        if fix_root_link:
            entities[0][0].set_root_pose(pose)
        else:
            entities[0][0].set_pose(pose)
        
        return entities[0][0]

    def add_robot(self, robot_config):
        self.robot = self.add_entity(robot_config, fix_root_link=True)
        self.active_joints = self.robot.get_active_joints()
        for joint in self.active_joints:
            joint.set_drive_property(
                stiffness= 1000,
                damping=200,
            )
        return self.robot

    def add_object(self, object_config):
        return self.add_entity(object_config, fix_root_link=False)

    def add_storage_bin(self, config=DEFAULT_STORAGE_BIN):
        return add_storage_bin_to_scene(self.scene, config)
       
    def set_robot_pose(self,object,arm):
        # arm_init_qpos = [4.71, 2.84, 0, 0.75, 4.62, 4.48, 4.88]
        # gripper_init_qpos = [0, 0, 0, 0, 0, 0]
        # init_qpos = arm_init_qpos + gripper_init_qpos

        qpos = arm 
        object.set_qpos(qpos)
            
    def visualize(self,robot):
        while not self.viewer.closed:
            for _ in range(4):  # render every 4 steps
                self.scene.step()
                qf = robot.compute_passive_force(
                    gravity=True,
                    coriolis_and_centrifugal=False,
                )
                robot.set_qf(qf)
            self.scene.update_render()
            self.viewer.render()
            
