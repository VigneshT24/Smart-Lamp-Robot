from pathlib import Path
import time

import mujoco as m # type: ignore
import mujoco.viewer # type: ignore


class LampSimulator:
    def __init__(self):
        urdf_path = (Path(__file__).resolve().parent.parent / "robot" / "dummy_lamp_5dof.urdf")

        self.model = m.MjModel.from_xml_path(str(urdf_path))

        self.data = m.MjData(self.model)

        self.light_geom_ids = []

        for i in range(self.model.ngeom):
            rgba = self.model.geom_rgba[i]

            if (abs(rgba[0] - 1.0) < 0.01 and abs(rgba[1] - 0.95) < 0.01  and abs(rgba[2] - 0.76) < 0.01):
                self.light_geom_ids.append(i)

        print("Light geoms:", self.light_geom_ids)

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)

        self.joints = {}

        for i in range(self.model.njnt):
            name = m.mj_id2name(self.model, m.mjtObj.mjOBJ_JOINT, i)

            self.joints[name] = i
            print(i, name)

    def set_light(self, on):
        with self.viewer.lock():
            for geom_id in self.light_geom_ids:
                if on:
                    self.model.geom_rgba[geom_id] = [1.0, 0.95, 0.25, 1.0]
                else:
                    self.model.geom_rgba[geom_id] = [0.08, 0.08, 0.08, 1.0]

        self.viewer.sync()

    # setting angle to a specific joint
    def set_joint(self, name, angle):
        joint_id = self.joints[name]

        qpos_index = self.model.jnt_qposadr[joint_id]
        self.data.qpos[qpos_index] = angle

        m.mj_forward(self.model, self.data)
        self.viewer.sync()

    def step(self, seconds=1.0):
        start = time.time()

        while time.time() - start < seconds:
            m.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(self.model.opt.timestep)

    def perform_action(self, action):
        if action == "LOOK_LEFT":
            self.move_pose({"base_yaw_joint": 0.8, "neck_yaw_joint": 0.5}, 0.6)

        elif action == "LOOK_RIGHT":
            self.move_pose({"base_yaw_joint": -0.8, "neck_yaw_joint": -0.5}, 0.6)

        elif action == "LOOK_CENTER":
            self.move_pose({"base_yaw_joint": 0.0, "neck_yaw_joint": 0.0}, 0.6)

        elif action == "NOD":
            self.move_pose({"head_pitch_joint": 0.35}, 0.25)
            self.move_pose({"head_pitch_joint": -0.25}, 0.25)

        elif action == "LOOK_UPPER_RIGHT":
            self.move_pose({"base_yaw_joint": -1.2, "neck_yaw_joint": 0.7, "head_pitch_joint": -0.4}, 0.6)

        elif action == "LOOK_UPPER_LEFT":
            self.move_pose({"base_yaw_joint": 1.2, "neck_yaw_joint": -0.7, "head_pitch_joint": -0.4}, 0.6)

        elif action == "LOOK_LOWER_RIGHT":
            self.move_pose({"base_yaw_joint": -1.2, "neck_yaw_joint": 0.7, "head_pitch_joint": 0.1}, 0.6)

        elif action == "LOOK_LOWER_LEFT":
            self.move_pose({"base_yaw_joint": 1.2, "neck_yaw_joint": -0.7, "head_pitch_joint": 0.1}, 0.6)

        elif action == "LOOK_UP":
            self.move_pose({"base_yaw_joint": 0.0, "neck_yaw_joint": 0.0, "head_pitch_joint": -0.4,}, 0.6)

        elif action == "LOOK_DOWN":
            self.move_pose({"base_yaw_joint": 0.0, "neck_yaw_joint": 0.0, "head_pitch_joint": 0.25,}, 0.6)

    # move multiple joints at once
    def move_pose(self, pose, seconds=1.0):
        start_positions = {}

        # save current joint positions
        for name, target_angle in pose.items():
            joint_id = self.joints[name]
            qpos_index = self.model.jnt_qposadr[joint_id]

            start_positions[name] = self.data.qpos[qpos_index]

        steps = int(seconds / self.model.opt.timestep)

        # smooth interpolation
        for step in range(steps):
            t = (step + 1) / steps

            for name, target_angle in pose.items():
                joint_id = self.joints[name]
                qpos_index = self.model.jnt_qposadr[joint_id]

                start_angle = start_positions[name]

                new_angle = start_angle + (target_angle - start_angle) * t

                self.data.qpos[qpos_index] = new_angle

            m.mj_forward(self.model, self.data)
            self.viewer.sync()

            time.sleep(self.model.opt.timestep)

    def close(self):
        self.viewer.close()