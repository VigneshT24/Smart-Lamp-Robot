from pathlib import Path
import time

import mujoco as m # type: ignore
import mujoco.viewer # type: ignore


class LampSimulator:
    def __init__(self):
        urdf_path = (Path(__file__).resolve().parent.parent / "robot" / "dummy_lamp_5dof.urdf")

        self.model = m.MjModel.from_xml_path(str(urdf_path))
        self.data = m.MjData(self.model)

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)

        self.joints = {}

        for i in range(self.model.njnt):
            name = m.mj_id2name(self.model, m.mjtObj.mjOBJ_JOINT, i)

            self.joints[name] = i
            print(i, name)

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