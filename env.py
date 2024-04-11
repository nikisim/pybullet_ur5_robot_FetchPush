import time
import math
import random

import numpy as np
import pybullet as p
import pybullet_data

from utilities import Models, Camera
from collections import namedtuple
from attrdict import AttrDict
from tqdm import tqdm


class FailToReachTargetError(RuntimeError):
    pass


class ClutteredPushGrasp:

    SIMULATION_STEP_DELAY = 1 / 240.

    def __init__(self, robot, models: Models, camera=None, vis=False) -> None:
        self.robot = robot
        self.vis = vis
        if self.vis:
            self.p_bar = tqdm(ncols=0, disable=False)
        self.camera = camera

        # define environment
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.planeID = p.loadURDF("plane.urdf")

        self.robot.load()
        self.robot.step_simulation = self.step_simulation

        # custom sliders to tune parameters (name of the parameter,range,initial value)
        self.xin = p.addUserDebugParameter("x", -0.224, 0.224, 0)
        self.yin = p.addUserDebugParameter("y", -0.224, 0.224, 0)
        self.zin = p.addUserDebugParameter("z", 0, 1., 0.5)
        self.rollId = p.addUserDebugParameter("roll", -3.14, 3.14, 0)
        self.pitchId = p.addUserDebugParameter("pitch", -3.14, 3.14, np.pi/2)
        self.yawId = p.addUserDebugParameter("yaw", -np.pi/2, np.pi/2, np.pi/2)
        self.gripper_opening_length_control = p.addUserDebugParameter("gripper_opening_length", 0, 0.085, 0.04)

        self.boxID = p.loadURDF("./urdf/simple-table.urdf",
                                [0.0, 0.0, 0.0])
                                # p.getQuaternionFromEuler([0, 1.5706453, 0]),
                                # p.getQuaternionFromEuler([0, 0, 0]),
                                # useFixedBase=True,
                                # flags=p.URDF_MERGE_FIXED_LINKS | p.URDF_USE_SELF_COLLISION)

        # Load the puck URDF
        # Adjust the basePosition so the puck is on the table. For example, if the table height is 0.75m, you might set z to 0.76m.
        self.puckId = p.loadURDF("./urdf/puck.urdf", basePosition=[0, 0.1, 0.25])

        table_size = (0.5, 0.5)  # Length and width of the table
        table_height = 0.03 + 0.05 / 2

        target_position = self.random_position_on_table(0.5,0.5, table_height)

        # Create a visual shape (red sphere) for the target
        target_visual_shape = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 1])
        self.target_body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=target_visual_shape, basePosition=target_position)

        # Optionally, you can adjust friction properties to make the puck slide more realistically
        p.changeDynamics(self.puckId , -1, lateralFriction=0.5)


        # For calculating the reward
        self.is_success = False

    # Function to generate a random position on the table
    def random_position_on_table(self, table_length, table_width, table_height):
        x = np.random.uniform(-table_length / 2, table_length / 2)
        y = np.random.uniform(-table_width / 2, table_width / 2)
        z = table_height  # Height of the table surface
        return [x, y, z]

    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """
        p.stepSimulation()
        if self.vis:
            time.sleep(self.SIMULATION_STEP_DELAY)
            self.p_bar.update(1)

    def read_debug_parameter(self):
        # read the value of task parameter
        x = p.readUserDebugParameter(self.xin)
        y = p.readUserDebugParameter(self.yin)
        z = p.readUserDebugParameter(self.zin)
        roll = p.readUserDebugParameter(self.rollId)
        pitch = p.readUserDebugParameter(self.pitchId)
        yaw = p.readUserDebugParameter(self.yawId)
        gripper_opening_length = p.readUserDebugParameter(self.gripper_opening_length_control)

        return x, y, z, roll, pitch, yaw, gripper_opening_length

    def step(self, action, control_method='joint'):
        """
        action: (x, y, z, roll, pitch, yaw, gripper_opening_length) for End Effector Position Control
                (a1, a2, a3, a4, a5, a6, a7, gripper_opening_length) for Joint Position Control
        control_method:  'end' for end effector position control
                         'joint' for joint position control
        """
        assert control_method in ('joint', 'end')
        self.robot.move_ee(action[:-1], control_method)
        self.robot.move_gripper(action[-1])
        for _ in range(120):  # Wait for a few steps
            self.step_simulation()

        reward = self.update_reward()

        if reward < 0.05:
            self.is_success = True

        done = True if self.is_success==True else False
        info = dict(is_success=self.is_success)
        return self.get_observation(), reward, done, info

    # Function to calculate Euclidean distance
    def euclidean_distance(self, position1, position2):
        return np.sqrt((position1[0] - position2[0])**2 + (position1[1] - position2[1])**2 + (position1[2] - position2[2])**2)


    def update_reward(self):
        reward = 0
        puck_position, _ = p.getBasePositionAndOrientation(self.puckId)
        target_position, _ = p.getBasePositionAndOrientation(self.target_body)
        
        distance = self.euclidean_distance(puck_position, target_position)
        reward = -distance
        print("Current reward:", reward)
        return reward

    def get_observation(self):
        obs = dict()
        if isinstance(self.camera, Camera):
            rgb, depth, seg = self.camera.shot()
            obs.update(dict(rgb=rgb, depth=depth, seg=seg))
        else:
            assert self.camera is None
        obs.update(self.robot.get_joint_obs())

        return obs

    # def reset_box(self):
    #     p.setJointMotorControl2(self.boxID, 0, p.POSITION_CONTROL, force=1)
    #     p.setJointMotorControl2(self.boxID, 1, p.VELOCITY_CONTROL, force=0)

    def reset(self):
        self.robot.reset()
        self.is_success = False
        # self.reset_box()
        new_target_position_target = self.random_position_on_table(0.5,0.5, 0.03 + 0.05 / 2)
        new_target_position_puck = self.random_position_on_table(0.5,0.5, 0.03 + 0.05 / 2) 
        p.resetBasePositionAndOrientation(self.target_body, new_target_position_target, [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.puckId, new_target_position_puck, [0, 0, 0, 1])
        return self.get_observation()

    def close(self):
        p.disconnect(self.physicsClient)
