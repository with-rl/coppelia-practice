# Copyright 2024 @with-RL
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from pynput import keyboard
from pynput.keyboard import Key, Listener

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


@dataclass
class Control:
    vel_X: float = 0
    vel_Y: float = 0
    vel_Z: float = 0
    arm_0: float = 0
    arm_1: float = 0
    arm_2: float = 0
    arm_3: float = 0
    arm_4: float = 0
    gripper: float = -0.04


class YouBot:
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.require("sim")
        self.control = Control()
        self.fk = FK()

    def on_press(self, key):
        delta = 1.0
        if key == Key.up:
            self.control.vel_X += delta
            self.control.vel_Z = 0
        if key == Key.down:
            self.control.vel_X -= delta
            self.control.vel_Z = 0
        if key == Key.left:
            self.control.vel_X = 0
            self.control.vel_Z += delta
        if key == Key.right:
            self.control.vel_X = 0
            self.control.vel_Z -= delta
        self.control.vel_X = min(max(self.control.vel_X, -10), 10)
        self.control.vel_Y = min(max(self.control.vel_Y, -5), 5)
        self.control.vel_Z = min(max(self.control.vel_Z, -np.pi), np.pi)

        delta = 0.1
        if key == keyboard.KeyCode.from_char("q"):
            self.control.arm_0 += delta
        if key == keyboard.KeyCode.from_char("a"):
            self.control.arm_0 -= delta
        self.control.arm_0 = min(max(self.control.arm_0, -np.pi), np.pi)

        if key == keyboard.KeyCode.from_char("w"):
            self.control.arm_1 += delta
        if key == keyboard.KeyCode.from_char("s"):
            self.control.arm_1 -= delta
        self.control.arm_1 = min(max(self.control.arm_1, -np.pi), np.pi)

        if key == keyboard.KeyCode.from_char("e"):
            self.control.arm_2 += delta
        if key == keyboard.KeyCode.from_char("d"):
            self.control.arm_2 -= delta
        self.control.arm_2 = min(max(self.control.arm_2, -np.pi), np.pi)

        if key == keyboard.KeyCode.from_char("r"):
            self.control.arm_3 += delta
        if key == keyboard.KeyCode.from_char("f"):
            self.control.arm_3 -= delta
        self.control.arm_3 = min(max(self.control.arm_3, -np.pi), np.pi)

        if key == keyboard.KeyCode.from_char("t"):
            self.control.arm_4 += delta
        if key == keyboard.KeyCode.from_char("g"):
            self.control.arm_4 -= delta
        self.control.arm_4 = min(max(self.control.arm_4, -np.pi), np.pi)

        delta = 0.01
        if key == keyboard.KeyCode.from_char("y"):
            self.control.gripper += delta
        if key == keyboard.KeyCode.from_char("h"):
            self.control.gripper -= delta
        self.control.gripper = min(max(self.control.gripper, -0.04), 0.04)

    def init_coppelia(self):
        # Wheel Joints: front left, rear left, rear right, front right
        self.wheels = []
        self.wheels.append(self.sim.getObject("/rollingJoint_fl"))
        self.wheels.append(self.sim.getObject("/rollingJoint_rl"))
        self.wheels.append(self.sim.getObject("/rollingJoint_fr"))
        self.wheels.append(self.sim.getObject("/rollingJoint_rr"))
        # Arm Joints
        self.arms = []
        for i in range(5):
            self.arms.append(self.sim.getObject(f"/youBotArmJoint{i}"))
        # Gripper Joint
        self.gripper = self.sim.getObject(f"/youBotGripperJoint2")
        # reference
        self.youBot_ref = self.sim.getObject("/youBot_ref")

    def control_car(self):
        self.sim.setJointTargetVelocity(
            self.wheels[0],
            -self.control.vel_X + self.control.vel_Z,
        )
        self.sim.setJointTargetVelocity(
            self.wheels[1],
            -self.control.vel_X + self.control.vel_Z,
        )
        self.sim.setJointTargetVelocity(
            self.wheels[2],
            -self.control.vel_X - self.control.vel_Z,
        )
        self.sim.setJointTargetVelocity(
            self.wheels[3],
            -self.control.vel_X - self.control.vel_Z,
        )

    def control_arm(self):
        self.sim.setJointTargetPosition(self.arms[0], self.control.arm_0)
        self.sim.setJointTargetPosition(self.arms[1], self.control.arm_1)
        self.sim.setJointTargetPosition(self.arms[2], self.control.arm_2)
        self.sim.setJointTargetPosition(self.arms[3], self.control.arm_3)
        self.sim.setJointTargetPosition(self.arms[4], self.control.arm_4)

    def control_gripper(self):
        self.sim.setJointTargetVelocity(self.gripper, self.control.gripper)

    def read_joints(self):
        joints = []
        # car ref
        O = self.sim.getObjectPosition(self.youBot_ref)
        R = self.sim.getObjectOrientation(self.youBot_ref)[2]
        joints.append((np.array(O), R))

        # arm joints
        for arm in self.arms:
            O = self.sim.getObjectPosition(arm)
            R = self.sim.getJointPosition(arm)
            joints.append((np.array(O), R))

        # gripper joint
        O = self.sim.getObjectPosition(self.gripper)
        R = self.sim.getJointPosition(self.gripper)
        joints.append((np.array(O), R))

        return joints

    def run_coppelia(self, sec):
        # key input
        Listener(on_press=self.on_press).start()
        # start simulation
        self.sim.setStepping(True)
        self.sim.startSimulation()
        arr = np.concatenate(
            (
                np.linspace(0, -np.pi / 4, 501),
                np.linspace(-np.pi / 4, np.pi / 4, 1001),
            )
        )
        for i, r in enumerate(arr):
            self.control.arm_0 = r
            self.control.arm_1 = r
            self.control.arm_2 = r
            self.control.arm_3 = r
            self.control.arm_4 = r
            self.control.vel_X = 0.25
            self.control.vel_Z = 1.0
            # car control
            self.control_car()
            # arm control
            self.control_arm()
            # read youBot_ref
            joints = self.read_joints()
            # update FK
            self.fk.update(joints, i == (len(arr) - 1))  # save at last
            # step
            self.sim.step()
        self.sim.stopSimulation()


class FK:
    D_CR_A0 = np.array([0.44122, 0.00000, 0.14467, 1])
    D_A0_A1 = np.array([0.03301, -0.03945, 0.10123, 1])
    D_A1_A2 = np.array([-0.0001, 0.009, 0.155, 1])
    D_A2_A3 = np.array([-0.00005, 0.05000, 0.13485, 1])
    D_A3_A4 = np.array([0.00055, -0.01903, 0.09633, 1])
    D_A4_GR = np.array([-0.00011, -0.02452, 0.09734, 1])

    def __init__(self):
        self.data = np.zeros((10240, 3))  # default values
        self.idx = 0
        # plot object
        self.plt_objects = [None] * 15

    def update(self, joints, save):
        OGR, OGR_hat = self.fk(joints)
        self.visualize(OGR, OGR_hat, save)

    def fk(self, joints):
        OCR = joints[0][0]
        RCR = joints[0][1]  # OA0 = joints[1][0]
        RA0 = joints[1][1]  # OA1 = joints[2][0]
        RA1 = joints[2][1]  # OA2 = joints[3][0]
        RA2 = joints[3][1]  # OA3 = joints[4][0]
        RA3 = joints[4][1]  # OA4 = joints[5][0]
        RA4 = joints[5][1]
        OGR = joints[6][0]  # RGR = joints[6][1]

        H_O0_CR = np.array(
            [
                [np.cos(RCR), -np.sin(RCR), 0, OCR[0]],
                [np.sin(RCR), np.cos(RCR), 0, OCR[1]],
                [0, 0, 1, OCR[2]],
                [0, 0, 0, 1],
            ]
        )
        H_CR_A0 = np.array(
            [
                [np.cos(RA0), -np.sin(RA0), 0, FK.D_CR_A0[0]],
                [np.sin(RA0), np.cos(RA0), 0, FK.D_CR_A0[1]],
                [0, 0, 1, FK.D_CR_A0[2]],
                [0, 0, 0, 1],
            ]
        )
        H_O0_A0 = H_O0_CR @ H_CR_A0
        H_A0_A1 = np.array(
            [
                [np.cos(RA1), 0, -np.sin(RA1), FK.D_A0_A1[0]],
                [0, 1, 0, FK.D_A0_A1[1]],
                [np.sin(RA1), 0, np.cos(RA1), FK.D_A0_A1[2]],
                [0, 0, 0, 1],
            ]
        )
        H_O0_A1 = H_O0_A0 @ H_A0_A1
        H_A1_A2 = np.array(
            [
                [np.cos(RA2), 0, -np.sin(RA2), FK.D_A1_A2[0]],
                [0, 1, 0, FK.D_A1_A2[1]],
                [np.sin(RA2), 0, np.cos(RA2), FK.D_A1_A2[2]],
                [0, 0, 0, 1],
            ]
        )
        H_O0_A2 = H_O0_A1 @ H_A1_A2
        H_A2_A3 = np.array(
            [
                [np.cos(RA3), 0, -np.sin(RA3), FK.D_A2_A3[0]],
                [0, 1, 0, FK.D_A2_A3[1]],
                [np.sin(RA3), 0, np.cos(RA3), FK.D_A2_A3[2]],
                [0, 0, 0, 1],
            ]
        )
        H_O0_A3 = H_O0_A2 @ H_A2_A3
        H_A3_A4 = np.array(
            [
                [np.cos(RA4), -np.sin(RA4), 0, FK.D_A3_A4[0]],
                [np.sin(RA4), np.cos(RA4), 0, FK.D_A3_A4[1]],
                [0, 0, 1, FK.D_A3_A4[2]],
                [0, 0, 0, 1],
            ]
        )
        H_O0_A4 = H_O0_A3 @ H_A3_A4
        OGR_hat = (H_O0_A4 @ FK.D_A4_GR)[:3]

        return OGR, OGR_hat

    def visualize(self, OGR, OGR_hat, save):
        # clear object
        for object in self.plt_objects:
            if object:
                object.remove()

        self.data[self.idx] = OGR - OGR_hat
        self.idx += 1

        plt.subplot(3, 1, 1)
        (self.plt_objects[0],) = plt.plot(
            self.data[: self.idx, 0], "-r", label="x error"
        )
        self.plt_objects[1] = plt.legend()
        plt.subplot(3, 1, 2)
        (self.plt_objects[2],) = plt.plot(
            self.data[: self.idx, 1], "-g", label="y error"
        )
        self.plt_objects[3] = plt.legend()
        plt.subplot(3, 1, 3)
        (self.plt_objects[4],) = plt.plot(
            self.data[: self.idx, 2], "-b", label="z error"
        )
        self.plt_objects[5] = plt.legend()
        if save:
            plt.savefig("forward_kinematics.png")
        plt.pause(0.001)


if __name__ == "__main__":
    client = YouBot()
    client.init_coppelia()
    client.run_coppelia(3600)
