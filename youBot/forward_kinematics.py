# Copyright 2024 @with-RL
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import matplotlib.pyplot as plt

from youBot import YouBot


class FKBot(YouBot):
    def __init__(self):
        super().__init__()
        self.fk = FK()
        self.angles = np.concatenate(
            (
                np.full(501, -np.pi / 2000),
                np.full(1001, np.pi / 2000),
            )
        )

    def init_coppelia(self):
        super().init_coppelia()
        # reference
        self.gripper_ref = self.sim.getObject("/Gripper_ref")

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

        # gripper ref
        O = self.sim.getObjectPosition(self.gripper_ref)
        R = 0
        joints.append((np.array(O), R))

        return joints

    def run_step(self, count):
        if count >= len(self.angles):
            self.run_flag = False
            return
        r = self.angles[count]
        self.control.arm_0 += r
        self.control.arm_1 -= r
        self.control.arm_2 += r
        self.control.arm_3 -= r
        self.control.arm_4 += r
        self.control.vel_X = 0.25
        self.control.vel_Z = 1.0
        # car control
        self.control_car()
        # arm control
        self.control_arm()
        # read joints
        joints = self.read_joints()
        # update FK
        self.fk.update(joints, count == (len(self.angles) - 1))  # save at last


class FK:
    A4 = np.array([-0.00052, 1.19959, 0.63208, 1])
    GR = np.array([-0.00052, 1.19959, 0.73208, 1])

    D_CA_A0 = np.array([0.44122, 0.00000, 0.14467, 1])
    D_A0_A1 = np.array([0.03301, -0.03945, 0.10123, 1])
    D_A1_A2 = np.array([-0.0001, 0.009, 0.155, 1])
    D_A2_A3 = np.array([-0.00005, 0.05000, 0.13485, 1])
    D_A3_A4 = np.array([0.00055, -0.01903, 0.09633, 1])
    D_A4_GR = np.array([0.00000, 0.00000, 0.10000, 1])

    def __init__(self):
        self.data = np.zeros((10240, 3))  # default values
        self.idx = 0
        # plot object
        self.plt_objects = [None] * 15

    def update(self, joints, save):
        OGR, OGR_hat = self.fk(joints)
        self.visualize(OGR, OGR_hat, save)

    def fk(self, joints):
        print(FK.GR - FK.A4)
        OCA = joints[0][0]
        RCA = joints[0][1]  # OA0 = joints[1][0]
        RA0 = joints[1][1]  # OA1 = joints[2][0]
        RA1 = joints[2][1]  # OA2 = joints[3][0]
        RA2 = joints[3][1]  # OA3 = joints[4][0]
        RA3 = joints[4][1]  # OA4 = joints[5][0]
        RA4 = joints[5][1]
        OGR = joints[6][0]  # RGR = joints[6][1]

        H_O0_CR = np.array(
            [
                [np.cos(RCA), -np.sin(RCA), 0, OCA[0]],
                [np.sin(RCA), np.cos(RCA), 0, OCA[1]],
                [0, 0, 1, OCA[2]],
                [0, 0, 0, 1],
            ]
        )
        H_CR_A0 = np.array(
            [
                [np.cos(RA0), -np.sin(RA0), 0, FK.D_CA_A0[0]],
                [np.sin(RA0), np.cos(RA0), 0, FK.D_CA_A0[1]],
                [0, 0, 1, FK.D_CA_A0[2]],
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
    client = FKBot()
    client.init_coppelia()
    client.run_coppelia()
