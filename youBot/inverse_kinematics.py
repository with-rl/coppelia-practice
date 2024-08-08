# Copyright 2024 @with-RL
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

from youBot import YouBot


class IKBot(YouBot):
    def __init__(self):
        super().__init__()
        self.ik = IK()
        self.target_thetas = None

    def init_coppelia(self):
        super().init_coppelia()
        # reference
        self.target = self.sim.getObject("/Target")

    def read_ref(self):
        OCA = self.sim.getObjectPosition(self.youBot_ref)
        RCA = self.sim.getObjectOrientation(self.youBot_ref)[2]
        return np.array(OCA), RCA

    def read_joints(self):
        joints = []
        # arm joints
        for arm in self.arms:
            R = self.sim.getJointPosition(arm)
            joints.append(R)

        return joints

    def read_target(self):
        OTR = self.sim.getObjectPosition(self.target)
        return np.array(OTR)

    def run_step(self, count):
        # car control
        self.control_car()
        # arm control
        self.control_arm()

        if self.target_thetas is None:
            OCA, RCA = self.read_ref()
            thetas = self.read_joints()
            OTR = self.read_target()
            self.target_thetas = self.ik.solve(thetas, OCA, RCA, OTR)
            return

        diff = self.control.arm_0 - self.target_thetas[0]
        self.control.arm_0 -= min(0.01, max(-0.01, diff))

        diff = self.control.arm_1 - self.target_thetas[1]
        self.control.arm_1 -= min(0.01, max(-0.01, diff))

        diff = self.control.arm_2 - self.target_thetas[2]
        self.control.arm_2 -= min(0.01, max(-0.01, diff))

        diff = self.control.arm_3 - self.target_thetas[3]
        self.control.arm_3 -= min(0.01, max(-0.01, diff))


class IK:
    D_CR_A0 = np.array([0.44122, 0.00000, 0.14467, 1])
    D_A0_A1 = np.array([0.03301, -0.03945, 0.10123, 1])
    D_A1_A2 = np.array([-0.0001, 0.009, 0.155, 1])
    D_A2_A3 = np.array([-0.00005, 0.05000, 0.13485, 1])
    D_A3_A4 = np.array([0.00055, -0.01903, 0.09633, 1])
    D_A4_GR = np.array([0.00000, 0.00000, 0.10000, 1])

    def __init__(self):
        pass

    def solve(self, thetas, OCA, RCA, OTR):
        target_thetas = fsolve(
            self.ik,
            thetas,
            [OCA, RCA, OTR],
        )
        return target_thetas

    def fk(self, theta, params):
        RA0, RA1, RA2, RA3, RA4 = theta
        OCA, RCA, _ = params

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
                [np.cos(RA0), -np.sin(RA0), 0, IK.D_CR_A0[0]],
                [np.sin(RA0), np.cos(RA0), 0, IK.D_CR_A0[1]],
                [0, 0, 1, IK.D_CR_A0[2]],
                [0, 0, 0, 1],
            ]
        )
        H_O0_A0 = H_O0_CR @ H_CR_A0
        H_A0_A1 = np.array(
            [
                [np.cos(RA1), 0, -np.sin(RA1), IK.D_A0_A1[0]],
                [0, 1, 0, IK.D_A0_A1[1]],
                [np.sin(RA1), 0, np.cos(RA1), IK.D_A0_A1[2]],
                [0, 0, 0, 1],
            ]
        )
        H_O0_A1 = H_O0_A0 @ H_A0_A1
        H_A1_A2 = np.array(
            [
                [np.cos(RA2), 0, -np.sin(RA2), IK.D_A1_A2[0]],
                [0, 1, 0, IK.D_A1_A2[1]],
                [np.sin(RA2), 0, np.cos(RA2), IK.D_A1_A2[2]],
                [0, 0, 0, 1],
            ]
        )
        H_O0_A2 = H_O0_A1 @ H_A1_A2
        H_A2_A3 = np.array(
            [
                [np.cos(RA3), 0, -np.sin(RA3), IK.D_A2_A3[0]],
                [0, 1, 0, IK.D_A2_A3[1]],
                [np.sin(RA3), 0, np.cos(RA3), IK.D_A2_A3[2]],
                [0, 0, 0, 1],
            ]
        )
        H_O0_A3 = H_O0_A2 @ H_A2_A3
        H_A3_A4 = np.array(
            [
                [np.cos(RA4), -np.sin(RA4), 0, IK.D_A3_A4[0]],
                [np.sin(RA4), np.cos(RA4), 0, IK.D_A3_A4[1]],
                [0, 0, 1, IK.D_A3_A4[2]],
                [0, 0, 0, 1],
            ]
        )
        H_O0_A4 = H_O0_A3 @ H_A3_A4
        return (H_O0_A4 @ IK.D_A4_GR)[:-1]

    def ik(self, thetas, params):
        _, _, OTR = params
        OTR_hat = self.fk(thetas, params)
        return np.linalg.norm(OTR_hat - OTR), 0, 0, 0, 0


if __name__ == "__main__":
    client = IKBot()
    client.init_coppelia()
    client.run_coppelia()
