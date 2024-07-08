# Copyright 2024 @with-RL
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Note: For kinematics testing
#     - modes of Joint_01 must be set to Kinematic mode.
#     - dynamic of Link_01 must be set to false.

from dataclasses import dataclass
import numpy as np
from scipy.optimize import fsolve
from pynput.keyboard import Key, Listener

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


@dataclass
class Control:
    O_0: tuple = (0, 1)  # 중심 점
    l_1: float = 0.5
    theta_1: float = -np.pi / 3  # 각도: theta_1 (계산이 필요한 값)
    C_0: tuple = (0, 0.5)  # 좌표


class Penduleum1D:
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.require("sim")
        self.control = Control()

    def fk(self, theta_1, l_1):
        H_01 = np.array(
            [
                [np.cos(theta_1), -np.sin(theta_1), self.control.O_0[0]],
                [np.sin(theta_1), np.cos(theta_1), self.control.O_0[1]],
                [0, 0, 1],
            ]
        )
        return np.matmul(H_01, np.array([l_1, 0, 1]).T)[:-1]

    def ik(self, theta, params):
        (theta_1,) = theta
        l_1, C_ref = params
        C_0 = self.fk(theta_1, l_1)
        J_1 = np.array(
            [
                -self.control.l_1 * np.sin(theta_1),
                self.control.l_1 * np.cos(theta_1),
            ]
        )
        # avoid inf
        if J_1[0] == 0:
            J_1[0] = -1e6
        if J_1[1] == 0:
            J_1[1] = -1e6

        J_1inv = 1 / J_1
        dX = np.array([C_ref[0] - C_0[0], C_ref[1] - C_0[1]])
        dq = J_1inv.dot(dX)
        dq %= 2 * np.pi

        return (theta_1 + dq / 2,)

    def on_press(self, key):
        if key == Key.space:
            # calculate random position
            theta_1 = self.control.theta_1 + 0.1
            self.control.C_0 = (
                self.control.l_1 * np.cos(theta_1) + self.control.O_0[0],
                self.control.l_1 * np.sin(theta_1) + self.control.O_0[1],
            )
            # calc inverse kinematics
            theta = self.ik(
                [self.control.theta_1], [self.control.l_1, self.control.C_0]
            )
            self.control.theta_1 = theta[0]

    def init_coppelia(self):
        self.joint_01 = self.sim.getObject("/Joint_01")
        self.dummy = self.sim.getObject("/Dummy")

    def read_dummy(self):
        return self.sim.getObjectPosition(self.dummy)

    def control_joint(self):
        self.sim.setJointTargetPosition(self.joint_01, self.control.theta_1)

    def run_coppelia(self, sec):
        # key input
        Listener(on_press=self.on_press).start()
        # start simulation
        self.sim.setStepping(True)
        self.sim.startSimulation()
        while (t := self.sim.getSimulationTime()) < sec:
            # control joint
            self.control_joint()
            # dummy postion
            real = self.read_dummy()
            print(f"real={real[1:]}, fk={self.control.C_0}")
            # step
            self.sim.step()
        self.sim.stopSimulation()


if __name__ == "__main__":
    client = Penduleum1D()
    client.init_coppelia()
    client.run_coppelia(100)
