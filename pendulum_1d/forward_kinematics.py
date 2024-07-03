# Copyright 2024 @with-RL
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Note: For kinematics testing, dynamic of Link_01 must be set to false.

from dataclasses import dataclass
import numpy as np
from pynput.keyboard import Key, Listener

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


@dataclass
class Control:
    O_0: tuple = (0, 1)
    theta_1: float = -np.pi / 2
    C_1: tuple = (0.5, 0, 1)
    C_0: tuple = (0, 0, 1)  # 계산이 필요한 값


class Penduleum1D:
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.require("sim")
        self.control = Control()
        self.fk()

    def fk(self):
        H_01 = np.array(
            [
                [np.cos(self.control.theta_1), -np.sin(self.control.theta_1), 0],
                [np.sin(self.control.theta_1), np.cos(self.control.theta_1), 1],
                [0, 0, 1],
            ]
        )
        self.control.C_0 = np.matmul(H_01, self.control.C_1)

    def on_press(self, key):
        if key == Key.space:
            self.control.theta_1 = (np.random.random() * 2 - 1) * np.pi
            self.fk()

    def init_coppelia(self):
        self.joint_01 = self.sim.getObject("/Joint_01")
        self.dummy = self.sim.getObject("/Dummy")

    def read_dummy(self):
        return self.sim.getObjectPosition(self.dummy)

    def control_joint(self):
        self.sim.setJointPosition(self.joint_01, self.control.theta_1)

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
            print(f"real={real[1:]}, fk={self.control.C_0[:-1]}")
            # step
            self.sim.step()
        self.sim.stopSimulation()


if __name__ == "__main__":
    client = Penduleum1D()
    client.init_coppelia()
    client.run_coppelia(100)
