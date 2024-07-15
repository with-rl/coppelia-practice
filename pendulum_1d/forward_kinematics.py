# Copyright 2024 @with-RL
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import dataclass
import numpy as np
from pynput.keyboard import Key, Listener

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


@dataclass
class Control:
    O0: tuple = (0, 1)  # 중심 점
    l1: float = 0.5
    theta1: float = -np.pi / 2  # 각도: theta1


class Penduleum1D:
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.require("sim")
        self.control = Control()

    def fk(self, theta1, l1):
        H_01 = np.array(
            [
                [np.cos(theta1), -np.sin(theta1), self.control.O0[0]],
                [np.sin(theta1), np.cos(theta1), self.control.O0[1]],
                [0, 0, 1],
            ]
        )
        return np.matmul(H_01, np.array([l1, 0, 1]).T)[:-1]

    def on_press(self, key):
        if key == Key.space:
            self.control.theta1 = (np.random.random() * 2 - 1) * np.pi

    def init_coppelia(self):
        self.joint_01 = self.sim.getObject("/Joint_01")
        self.dummy = self.sim.getObject("/Dummy")
        self.sim.setObjectInt32Param(
            self.joint_01,
            self.sim.jointintparam_dynctrlmode,
            self.sim.jointdynctrl_position,
        )

    def read_dummy(self):
        return self.sim.getObjectPosition(self.dummy)

    def control_joint(self):
        self.sim.setJointTargetPosition(self.joint_01, self.control.theta1)

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
            C0 = self.fk(self.control.theta1, self.control.l1)
            print(f"real={real[1:]}, fk={C0}")
            # step
            self.sim.step()
        self.sim.stopSimulation()


if __name__ == "__main__":
    client = Penduleum1D()
    client.init_coppelia()
    client.run_coppelia(100)
