# Copyright 2024 @with-RL
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import dataclass
from pynput.keyboard import Key, Listener

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


@dataclass
class Control:
    force_1: float = 0


class Penduleum1D:
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.require("sim")
        self.control = Control()

    def on_press(self, key):
        if key == Key.left:
            self.control.force_1 -= 1
        if key == Key.right:
            self.control.force_1 += 1
        self.control.force_1 = min(max(self.control.force_1, -10), 10)

    def init_coppelia(self):
        self.joint_01 = self.sim.getObject("/Joint_01")

    def control_joint(self):
        self.sim.setJointTargetForce(self.joint_01, self.control.force_1)

    def run_coppelia(self, sec):
        # key input
        Listener(on_press=self.on_press).start()
        # start simulation
        self.sim.setStepping(True)
        self.sim.startSimulation()
        while (t := self.sim.getSimulationTime()) < sec:
            # control joint
            self.control_joint()
            # step
            self.sim.step()
        self.sim.stopSimulation()


if __name__ == "__main__":
    client = Penduleum1D()
    client.init_coppelia()
    client.run_coppelia(100)
