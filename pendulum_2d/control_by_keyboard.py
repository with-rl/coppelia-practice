# Copyright 2024 @with-RL
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import dataclass
from pynput.keyboard import Key, Listener

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


@dataclass
class Control:
    tau1: float = 0
    tau2: float = 0


class Penduleum1D:
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.require("sim")
        self.control = Control()

    def on_press(self, key):
        if key == Key.left:
            self.control.tau1 -= 1
        if key == Key.right:
            self.control.tau1 += 1
        self.control.tau1 = min(max(self.control.tau1, -10), 10)

        if key == Key.up:
            self.control.tau2 -= 1
        if key == Key.down:
            self.control.tau2 += 1
        self.control.tau2 = min(max(self.control.tau2, -10), 10)

    def init_coppelia(self):
        self.joint1 = self.sim.getObject("/Joint_01")
        self.joint2 = self.sim.getObject("/Joint_02")

    def control_joint(self):
        self.sim.setJointTargetForce(self.joint1, self.control.tau1)
        self.sim.setJointTargetForce(self.joint2, self.control.tau2)

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
