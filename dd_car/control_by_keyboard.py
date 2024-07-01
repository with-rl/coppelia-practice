from dataclasses import dataclass
import numpy as np
from pynput.keyboard import Key, Listener

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


@dataclass
class Control:
    wheel_L: float = 0
    wheel_R: float = 0


class DDCar:
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.require("sim")
        self.control = Control()

    def on_press(self, key):
        delta = 0.5
        if key == Key.up:
            self.control.wheel_L += delta
            self.control.wheel_R += delta
        if key == Key.down:
            self.control.wheel_L -= delta
            self.control.wheel_R -= delta
        if key == Key.left:
            self.control.wheel_L -= delta
            self.control.wheel_R += delta
        if key == Key.right:
            self.control.wheel_L += delta
            self.control.wheel_R -= delta
        self.control.wheel_L = min(max(self.control.wheel_L, -5), 5)
        self.control.wheel_R = min(max(self.control.wheel_R, -5), 5)

    def init_coppelia(self):
        self.joint_left = self.sim.getObject("/Joint_left")
        self.joint_right = self.sim.getObject("/Joint_right")

        # velocity control mode
        self.sim.setObjectInt32Param(
            self.joint_left,
            self.sim.jointintparam_dynctrlmode,
            self.sim.jointdynctrl_velocity,
        )
        self.sim.setObjectInt32Param(
            self.joint_right,
            self.sim.jointintparam_dynctrlmode,
            self.sim.jointdynctrl_velocity,
        )

    def control_car(self):
        self.sim.setJointTargetVelocity(self.joint_left, self.control.wheel_L)
        self.sim.setJointTargetVelocity(self.joint_right, self.control.wheel_R)

    def run_coppelia(self, sec):
        # key input
        Listener(on_press=self.on_press).start()
        # start simulation
        self.sim.setStepping(True)
        self.sim.startSimulation()
        while (t := self.sim.getSimulationTime()) < sec:
            # velocity control
            self.control_car()
            # step
            self.sim.step()
        self.sim.stopSimulation()


if __name__ == "__main__":
    client = DDCar()
    client.init_coppelia()
    client.run_coppelia(100)
