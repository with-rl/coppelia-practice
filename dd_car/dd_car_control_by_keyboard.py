import numpy as np
from pynput.keyboard import Key, Listener

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


class DDCar:
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.require("sim")
        self.config = np.zeros(2)

    def on_press(self, key):
        if key == Key.up:
            self.config += 1
        if key == Key.down:
            self.config -= 1
        if key == Key.left:
            self.config[0] -= 1
            self.config[1] += 1
        if key == Key.right:
            self.config[0] += 1
            self.config[1] -= 1
        self.config = np.clip(self.config, -5, 5)
        print(self.config)

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

    def run_coppelia(self, sec):
        # key input
        Listener(on_press=self.on_press).start()
        # start simulation
        self.sim.setStepping(True)
        self.sim.startSimulation()
        while (t := self.sim.getSimulationTime()) < sec:
            # velocity control
            self.sim.setJointTargetVelocity(self.joint_left, self.config[0])
            self.sim.setJointTargetVelocity(self.joint_right, self.config[1])
            self.sim.step()
        self.sim.stopSimulation()


if __name__ == "__main__":
    client = DDCar()
    client.init_coppelia()
    client.run_coppelia(100)
