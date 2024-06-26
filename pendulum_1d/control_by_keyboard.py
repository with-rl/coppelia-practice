import numpy as np
from pynput.keyboard import Key, Listener

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


class Denduleum1D:
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.require("sim")
        self.config = np.zeros(1)

    def on_press(self, key):
        if key == Key.left:
            self.config -= 1
        if key == Key.right:
            self.config += 1
        self.config = np.clip(self.config, -10, 10)
        print(self.config)

    def init_coppelia(self):
        self.joint_01 = self.sim.getObject("/Joint_01")

        # force control mode
        self.sim.setObjectInt32Param(
            self.joint_01,
            self.sim.jointintparam_dynctrlmode,
            self.sim.jointdynctrl_force,
        )

    def run_coppelia(self, sec):
        # key input
        Listener(on_press=self.on_press).start()
        # start simulation
        self.sim.setStepping(True)
        self.sim.startSimulation()
        while (t := self.sim.getSimulationTime()) < sec:
            # force control
            self.sim.setJointTargetForce(self.joint_01, self.config[0])
            self.sim.step()
        self.sim.stopSimulation()


if __name__ == "__main__":
    client = Denduleum1D()
    client.init_coppelia()
    client.run_coppelia(100)
