# Copyright 2024 @with-RL
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from pynput.keyboard import Key, Listener

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


@dataclass
class Control:
    vel_X: float = 0
    vel_Y: float = 0
    vel_Z: float = 0


class YouBot:
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.require("sim")
        self.control = Control()
        self.grid = Grid()

    def on_press(self, key):
        deltaX, deltaZ = 1.0, np.pi / 10
        if key == Key.up:
            self.control.vel_X += deltaX
            self.control.vel_Z += min(deltaZ, abs(self.control.vel_Z)) * (
                -1 if self.control.vel_Z > 0 else 1
            )
        if key == Key.down:
            self.control.vel_X -= deltaX
            self.control.vel_Z += min(deltaZ, abs(self.control.vel_Z)) * (
                -1 if self.control.vel_Z > 0 else 1
            )
        if key == Key.left:
            self.control.vel_X += min(deltaX, abs(self.control.vel_X)) * (
                -1 if self.control.vel_X > 0 else 1
            )
            self.control.vel_Z += deltaZ
        if key == Key.right:
            self.control.vel_X += min(deltaX, abs(self.control.vel_X)) * (
                -1 if self.control.vel_X > 0 else 1
            )
            self.control.vel_Z -= deltaZ
        self.control.vel_X = min(max(self.control.vel_X, -20), 20)
        self.control.vel_Y = 0
        self.control.vel_Z = min(max(self.control.vel_Z, -np.pi), np.pi)

    def init_coppelia(self):
        # Wheel Joints: front left, rear left, rear right, front right
        self.wheels = []
        self.wheels.append(self.sim.getObject("/rollingJoint_fl"))
        self.wheels.append(self.sim.getObject("/rollingJoint_rl"))
        self.wheels.append(self.sim.getObject("/rollingJoint_fr"))
        self.wheels.append(self.sim.getObject("/rollingJoint_rr"))
        # lidar
        self.lidars = []
        for i in range(1, 14):
            self.lidars.append(self.sim.getObject(f"/lidar_{i:02d}"))
        # reference
        self.youBot_ref = self.sim.getObject("/youBot_ref")

    def control_car(self):
        self.sim.setJointTargetVelocity(
            self.wheels[0],
            -self.control.vel_X + self.control.vel_Z,
        )
        self.sim.setJointTargetVelocity(
            self.wheels[1],
            -self.control.vel_X + self.control.vel_Z,
        )
        self.sim.setJointTargetVelocity(
            self.wheels[2],
            -self.control.vel_X - self.control.vel_Z,
        )
        self.sim.setJointTargetVelocity(
            self.wheels[3],
            -self.control.vel_X - self.control.vel_Z,
        )

    def read_lidars(self):
        scan = []
        for id in self.lidars:
            scan.append(self.sim.readProximitySensor(id))
        return scan

    def read_ref(self):
        x, y = self.sim.getObjectPosition(self.youBot_ref)[:2]
        theta = self.sim.getObjectOrientation(self.youBot_ref)[2]
        return x, y, theta

    def run_coppelia(self, sec):
        # key input
        Listener(on_press=self.on_press).start()
        # start simulation
        self.sim.setStepping(True)
        self.sim.startSimulation()
        while (t := self.sim.getSimulationTime()) < sec:
            # car control
            self.control_car()
            # read lidars
            scan = self.read_lidars()
            # read youBot_ref
            loc = self.read_ref()
            # update grid
            self.grid.update(loc, scan)
            # step
            self.sim.step()
        self.sim.stopSimulation()


class Grid:
    def __init__(self):
        self.grid = np.zeros((100, 100))
        # plot grid
        r = np.linspace(-5, 5, 101)
        p = np.linspace(-5, 5, 101)
        self.R, self.P = np.meshgrid(r, p)
        # plot object
        self.plt_objects = [None] * 15  # grid, robot, scans (13)
        # scan theta
        delta = np.pi / 12
        self.sacn_theta = np.array([-np.pi / 2 + delta * i for i in range(13)])
        # min distance
        self.min_dist = (2 * (0.05**2)) ** 0.5

    def update(self, loc, scan):
        self.mapping(loc, scan)
        self.save()
        self.visualize(loc, scan)

    def mapping(self, loc, scan):
        x, y, theta = loc
        rx = x + 0.275 * np.cos(theta)
        ry = y + 0.275 * np.sin(theta)
        for j in range(100):  # y축
            gy = j * 0.1 + 0.05 - 5
            for i in range(100):  # x축
                gx = i * 0.1 + 0.05 - 5
                dist = ((gx - rx) ** 2 + (gy - ry) ** 2) ** 0.5
                if dist < 2.35:  # lidar 거리 안에 있는 것만 계산
                    self.grid[j, i] += self.inverseSensorModel(
                        rx, ry, theta, gx, gy, dist, scan
                    )
                else:
                    # self.grid[j, i] += 0
                    pass
        np.clip(self.grid, -5, 5, out=self.grid)

    def inverseSensorModel(self, rx, ry, rtheta, gx, gy, gd, scan):
        # theta 계산
        dx = gx - rx
        dy = gy - ry
        dd = (dx**2 + dy**2) ** 0.5
        gtheta = np.arccos(dx / dd) * (1 if dy > 0 else -1)
        dtheta = gtheta - rtheta
        while dtheta > np.pi:
            dtheta -= np.pi * 2
        while dtheta < -np.pi:
            dtheta += np.pi * 2
        # boundary 내부에 있는 것만 계산
        delta = np.pi / 12
        boundary = np.pi / 2 + delta / 2

        if -boundary < dtheta < boundary:
            idx = np.argmin(np.abs(self.sacn_theta - dtheta))
            res, dist, _, _, _ = scan[idx]  # res, dist, point, obj, n
            if res == 1 and abs(dist - gd) < self.min_dist:
                return 0.5
            elif res == 0 or dist > gd:  # not occupyted
                return -0.5
        return 0

    def save(self):
        with open("youBot/mapping.npy", "wb") as f:
            np.save(f, self.grid)

    def visualize(self, loc, scan):
        x, y, theta = loc
        # clear object
        for object in self.plt_objects:
            if object:
                object.remove()
        # grid
        grid = -self.grid + 5
        self.plt_objects[0] = plt.pcolor(self.R, self.P, grid, cmap="gray")
        # robot
        (self.plt_objects[1],) = plt.plot(
            x, y, color="green", marker="o", markersize=10
        )
        # scan
        rx = x + 0.275 * np.cos(theta)
        ry = y + 0.275 * np.sin(theta)
        for i, data in enumerate(scan):
            res, dist, _, _, _ = data  # res, dist, point, obj, n
            style = "--r" if res == 1 else "--b"
            dist = dist if res == 1 else 2.20

            ti = theta + self.sacn_theta[i]
            xi = rx + dist * np.cos(ti)
            yi = ry + dist * np.sin(ti)
            (self.plt_objects[2 + i],) = plt.plot([rx, xi], [ry, yi], style)

        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.gca().set_aspect("equal")
        plt.pause(0.001)


if __name__ == "__main__":
    client = YouBot()
    client.init_coppelia()
    client.run_coppelia(3600)
