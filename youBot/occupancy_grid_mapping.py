# Copyright 2024 @with-RL
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import matplotlib.pyplot as plt

from youBot import YouBot


class MappingBot(YouBot):
    def __init__(self):
        super().__init__()
        self.grid = Grid()

    def read_ref(self):
        x, y = self.sim.getObjectPosition(self.youBot_ref)[:2]
        theta = self.sim.getObjectOrientation(self.youBot_ref)[2]
        return x, y, theta

    def run_step(self, count):
        # car control
        self.control_car()
        # read lidars
        scan = self.read_lidars()
        # read youBot_ref
        loc = self.read_ref()
        # update grid
        self.grid.update(loc, scan)


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
        self.delta = np.pi / 12
        self.sacn_theta = np.array([-np.pi / 2 + self.delta * i for i in range(13)])
        self.boundary = np.pi / 2 + self.delta / 2
        # min distance
        self.min_dist = (2 * (0.05**2)) ** 0.5

    def update(self, loc, scan):
        self.mapping(loc, scan)
        self.save()
        self.visualize(loc, scan)

    def mapping(self, loc, scan):
        x, y, theta = loc
        # scan position
        rx = x + 0.275 * np.cos(theta)
        ry = y + 0.275 * np.sin(theta)
        # range
        dist = 2.25
        i_min = max(0, int((rx - dist) // 0.1 + 50))
        i_max = min(99, int((rx + dist) // 0.1 + 50))
        j_min = max(0, int((ry - dist) // 0.1 + 50))
        j_max = min(99, int((ry + dist) // 0.1 + 50))
        # sub grid
        sub_grid = self.grid[j_min : j_max + 1, i_min : i_max + 1]
        # x distance
        gx = np.arange(i_min, i_max + 1) * 0.1 + 0.05 - 5
        gx = np.repeat(gx.reshape(1, -1), sub_grid.shape[0], axis=0)
        dx = gx - rx
        # y distance
        gy = np.arange(j_min, j_max + 1) * 0.1 + 0.05 - 5
        gy = np.repeat(gy.reshape(1, -1).T, sub_grid.shape[1], axis=1)
        dy = gy - ry
        # distance
        gd = (dx**2 + dy**2) ** 0.5
        # theta diff
        gtheta = np.arccos(dx / gd) * ((dy > 0) * 2 - 1)
        dtheta = gtheta - theta
        while np.pi < np.max(dtheta):
            dtheta -= (np.pi < dtheta) * 2 * np.pi
        while np.min(dtheta) < -np.pi:
            dtheta += (dtheta < -np.pi) * 2 * np.pi
        # inverse sensor model
        for i in range(13):
            res, dist, _, _, _ = scan[i]
            if res == 0:
                area = (
                    (gd <= 2.25)
                    * (-self.boundary + self.delta * i <= dtheta)
                    * (dtheta <= -self.boundary + self.delta * (i + 1))
                )
                sub_grid[area] -= 0.5
            else:
                dist = min(2.25, dist)
                detect_area = (
                    (np.abs(gd - dist) < self.min_dist)
                    * (-self.boundary + self.delta * i <= dtheta)
                    * (dtheta <= -self.boundary + self.delta * (i + 1))
                )
                sub_grid[detect_area] += 0.5

                free_area = (
                    (gd <= dist - self.min_dist)
                    * (-self.boundary + self.delta * i <= dtheta)
                    * (dtheta <= -self.boundary + self.delta * (i + 1))
                )
                sub_grid[free_area] -= 0.5
        np.clip(self.grid, -5, 5, out=self.grid)

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
    client = MappingBot()
    client.init_coppelia()
    client.run_coppelia()
