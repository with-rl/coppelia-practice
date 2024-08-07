# Copyright 2024 @with-RL
#
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

from youBot import YouBot


class KeyboardBot(YouBot):
    def run_step(self, count):
        # car control
        self.control_car()
        # arm control
        self.control_arm()
        # arm gripper
        self.control_gripper()
        # read lidarqa
        self.read_lidars()
        # read camera
        self.read_camera_1()


if __name__ == "__main__":
    client = KeyboardBot()
    client.init_coppelia()
    client.run_coppelia()
