#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sim as vrep
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import cv2


class VrepSim(object):

    def __init__(self):
        # Close any open connections
        vrep.simxFinish(-1)

        # Create Var for client connection
        self.clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)

        if self.clientID != -1:
            print('Connected to remote API server')

            self.joint_array = []
            self.joint_org_position = []
            error_code, self.input_cam = vrep.simxGetObjectHandle(self.clientID, 'input_camera',
                                                                  vrep.simx_opmode_oneshot_wait)
            error_code, self.hand = vrep.simxGetObjectHandle(self.clientID, 'Baxter_rightArm_camera',
                                                             vrep.simx_opmode_oneshot_wait)
            error_code, self.target = vrep.simxGetObjectHandle(self.clientID, 'right_target',
                                                               vrep.simx_opmode_oneshot_wait)
            error_code, self.video_cam = vrep.simxGetObjectHandle(self.clientID, 'video_camera',
                                                                  vrep.simx_opmode_oneshot_wait)
            error_code, self.main_target = vrep.simxGetObjectHandle(self.clientID, 'target',
                                                                    vrep.simx_opmode_oneshot_wait)
            error, self.right_arm_collision_target = vrep.simxGetCollisionHandle(self.clientID,
                                                                                 "right_arm_collision_target#",
                                                                                 vrep.simx_opmode_blocking)
            error, self.right_arm_collision_table = vrep.simxGetCollisionHandle(self.clientID,
                                                                                "right_arm_collision_table#",
                                                                                vrep.simx_opmode_blocking)

            # Used to translate action to joint array position
            self.joint_switch = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4, 9: 4, 10: 5, 11: 5, 12: 6, 13: 6}

            for x in range(1, 8):
                error_code, joint = vrep.simxGetObjectHandle(self.clientID, 'Baxter_rightArm_joint' + str(x),
                                                             vrep.simx_opmode_oneshot_wait)
                self.joint_array.append(joint)

            for x in range(0, 7):
                vrep.simxGetJointPosition(self.clientID, self.joint_array[x], vrep.simx_opmode_streaming)

            for x in range(0, 7):
                error_code, temp_pos = vrep.simxGetJointPosition(self.clientID, self.joint_array[x],
                                                                 vrep.simx_opmode_buffer)
                self.joint_org_position.append(temp_pos)

            error_code, self.xyz_hand = vrep.simxGetObjectPosition(self.clientID, self.hand, -1,
                                                                   vrep.simx_opmode_streaming)
            error_code, self.xyz_target = vrep.simxGetObjectPosition(self.clientID, self.target, -1,
                                                                     vrep.simx_opmode_streaming)
            error_code, self.xyz_main_target = vrep.simxGetObjectPosition(self.clientID, self.main_target, -1,
                                                                          vrep.simx_opmode_streaming)
            vrep.simxGetVisionSensorImage(self.clientID, self.input_cam, 0, vrep.simx_opmode_streaming)
            vrep.simxGetVisionSensorImage(self.clientID, self.video_cam, 0, vrep.simx_opmode_streaming)
            error_code, self.right_arm_collision_state_target = vrep.simxReadCollision(self.clientID,
                                                                                       self.right_arm_collision_target,
                                                                                       vrep.simx_opmode_streaming)
            error_code, self.right_arm_collision_state_table = vrep.simxReadCollision(self.clientID,
                                                                                      self.right_arm_collision_table,
                                                                                      vrep.simx_opmode_streaming)

            time.sleep(1)
        else:
            print('Failed connecting to remote API server')
            sys.exit('Could not connect')

    def move_joint(self, action):

        if action == 0 or action % 2 == 0:
            move_interval = 0.03
        else:
            move_interval = -0.03

        joint_num = self.joint_switch.get(action, -1)

        error_code, position = vrep.simxGetJointPosition(self.clientID, self.joint_array[joint_num],
                                                         vrep.simx_opmode_buffer)

        error_code = vrep.simxSetJointTargetPosition(self.clientID, self.joint_array[joint_num],
                                                     position + move_interval, vrep.simx_opmode_oneshot_wait)

        return error_code

    def step_right(self, action):
        """Applies an array of actions to all right joint positions.
           Args:
               action (list): increments to add to robot position (-0.1, 0, 0.1)
        """

        # get position of arm, increment by values, then move robot
        start_position = []

        for x in range(0, 7):
            error_code, temp_pos = vrep.simxGetJointPosition(self.clientID, self.joint_array[x],
                                                             vrep.simx_opmode_buffer)
            start_position.append(temp_pos)
            error_code = vrep.simxSetJointTargetPosition(self.clientID, self.joint_array[x],
                                                         start_position[x] + action[x], vrep.simx_opmode_oneshot_wait)

    def calc_distance(self):

        error_code, self.xyz_hand = vrep.simxGetObjectPosition(self.clientID, self.hand, -1, vrep.simx_opmode_buffer)

        error_code, self.xyz_target = vrep.simxGetObjectPosition(self.clientID, self.target, -1,
                                                                 vrep.simx_opmode_buffer)

        # need to check if this formula is calculating distance properly
        distance = math.sqrt(
            pow((self.xyz_hand[0] - self.xyz_target[0]), 2) + pow((self.xyz_hand[1] - self.xyz_target[1]), 2) + pow(
                (self.xyz_hand[2] - self.xyz_target[2]), 2))

        return distance

    def get_collision_state(self):

        error_code, self.right_arm_collision_state_target = vrep.simxReadCollision(self.clientID,
                                                                                   self.right_arm_collision_target,
                                                                                   vrep.simx_opmode_buffer)
        error_code, self.right_arm_collision_state_table = vrep.simxReadCollision(self.clientID,
                                                                                  self.right_arm_collision_table,
                                                                                  vrep.simx_opmode_buffer)

        if self.right_arm_collision_state_target or self.right_arm_collision_state_table:

            return True

        else:

            return False

    def get_input_image(self):

        error_code, resolution, image = vrep.simxGetVisionSensorImage(self.clientID, self.input_cam, 0,
                                                                      vrep.simx_opmode_buffer)

        image = np.array(image, dtype=np.uint8)

        image.resize([resolution[0], resolution[1], 3])

        image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

        return image

    def get_video_image(self):

        error_code, resolution, image = vrep.simxGetVisionSensorImage(self.clientID, self.video_cam, 0,
                                                                      vrep.simx_opmode_buffer)

        image = np.array(image, dtype=np.uint8)

        image.resize([resolution[0], resolution[1], 3])

        image = cv2.rotate(image, cv2.ROTATE_180)

        return image

    def display_image(self):

        image = self.get_input_image()

        plt.imshow(image)

    def reset_sim(self):

        for x in range(0, 7):
            vrep.simxSetJointTargetPosition(self.clientID, self.joint_array[x], self.joint_org_position[x],
                                            vrep.simx_opmode_oneshot_wait)

        time.sleep(1)

