"""
pose_utils.py
-------------
Visualization, smoothing, and helper functions.
"""

import cv2
import numpy as np
from collections import deque

def draw_pose_axes(img, pitch, yaw, roll, nose_point, length=100):
    """
    Draw 3D axes on the face image according to yaw, pitch, roll.

    Args:
        img (np.array): Image frame.
        pitch, yaw, roll (float): Angles in degrees.
        nose_point (tuple): (x,y) coordinates of nose center.
    """
    # Convert to radians
    pitch = -pitch
    h, w, _ = img.shape
    cx, cy = int(nose_point[0] * w), int(nose_point[1] * h)

    pitch, yaw, roll = np.radians([pitch, yaw, roll])
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(pitch), -np.sin(pitch)],
                    [0, np.sin(pitch),  np.cos(pitch)]])
    R_y = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                    [0, 1, 0],
                    [-np.sin(yaw), 0, np.cos(yaw)]])
    R_z = np.array([[np.cos(roll), -np.sin(roll), 0],
                    [np.sin(roll),  np.cos(roll), 0],
                    [0, 0, 1]])
    R = R_z @ R_y @ R_x

    # Axis points
    axis_x = R @ np.array([length, 0, 0])
    axis_y = R @ np.array([0, -length, 0])
    axis_z = R @ np.array([0, 0, -length])

    def project(pt): return int(cx + pt[0]), int(cy + pt[1])
    x2, y2, z2 = project(axis_x), project(axis_y), project(axis_z)

    # Draw axes
    cv2.arrowedLine(img, (cx, cy), x2, (0, 0, 255), 3)   # roll → red
    cv2.arrowedLine(img, (cx, cy), y2, (0, 255, 0), 3)   # pitch → green
    cv2.arrowedLine(img, (cx, cy), z2, (255, 0, 0), 3)   # yaw → blue
    return img

class PoseSmoother:
    """
    Temporal smoothing of pose predictions using a sliding window average.
    """
    def __init__(self, window=5):
        self.pitch_hist = deque(maxlen=window)
        self.yaw_hist = deque(maxlen=window)
        self.roll_hist = deque(maxlen=window)

    def update(self, pitch, yaw, roll):
        self.pitch_hist.append(pitch)
        self.yaw_hist.append(yaw)
        self.roll_hist.append(roll)

        smooth_pitch = np.mean(self.pitch_hist)
        smooth_yaw   = np.mean(self.yaw_hist)
        smooth_roll  = np.mean(self.roll_hist)
        return smooth_pitch, smooth_yaw, smooth_roll