"""
dataset_builder.py
-------------------
Extracts 7 selected face landmarks using Mediapipe from the AFLW2000 dataset
and saves them along with ground truth pitch, yaw, roll angles to a CSV file.
"""

import os
import cv2
import pandas as pd
import numpy as np
from scipy.io import loadmat
import mediapipe as mp

# Mediapipe face mesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# 7 key landmarks (nose, eyes, mouth corners, chin)
SELECTED_LANDMARKS = [1, 33, 263, 10, 234, 454, 152]

def build_dataset(dataset_folder: str, output_csv: str = "dataset_features_7points.csv") -> None:
    """
    Build dataset by extracting facial landmarks and head pose labels.

    Args:
        dataset_folder (str): Path to AFLW2000 dataset.
        output_csv (str): Path to save processed dataset.

    Returns:
        None
    """
    columns = ["image_name", "pitch", "yaw", "roll"] + \
              [f"landmark_{i}x" for i in SELECTED_LANDMARKS] + \
              [f"landmark_{i}y" for i in SELECTED_LANDMARKS]
    df = pd.DataFrame(columns=columns)

    for file_name in os.listdir(dataset_folder):
        if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(dataset_folder, file_name)
            mat_path = os.path.join(dataset_folder, os.path.splitext(file_name)[0] + ".mat")

            # Load ground truth pose
            data = loadmat(mat_path)
            pose_para = data["Pose_Para"][0][:3]
            pitch, yaw, roll = np.degrees(pose_para[0]), np.degrees(pose_para[1]), np.degrees(pose_para[2])

            # Extract landmarks
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)

            landmarks_x, landmarks_y = [None] * len(SELECTED_LANDMARKS), [None] * len(SELECTED_LANDMARKS)
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                for idx, lm_index in enumerate(SELECTED_LANDMARKS):
                    lm = face_landmarks.landmark[lm_index]
                    landmarks_x[idx] = lm.x
                    landmarks_y[idx] = lm.y

            row = [file_name, pitch, yaw, roll] + landmarks_x + landmarks_y
            df.loc[len(df)] = row

    df.dropna(axis=0, how='any', inplace=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Dataset saved: {output_csv}")