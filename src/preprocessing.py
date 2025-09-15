"""
preprocessing.py
----------------
Feature normalization and data preparation.
"""

import pandas as pd
import numpy as np
from src.dataset_builder import SELECTED_LANDMARKS

def normalize_features(row: pd.Series) -> pd.Series:
    """
    Normalize landmark coordinates relative to nose and inter-eye distance.

    Args:
        row (pd.Series): Row containing landmark x,y coordinates.

    Returns:
        pd.Series: Normalized features (x,y for each landmark).
    """
    nose_x, nose_y = row["landmark_1x"], row["landmark_1y"]
    eye_left_x, eye_left_y = row["landmark_33x"], row["landmark_33y"]
    eye_right_x, eye_right_y = row["landmark_263x"], row["landmark_263y"]

    # Eye distance used as scaling factor
    eye_dist = np.sqrt((eye_right_x - eye_left_x) ** 2 + (eye_right_y - eye_left_y) ** 2)
    if eye_dist == 0: 
        eye_dist = 1e-6

    feats = []
    for i in SELECTED_LANDMARKS:
        x, y = row[f"landmark_{i}x"], row[f"landmark_{i}y"]
        feats.append((x - nose_x) / eye_dist)
        feats.append((y - nose_y) / eye_dist)

    return pd.Series(feats)