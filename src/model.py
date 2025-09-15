"""
model.py
--------
Train Random Forest regressors for head pose prediction.
"""

from sklearn.ensemble import RandomForestRegressor

def build_rf():
    """
    Build a Random Forest Regressor with predefined parameters.
    """
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )