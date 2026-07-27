import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("combined_training_data.csv")

# Drop rows without labels
df = df.dropna(subset=["good_detection"])

# =========================
# SELECT FEATURES
# =========================
features = [
    "r",                   # radius
    "score",               # ransac score
    "inliers",             # ransac inliers
    "flow_dist",           # optical flow distance
    "motion_dist",         # movement
    "motion_speed",        # speed
    "mean_ang_deg",        # rotation
    "rotation_score",      # rotation confidence
    "rot_n"                # number of rotation points
]

# Fill missing values
X = df[features].fillna(0)

# Target
y = df["good_detection"]

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# TRAIN MODEL
# =========================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# =========================
# EVALUATE
# =========================
y_pred = model.predict(X_test)

print("\n=== MODEL RESULTS ===")
print(classification_report(y_test, y_pred))

# =========================
# SAVE MODEL
# =========================
joblib.dump(model, "confidence_model.pkl")

print("\nModel saved as confidence_model.pkl")