import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("phase3_training_data2.csv")

features = [
    "r",
    "score",
    "inliers",
    "flow_dist",
    "motion_dist",
    "motion_speed",
    "mean_ang_deg",
    "rotation_score",
    "rot_n"
]

# Fill missing values
X = df[features].fillna(0)

# Targets (correction offsets)
y_x = df["error_x"]
y_y = df["error_y"]

# =========================
# TRAIN / TEST SPLIT 
# =========================
X_train, X_test, y_x_train, y_x_test, y_y_train, y_y_test = train_test_split(
    X, y_x, y_y, test_size=0.2, random_state=42
)

# =========================
# NORMALIZE FEATURES
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# TRAIN MODELS
# =========================
model_x = RandomForestRegressor(n_estimators=200, random_state=42)
model_y = RandomForestRegressor(n_estimators=200, random_state=42)

model_x.fit(X_train_scaled, y_x_train)
model_y.fit(X_train_scaled, y_y_train)

# =========================
# EVALUATION
# =========================
y_x_pred = model_x.predict(X_test_scaled)
y_y_pred = model_y.predict(X_test_scaled)

mse_x = mean_squared_error(y_x_test, y_x_pred)
mse_y = mean_squared_error(y_y_test, y_y_pred)

rmse_x = np.sqrt(mse_x)
rmse_y = np.sqrt(mse_y)

print("\n===== MODEL PERFORMANCE =====")
print(f"MSE X: {mse_x:.4f}")
print(f"MSE Y: {mse_y:.4f}")
print(f"RMSE X: {rmse_x:.4f}")
print(f"RMSE Y: {rmse_y:.4f}")

# =========================
# FEATURE IMPORTANCE
# =========================
print("\n===== FEATURE IMPORTANCE (X MODEL) =====")
for name, importance in zip(features, model_x.feature_importances_):
    print(f"{name}: {importance:.4f}")

print("\n===== FEATURE IMPORTANCE (Y MODEL) =====")
for name, importance in zip(features, model_y.feature_importances_):
    print(f"{name}: {importance:.4f}")

# =========================
# SAVE MODELS + SCALER
# =========================
joblib.dump(model_x, "model_dx.pkl")
joblib.dump(model_y, "model_dy.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModels and scaler saved successfully.")