# -*- coding: utf-8 -*-

import os
import re
import csv
from pathlib import Path
from datetime import datetime

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ultralytics import YOLO


np.random.seed(42)


# =========================================================
# PROJECT PATHS
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent

# If this script lives in /scripts, project root is one folder above.
if SCRIPT_DIR.name == "scripts":
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    PROJECT_ROOT = SCRIPT_DIR


# =========================================================
# SETTINGS
# =========================================================

# ---------------------------------------------------------
# INPUT / OUTPUT
# ---------------------------------------------------------
#
# Change INPUT_DIR when testing another hurricane.
#
# Examples:
#   IAN_FULL_DISK_SEQUENCE
#   MICHAEL_FULL_DISK_SEQUENCE
#   MILTON_FULL_DISK_SEQUENCE
#
INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "IAN_FULL_DISK_SEQUENCE"

OUTPUT_DIR = PROJECT_ROOT / "ML_PHASE4_OUTPUT_IAN_YOLO_ROI"

SHOW_PLOTS = False
SAVE_SEGMENTED = False
SAVE_LOCALS = False

# Save the crop given to YOLO so we can visually inspect it.
SAVE_YOLO_ROIS = True


# =========================================================
# YOLO SETTINGS
# =========================================================

YOLO_MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "runs"
    / "hurricane_eye"
    / "yolo1280_final"
    / "weights"
    / "best.pt"
)

# Your model was trained at 1280.
YOLO_IMGSZ = 1280

# M3 Pro acceleration.
YOLO_DEVICE = "mps"

# We can afford a somewhat lower confidence threshold inside
# a storm-centered ROI because the search area is constrained.
YOLO_CONF_THRESHOLD = 0.10

# Size of crop around the approximate storm/eye location.
#
# This is HALF the width/height.
# 512 => approximately 1024 x 1024 crop.
YOLO_ROI_HALF_SIZE = 512

# If coarse radius is large, allow ROI to expand.
YOLO_ROI_RADIUS_MULTIPLIER = 6.0

# If more than one YOLO eye appears inside the crop:
# favor the one closest to the expected/coarse center.
YOLO_DISTANCE_WEIGHT = 0.002

# Visual color for YOLO detection.
YOLO_COLOR = (0, 255, 255)


# =========================================================
# ORIGINAL PIPELINE SETTINGS
# =========================================================

GAUSSIAN_BLUR_KSIZE = (5, 5)

KMEANS_K = 4
KMEANS_ATTEMPTS = 3
KMEANS_MAX_ITER = 10
KMEANS_EPS = 0.5


# =========================================================
# CORE VISUAL COLORS
# =========================================================

COARSE_COLOR = (255, 105, 180)       # hot pink
RANSAC_COLOR = (255, 255, 0)         # cyan
TRACK_COLOR = (0, 255, 0)            # green

FLOW_COLOR = (0, 0, 255)             # red
VELOCITY_COLOR = (0, 255, 255)       # yellow
ROTATION_COLOR = (0, 165, 255)       # orange

ML_CORRECTION_COLOR = (255, 0, 200)  # magenta
TEXT_COLOR = (255, 255, 255)

FALLBACK_COLORS = {
    "ml_correction": (255, 0, 255),
    "optical_flow": (255, 255, 255),
    "previous_center": (180, 180, 180),
    "none": (0, 255, 0),
}


# =========================================================
# TEMPORAL / TRACKING
# =========================================================

USE_TEMPORAL_SMOOTHING = True
SMOOTHING_ALPHA = 0.35

MAX_REASONABLE_JUMP = 80.0

USE_PREVIOUS_GUIDANCE = True
GUIDED_SEARCH_RADIUS_FACTOR = 1.2
FALLBACK_TO_GLOBAL_COARSE = True

RANSAC_CENTER_PENALTY = 1.2
RANSAC_RADIUS_PENALTY = 0.6


# =========================================================
# OPTICAL FLOW
# =========================================================

USE_OPTICAL_FLOW = True

FLOW_MAX_CORNERS = 50
FLOW_QUALITY_LEVEL = 0.01
FLOW_MIN_DISTANCE = 5
FLOW_BLOCK_SIZE = 7
FLOW_WIN_SIZE = (21, 21)
FLOW_MAX_LEVEL = 3


# =========================================================
# ROTATION
# =========================================================

USE_ROTATION_ESTIMATION = True

ROT_INNER_RADIUS_FACTOR = 0.90
ROT_OUTER_RADIUS_FACTOR = 1.60

ROT_MAX_CORNERS = 80
ROT_QUALITY_LEVEL = 0.01
ROT_MIN_DISTANCE = 5
ROT_BLOCK_SIZE = 7

ROT_MIN_POINTS = 6
ROT_MIN_RADIUS_FOR_POINT = 6.0

ROT_CCW_THRESHOLD = 0.60
ROT_DRAW_MAX_ARROWS = 12


# =========================================================
# ORIGINAL ML CORRECTION
# =========================================================

CONFIDENCE_THRESHOLD = 0.5
USE_CONFIDENCE_FALLBACK = True

MIN_INLIERS_FOR_ML = 20
MIN_FLOW_POINTS = 8


# =========================================================
# REAL-WORLD CONVERSION
# =========================================================

KM_PER_PIXEL = 2.0


# =========================================================
# LOAD MODELS
# =========================================================

print("Loading models...")

confidence_model = joblib.load(PROJECT_ROOT / "models" / "confidence_model.pkl")

model_dx = joblib.load(PROJECT_ROOT / "models" / "model_dx.pkl")
model_dy = joblib.load(PROJECT_ROOT / "models" / "model_dy.pkl")
scaler = joblib.load(PROJECT_ROOT / "models" / "scaler.pkl")

yolo_model = YOLO(str(YOLO_MODEL_PATH))

print(f"YOLO model loaded:")
print(YOLO_MODEL_PATH)


# =========================================================
# HELPERS
# =========================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def show(title, img, cmap="gray"):
    plt.figure(figsize=(9, 6))
    plt.title(title)
    plt.axis("off")

    if cmap is None:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap=cmap)

    plt.show()


def natural_key(path):
    base = os.path.basename(str(path))
    parts = re.split(r"(\d+)", base)

    return [
        int(p) if p.isdigit() else p.lower()
        for p in parts
    ]


def list_image_files(folder):
    exts = (
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp"
    )

    folder = Path(folder)

    files = [
        str(folder / f)
        for f in os.listdir(folder)
        if f.lower().endswith(exts)
    ]

    return sorted(files, key=natural_key)


def parse_timestamp_from_filename(filename):

    base = os.path.basename(filename)

    patterns = [
        (
            r"(\d{8})[_-](\d{4})",
            "%Y%m%d%H%M"
        ),
        (
            r"(\d{4})[-_](\d{2})[-_](\d{2})[_-](\d{2})[-_](\d{2})",
            "%Y%m%d%H%M"
        ),
    ]

    for pattern, fmt in patterns:

        match = re.search(pattern, base)

        if match:

            parts = "".join(match.groups())

            try:
                return datetime.strptime(parts, fmt)

            except ValueError:
                pass

    return None


def sort_files_by_time_or_name(files):

    enriched = []

    for f in files:
        ts = parse_timestamp_from_filename(f)
        enriched.append((f, ts))

    if all(ts is not None for _, ts in enriched):
        enriched.sort(key=lambda x: x[1])

    else:
        enriched.sort(
            key=lambda x: natural_key(x[0])
        )

    return enriched


# =========================================================
# K-MEANS COARSE DETECTORS
# =========================================================

def run_local_kmeans_mask(roi):

    blur = cv2.GaussianBlur(
        roi,
        (7, 7),
        0
    )

    Z = blur.reshape((-1, 1)).astype(
        np.float32
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS
        + cv2.TERM_CRITERIA_MAX_ITER,
        25,
        0.5,
    )

    _, labels, centers = cv2.kmeans(
        Z,
        4,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS,
    )

    centers = centers.flatten()

    darkest_cluster = np.argmin(centers)

    mask = (
        labels.flatten()
        == darkest_cluster
    )

    mask_img = (
        mask.reshape(roi.shape)
        .astype(np.uint8)
        * 255
    )

    return mask_img


def find_eye_candidate_global(img_gray):

    h, w = img_gray.shape

    cx = w // 2
    cy = h // 2

    search_radius = int(
        min(h, w) * 0.35
    )

    x1 = max(
        cx - search_radius,
        0
    )

    y1 = max(
        cy - search_radius,
        0
    )

    x2 = min(
        cx + search_radius,
        w
    )

    y2 = min(
        cy + search_radius,
        h
    )

    roi = img_gray[
        y1:y2,
        x1:x2
    ]

    mask_img = run_local_kmeans_mask(
        roi
    )

    contours, _ = cv2.findContours(
        mask_img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    largest = max(
        contours,
        key=cv2.contourArea
    )

    (x, y), r = cv2.minEnclosingCircle(
        largest
    )

    return (
        (x + x1, y + y1),
        r
    )


def find_eye_candidate_guided(
    img_gray,
    prev_center,
    prev_radius
):

    h, w = img_gray.shape

    px, py = prev_center

    search_radius = int(
        max(
            20,
            prev_radius
            * GUIDED_SEARCH_RADIUS_FACTOR
        )
    )

    x1 = max(
        int(px) - search_radius,
        0
    )

    y1 = max(
        int(py) - search_radius,
        0
    )

    x2 = min(
        int(px) + search_radius,
        w
    )

    y2 = min(
        int(py) + search_radius,
        h
    )

    roi = img_gray[
        y1:y2,
        x1:x2
    ]

    if roi.size == 0:
        return None

    mask_img = run_local_kmeans_mask(
        roi
    )

    contours, _ = cv2.findContours(
        mask_img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    best = None
    best_score = -1e18

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area <= 0:
            continue

        (x, y), r = cv2.minEnclosingCircle(
            cnt
        )

        gx = x + x1
        gy = y + y1

        center_dist = np.hypot(
            gx - px,
            gy - py
        )

        radius_diff = abs(
            r - prev_radius
        )

        score = (
            area
            - 1.2 * center_dist
            - 0.8 * radius_diff
        )

        if score > best_score:

            best_score = score

            best = (
                (gx, gy),
                r
            )

    return best


def find_eye_candidate(
    img_gray,
    prev_center=None,
    prev_radius=None
):

    if (
        USE_PREVIOUS_GUIDANCE
        and prev_center is not None
        and prev_radius is not None
    ):

        guided = find_eye_candidate_guided(
            img_gray,
            prev_center,
            prev_radius
        )

        if guided is not None:
            return guided

        if FALLBACK_TO_GLOBAL_COARSE:
            return find_eye_candidate_global(
                img_gray
            )

        return None

    return find_eye_candidate_global(
        img_gray
    )


# =========================================================
# YOLO ROI
# =========================================================

def build_yolo_roi(
    img_gray,
    coarse_cand,
    prev_center=None
):
    """
    Builds a storm-centered square crop for YOLO.

    Returns:
        crop
        bounds = (x1, y1, x2, y2)
        expected_local_center
    """

    if coarse_cand is None:

        if prev_center is None:
            return None, None, None

        cx, cy = prev_center
        coarse_r = YOLO_ROI_HALF_SIZE / 4

    else:

        (cx, cy), coarse_r = coarse_cand

    # If we already have a tracked center,
    # bias toward it.
    if prev_center is not None:

        cx = (
            0.65 * prev_center[0]
            + 0.35 * cx
        )

        cy = (
            0.65 * prev_center[1]
            + 0.35 * cy
        )

    half_size = int(
        max(
            YOLO_ROI_HALF_SIZE,
            coarse_r
            * YOLO_ROI_RADIUS_MULTIPLIER
        )
    )

    h, w = img_gray.shape

    x1 = max(
        int(cx) - half_size,
        0
    )

    y1 = max(
        int(cy) - half_size,
        0
    )

    x2 = min(
        int(cx) + half_size,
        w
    )

    y2 = min(
        int(cy) + half_size,
        h
    )

    crop = img_gray[
        y1:y2,
        x1:x2
    ].copy()

    if crop.size == 0:
        return None, None, None

    local_expected = (
        cx - x1,
        cy - y1
    )

    return (
        crop,
        (x1, y1, x2, y2),
        local_expected,
    )


def detect_eye_yolo_roi(
    img_gray,
    coarse_cand,
    prev_center=None,
    frame_index=None,
    roi_output_dir=None
):
    """
    Runs YOLO on a smaller storm-centered crop.

    Returns dictionary:
        found
        x
        y
        r
        confidence
        bbox
        roi_bounds
    """

    result = {
        "found": False,
        "x": np.nan,
        "y": np.nan,
        "r": np.nan,
        "confidence": np.nan,
        "bbox": None,
        "roi_bounds": None,
    }

    crop, bounds, expected_local = build_yolo_roi(
        img_gray,
        coarse_cand,
        prev_center
    )

    if crop is None:
        return result

    x1, y1, x2, y2 = bounds

    result["roi_bounds"] = bounds

    # YOLO normally expects 3-channel image.
    crop_bgr = cv2.cvtColor(
        crop,
        cv2.COLOR_GRAY2BGR
    )

    predictions = yolo_model.predict(
        source=crop_bgr,
        imgsz=YOLO_IMGSZ,
        conf=YOLO_CONF_THRESHOLD,
        device=YOLO_DEVICE,
        verbose=False,
    )

    if (
        predictions is None
        or len(predictions) == 0
    ):
        return result

    pred = predictions[0]

    if (
        pred.boxes is None
        or len(pred.boxes) == 0
    ):

        if (
            SAVE_YOLO_ROIS
            and roi_output_dir is not None
            and frame_index is not None
        ):

            roi_vis = crop_bgr.copy()

            cv2.putText(
                roi_vis,
                "YOLO: NO DETECTION",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            cv2.imwrite(
                str(
                    Path(roi_output_dir)
                    / f"{frame_index:03d}_roi_no_detection.png"
                ),
                roi_vis
            )

        return result

    expected_x, expected_y = expected_local

    best_score = -1e18
    best_detection = None

    for box in pred.boxes:

        coords = (
            box.xyxy[0]
            .detach()
            .cpu()
            .numpy()
        )

        bx1, by1, bx2, by2 = coords

        conf = float(
            box.conf[0]
            .detach()
            .cpu()
            .item()
        )

        bx = (
            bx1 + bx2
        ) / 2.0

        by = (
            by1 + by2
        ) / 2.0

        bw = bx2 - bx1
        bh = by2 - by1

        distance = np.hypot(
            bx - expected_x,
            by - expected_y
        )

        # Higher confidence is good.
        # Greater distance from expected center is penalized.
        score = (
            conf
            - YOLO_DISTANCE_WEIGHT
            * distance
        )

        if score > best_score:

            best_score = score

            best_detection = (
                bx1,
                by1,
                bx2,
                by2,
                bx,
                by,
                bw,
                bh,
                conf,
            )

    if best_detection is None:
        return result

    (
        bx1,
        by1,
        bx2,
        by2,
        bx,
        by,
        bw,
        bh,
        conf,
    ) = best_detection

    global_x = bx + x1
    global_y = by + y1

    # Approximate radius from YOLO rectangle.
    radius = (
        bw + bh
    ) / 4.0

    result["found"] = True
    result["x"] = float(global_x)
    result["y"] = float(global_y)
    result["r"] = float(radius)
    result["confidence"] = float(conf)

    result["bbox"] = (
        float(bx1 + x1),
        float(by1 + y1),
        float(bx2 + x1),
        float(by2 + y1),
    )

    # -----------------------------------------------------
    # SAVE ROI VISUALIZATION
    # -----------------------------------------------------
    if (
        SAVE_YOLO_ROIS
        and roi_output_dir is not None
        and frame_index is not None
    ):

        roi_vis = crop_bgr.copy()

        cv2.rectangle(
            roi_vis,
            (int(bx1), int(by1)),
            (int(bx2), int(by2)),
            YOLO_COLOR,
            2,
        )

        cv2.drawMarker(
            roi_vis,
            (int(bx), int(by)),
            YOLO_COLOR,
            cv2.MARKER_CROSS,
            16,
            2,
        )

        cv2.putText(
            roi_vis,
            f"YOLO eye {conf:.2f}",
            (
                max(5, int(bx1)),
                max(20, int(by1) - 5),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            YOLO_COLOR,
            2,
        )

        cv2.imwrite(
            str(
                Path(roi_output_dir)
                / f"{frame_index:03d}_roi_yolo.png"
            ),
            roi_vis,
        )

    return result


# =========================================================
# LOCAL CROP INSIDE CIRCLE
# =========================================================

def get_local_crop_inside_circle(
    img,
    center,
    radius,
    scale=0.9
):

    h, w = img.shape

    cx, cy = center

    local_r = max(
        8,
        int(radius * scale)
    )

    x1 = max(
        int(cx) - local_r,
        0
    )

    y1 = max(
        int(cy) - local_r,
        0
    )

    x2 = min(
        int(cx) + local_r,
        w
    )

    y2 = min(
        int(cy) + local_r,
        h
    )

    crop = img[
        y1:y2,
        x1:x2
    ].copy()

    yy, xx = np.ogrid[
        :crop.shape[0],
        :crop.shape[1]
    ]

    local_cx = cx - x1
    local_cy = cy - y1

    dist = np.sqrt(
        (xx - local_cx) ** 2
        + (yy - local_cy) ** 2
    )

    mask = np.zeros_like(
        crop,
        dtype=np.uint8
    )

    mask[
        dist <= local_r
    ] = 255

    crop_masked = cv2.bitwise_and(
        crop,
        crop,
        mask=mask
    )

    return (
        crop_masked,
        mask,
        (x1, y1, x2, y2),
        (local_cx, local_cy),
        local_r,
    )


# =========================================================
# RANSAC
# =========================================================

def circle_from_3pts(
    p1,
    p2,
    p3
):

    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    temp = x2**2 + y2**2

    bc = (
        x1**2
        + y1**2
        - temp
    ) / 2.0

    cd = (
        temp
        - x3**2
        - y3**2
    ) / 2.0

    det = (
        (x1 - x2)
        * (y2 - y3)
        - (x2 - x3)
        * (y1 - y2)
    )

    if abs(det) < 1e-8:
        return None

    cx = (
        bc * (y2 - y3)
        - cd * (y1 - y2)
    ) / det

    cy = (
        (x1 - x2)
        * cd
        - (x2 - x3)
        * bc
    ) / det

    r = np.hypot(
        x1 - cx,
        y1 - cy
    )

    return cx, cy, r


def find_eye_candidate_ransac_local(
    img_gray,
    coarse_cand,
    prev_center=None,
    prev_radius=None,
    n_iter=2500,
    dist_thresh=2.5,
    min_inliers=25,
):

    if coarse_cand is None:
        return (
            None,
            None,
            None,
            None,
            None,
            0,
        )

    (cx, cy), coarse_r = coarse_cand

    (
        crop,
        circle_mask,
        bounds,
        local_center,
        local_r,
    ) = get_local_crop_inside_circle(
        img_gray,
        (cx, cy),
        coarse_r,
        scale=0.9,
    )

    blur = cv2.GaussianBlur(
        crop,
        (7, 7),
        0
    )

    edges = cv2.Canny(
        blur,
        40,
        120
    )

    points = np.column_stack(
        np.where(edges > 0)
    )

    if len(points) < 3:

        return (
            None,
            crop,
            circle_mask,
            edges,
            None,
            0,
        )

    points_xy = np.column_stack(
        (
            points[:, 1],
            points[:, 0]
        )
    ).astype(np.float32)

    min_r = int(
        coarse_r * 0.35
    )

    max_r = max(
        min_r + 2,
        int(coarse_r * 0.75)
    )

    lc_x, lc_y = local_center

    best_circle = None
    best_score = -1e18
    best_inliers = 0

    for _ in range(n_iter):

        idx = np.random.choice(
            len(points_xy),
            3,
            replace=False
        )

        p1, p2, p3 = points_xy[idx]

        circle = circle_from_3pts(
            p1,
            p2,
            p3
        )

        if circle is None:
            continue

        cx_fit, cy_fit, r_fit = circle

        if (
            r_fit < min_r
            or r_fit > max_r
        ):
            continue

        d = np.sqrt(
            (
                points_xy[:, 0]
                - cx_fit
            ) ** 2
            + (
                points_xy[:, 1]
                - cy_fit
            ) ** 2
        )

        inliers = np.where(
            np.abs(
                d - r_fit
            ) < dist_thresh
        )[0]

        if len(inliers) < min_inliers:
            continue

        center_dist_local = np.hypot(
            cx_fit - lc_x,
            cy_fit - lc_y
        )

        score = (
            len(inliers)
            - 0.8
            * center_dist_local
        )

        if (
            prev_center is not None
            and prev_radius is not None
        ):

            x1, y1, _, _ = bounds

            gx = cx_fit + x1
            gy = cy_fit + y1

            center_dist_prev = np.hypot(
                gx - prev_center[0],
                gy - prev_center[1]
            )

            radius_diff_prev = abs(
                r_fit
                - prev_radius
            )

            score -= (
                RANSAC_CENTER_PENALTY
                * center_dist_prev
            )

            score -= (
                RANSAC_RADIUS_PENALTY
                * radius_diff_prev
            )

        if score > best_score:

            best_score = score

            best_circle = (
                cx_fit,
                cy_fit,
                r_fit
            )

            best_inliers = len(
                inliers
            )

    if best_circle is None:

        return (
            None,
            crop,
            circle_mask,
            edges,
            None,
            0,
        )

    x1, y1, _, _ = bounds

    (
        cx_fit,
        cy_fit,
        r_fit
    ) = best_circle

    return (
        (
            (
                cx_fit + x1,
                cy_fit + y1
            ),
            r_fit
        ),
        crop,
        circle_mask,
        edges,
        best_score,
        best_inliers,
    )


# =========================================================
# OPTICAL FLOW
# =========================================================

def estimate_optical_flow(
    prev_img,
    curr_img,
    prev_center,
    prev_radius
):

    if (
        prev_center is None
        or prev_radius is None
    ):
        return np.nan, np.nan, 0

    px, py = prev_center

    r = int(
        max(
            12,
            prev_radius * 1.0
        )
    )

    h, w = prev_img.shape

    x1 = max(
        int(px) - r,
        0
    )

    y1 = max(
        int(py) - r,
        0
    )

    x2 = min(
        int(px) + r,
        w
    )

    y2 = min(
        int(py) + r,
        h
    )

    prev_crop = prev_img[
        y1:y2,
        x1:x2
    ]

    curr_crop = curr_img[
        y1:y2,
        x1:x2
    ]

    if (
        prev_crop.size == 0
        or curr_crop.size == 0
    ):
        return np.nan, np.nan, 0

    pts_prev = cv2.goodFeaturesToTrack(
        prev_crop,
        maxCorners=FLOW_MAX_CORNERS,
        qualityLevel=FLOW_QUALITY_LEVEL,
        minDistance=FLOW_MIN_DISTANCE,
        blockSize=FLOW_BLOCK_SIZE,
    )

    if (
        pts_prev is None
        or len(pts_prev) == 0
    ):
        return np.nan, np.nan, 0

    (
        pts_curr,
        status,
        err,
    ) = cv2.calcOpticalFlowPyrLK(
        prev_crop,
        curr_crop,
        pts_prev,
        None,
        winSize=FLOW_WIN_SIZE,
        maxLevel=FLOW_MAX_LEVEL,
        criteria=(
            cv2.TERM_CRITERIA_EPS
            | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    )

    if (
        pts_curr is None
        or status is None
    ):
        return np.nan, np.nan, 0

    good_prev = pts_prev[
        status.flatten() == 1
    ]

    good_curr = pts_curr[
        status.flatten() == 1
    ]

    if len(good_prev) == 0:
        return np.nan, np.nan, 0

    motion = (
        good_curr
        - good_prev
    )

    dx = float(
        np.median(
            motion[:, 0, 0]
        )
    )

    dy = float(
        np.median(
            motion[:, 0, 1]
        )
    )

    n_tracked = len(
        good_prev
    )

    return dx, dy, n_tracked


# =========================================================
# ROTATIONAL FLOW
# =========================================================

def estimate_rotational_flow(
    prev_img,
    curr_img,
    center,
    radius
):

    result = {
        "rotation_label": "Uncertain",
        "rotation_score": np.nan,
        "mean_ang_deg": np.nan,
        "ccw_fraction": np.nan,
        "cw_fraction": np.nan,
        "rot_n": 0,
        "pairs": [],
    }

    if (
        center is None
        or radius is None
    ):
        return result

    cx, cy = center

    inner_r = max(
        8.0,
        radius
        * ROT_INNER_RADIUS_FACTOR
    )

    outer_r = max(
        inner_r + 4.0,
        radius
        * ROT_OUTER_RADIUS_FACTOR
    )

    h, w = prev_img.shape

    x1 = max(
        int(cx - outer_r),
        0
    )

    y1 = max(
        int(cy - outer_r),
        0
    )

    x2 = min(
        int(cx + outer_r),
        w
    )

    y2 = min(
        int(cy + outer_r),
        h
    )

    prev_crop = prev_img[
        y1:y2,
        x1:x2
    ]

    curr_crop = curr_img[
        y1:y2,
        x1:x2
    ]

    if (
        prev_crop.size == 0
        or curr_crop.size == 0
    ):
        return result

    yy, xx = np.ogrid[
        :prev_crop.shape[0],
        :prev_crop.shape[1]
    ]

    local_cx = cx - x1
    local_cy = cy - y1

    dist = np.sqrt(
        (xx - local_cx) ** 2
        + (yy - local_cy) ** 2
    )

    annulus_mask = np.zeros_like(
        prev_crop,
        dtype=np.uint8
    )

    annulus_mask[
        (dist >= inner_r)
        & (dist <= outer_r)
    ] = 255

    pts_prev = cv2.goodFeaturesToTrack(
        prev_crop,
        maxCorners=ROT_MAX_CORNERS,
        qualityLevel=ROT_QUALITY_LEVEL,
        minDistance=ROT_MIN_DISTANCE,
        blockSize=ROT_BLOCK_SIZE,
        mask=annulus_mask,
    )

    if (
        pts_prev is None
        or len(pts_prev) == 0
    ):
        return result

    (
        pts_curr,
        status,
        err,
    ) = cv2.calcOpticalFlowPyrLK(
        prev_crop,
        curr_crop,
        pts_prev,
        None,
        winSize=FLOW_WIN_SIZE,
        maxLevel=FLOW_MAX_LEVEL,
        criteria=(
            cv2.TERM_CRITERIA_EPS
            | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    )

    if (
        pts_curr is None
        or status is None
    ):
        return result

    good_prev = pts_prev[
        status.flatten() == 1
    ]

    good_curr = pts_curr[
        status.flatten() == 1
    ]

    if len(good_prev) < ROT_MIN_POINTS:
        return result

    ccw_count = 0
    cw_count = 0

    ang_list = []
    pairs = []

    for p_prev, p_curr in zip(
        good_prev,
        good_curr
    ):

        x_prev = float(
            p_prev[0, 0] + x1
        )

        y_prev = float(
            p_prev[0, 1] + y1
        )

        x_curr = float(
            p_curr[0, 0] + x1
        )

        y_curr = float(
            p_curr[0, 1] + y1
        )

        rx = x_prev - cx
        ry = y_prev - cy

        vx = x_curr - x_prev
        vy = y_curr - y_prev

        radial_dist = np.hypot(
            rx,
            ry
        )

        if (
            radial_dist
            < ROT_MIN_RADIUS_FOR_POINT
        ):
            continue

        cross = (
            rx * (-vy)
            - (-ry) * vx
        )

        theta1 = np.arctan2(
            -(y_prev - cy),
            x_prev - cx
        )

        theta2 = np.arctan2(
            -(y_curr - cy),
            x_curr - cx
        )

        dtheta = (
            theta2
            - theta1
        )

        dtheta = (
            dtheta
            + np.pi
        ) % (
            2 * np.pi
        ) - np.pi

        if cross > 0:
            ccw_count += 1

        elif cross < 0:
            cw_count += 1

        ang_list.append(
            np.degrees(
                dtheta
            )
        )

        pairs.append(
            (
                (x_prev, y_prev),
                (x_curr, y_curr)
            )
        )

    valid_n = (
        ccw_count
        + cw_count
    )

    if (
        valid_n < ROT_MIN_POINTS
        or len(ang_list) == 0
    ):
        return result

    ccw_fraction = (
        ccw_count
        / valid_n
    )

    cw_fraction = (
        cw_count
        / valid_n
    )

    mean_ang_deg = float(
        np.median(
            ang_list
        )
    )

    if (
        ccw_fraction
        >= ROT_CCW_THRESHOLD
    ):

        rotation_label = "CCW"

    elif (
        cw_fraction
        >= ROT_CCW_THRESHOLD
    ):

        rotation_label = "CW"

    else:
        rotation_label = "Uncertain"

    result["rotation_label"] = (
        rotation_label
    )

    result["rotation_score"] = max(
        ccw_fraction,
        cw_fraction
    )

    result["mean_ang_deg"] = (
        mean_ang_deg
    )

    result["ccw_fraction"] = (
        ccw_fraction
    )

    result["cw_fraction"] = (
        cw_fraction
    )

    result["rot_n"] = (
        valid_n
    )

    result["pairs"] = pairs

    return result


# =========================================================
# MOTION
# =========================================================

def compute_motion_metrics(
    prev_x,
    prev_y,
    curr_x,
    curr_y,
    dt_minutes=None
):

    if any(
        np.isnan(v)
        for v in [
            prev_x,
            prev_y,
            curr_x,
            curr_y,
        ]
    ):

        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    dx = curr_x - prev_x
    dy = curr_y - prev_y

    dist = float(
        np.hypot(
            dx,
            dy
        )
    )

    angle_deg = float(
        np.degrees(
            np.arctan2(
                dy,
                dx
            )
        )
    )

    if (
        dt_minutes is not None
        and dt_minutes > 0
    ):

        speed = (
            dist
            / dt_minutes
        )

    else:
        speed = dist

    return (
        dx,
        dy,
        dist,
        angle_deg,
        speed,
    )


def angle_to_compass(
    dx,
    dy
):

    if (
        np.isnan(dx)
        or np.isnan(dy)
    ):
        return ""

    angle = np.degrees(
        np.arctan2(
            -dy,
            dx
        )
    )

    angle = (
        angle
        + 360
    ) % 360

    directions = [
        "E",
        "NE",
        "N",
        "NW",
        "W",
        "SW",
        "S",
        "SE",
    ]

    idx = int(
        (
            angle
            + 22.5
        ) // 45
    ) % 8

    return directions[idx]


def draw_velocity_arrow(
    img,
    start_xy,
    dx,
    dy,
    color=VELOCITY_COLOR,
    scale=3.0,
):

    if (
        np.isnan(dx)
        or np.isnan(dy)
    ):
        return img

    x0 = int(
        start_xy[0]
    )

    y0 = int(
        start_xy[1]
    )

    x1 = int(
        x0
        + dx * scale
    )

    y1 = int(
        y0
        + dy * scale
    )

    cv2.arrowedLine(
        img,
        (x0, y0),
        (x1, y1),
        color,
        2,
        tipLength=0.25,
    )

    return img


def draw_rotation_vectors(
    img,
    pairs,
    center=None,
    bins=12,
    scale=5.0,
):

    if (
        not pairs
        or center is None
    ):
        return img

    cx, cy = center

    bin_vectors = {}

    for (
        (x0, y0),
        (x1, y1)
    ) in pairs:

        dx = x1 - x0
        dy = y1 - y0

        mag = np.hypot(
            dx,
            dy
        )

        if mag < 0.2:
            continue

        angle = np.arctan2(
            y0 - cy,
            x0 - cx
        )

        bin_id = int(
            (
                angle
                + np.pi
            )
            / (
                2 * np.pi
            )
            * bins
        )

        if (
            bin_id not in bin_vectors
            or mag
            > bin_vectors[bin_id][2]
        ):

            bin_vectors[
                bin_id
            ] = (
                (x0, y0),
                (dx, dy),
                mag,
            )

    for (
        (x0, y0),
        (dx, dy),
        mag,
    ) in bin_vectors.values():

        x_end = (
            x0
            + dx * scale
        )

        y_end = (
            y0
            + dy * scale
        )

        cv2.circle(
            img,
            (
                int(x0),
                int(y0)
            ),
            2,
            ROTATION_COLOR,
            -1,
        )

        cv2.arrowedLine(
            img,
            (
                int(x0),
                int(y0)
            ),
            (
                int(x_end),
                int(y_end)
            ),
            ROTATION_COLOR,
            1,
            tipLength=0.45,
        )

    return img


# =========================================================
# PROCESS ONE FRAME
# =========================================================

def process_frame(
    image_path,
    frame_index,
    prev_center=None,
    prev_radius=None,
    roi_output_dir=None,
):

    img_gray = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if img_gray is None:

        raise FileNotFoundError(
            f"Could not read '{image_path}'"
        )

    blur = cv2.GaussianBlur(
        img_gray,
        GAUSSIAN_BLUR_KSIZE,
        0
    )

    # -----------------------------------------------------
    # K-MEANS SEGMENTATION
    # -----------------------------------------------------
    Z = blur.reshape(
        (-1, 1)
    ).astype(
        np.float32
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS
        + cv2.TERM_CRITERIA_MAX_ITER,
        KMEANS_MAX_ITER,
        KMEANS_EPS,
    )

    (
        _,
        labels,
        centers,
    ) = cv2.kmeans(
        Z,
        KMEANS_K,
        None,
        criteria,
        KMEANS_ATTEMPTS,
        cv2.KMEANS_RANDOM_CENTERS,
    )

    seg = (
        np.uint8(
            centers
        )[
            labels.flatten()
        ]
        .reshape(
            blur.shape
        )
    )

    # -----------------------------------------------------
    # STAGE 1:
    # ORIGINAL COARSE LOCALIZATION
    # -----------------------------------------------------
    coarse_cand = find_eye_candidate(
        seg,
        prev_center=prev_center,
        prev_radius=prev_radius,
    )

    # -----------------------------------------------------
    # STAGE 2:
    # YOLO INSIDE ROI
    # -----------------------------------------------------
    yolo_result = detect_eye_yolo_roi(
        img_gray,
        coarse_cand,
        prev_center=prev_center,
        frame_index=frame_index,
        roi_output_dir=roi_output_dir,
    )

    # -----------------------------------------------------
    # STAGE 3:
    # ORIGINAL RANSAC FALLBACK
    # -----------------------------------------------------
    (
        ransac_cand,
        ransac_crop,
        ransac_mask,
        ransac_edges,
        ransac_score,
        ransac_inliers,
    ) = find_eye_candidate_ransac_local(
        seg,
        coarse_cand,
        prev_center=prev_center,
        prev_radius=prev_radius,
    )

    vis = cv2.cvtColor(
        img_gray,
        cv2.COLOR_GRAY2BGR
    )

    # Draw coarse result.
    if coarse_cand is not None:

        (cx, cy), cr = (
            coarse_cand
        )

        cv2.circle(
            vis,
            (
                int(cx),
                int(cy)
            ),
            int(cr),
            COARSE_COLOR,
            1,
        )

        cv2.drawMarker(
            vis,
            (
                int(cx),
                int(cy)
            ),
            COARSE_COLOR,
            cv2.MARKER_CROSS,
            12,
            1,
        )

    # Draw ROI rectangle.
    if (
        yolo_result["roi_bounds"]
        is not None
    ):

        rx1, ry1, rx2, ry2 = (
            yolo_result[
                "roi_bounds"
            ]
        )

        cv2.rectangle(
            vis,
            (
                int(rx1),
                int(ry1)
            ),
            (
                int(rx2),
                int(ry2)
            ),
            (150, 150, 150),
            1,
        )

    # -----------------------------------------------------
    # CHOOSE FINAL DETECTOR
    # -----------------------------------------------------

    found = False

    x = np.nan
    y = np.nan
    r = np.nan

    detection_source = "none"

    yolo_confidence = np.nan

    # YOLO FIRST
    if yolo_result["found"]:

        found = True

        x = yolo_result["x"]
        y = yolo_result["y"]
        r = yolo_result["r"]

        yolo_confidence = (
            yolo_result[
                "confidence"
            ]
        )

        detection_source = "yolo"

        bx1, by1, bx2, by2 = (
            yolo_result[
                "bbox"
            ]
        )

        cv2.rectangle(
            vis,
            (
                int(bx1),
                int(by1)
            ),
            (
                int(bx2),
                int(by2)
            ),
            YOLO_COLOR,
            2,
        )

        cv2.drawMarker(
            vis,
            (
                int(x),
                int(y)
            ),
            YOLO_COLOR,
            cv2.MARKER_CROSS,
            16,
            2,
        )

        cv2.putText(
            vis,
            (
                f"Frame {frame_index:03d} "
                f"YOLO Eye "
                f"({x:.0f}, {y:.0f}) "
                f"conf={yolo_confidence:.2f}"
            ),
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )

    # RANSAC FALLBACK
    elif ransac_cand is not None:

        (x, y), r = (
            ransac_cand
        )

        found = True

        detection_source = (
            "ransac"
        )

        cv2.circle(
            vis,
            (
                int(x),
                int(y)
            ),
            int(r),
            RANSAC_COLOR,
            2,
        )

        cv2.drawMarker(
            vis,
            (
                int(x),
                int(y)
            ),
            RANSAC_COLOR,
            cv2.MARKER_CROSS,
            14,
            2,
        )

        cv2.putText(
            vis,
            (
                f"Frame {frame_index:03d} "
                f"RANSAC fallback "
                f"({x:.0f}, {y:.0f})"
            ),
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )

    else:

        cv2.putText(
            vis,
            (
                f"Frame {frame_index:03d} "
                f"No eye found"
            ),
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # Previous center.
    if prev_center is not None:

        cv2.drawMarker(
            vis,
            (
                int(
                    prev_center[0]
                ),
                int(
                    prev_center[1]
                ),
            ),
            TRACK_COLOR,
            cv2.MARKER_TILTED_CROSS,
            14,
            2,
        )

    return {

        "img_gray": img_gray,

        "seg": seg,

        "vis": vis,

        "coarse_cand": coarse_cand,

        "ransac_crop": ransac_crop,
        "ransac_mask": ransac_mask,
        "ransac_edges": ransac_edges,

        "found": found,

        "x": (
            float(x)
            if found
            else np.nan
        ),

        "y": (
            float(y)
            if found
            else np.nan
        ),

        "r": (
            float(r)
            if found
            else np.nan
        ),

        "score": (
            float(
                ransac_score
            )
            if ransac_score
            is not None
            else np.nan
        ),

        "inliers": int(
            ransac_inliers
        ),

        "detection_source":
            detection_source,

        "yolo_confidence":
            yolo_confidence,

        "yolo_found":
            yolo_result[
                "found"
            ],

        "yolo_bbox":
            yolo_result[
                "bbox"
            ],
    }


# =========================================================
# TEMPORAL SMOOTHING
# =========================================================

def smooth_tracks(results):

    prev_x = None
    prev_y = None
    prev_r = None

    for row in results:

        if not row["found"]:

            row["x_smooth"] = np.nan
            row["y_smooth"] = np.nan
            row["r_smooth"] = np.nan
            row["jump_flag"] = False

            continue

        x = row["x"]
        y = row["y"]
        r = row["r"]

        if prev_x is None:

            xs = x
            ys = y
            rs = r

            jump_flag = False

        else:

            raw_jump = np.hypot(
                x - prev_x,
                y - prev_y
            )

            jump_flag = (
                raw_jump
                > MAX_REASONABLE_JUMP
            )

            if USE_TEMPORAL_SMOOTHING:

                xs = (
                    SMOOTHING_ALPHA
                    * x
                    + (
                        1
                        - SMOOTHING_ALPHA
                    )
                    * prev_x
                )

                ys = (
                    SMOOTHING_ALPHA
                    * y
                    + (
                        1
                        - SMOOTHING_ALPHA
                    )
                    * prev_y
                )

                rs = (
                    SMOOTHING_ALPHA
                    * r
                    + (
                        1
                        - SMOOTHING_ALPHA
                    )
                    * prev_r
                )

            else:

                xs = x
                ys = y
                rs = r

        row["x_smooth"] = float(xs)
        row["y_smooth"] = float(ys)
        row["r_smooth"] = float(rs)

        row["jump_flag"] = (
            jump_flag
        )

        prev_x = xs
        prev_y = ys
        prev_r = rs

    return results


# =========================================================
# CSV
# =========================================================

def write_results_csv(
    results,
    csv_path
):

    fieldnames = [

        "frame_index",
        "filename",
        "timestamp",

        "found",

        "detection_source",

        "x",
        "y",
        "r",

        "yolo_found",
        "yolo_confidence",

        "confidence",
        "confidence_label",

        "score",
        "inliers",

        "flow_dx",
        "flow_dy",
        "flow_dist",
        "flow_n",

        "motion_dx",
        "motion_dy",
        "motion_dist",

        "motion_angle_deg",
        "motion_direction",
        "motion_speed",
        "speed_kmh",

        "rotation_label",
        "rotation_score",
        "mean_ang_deg",
        "rotation_rate",

        "ccw_fraction",
        "cw_fraction",
        "rot_n",

        "used_fallback",
        "fallback_source",

    ]

    with open(
        csv_path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in results:

            writer.writerow(
                {
                    k: row.get(
                        k,
                        ""
                    )
                    for k
                    in fieldnames
                }
            )


# =========================================================
# TRACK PLOT
# =========================================================

def save_track_plot(
    results,
    out_path
):

    xs = [
        r["x"]
        for r in results
        if (
            r["found"]
            and not np.isnan(
                r["x"]
            )
        )
    ]

    ys = [
        r["y"]
        for r in results
        if (
            r["found"]
            and not np.isnan(
                r["y"]
            )
        )
    ]

    if len(xs) < 2:
        return

    plt.figure(
        figsize=(8, 8)
    )

    plt.plot(
        xs,
        ys,
        marker="o"
    )

    plt.gca().invert_yaxis()

    plt.title(
        "Hybrid YOLO ROI Eye Track"
    )

    plt.xlabel("X")
    plt.ylabel("Y")

    plt.grid(True)

    plt.savefig(
        out_path,
        bbox_inches="tight"
    )

    plt.close()


# =========================================================
# LEGEND
# =========================================================

def draw_color_legend(
    img,
    x0=400,
    y0=120
):

    dy = 28
    box = 14

    legend_items = [

        (
            "YOLO eye",
            YOLO_COLOR
        ),

        (
            "Coarse locator",
            COARSE_COLOR
        ),

        (
            "RANSAC fallback",
            RANSAC_COLOR
        ),

        (
            "Previous center",
            TRACK_COLOR
        ),

        (
            "Storm motion",
            VELOCITY_COLOR
        ),

        (
            "Eyewall rotation",
            ROTATION_COLOR
        ),

        (
            "Optical flow",
            FLOW_COLOR
        ),

        (
            "ML correction",
            ML_CORRECTION_COLOR
        ),
    ]

    cv2.putText(
        img,
        "Color Key",
        (
            x0,
            y0 - 12
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )

    for i, (
        label,
        color
    ) in enumerate(
        legend_items
    ):

        yy = (
            y0
            + i * dy
        )

        cv2.rectangle(
            img,
            (
                x0,
                yy - box + 2
            ),
            (
                x0 + box,
                yy + 2
            ),
            color,
            -1,
        )

        cv2.putText(
            img,
            label,
            (
                x0 + 24,
                yy
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )

    return img


# =========================================================
# HUD
# =========================================================

def draw_hud(
    img,
    x,
    y,
    speed_kmh,
    direction,
    rotation_label,
    rotation_rate,
    confidence,
    detection_source,
    fallback_source,
):

    x0 = 20
    y0 = 80

    w = 360
    h = 195

    cv2.rectangle(
        img,
        (
            x0,
            y0
        ),
        (
            x0 + w,
            y0 + h
        ),
        (25, 25, 25),
        -1,
    )

    cv2.rectangle(
        img,
        (
            x0,
            y0
        ),
        (
            x0 + w,
            y0 + h
        ),
        (200, 200, 200),
        2,
    )

    line_y = y0 + 25
    dy = 22

    def put(
        text,
        color=(255, 255, 255)
    ):

        nonlocal line_y

        cv2.putText(
            img,
            text,
            (
                x0 + 10,
                line_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

        line_y += dy

    put(
        "STORM TRACKING HUD"
    )

    if (
        not np.isnan(x)
        and not np.isnan(y)
    ):

        put(
            f"X,Y: {int(x)}, {int(y)}"
        )

    else:

        put(
            "X,Y: unavailable",
            (0, 0, 255)
        )

    put(
        f"Detector: {detection_source.upper()}",
        (
            YOLO_COLOR
            if detection_source == "yolo"
            else RANSAC_COLOR
        ),
    )

    if not np.isnan(
        speed_kmh
    ):

        put(
            f"Speed: {speed_kmh:.1f} km/h",
            (0, 255, 255)
        )

    put(
        f"Dir: {direction}",
        (0, 255, 255)
    )

    if not np.isnan(
        rotation_rate
    ):

        put(
            (
                f"Rotation: "
                f"{rotation_label} "
                f"{rotation_rate:.2f} deg/min"
            ),
            ROTATION_COLOR,
        )

    conf_color = (
        (0, 255, 0)
        if confidence >= 0.5
        else (0, 0, 255)
    )

    put(
        f"Confidence: {confidence * 100:.0f}%",
        conf_color,
    )

    if (
        fallback_source
        != "none"
    ):

        put(
            f"Fallback: {fallback_source}",
            ML_CORRECTION_COLOR,
        )


# =========================================================
# MAIN
# =========================================================

def main():

    ensure_dir(
        OUTPUT_DIR
    )

    overlays_dir = (
        OUTPUT_DIR
        / "overlays"
    )

    yolo_roi_dir = (
        OUTPUT_DIR
        / "yolo_rois"
    )

    ensure_dir(
        overlays_dir
    )

    ensure_dir(
        yolo_roi_dir
    )

    files = list_image_files(
        INPUT_DIR
    )

    if not files:

        raise FileNotFoundError(
            (
                "No image files "
                f"found in folder: "
                f"{INPUT_DIR}"
            )
        )

    print("\n====================================")
    print("PHASE 4: YOLO ROI HYBRID PIPELINE")
    print("====================================")

    print(
        f"Input folder: {INPUT_DIR}"
    )

    print(
        f"Files found: {len(files)}"
    )

    print(
        f"YOLO model: {YOLO_MODEL_PATH}"
    )

    print(
        f"YOLO confidence threshold: "
        f"{YOLO_CONF_THRESHOLD}"
    )

    print(
        f"ROI half-size: "
        f"{YOLO_ROI_HALF_SIZE}"
    )

    files_with_time = (
        sort_files_by_time_or_name(
            files
        )
    )

    results = []

    prev_center_for_guidance = None
    prev_radius_for_guidance = None

    prev_img_gray_for_flow = None

    prev_motion_speed = None
    prev_flow_dist = None
    prev_rotation_score = None

    total_frames = 0

    yolo_used = 0
    ransac_used = 0
    ml_used = 0

    lost_frames = 0
    flow_failures = 0

    def smooth(
        prev,
        curr,
        alpha=0.3
    ):

        if (
            prev is None
            or np.isnan(prev)
            or np.isnan(curr)
        ):

            return curr

        return (
            alpha * curr
            + (
                1 - alpha
            ) * prev
        )

    for idx, (
        image_path,
        timestamp
    ) in enumerate(
        files_with_time,
        start=1
    ):

        total_frames += 1

        print(
            "\n------------------------------------"
        )

        print(
            f"[{idx}] Processing "
            f"{os.path.basename(image_path)}"
        )

        out = process_frame(
            image_path,
            idx,
            prev_center=
                prev_center_for_guidance,
            prev_radius=
                prev_radius_for_guidance,
            roi_output_dir=
                yolo_roi_dir,
        )

        timestamp_str = (
            timestamp.isoformat()
            if timestamp
            else ""
        )

        if (
            out["detection_source"]
            == "yolo"
        ):

            yolo_used += 1

        elif (
            out["detection_source"]
            == "ransac"
        ):

            ransac_used += 1

        # =================================================
        # OPTICAL FLOW
        # =================================================

        flow_dx = np.nan
        flow_dy = np.nan

        flow_n = 0
        flow_dist = np.nan

        if (
            prev_img_gray_for_flow
            is not None
            and
            prev_center_for_guidance
            is not None
            and
            prev_radius_for_guidance
            is not None
        ):

            (
                flow_dx,
                flow_dy,
                flow_n,
            ) = estimate_optical_flow(
                prev_img_gray_for_flow,
                out["img_gray"],
                prev_center_for_guidance,
                prev_radius_for_guidance,
            )

            if flow_n == 0:
                flow_failures += 1

            if not np.isnan(
                flow_dx
            ):

                flow_dist = float(
                    np.hypot(
                        flow_dx,
                        flow_dy
                    )
                )

                px, py = (
                    prev_center_for_guidance
                )

                cv2.arrowedLine(
                    out["vis"],
                    (
                        int(px),
                        int(py)
                    ),
                    (
                        int(
                            px
                            + flow_dx
                        ),
                        int(
                            py
                            + flow_dy
                        ),
                    ),
                    FLOW_COLOR,
                    2,
                )

        # =================================================
        # MOTION
        # =================================================

        motion_dx = np.nan
        motion_dy = np.nan

        motion_dist = np.nan
        motion_angle_deg = np.nan
        motion_speed = np.nan

        motion_direction = ""

        dt_minutes = None

        if idx > 1:

            prev_ts = (
                files_with_time[
                    idx - 2
                ][1]
            )

            if (
                prev_ts
                and timestamp
            ):

                dt_minutes = (
                    (
                        timestamp
                        - prev_ts
                    )
                    .total_seconds()
                    / 60.0
                )

        if (
            prev_center_for_guidance
            is not None
            and out["found"]
        ):

            (
                motion_dx,
                motion_dy,
                motion_dist,
                motion_angle_deg,
                motion_speed,
            ) = compute_motion_metrics(

                prev_center_for_guidance[0],
                prev_center_for_guidance[1],

                out["x"],
                out["y"],

                dt_minutes=
                    dt_minutes,
            )

            motion_direction = (
                angle_to_compass(
                    motion_dx,
                    motion_dy
                )
            )

            draw_velocity_arrow(
                out["vis"],
                prev_center_for_guidance,
                motion_dx,
                motion_dy,
            )

        # =================================================
        # REAL WORLD SPEED
        # =================================================

        if not np.isnan(
            motion_speed
        ):

            speed_kmh = (
                motion_speed
                * KM_PER_PIXEL
                * 60
            )

        else:

            speed_kmh = np.nan

        # =================================================
        # ROTATION
        # =================================================

        rotation_label = (
            "Uncertain"
        )

        rotation_score = np.nan
        mean_ang_deg = np.nan

        ccw_fraction = np.nan
        cw_fraction = np.nan

        rot_n = 0

        rotation_rate = np.nan

        if (
            prev_img_gray_for_flow
            is not None
            and
            prev_center_for_guidance
            is not None
            and
            prev_radius_for_guidance
            is not None
        ):

            rot = estimate_rotational_flow(

                prev_img_gray_for_flow,

                out["img_gray"],

                prev_center_for_guidance,

                prev_radius_for_guidance,
            )

            rotation_label = (
                rot[
                    "rotation_label"
                ]
            )

            rotation_score = (
                rot[
                    "rotation_score"
                ]
            )

            mean_ang_deg = (
                rot[
                    "mean_ang_deg"
                ]
            )

            ccw_fraction = (
                rot[
                    "ccw_fraction"
                ]
            )

            cw_fraction = (
                rot[
                    "cw_fraction"
                ]
            )

            rot_n = (
                rot[
                    "rot_n"
                ]
            )

            draw_rotation_vectors(
                out["vis"],
                rot["pairs"],
                center=
                    prev_center_for_guidance,
            )

        if not np.isnan(
            mean_ang_deg
        ):

            if (
                dt_minutes
                and dt_minutes > 0
            ):

                rotation_rate = (
                    mean_ang_deg
                    / dt_minutes
                )

            else:

                rotation_rate = (
                    mean_ang_deg
                )

        # =================================================
        # SMOOTH ORIGINAL FEATURES
        # =================================================

        motion_speed_s = smooth(
            prev_motion_speed,
            motion_speed
        )

        flow_dist_s = smooth(
            prev_flow_dist,
            flow_dist
        )

        rotation_score_s = smooth(
            prev_rotation_score,
            rotation_score
        )

        # =================================================
        # CONFIDENCE / ML CORRECTION
        # =================================================

        used_fallback = False
        fallback_source = "none"

        confidence = 0.0

        # -------------------------------------------------
        # IF YOLO SUCCEEDED:
        #
        # Use YOLO confidence.
        # DO NOT apply old dx/dy correction model.
        # -------------------------------------------------
        if (
            out["detection_source"]
            == "yolo"
        ):

            confidence = float(
                out[
                    "yolo_confidence"
                ]
            )

        # -------------------------------------------------
        # IF RANSAC WAS USED:
        #
        # Preserve your original ML correction logic.
        # -------------------------------------------------
        elif (
            out["detection_source"]
            == "ransac"
            and out["found"]
        ):

            features_vec = [

                out["r"],
                out["score"],
                out["inliers"],

                flow_dist_s,
                motion_dist,
                motion_speed_s,

                mean_ang_deg,
                rotation_score_s,
                rot_n,
            ]

            features_vec = [

                0
                if pd.isna(x)
                else x

                for x
                in features_vec
            ]

            X = pd.DataFrame(
                [features_vec],
                columns=[
                    "r",
                    "score",
                    "inliers",
                    "flow_dist",
                    "motion_dist",
                    "motion_speed",
                    "mean_ang_deg",
                    "rotation_score",
                    "rot_n",
                ],
            )

            X_scaled = scaler.transform(
                X
            )

            dx_corr = np.clip(
                model_dx.predict(
                    X_scaled
                )[0],
                -50,
                50,
            )

            dy_corr = np.clip(
                model_dy.predict(
                    X_scaled
                )[0],
                -50,
                50,
            )

            corrected_x = (
                out["x"]
                + dx_corr
            )

            corrected_y = (
                out["y"]
                + dy_corr
            )

            confidence = float(
                confidence_model
                .predict_proba(
                    [features_vec]
                )[0][1]
            )

            if (
                USE_CONFIDENCE_FALLBACK
                and
                confidence
                < CONFIDENCE_THRESHOLD
            ):

                # ===== ML CORRECTION =====
                if (
                    not np.isnan(
                        dx_corr
                    )
                    and
                    not np.isnan(
                        dy_corr
                    )
                ):

                    out["x"] = (
                        corrected_x
                    )

                    out["y"] = (
                        corrected_y
                    )

                    fallback_source = (
                        "ml_correction"
                    )

                    used_fallback = True

                    ml_used += 1

                # ===== OPTICAL FLOW =====
                elif (
                    prev_center_for_guidance
                    is not None
                    and
                    not np.isnan(
                        flow_dx
                    )
                    and
                    not np.isnan(
                        flow_dy
                    )
                ):

                    out["x"] = (
                        prev_center_for_guidance[0]
                        + flow_dx
                    )

                    out["y"] = (
                        prev_center_for_guidance[1]
                        + flow_dy
                    )

                    fallback_source = (
                        "optical_flow"
                    )

                    used_fallback = True

                # ===== HOLD PREVIOUS =====
                elif (
                    prev_center_for_guidance
                    is not None
                ):

                    out["x"] = (
                        prev_center_for_guidance[0]
                    )

                    out["y"] = (
                        prev_center_for_guidance[1]
                    )

                    fallback_source = (
                        "previous_center"
                    )

                    used_fallback = True

        else:

            confidence = 0.0

        if not out["found"]:

            lost_frames += 1

        confidence_label = (

            "high"
            if confidence >= 0.75

            else "medium"
            if confidence >= 0.50

            else "low"
        )

        # =================================================
        # HUD
        # =================================================

        draw_hud(

            out["vis"],

            out["x"],
            out["y"],

            speed_kmh,
            motion_direction,

            rotation_label,
            rotation_rate,

            confidence,

            out[
                "detection_source"
            ],

            fallback_source,
        )

        # =================================================
        # FALLBACK MARKER
        # =================================================

        if (
            fallback_source
            != "none"
        ):

            color = (
                FALLBACK_COLORS.get(
                    fallback_source,
                    ML_CORRECTION_COLOR
                )
            )

            cv2.drawMarker(
                out["vis"],
                (
                    int(out["x"]),
                    int(out["y"])
                ),
                color,
                cv2.MARKER_STAR,
                20,
                2,
            )

        # =================================================
        # LEGEND
        # =================================================

        draw_color_legend(
            out["vis"]
        )

        # =================================================
        # SAVE OVERLAY
        # =================================================

        overlay_path = (
            overlays_dir
            / f"{idx:03d}.png"
        )

        cv2.imwrite(
            str(overlay_path),
            out["vis"],
        )

        # =================================================
        # SAVE ROW
        # =================================================

        results.append({

            "frame_index":
                idx,

            "filename":
                os.path.basename(
                    image_path
                ),

            "timestamp":
                timestamp_str,

            "found":
                out["found"],

            "detection_source":
                out[
                    "detection_source"
                ],

            "x":
                out["x"],

            "y":
                out["y"],

            "r":
                out["r"],

            "yolo_found":
                out[
                    "yolo_found"
                ],

            "yolo_confidence":
                out[
                    "yolo_confidence"
                ],

            "confidence":
                confidence,

            "confidence_label":
                confidence_label,

            "score":
                out["score"],

            "inliers":
                out["inliers"],

            "flow_dx":
                flow_dx,

            "flow_dy":
                flow_dy,

            "flow_dist":
                flow_dist,

            "flow_n":
                flow_n,

            "motion_dx":
                motion_dx,

            "motion_dy":
                motion_dy,

            "motion_dist":
                motion_dist,

            "motion_angle_deg":
                motion_angle_deg,

            "motion_direction":
                motion_direction,

            "motion_speed":
                motion_speed,

            "speed_kmh":
                speed_kmh,

            "rotation_label":
                rotation_label,

            "rotation_score":
                rotation_score,

            "mean_ang_deg":
                mean_ang_deg,

            "rotation_rate":
                rotation_rate,

            "ccw_fraction":
                ccw_fraction,

            "cw_fraction":
                cw_fraction,

            "rot_n":
                rot_n,

            "used_fallback":
                used_fallback,

            "fallback_source":
                fallback_source,
        })

        # =================================================
        # UPDATE TRACKING STATE
        # =================================================

        prev_img_gray_for_flow = (
            out["img_gray"]
        )

        if out["found"]:

            prev_center_for_guidance = (
                out["x"],
                out["y"]
            )

            prev_radius_for_guidance = max(
                out["r"],
                8.0
            )

        prev_motion_speed = (
            motion_speed_s
        )

        prev_flow_dist = (
            flow_dist_s
        )

        prev_rotation_score = (
            rotation_score_s
        )

        print(
            f"Detector: "
            f"{out['detection_source']}"
        )

        print(
            f"Eye: "
            f"({out['x']:.1f}, "
            f"{out['y']:.1f})"
        )

        print(
            f"YOLO conf: "
            f"{out['yolo_confidence']}"
        )

        print(
            f"Final confidence: "
            f"{confidence:.3f}"
        )

        print(
            f"Fallback: "
            f"{fallback_source}"
        )

    # =====================================================
    # SAVE FINAL DATA
    # =====================================================

    csv_path = (
        OUTPUT_DIR
        / "phase4_results.csv"
    )

    write_results_csv(
        results,
        csv_path
    )

    track_path = (
        OUTPUT_DIR
        / "phase4_track.png"
    )

    save_track_plot(
        results,
        track_path
    )

    # =====================================================
    # DIAGNOSTICS
    # =====================================================

    print("\n====================================")
    print("PHASE 4 COMPLETE")
    print("====================================")

    print(
        f"Frames processed: "
        f"{total_frames}"
    )

    print(
        f"YOLO used: "
        f"{yolo_used}"
    )

    print(
        f"RANSAC fallback used: "
        f"{ransac_used}"
    )

    print(
        f"ML corrections used: "
        f"{ml_used}"
    )

    print(
        f"Lost frames: "
        f"{lost_frames}"
    )

    print(
        f"Flow failures: "
        f"{flow_failures}"
    )

    print(
        f"\nCSV saved:"
    )

    print(
        csv_path
    )

    print(
        f"\nTrack plot saved:"
    )

    print(
        track_path
    )

    print(
        f"\nOverlays saved:"
    )

    print(
        overlays_dir
    )

    print(
        f"\nYOLO ROI images saved:"
    )

    print(
        yolo_roi_dir
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    print(
        "STARTING PHASE 4 "
        "YOLO ROI PIPELINE"
    )

    main()