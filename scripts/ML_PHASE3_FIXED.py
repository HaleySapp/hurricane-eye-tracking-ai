import os
import re
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import joblib
import pandas as pd

np.random.seed(42)  # makes RANSAC repeatable

# =========================================================
# SETTINGS
# =========================================================
INPUT_DIR = "IAN_FULL_DISK_SEQUENCE"
OUTPUT_DIR = "ML_PHASE3_OUTPUT_IAN2_FIXED"

SHOW_PLOTS = False
SAVE_SEGMENTED = False
SAVE_LOCALS = False

GAUSSIAN_BLUR_KSIZE = (5, 5)

KMEANS_K = 4
KMEANS_ATTEMPTS = 3
KMEANS_MAX_ITER = 10
KMEANS_EPS = 0.5

# =========================
# CORE VISUAL COLORS
# =========================
COARSE_COLOR = (255, 105, 180)      # hot pink
RANSAC_COLOR = (255, 255, 0)        # cyan (final eye)
TRACK_COLOR = (0, 255, 0)           # green (previous center)

FLOW_COLOR = (0, 0, 255)           # red (optical flow motion)
VELOCITY_COLOR = (0, 255, 255)      # yellow (motion)
ROTATION_COLOR = (0, 165, 255)      # orange (rotation)

ML_CORRECTION_COLOR = (255, 0, 200) # strong magenta (ML)

TEXT_COLOR = (255, 255, 255)        # white

FALLBACK_COLORS = {
    "ml_correction": (255, 0, 255),     # 💜 magenta 
    "optical_flow": (255, 255, 255),    # white (distinct from flow red)
    "previous_center": (180, 180, 180), # light gray (distinct from track green)
    "none": (0, 255, 0)                
}



STATE_COLORS = {
    "locked": (0, 255, 0),       # green
    "ml": (255, 0, 200),         # magenta
    "flow": (255, 100, 0),       # orange-blue
    "hold": (255, 255, 255),     # white
    "lost": (0, 0, 255)          # red
}

# Temporal smoothing
USE_TEMPORAL_SMOOTHING = True
SMOOTHING_ALPHA = 0.35

# Jump guard
MAX_REASONABLE_JUMP = 80.0  # pixels

# Tracking guidance
USE_PREVIOUS_GUIDANCE = True
GUIDED_SEARCH_RADIUS_FACTOR = 1.2
FALLBACK_TO_GLOBAL_COARSE = True
RANSAC_CENTER_PENALTY = 1.2
RANSAC_RADIUS_PENALTY = 0.6

# Optical flow
USE_OPTICAL_FLOW = True
FLOW_MAX_CORNERS = 50
FLOW_QUALITY_LEVEL = 0.01
FLOW_MIN_DISTANCE = 5
FLOW_BLOCK_SIZE = 7
FLOW_WIN_SIZE = (21, 21)
FLOW_MAX_LEVEL = 3

# Rotational flow around eye
USE_ROTATION_ESTIMATION = True
ROT_INNER_RADIUS_FACTOR = 0.90
ROT_OUTER_RADIUS_FACTOR = 1.60
ROT_MAX_CORNERS = 80
ROT_QUALITY_LEVEL = 0.01
ROT_MIN_DISTANCE = 5
ROT_BLOCK_SIZE = 7
ROT_MIN_POINTS = 6
ROT_MIN_RADIUS_FOR_POINT = 6.0
ROT_CCW_THRESHOLD = 0.60    # at least 60% of valid points must agree
ROT_DRAW_MAX_ARROWS = 12

# ML correction
CONFIDENCE_THRESHOLD = 0.5
USE_CONFIDENCE_FALLBACK = True

# Minimum inliers for ML correction to be applied (to avoid crazy corrections on very bad detections)
MIN_INLIERS_FOR_ML = 20
MIN_FLOW_POINTS = 8

# =========================
# REAL WORLD CONVERSION
# =========================
KM_PER_PIXEL = 2.0   # approx for GOES full disk

confidence_model = joblib.load("confidence_model.pkl")

model_dx = joblib.load("model_dx.pkl")
model_dy = joblib.load("model_dy.pkl")
scaler = joblib.load("scaler.pkl")

# =========================================================
# HELPERS
# =========================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def show(title: str, img: np.ndarray, cmap: str | None = "gray") -> None:
    plt.figure(figsize=(9, 6))
    plt.title(title)
    plt.axis("off")
    if cmap is None:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap=cmap)
    plt.show()


def natural_key(path: str):
    base = os.path.basename(path)
    parts = re.split(r"(\d+)", base)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def list_image_files(folder: str):
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(exts)
    ]
    return sorted(files, key=natural_key)


def parse_timestamp_from_filename(filename: str):
    """
    Tries to parse timestamps like:
    20220928_1200
    20220928-1200
    2022-09-28_12-00
    2022_09_28_1200
    Returns datetime or None.
    """
    base = os.path.basename(filename)

    patterns = [
        (r"(\d{8})[_-](\d{4})", "%Y%m%d%H%M"),
        (r"(\d{4})[-_](\d{2})[-_](\d{2})[_-](\d{2})[-_](\d{2})", "%Y%m%d%H%M"),
    ]

    for pattern, fmt in patterns:
        m = re.search(pattern, base)
        if m:
            parts = "".join(m.groups())
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
        enriched.sort(key=lambda x: natural_key(x[0]))

    return enriched


# =========================================================
# COARSE DETECTORS
# =========================================================
def run_local_kmeans_mask(roi: np.ndarray):
    blur = cv2.GaussianBlur(roi, (7, 7), 0)
    Z = blur.reshape((-1, 1)).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        25,
        0.5
    )

    _, labels, centers = cv2.kmeans(
        Z,
        4,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    centers = centers.flatten()
    darkest_cluster = np.argmin(centers)

    mask = (labels.flatten() == darkest_cluster)
    mask_img = mask.reshape(roi.shape).astype(np.uint8) * 255
    return mask_img


def find_eye_candidate_global(img_gray: np.ndarray):
    h, w = img_gray.shape

    cx = w // 2
    cy = h // 2

    search_radius = int(min(h, w) * 0.35)

    x1 = max(cx - search_radius, 0)
    y1 = max(cy - search_radius, 0)
    x2 = min(cx + search_radius, w)
    y2 = min(cy + search_radius, h)

    roi = img_gray[y1:y2, x1:x2]
    mask_img = run_local_kmeans_mask(roi)

    contours, _ = cv2.findContours(
        mask_img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    (x, y), r = cv2.minEnclosingCircle(largest)

    return ((x + x1, y + y1), r)


def find_eye_candidate_guided(img_gray: np.ndarray, prev_center, prev_radius):
    """
    Guided coarse detector that searches near the previous smoothed center.
    """
    h, w = img_gray.shape
    px, py = prev_center

    search_radius = int(max(20, prev_radius * GUIDED_SEARCH_RADIUS_FACTOR))

    x1 = max(int(px) - search_radius, 0)
    y1 = max(int(py) - search_radius, 0)
    x2 = min(int(px) + search_radius, w)
    y2 = min(int(py) + search_radius, h)

    roi = img_gray[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    mask_img = run_local_kmeans_mask(roi)

    contours, _ = cv2.findContours(
        mask_img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    best = None
    best_score = -1e18

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= 0:
            continue

        (x, y), r = cv2.minEnclosingCircle(cnt)

        gx = x + x1
        gy = y + y1

        center_dist = np.hypot(gx - px, gy - py)
        radius_diff = abs(r - prev_radius)

        score = area - 1.2 * center_dist - 0.8 * radius_diff

        if score > best_score:
            best_score = score
            best = ((gx, gy), r)

    return best


def find_eye_candidate(img_gray: np.ndarray, prev_center=None, prev_radius=None):
    if USE_PREVIOUS_GUIDANCE and prev_center is not None and prev_radius is not None:
        guided = find_eye_candidate_guided(img_gray, prev_center, prev_radius)
        if guided is not None:
            return guided
        if FALLBACK_TO_GLOBAL_COARSE:
            return find_eye_candidate_global(img_gray)
        return None

    return find_eye_candidate_global(img_gray)


# =========================================================
# LOCAL CROP INSIDE CIRCLE
# =========================================================
def get_local_crop_inside_circle(
    img: np.ndarray,
    center: tuple[float, float],
    radius: float,
    scale: float = 0.9
):
    h, w = img.shape
    cx, cy = center
    local_r = max(8, int(radius * scale))

    x1 = max(int(cx) - local_r, 0)
    y1 = max(int(cy) - local_r, 0)
    x2 = min(int(cx) + local_r, w)
    y2 = min(int(cy) + local_r, h)

    crop = img[y1:y2, x1:x2].copy()

    yy, xx = np.ogrid[:crop.shape[0], :crop.shape[1]]
    local_cx = cx - x1
    local_cy = cy - y1
    dist = np.sqrt((xx - local_cx) ** 2 + (yy - local_cy) ** 2)

    mask = np.zeros_like(crop, dtype=np.uint8)
    mask[dist <= local_r] = 255

    crop_masked = cv2.bitwise_and(crop, crop, mask=mask)

    return crop_masked, mask, (x1, y1, x2, y2), (local_cx, local_cy), local_r


# =========================================================
# RANSAC CIRCLE FIT
# =========================================================
def circle_from_3pts(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    temp = x2**2 + y2**2
    bc = (x1**2 + y1**2 - temp) / 2.0
    cd = (temp - x3**2 - y3**2) / 2.0
    det = (x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2)

    if abs(det) < 1e-8:
        return None

    cx = (bc * (y2 - y3) - cd * (y1 - y2)) / det
    cy = ((x1 - x2) * cd - (x2 - x3) * bc) / det
    r = np.hypot(x1 - cx, y1 - cy)

    return cx, cy, r


def find_eye_candidate_ransac_local(
    img_gray: np.ndarray,
    coarse_cand,
    prev_center=None,
    prev_radius=None,
    n_iter: int = 2500,
    dist_thresh: float = 2.5,
    min_inliers: int = 25,
):
    if coarse_cand is None:
        return None, None, None, None, None, 0

    (cx, cy), coarse_r = coarse_cand

    crop, circle_mask, bounds, local_center, local_r = get_local_crop_inside_circle(
        img_gray, (cx, cy), coarse_r, scale=0.9
    )

    blur = cv2.GaussianBlur(crop, (7, 7), 0)
    edges = cv2.Canny(blur, 40, 120)

    points = np.column_stack(np.where(edges > 0))
    if len(points) < 3:
        return None, crop, circle_mask, edges, None, 0

    points_xy = np.column_stack((points[:, 1], points[:, 0])).astype(np.float32)

    min_r = int(coarse_r * 0.35)
    max_r = max(min_r + 2, int(coarse_r * 0.75))

    lc_x, lc_y = local_center
    best_circle = None
    best_score = -1e18
    best_inliers = 0

    for _ in range(n_iter):
        idx = np.random.choice(len(points_xy), 3, replace=False)
        p1, p2, p3 = points_xy[idx]

        circle = circle_from_3pts(p1, p2, p3)
        if circle is None:
            continue

        cx_fit, cy_fit, r_fit = circle

        if r_fit < min_r or r_fit > max_r:
            continue

        d = np.sqrt((points_xy[:, 0] - cx_fit) ** 2 + (points_xy[:, 1] - cy_fit) ** 2)
        inliers = np.where(np.abs(d - r_fit) < dist_thresh)[0]

        if len(inliers) < min_inliers:
            continue

        center_dist_local = np.hypot(cx_fit - lc_x, cy_fit - lc_y)
        score = len(inliers) - 0.8 * center_dist_local

        if prev_center is not None and prev_radius is not None:
            x1, y1, _, _ = bounds
            gx = cx_fit + x1
            gy = cy_fit + y1

            center_dist_prev = np.hypot(gx - prev_center[0], gy - prev_center[1])
            radius_diff_prev = abs(r_fit - prev_radius)

            score -= RANSAC_CENTER_PENALTY * center_dist_prev
            score -= RANSAC_RADIUS_PENALTY * radius_diff_prev

        if score > best_score:
            best_score = score
            best_circle = (cx_fit, cy_fit, r_fit)
            best_inliers = len(inliers)

    if best_circle is None:
        return None, crop, circle_mask, edges, None, 0

    x1, y1, x2, y2 = bounds
    cx_fit, cy_fit, r_fit = best_circle

    return ((cx_fit + x1, cy_fit + y1), r_fit), crop, circle_mask, edges, best_score, best_inliers


# =========================================================
# OPTICAL FLOW
# =========================================================
def estimate_optical_flow(prev_img, curr_img, prev_center, prev_radius):
    """
    Estimate local motion near the previous detected eye/core center
    using Lucas-Kanade optical flow.
    Returns (dx, dy, n_tracked) or (nan, nan, 0) if it fails.
    """
    if prev_center is None or prev_radius is None:
        return np.nan, np.nan, 0

    px, py = prev_center
    r = int(max(12, prev_radius * 1.0))

    h, w = prev_img.shape

    x1 = max(int(px) - r, 0)
    y1 = max(int(py) - r, 0)
    x2 = min(int(px) + r, w)
    y2 = min(int(py) + r, h)

    prev_crop = prev_img[y1:y2, x1:x2]
    curr_crop = curr_img[y1:y2, x1:x2]

    if prev_crop.size == 0 or curr_crop.size == 0:
        return np.nan, np.nan, 0

    pts_prev = cv2.goodFeaturesToTrack(
        prev_crop,
        maxCorners=FLOW_MAX_CORNERS,
        qualityLevel=FLOW_QUALITY_LEVEL,
        minDistance=FLOW_MIN_DISTANCE,
        blockSize=FLOW_BLOCK_SIZE
    )

    if pts_prev is None or len(pts_prev) == 0:
        return np.nan, np.nan, 0

    pts_curr, status, err = cv2.calcOpticalFlowPyrLK(
        prev_crop,
        curr_crop,
        pts_prev,
        None,
        winSize=FLOW_WIN_SIZE,
        maxLevel=FLOW_MAX_LEVEL,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    if pts_curr is None or status is None:
        return np.nan, np.nan, 0

    good_prev = pts_prev[status.flatten() == 1]
    good_curr = pts_curr[status.flatten() == 1]

    if len(good_prev) == 0:
        return np.nan, np.nan, 0

    motion = good_curr - good_prev
    dx = float(np.median(motion[:, 0, 0]))
    dy = float(np.median(motion[:, 0, 1]))
    n_tracked = len(good_prev)

    return dx, dy, n_tracked


def estimate_rotational_flow(prev_img, curr_img, center, radius):
    """
    Tracks points in an annulus around the eye and estimates whether
    motion is mostly counterclockwise or clockwise around the center.
    Returns a dictionary with rotation stats and tracked point pairs.
    """
    result = {
        "rotation_label": "Uncertain",
        "rotation_score": np.nan,
        "mean_ang_deg": np.nan,
        "ccw_fraction": np.nan,
        "cw_fraction": np.nan,
        "rot_n": 0,
        "pairs": [],
    }

    if center is None or radius is None:
        return result

    cx, cy = center
    inner_r = max(8.0, radius * ROT_INNER_RADIUS_FACTOR)
    outer_r = max(inner_r + 4.0, radius * ROT_OUTER_RADIUS_FACTOR)

    h, w = prev_img.shape

    x1 = max(int(cx - outer_r), 0)
    y1 = max(int(cy - outer_r), 0)
    x2 = min(int(cx + outer_r), w)
    y2 = min(int(cy + outer_r), h)

    prev_crop = prev_img[y1:y2, x1:x2]
    curr_crop = curr_img[y1:y2, x1:x2]

    if prev_crop.size == 0 or curr_crop.size == 0:
        return result

    # Build annulus mask
    yy, xx = np.ogrid[:prev_crop.shape[0], :prev_crop.shape[1]]
    local_cx = cx - x1
    local_cy = cy - y1
    dist = np.sqrt((xx - local_cx) ** 2 + (yy - local_cy) ** 2)

    annulus_mask = np.zeros_like(prev_crop, dtype=np.uint8)
    annulus_mask[(dist >= inner_r) & (dist <= outer_r)] = 255

    pts_prev = cv2.goodFeaturesToTrack(
        prev_crop,
        maxCorners=ROT_MAX_CORNERS,
        qualityLevel=ROT_QUALITY_LEVEL,
        minDistance=ROT_MIN_DISTANCE,
        blockSize=ROT_BLOCK_SIZE,
        mask=annulus_mask
    )

    if pts_prev is None or len(pts_prev) == 0:
        return result

    pts_curr, status, err = cv2.calcOpticalFlowPyrLK(
        prev_crop,
        curr_crop,
        pts_prev,
        None,
        winSize=FLOW_WIN_SIZE,
        maxLevel=FLOW_MAX_LEVEL,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    if pts_curr is None or status is None:
        return result

    good_prev = pts_prev[status.flatten() == 1]
    good_curr = pts_curr[status.flatten() == 1]

    if len(good_prev) < ROT_MIN_POINTS:
        return result

    ccw_count = 0
    cw_count = 0
    ang_list = []
    pairs = []

    for p_prev, p_curr in zip(good_prev, good_curr):
        x_prev = float(p_prev[0, 0] + x1)
        y_prev = float(p_prev[0, 1] + y1)
        x_curr = float(p_curr[0, 0] + x1)
        y_curr = float(p_curr[0, 1] + y1)

        rx = x_prev - cx
        ry = y_prev - cy
        vx = x_curr - x_prev
        vy = y_curr - y_prev

        radial_dist = np.hypot(rx, ry)
        if radial_dist < ROT_MIN_RADIUS_FOR_POINT:
            continue

        # 2D cross product sign tells CW vs CCW in image coordinates.
        # Since y increases downward in image coordinates, we flip vy.
        cross = rx * (-vy) - (-ry) * vx

        theta1 = np.arctan2(-(y_prev - cy), x_prev - cx)
        theta2 = np.arctan2(-(y_curr - cy), x_curr - cx)
        dtheta = theta2 - theta1
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]

        if cross > 0:
            ccw_count += 1
        elif cross < 0:
            cw_count += 1

        ang_list.append(np.degrees(dtheta))
        pairs.append(((x_prev, y_prev), (x_curr, y_curr)))

    valid_n = ccw_count + cw_count
    if valid_n < ROT_MIN_POINTS or len(ang_list) == 0:
        return result

    ccw_fraction = ccw_count / valid_n
    cw_fraction = cw_count / valid_n
    mean_ang_deg = float(np.median(ang_list))

    if ccw_fraction >= ROT_CCW_THRESHOLD:
        rotation_label = "CCW"
    elif cw_fraction >= ROT_CCW_THRESHOLD:
        rotation_label = "CW"
    else:
        rotation_label = "Uncertain"

    result["rotation_label"] = rotation_label
    result["rotation_score"] = max(ccw_fraction, cw_fraction)
    result["mean_ang_deg"] = mean_ang_deg
    result["ccw_fraction"] = ccw_fraction
    result["cw_fraction"] = cw_fraction
    result["rot_n"] = valid_n
    result["pairs"] = pairs

    return result


# =========================================================
# VELOCITY / MOTION HELPERS
# =========================================================
def compute_motion_metrics(prev_x, prev_y, curr_x, curr_y, dt_minutes=None):
    """
    Compute dx, dy, distance, direction angle, and speed.
    If dt_minutes is given, speed is pixels per minute.
    Otherwise speed is pixels per frame.
    """
    if any(np.isnan(v) for v in [prev_x, prev_y, curr_x, curr_y]):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    dx = curr_x - prev_x
    dy = curr_y - prev_y
    dist = float(np.hypot(dx, dy))
    angle_deg = float(np.degrees(np.arctan2(dy, dx)))

    if dt_minutes is not None and dt_minutes > 0:
        speed = dist / dt_minutes
    else:
        speed = dist

    return dx, dy, dist, angle_deg, speed


def angle_to_compass(dx, dy):
    """
    Convert image-coordinate motion into a simple compass direction.
    Since image y increases downward, use -dy for north/south logic.
    """
    if np.isnan(dx) or np.isnan(dy):
        return ""

    angle = np.degrees(np.arctan2(-dy, dx))
    angle = (angle + 360) % 360

    directions = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    idx = int((angle + 22.5) // 45) % 8
    return directions[idx]


def draw_velocity_arrow(img, start_xy, dx, dy, color=VELOCITY_COLOR, scale=3.0):
    """
    Draw a velocity arrow on the overlay image.
    """
    if np.isnan(dx) or np.isnan(dy):
        return img

    x0, y0 = int(start_xy[0]), int(start_xy[1])
    x1 = int(x0 + dx * scale)
    y1 = int(y0 + dy * scale)

    cv2.arrowedLine(img, (x0, y0), (x1, y1), color, 2, tipLength=0.25)
    return img


def draw_rotation_vectors(img, pairs, center=None, bins=12, scale=5.0):
    """
    Draw rotation arrows evenly spaced around the eyewall.
    Groups vectors by angle around the center so the motion
    looks like a circular swirl instead of random arrows.
    """

    if not pairs or center is None:
        return img

    cx, cy = center

    # store best vector for each angular bin
    bin_vectors = {}

    for (x0, y0), (x1, y1) in pairs:

        dx = x1 - x0
        dy = y1 - y0

        mag = np.hypot(dx, dy)
        if mag < 0.2:
            continue

        angle = np.arctan2(y0 - cy, x0 - cx)
        bin_id = int((angle + np.pi) / (2*np.pi) * bins)

        if bin_id not in bin_vectors or mag > bin_vectors[bin_id][2]:
            bin_vectors[bin_id] = ((x0, y0), (dx, dy), mag)

    for (x0, y0), (dx, dy), mag in bin_vectors.values():

        x_end = x0 + dx * scale
        y_end = y0 + dy * scale

        cv2.circle(
            img,
            (int(x0), int(y0)),
            2,
            ROTATION_COLOR,
            -1
        )

        cv2.arrowedLine(
            img,
            (int(x0), int(y0)),
            (int(x_end), int(y_end)),
            ROTATION_COLOR,
            1,
            tipLength=0.45
        )

    return img


# =========================================================
# PIPELINE FOR ONE FRAME
# =========================================================
def process_frame(image_path: str, frame_index: int, prev_center=None, prev_radius=None):
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"Could not read '{image_path}'")

    blur = cv2.GaussianBlur(img_gray, GAUSSIAN_BLUR_KSIZE, 0)

    # K-means segmentation
    Z = blur.reshape((-1, 1)).astype(np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        KMEANS_MAX_ITER,
        KMEANS_EPS
    )

    _, labels, centers = cv2.kmeans(
        Z,
        KMEANS_K,
        None,
        criteria,
        KMEANS_ATTEMPTS,
        cv2.KMEANS_RANDOM_CENTERS
    )

    seg = np.uint8(centers)[labels.flatten()].reshape(blur.shape)

    # Stage 1: coarse detection
    coarse_cand = find_eye_candidate(seg, prev_center=prev_center, prev_radius=prev_radius)

    # Stage 2: RANSAC local refinement
    ransac_cand, ransac_crop, ransac_mask, ransac_edges, ransac_score, ransac_inliers = \
        find_eye_candidate_ransac_local(
            seg,
            coarse_cand,
            prev_center=prev_center,
            prev_radius=prev_radius
        )

    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    if coarse_cand is not None:
        (cx, cy), cr = coarse_cand
        cv2.circle(vis, (int(cx), int(cy)), int(cr), COARSE_COLOR, 1)
        cv2.drawMarker(vis, (int(cx), int(cy)), COARSE_COLOR, cv2.MARKER_CROSS, 12, 1)

    found = False
    x = y = r = np.nan

    if ransac_cand is not None:
        (x, y), r = ransac_cand
        found = True

        cv2.circle(vis, (int(x), int(y)), int(r), RANSAC_COLOR, 2)
        cv2.drawMarker(vis, (int(x), int(y)), RANSAC_COLOR, cv2.MARKER_CROSS, 14, 2)
        cv2.putText(
            vis,
            f"Frame {frame_index:03d}  Eye: ({x:.0f}, {y:.0f})  r={r:.0f}",
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
            f"Frame {frame_index:03d}  No RANSAC eye found",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )

    if prev_center is not None:
        cv2.drawMarker(
            vis,
            (int(prev_center[0]), int(prev_center[1])),
            TRACK_COLOR,
            cv2.MARKER_TILTED_CROSS,
            14,
            2
        )

    return {
        "img_gray": img_gray,
        "seg": seg,
        "vis": vis,
        "ransac_crop": ransac_crop,
        "ransac_mask": ransac_mask,
        "ransac_edges": ransac_edges,
        "found": found,
        "x": float(x) if found else np.nan,
        "y": float(y) if found else np.nan,
        "r": float(r) if found else np.nan,
        "score": float(ransac_score) if ransac_score is not None else np.nan,
        "inliers": int(ransac_inliers),
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
            xs, ys, rs = x, y, r
            jump_flag = False
        else:
            raw_jump = np.hypot(x - prev_x, y - prev_y)
            jump_flag = raw_jump > MAX_REASONABLE_JUMP

            if USE_TEMPORAL_SMOOTHING:
                xs = SMOOTHING_ALPHA * x + (1 - SMOOTHING_ALPHA) * prev_x
                ys = SMOOTHING_ALPHA * y + (1 - SMOOTHING_ALPHA) * prev_y
                rs = SMOOTHING_ALPHA * r + (1 - SMOOTHING_ALPHA) * prev_r
            else:
                xs, ys, rs = x, y, r

        row["x_smooth"] = float(xs)
        row["y_smooth"] = float(ys)
        row["r_smooth"] = float(rs)
        row["jump_flag"] = jump_flag

        prev_x, prev_y, prev_r = xs, ys, rs

    return results


def add_frame_differences(results):
    prev = None
    for row in results:
        for key in [
            "dx", "dy", "dr", "dist_moved",
            "dx_smooth", "dy_smooth", "dr_smooth", "dist_moved_smooth"
        ]:
            row[key] = np.nan

        if prev is not None:
            if row["found"] and prev["found"]:
                row["dx"] = row["x"] - prev["x"]
                row["dy"] = row["y"] - prev["y"]
                row["dr"] = row["r"] - prev["r"]
                row["dist_moved"] = float(np.hypot(row["dx"], row["dy"]))

            if (
                not np.isnan(row["x_smooth"]) and not np.isnan(prev["x_smooth"])
                and not np.isnan(row["y_smooth"]) and not np.isnan(prev["y_smooth"])
            ):
                row["dx_smooth"] = row["x_smooth"] - prev["x_smooth"]
                row["dy_smooth"] = row["y_smooth"] - prev["y_smooth"]
                row["dr_smooth"] = row["r_smooth"] - prev["r_smooth"]
                row["dist_moved_smooth"] = float(np.hypot(row["dx_smooth"], row["dy_smooth"]))

        prev = row

    return results


# =========================================================
# CSV + TRACK PLOT
# =========================================================
def write_results_csv(results, csv_path):
    fieldnames = [
        "frame_index",
        "filename",
        "timestamp",
        "found",
        "x", "y", "r",
        "score", "inliers",
        "flow_dx", "flow_dy", "flow_dist", "flow_n", "ransac_vs_flow_error",
        "motion_dx", "motion_dy", "motion_dist",
        "motion_angle_deg", "motion_direction", "motion_speed",
        "rotation_label", "rotation_score", "mean_ang_deg", "confidence", "confidence_label",
        "used_fallback", "fallback_source",
        "ccw_fraction", "cw_fraction", "rot_n",
        "x_smooth", "y_smooth", "r_smooth",
        "jump_flag",
        "dx", "dy", "dr", "dist_moved",
        "dx_smooth", "dy_smooth", "dr_smooth", "dist_moved_smooth",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def save_track_plot(results, out_path):
    xs = [r["x_smooth"] for r in results if not np.isnan(r["x_smooth"])]
    ys = [r["y_smooth"] for r in results if not np.isnan(r["y_smooth"])]

    if len(xs) < 2:
        return

    plt.figure(figsize=(8, 8))
    plt.plot(xs, ys, marker="o")
    plt.gca().invert_yaxis()  # image coordinates
    plt.title("Smoothed Eye Center Track")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def draw_color_legend(img, x0=20, y0=220):
    dy = 28
    box = 14

    legend_items = [
        ("Coarse eye", COARSE_COLOR),
        ("RANSAC eye", RANSAC_COLOR),
        ("Prev center", TRACK_COLOR),
        ("Storm motion", VELOCITY_COLOR),
        ("Eyewall rotation", ROTATION_COLOR),
        ("Optical flow prediction", FLOW_COLOR),
        ("ML correction", (255, 0, 255)),
        ("Optical flow fallback", (0, 140, 255)),
        ("Previous center fallback", (255, 50, 50)),
    ]

    cv2.putText(
        img,
        "Color Key",
        (x0, y0 - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        TEXT_COLOR,
        2,
        cv2.LINE_AA
    )

    for i, (label, color) in enumerate(legend_items):
        yy = y0 + i * dy

        cv2.rectangle(
            img,
            (x0, yy - box + 2),
            (x0 + box, yy + 2),
            color,
            -1
        )

        cv2.putText(
            img,
            label,
            (x0 + 24, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            TEXT_COLOR,
            2,
            cv2.LINE_AA
        )

    return img

# =========================================================
# HUD DRAWING
# =========================================================
def draw_hud(img, x, y, speed_kmh, direction,
             rotation_label, rotation_rate,
             confidence, fallback_source):

    # HUD box
    x0, y0 = 20, 80
    w, h = 320, 170

    # Background (dark panel)
    cv2.rectangle(img, (x0, y0), (x0+w, y0+h), (25, 25, 25), -1)
    cv2.rectangle(img, (x0, y0), (x0+w, y0+h), (200, 200, 200), 2)

    line_y = y0 + 25
    dy = 22

    def put(text, color=(255,255,255)):
        nonlocal line_y
        cv2.putText(img, text, (x0+10, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        line_y += dy

    # =========================
    # DATA
    # =========================
    put("STORM TRACKING HUD")
    put(f"X,Y: {int(x)}, {int(y)}")

    if not np.isnan(speed_kmh):
        put(f"Speed: {speed_kmh:.1f} km/h", (0,255,255))

    put(f"Dir: {direction}", (0,255,255))

    if not np.isnan(rotation_rate):
        put(f"Rotation: {rotation_label} {rotation_rate:.2f} deg/min", (0,165,255))

    # Confidence
    conf_color = (0,255,0) if confidence >= 0.5 else (0,0,255)
    put(f"Confidence: {confidence*100:.0f}%", conf_color)

    # =========================
    # MODE / STATE
    # =========================
    label_map = {
        "ml_correction": "ML",
        "optical_flow": "FLOW",
        "previous_center": "HOLD",
        "none": "LOCKED"
    }

    color_map = {
        "ml_correction": (255, 0, 255),
        "optical_flow": (0, 140, 255),
        "previous_center": (255, 50, 50),
        "none": (0, 255, 0)
    }

    mode_label = label_map.get(fallback_source, fallback_source)
    mode_color = color_map.get(fallback_source, (255,255,255))

    put(f"Mode: {mode_label}", mode_color)

# =========================================================
# MAIN
# =========================================================
def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(os.path.join(OUTPUT_DIR, "overlays"))

    files = list_image_files(INPUT_DIR)
    if not files:
        raise FileNotFoundError(f"No image files found in folder: {INPUT_DIR}")

    print("MAIN STARTED")
    print(f"Files found: {len(files)}")

    files_with_time = sort_files_by_time_or_name(files)

    results = []

    prev_center_for_guidance = None
    prev_radius_for_guidance = None
    prev_img_gray_for_flow = None

    prev_motion_speed = None
    prev_flow_dist = None
    prev_rotation_score = None

    total_frames = 0
    ml_used = 0
    bad_frames = 0
    flow_failures = 0

    def smooth(prev, curr, alpha=0.3):
        if prev is None or np.isnan(prev) or np.isnan(curr):
            return curr
        return alpha * curr + (1 - alpha) * prev

    for idx, (image_path, timestamp) in enumerate(files_with_time, start=1):
        total_frames += 1
        print(f"\n[{idx}] Processing {os.path.basename(image_path)}")

        out = process_frame(
            image_path,
            idx,
            prev_center=prev_center_for_guidance,
            prev_radius=prev_radius_for_guidance
        )

        timestamp_str = timestamp.isoformat() if timestamp else ""

        # =========================
        # OPTICAL FLOW
        # =========================
        flow_dx = flow_dy = np.nan
        flow_n = 0
        flow_dist = np.nan

        if prev_img_gray_for_flow is not None and prev_center_for_guidance is not None:
            flow_dx, flow_dy, flow_n = estimate_optical_flow(
                prev_img_gray_for_flow,
                out["img_gray"],
                prev_center_for_guidance,
                prev_radius_for_guidance
            )

            if flow_n == 0:
                flow_failures += 1

            if not np.isnan(flow_dx):
                flow_dist = float(np.hypot(flow_dx, flow_dy))

                # draw flow arrow
                px, py = prev_center_for_guidance
                cv2.arrowedLine(
                    out["vis"],
                    (int(px), int(py)),
                    (int(px + flow_dx), int(py + flow_dy)),
                    FLOW_COLOR,
                    2
                )

        # =========================
        # MOTION
        # =========================
        motion_dx = motion_dy = motion_dist = np.nan
        motion_angle_deg = motion_speed = np.nan
        motion_direction = ""

        dt_minutes = None
        if idx > 1:
            prev_ts = files_with_time[idx - 2][1]
            if prev_ts and timestamp:
                dt_minutes = (timestamp - prev_ts).total_seconds() / 60.0

        if prev_center_for_guidance is not None and out["found"]:
            motion_dx, motion_dy, motion_dist, motion_angle_deg, motion_speed = compute_motion_metrics(
                prev_center_for_guidance[0],
                prev_center_for_guidance[1],
                out["x"],
                out["y"],
                dt_minutes=dt_minutes
            )

            motion_direction = angle_to_compass(motion_dx, motion_dy)

            draw_velocity_arrow(out["vis"], prev_center_for_guidance, motion_dx, motion_dy)

        # =========================
        # REAL WORLD SPEED
        # =========================
        if not np.isnan(motion_speed):
            speed_kmh = motion_speed * KM_PER_PIXEL * 60
        else:
            speed_kmh = np.nan

        # =========================
        # DISTANCE
        # =========================
        if not np.isnan(motion_dist):
            distance_px = motion_dist
            distance_km = motion_dist * KM_PER_PIXEL
        else:
            distance_px = np.nan
            distance_km = np.nan
        # =========================
        # ROTATION
        # =========================
        rotation_label = "Uncertain"
        rotation_score = mean_ang_deg = np.nan
        ccw_fraction = cw_fraction = np.nan
        rot_n = 0
        rotation_rate = np.nan

        if prev_img_gray_for_flow is not None and prev_center_for_guidance is not None:
            rot = estimate_rotational_flow(
                prev_img_gray_for_flow,
                out["img_gray"],
                prev_center_for_guidance,
                prev_radius_for_guidance
            )

            rotation_label = rot["rotation_label"]
            rotation_score = rot["rotation_score"]
            mean_ang_deg = rot["mean_ang_deg"]
            ccw_fraction = rot["ccw_fraction"]
            cw_fraction = rot["cw_fraction"]
            rot_n = rot["rot_n"]

            draw_rotation_vectors(out["vis"], rot["pairs"], center=prev_center_for_guidance)

        # =========================
        # ROTATION RATE (deg/min)
        # =========================
        if not np.isnan(mean_ang_deg):
            if dt_minutes and dt_minutes > 0:
                rotation_rate = mean_ang_deg / dt_minutes
            else:
                rotation_rate = mean_ang_deg
        # =========================
        # SMOOTH FEATURES
        # =========================
        motion_speed_s = smooth(prev_motion_speed, motion_speed)
        flow_dist_s = smooth(prev_flow_dist, flow_dist)
        rotation_score_s = smooth(prev_rotation_score, rotation_score)

        # =========================
        # FEATURES
        # =========================
        features_vec = [
            out["r"], out["score"], out["inliers"],
            flow_dist_s, motion_dist, motion_speed_s,
            mean_ang_deg, rotation_score_s, rot_n
        ]

        features_vec = [0 if pd.isna(x) else x for x in features_vec]

        X = pd.DataFrame([features_vec], columns=[
            "r", "score", "inliers",
            "flow_dist", "motion_dist", "motion_speed",
            "mean_ang_deg", "rotation_score", "rot_n"
        ])
        X_scaled = scaler.transform(X)

        dx = np.clip(model_dx.predict(X_scaled)[0], -50, 50)
        dy = np.clip(model_dy.predict(X_scaled)[0], -50, 50)

        corrected_x = out["x"] + dx
        corrected_y = out["y"] + dy

        confidence = confidence_model.predict_proba([features_vec])[0][1]
        confidence_pct = confidence * 100

        used_fallback = False
        fallback_source = "none"

        if USE_CONFIDENCE_FALLBACK and out["found"]:

            if confidence < CONFIDENCE_THRESHOLD:

                # ===== ML FIRST =====
                if not np.isnan(dx) and not np.isnan(dy):
                    out["x"] = corrected_x
                    out["y"] = corrected_y
                    fallback_source = "ml_correction"
                    used_fallback = True
                    ml_used += 1

                # ===== FLOW SECOND =====
                elif (
                    prev_center_for_guidance is not None
                    and not np.isnan(flow_dx)
                    and not np.isnan(flow_dy)
                ):
                    out["x"] = prev_center_for_guidance[0] + flow_dx
                    out["y"] = prev_center_for_guidance[1] + flow_dy
                    fallback_source = "optical_flow"
                    used_fallback = True

                # ===== LAST RESORT =====
                elif prev_center_for_guidance is not None:
                    out["x"] = prev_center_for_guidance[0]
                    out["y"] = prev_center_for_guidance[1]
                    fallback_source = "previous_center"
                    used_fallback = True

        if not out["found"] or out["inliers"] < MIN_INLIERS_FOR_ML:
            bad_frames += 1

        # If ML correction was not possible and flow also failed, count as bad frame.
        if not out["found"]:
            state = "lost"

        elif used_fallback:
            if fallback_source == "ml_correction":
                state = "ml"
            elif fallback_source == "optical_flow":
                state = "flow"
            else:
                state = "hold"

        else:
            state = "locked"

        print(f"x={out['x']:.1f}, y={out['y']:.1f}, dx={dx:.2f}, dy={dy:.2f}, conf={confidence:.2f}, fallback={fallback_source}")

        # =========================
        # TEXT OVERLAY (HUD ONLY)
        # =========================
        draw_hud(
            out["vis"],
            out["x"], out["y"],
            speed_kmh,
            motion_direction,
            rotation_label,
            rotation_rate,
            confidence,
            fallback_source
        )

        # =========================
        # DRAW ML / FALLBACK MARKER
        # =========================
        if fallback_source != "none":
            color = FALLBACK_COLORS.get(fallback_source, ML_CORRECTION_COLOR)

            cv2.drawMarker(
                out["vis"],
                (int(out["x"]), int(out["y"])),
                color,
                cv2.MARKER_STAR,
                20,
                2
            )

        print(f"FALLBACK SOURCE {fallback_source}")

        # =========================
        # SAVE
        # =========================
        draw_color_legend(out["vis"], x0=400, y0=120)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "overlays", f"{idx:03d}.png"), out["vis"])

        results.append({
            "frame_index": idx,
            "x": out["x"],
            "y": out["y"],
            "confidence": confidence
        })

        # update state
        prev_img_gray_for_flow = out["img_gray"]
        if out["found"]:
            prev_center_for_guidance = (out["x"], out["y"])
            prev_radius_for_guidance = out["r"]

        prev_motion_speed = motion_speed_s
        prev_flow_dist = flow_dist_s
        prev_rotation_score = rotation_score_s

    print("\n=== DIAGNOSTICS ===")
    print(f"Frames: {total_frames}")
    print(f"ML used: {ml_used}")
    print(f"Bad frames: {bad_frames}")
    print(f"Flow failures: {flow_failures}")




if __name__ == "__main__":
    print("STARTING PIPELINE")
    main()
