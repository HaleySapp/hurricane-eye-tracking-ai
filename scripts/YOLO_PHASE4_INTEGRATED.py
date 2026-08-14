from pathlib import Path
from datetime import datetime
import csv
import math
import re

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = Path("data/yolo_export/images")

OUTPUT_DIR = Path("results/phase4_integrated_milton")
OVERLAY_DIR = OUTPUT_DIR / "overlays"
CSV_PATH = OUTPUT_DIR / "phase4_integrated_results.csv"

MODEL_PATH = Path(
    "runs/detect/runs/tiled_yolo/eye_detector_v1/weights/best.pt"
)

STORM_NAME = "milton"

TILE_SIZE = 1024
OVERLAP = 256
STRIDE = TILE_SIZE - OVERLAP

YOLO_CONF_THRESHOLD = 0.25

MERGE_DISTANCE = 100.0
MAX_TRACK_DISTANCE = 300.0

# Local region used for optical flow / rotation.
# YOLO is used to locate the eye center.
ANALYSIS_RADIUS = 80.0

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ============================================================
# OPTICAL FLOW CONFIG
# ============================================================

FLOW_MAX_CORNERS = 50
FLOW_QUALITY_LEVEL = 0.01
FLOW_MIN_DISTANCE = 5
FLOW_BLOCK_SIZE = 7
FLOW_WIN_SIZE = (21, 21)
FLOW_MAX_LEVEL = 3


# ============================================================
# ROTATION CONFIG
# ============================================================

ROT_INNER_RADIUS_FACTOR = 0.90
ROT_OUTER_RADIUS_FACTOR = 1.60

ROT_MAX_CORNERS = 80
ROT_QUALITY_LEVEL = 0.01
ROT_MIN_DISTANCE = 5
ROT_BLOCK_SIZE = 7

ROT_MIN_POINTS = 6
ROT_MIN_RADIUS_FOR_POINT = 6.0

ROT_CCW_THRESHOLD = 0.60


# ============================================================
# COLORS
# ============================================================

YOLO_COLOR = (0, 0, 255)
TRACK_COLOR = (0, 255, 0)
FLOW_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)


# ============================================================
# HELPERS
# ============================================================

def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)


def parse_timestamp_from_filename(filename):
    match = re.search(
        r"_(\d{8})_(\d{4})",
        filename
    )

    if not match:
        return None

    return datetime.strptime(
        match.group(1) + match.group(2),
        "%Y%m%d%H%M"
    )


def get_storm_files():
    files = list(
        INPUT_DIR.glob(
            f"*{STORM_NAME}_full_*.png"
        )
    )

    files.sort(
        key=lambda p:
        parse_timestamp_from_filename(p.name)
        or datetime.min
    )

    return files


def get_tile_positions(length):
    if length <= TILE_SIZE:
        return [0]

    positions = list(
        range(
            0,
            length - TILE_SIZE + 1,
            STRIDE
        )
    )

    last = length - TILE_SIZE

    if positions[-1] != last:
        positions.append(last)

    return positions


def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )


# ============================================================
# YOLO FULL-DISK DETECTION
# ============================================================

def detect_full_disk(model, image_path):

    pil_image = Image.open(
        image_path
    ).convert("RGB")

    width, height = pil_image.size

    x_positions = get_tile_positions(width)
    y_positions = get_tile_positions(height)

    detections = []

    for tile_y in y_positions:
        for tile_x in x_positions:

            tile = pil_image.crop(
                (
                    tile_x,
                    tile_y,
                    tile_x + TILE_SIZE,
                    tile_y + TILE_SIZE
                )
            )

            results = model.predict(
                source=tile,
                imgsz=TILE_SIZE,
                conf=YOLO_CONF_THRESHOLD,
                device=DEVICE,
                verbose=False
            )

            result = results[0]

            if result.boxes is None:
                continue

            for box in result.boxes:

                confidence = float(
                    box.conf[0].cpu()
                )

                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .cpu()
                    .tolist()
                )

                detections.append(
                    {
                        "x1": x1 + tile_x,
                        "y1": y1 + tile_y,
                        "x2": x2 + tile_x,
                        "y2": y2 + tile_y,
                        "confidence": confidence
                    }
                )

    return detections


# ============================================================
# DUPLICATE MERGING
# ============================================================

def merge_detections(detections):

    if not detections:
        return []

    detections = sorted(
        detections,
        key=lambda d: d["confidence"],
        reverse=True
    )

    merged = []

    for detection in detections:

        cx = (
            detection["x1"] +
            detection["x2"]
        ) / 2

        cy = (
            detection["y1"] +
            detection["y2"]
        ) / 2

        duplicate = False

        for existing in merged:

            ex = (
                existing["x1"] +
                existing["x2"]
            ) / 2

            ey = (
                existing["y1"] +
                existing["y2"]
            ) / 2

            dist = euclidean_distance(
                cx, cy,
                ex, ey
            )

            if dist <= MERGE_DISTANCE:
                duplicate = True
                break

        if not duplicate:
            merged.append(detection)

    return merged


# ============================================================
# TEMPORAL SELECTION
# ============================================================

def select_tracked_detection(
    candidates,
    previous_center
):

    if not candidates:
        return None

    if previous_center is None:
        return max(
            candidates,
            key=lambda d: d["confidence"]
        )

    px, py = previous_center

    scored = []

    for candidate in candidates:

        cx = (
            candidate["x1"] +
            candidate["x2"]
        ) / 2

        cy = (
            candidate["y1"] +
            candidate["y2"]
        ) / 2

        dist = euclidean_distance(
            px, py,
            cx, cy
        )

        scored.append(
            (
                dist,
                candidate
            )
        )

    scored.sort(
        key=lambda item: item[0]
    )

    best_distance, best = scored[0]

    if best_distance > MAX_TRACK_DISTANCE:
        return None

    return best


# ============================================================
# OPTICAL FLOW
# ============================================================

def estimate_optical_flow(
    prev_img,
    curr_img,
    center,
    radius
):

    if center is None:
        return np.nan, np.nan, 0

    px, py = center

    r = int(
        max(
            12,
            radius
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
        blockSize=FLOW_BLOCK_SIZE
    )

    if (
        pts_prev is None
        or len(pts_prev) == 0
    ):
        return np.nan, np.nan, 0

    pts_curr, status, _ = (
        cv2.calcOpticalFlowPyrLK(
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
                0.01
            )
        )
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

    motion = good_curr - good_prev

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

    return (
        dx,
        dy,
        len(good_prev)
    )


# ============================================================
# ROTATION V2
# REMOVE STORM TRANSLATION FIRST
# ============================================================

def estimate_rotational_flow(
    prev_img,
    curr_img,
    previous_center,
    current_center,
    radius
):

    result = {
        "rotation_label": "Uncertain",
        "rotation_score": np.nan,
        "mean_ang_deg": np.nan,
        "ccw_fraction": np.nan,
        "cw_fraction": np.nan,
        "rot_n": 0,
        "center_dx": np.nan,
        "center_dy": np.nan,
    }

    if (
        previous_center is None
        or current_center is None
    ):
        return result

    prev_cx, prev_cy = previous_center
    curr_cx, curr_cy = current_center

    # --------------------------------------------------------
    # STORM TRANSLATION
    # --------------------------------------------------------

    center_dx = curr_cx - prev_cx
    center_dy = curr_cy - prev_cy

    result["center_dx"] = center_dx
    result["center_dy"] = center_dy

    inner_r = max(
        8.0,
        radius *
        ROT_INNER_RADIUS_FACTOR
    )

    outer_r = max(
        inner_r + 4.0,
        radius *
        ROT_OUTER_RADIUS_FACTOR
    )

    h, w = prev_img.shape

    x1 = max(
        int(prev_cx - outer_r),
        0
    )

    y1 = max(
        int(prev_cy - outer_r),
        0
    )

    x2 = min(
        int(prev_cx + outer_r),
        w
    )

    y2 = min(
        int(prev_cy + outer_r),
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

    local_cx = prev_cx - x1
    local_cy = prev_cy - y1

    dist = np.sqrt(
        (xx - local_cx) ** 2
        +
        (yy - local_cy) ** 2
    )

    mask = np.zeros_like(
        prev_crop,
        dtype=np.uint8
    )

    mask[
        (dist >= inner_r)
        &
        (dist <= outer_r)
    ] = 255

    pts_prev = cv2.goodFeaturesToTrack(
        prev_crop,
        maxCorners=ROT_MAX_CORNERS,
        qualityLevel=ROT_QUALITY_LEVEL,
        minDistance=ROT_MIN_DISTANCE,
        blockSize=ROT_BLOCK_SIZE,
        mask=mask
    )

    if (
        pts_prev is None
        or len(pts_prev) == 0
    ):
        return result

    pts_curr, status, _ = (
        cv2.calcOpticalFlowPyrLK(
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
                0.01
            )
        )
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

    angular_changes = []

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

        x_curr_raw = float(
            p_curr[0, 0] + x1
        )

        y_curr_raw = float(
            p_curr[0, 1] + y1
        )

        # ----------------------------------------------------
        # REMOVE WHOLE-STORM TRANSLATION
        # ----------------------------------------------------

        x_curr = (
            x_curr_raw -
            center_dx
        )

        y_curr = (
            y_curr_raw -
            center_dy
        )

        rx = x_prev - prev_cx
        ry = y_prev - prev_cy

        vx = x_curr - x_prev
        vy = y_curr - y_prev

        radial_dist = np.hypot(
            rx,
            ry
        )

        if (
            radial_dist
            <
            ROT_MIN_RADIUS_FOR_POINT
        ):
            continue

        # Cartesian sign correction because image y increases downward
        cross = (
            rx * (-vy)
            -
            (-ry) * vx
        )

        theta1 = np.arctan2(
            -(y_prev - prev_cy),
            x_prev - prev_cx
        )

        theta2 = np.arctan2(
            -(y_curr - prev_cy),
            x_curr - prev_cx
        )

        dtheta = theta2 - theta1

        dtheta = (
            dtheta + np.pi
        ) % (
            2 * np.pi
        ) - np.pi

        if cross > 0:
            ccw_count += 1

        elif cross < 0:
            cw_count += 1

        angular_changes.append(
            np.degrees(dtheta)
        )

    valid_n = (
        ccw_count +
        cw_count
    )

    if (
        valid_n < ROT_MIN_POINTS
        or not angular_changes
    ):
        return result

    ccw_fraction = (
        ccw_count /
        valid_n
    )

    cw_fraction = (
        cw_count /
        valid_n
    )

    mean_ang_deg = float(
        np.median(
            angular_changes
        )
    )

    if (
        ccw_fraction
        >= ROT_CCW_THRESHOLD
    ):
        label = "CCW"

    elif (
        cw_fraction
        >= ROT_CCW_THRESHOLD
    ):
        label = "CW"

    else:
        label = "Uncertain"

    result.update(
        {
            "rotation_label": label,
            "rotation_score": max(
                ccw_fraction,
                cw_fraction
            ),
            "mean_ang_deg": mean_ang_deg,
            "ccw_fraction": ccw_fraction,
            "cw_fraction": cw_fraction,
            "rot_n": valid_n,
        }
    )

    return result


# ============================================================
# MOTION
# ============================================================

def compute_motion(
    previous_center,
    current_center,
    dt_minutes
):

    if (
        previous_center is None
        or current_center is None
        or dt_minutes is None
        or dt_minutes <= 0
    ):
        return {
            "dx": np.nan,
            "dy": np.nan,
            "distance_px": np.nan,
            "speed_px_per_min": np.nan
        }

    px, py = previous_center
    cx, cy = current_center

    dx = cx - px
    dy = cy - py

    distance_px = float(
        np.hypot(
            dx,
            dy
        )
    )

    speed = (
        distance_px /
        dt_minutes
    )

    return {
        "dx": dx,
        "dy": dy,
        "distance_px": distance_px,
        "speed_px_per_min": speed
    }


# ============================================================
# OVERLAY
# ============================================================

def draw_overlay(
    image_path,
    frame_index,
    current_center,
    previous_center,
    confidence,
    motion,
    flow_dx,
    flow_dy,
    flow_n,
    rotation,
    dt_minutes
):

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR
    )

    if image is None:
        return None

    if current_center is not None:

        cx, cy = current_center

        cv2.circle(
            image,
            (
                int(cx),
                int(cy)
            ),
            18,
            YOLO_COLOR,
            5
        )

        cv2.drawMarker(
            image,
            (
                int(cx),
                int(cy)
            ),
            YOLO_COLOR,
            cv2.MARKER_CROSS,
            30,
            4
        )

        cv2.circle(
            image,
            (
                int(cx),
                int(cy)
            ),
            int(ANALYSIS_RADIUS),
            TRACK_COLOR,
            2
        )

    if previous_center is not None:

        px, py = previous_center

        cv2.drawMarker(
            image,
            (
                int(px),
                int(py)
            ),
            TRACK_COLOR,
            cv2.MARKER_TILTED_CROSS,
            24,
            3
        )

        if (
            not np.isnan(flow_dx)
            and
            not np.isnan(flow_dy)
        ):

            cv2.arrowedLine(
                image,
                (
                    int(px),
                    int(py)
                ),
                (
                    int(px + flow_dx * 8),
                    int(py + flow_dy * 8)
                ),
                FLOW_COLOR,
                3,
                tipLength=0.25
            )

    x0 = 30
    y0 = 50

    cv2.rectangle(
        image,
        (20, 20),
        (760, 400),
        (25, 25, 25),
        -1
    )

    lines = [
        "PHASE 4 - TILED YOLO + ROTATION V2",
        f"Frame: {frame_index}",
        f"Storm: {STORM_NAME.upper()}",
    ]

    if current_center is not None:

        lines.extend(
            [
                (
                    f"Center: "
                    f"({current_center[0]:.1f}, "
                    f"{current_center[1]:.1f})"
                ),
                (
                    f"YOLO confidence: "
                    f"{confidence:.3f}"
                )
            ]
        )

    else:

        lines.append(
            "YOLO: NO TRACKED DETECTION"
        )

    if (
        dt_minutes is not None
        and
        not np.isnan(
            motion["distance_px"]
        )
    ):

        lines.extend(
            [
                (
                    f"dt: "
                    f"{dt_minutes:.1f} min"
                ),
                (
                    f"Motion: "
                    f"{motion['distance_px']:.2f} px"
                ),
                (
                    f"Pixel speed: "
                    f"{motion['speed_px_per_min']:.3f} px/min"
                )
            ]
        )

    if flow_n > 0:

        lines.append(
            (
                f"Optical flow: "
                f"dx={flow_dx:.2f}, "
                f"dy={flow_dy:.2f}, "
                f"n={flow_n}"
            )
        )

    lines.extend(
        [
            (
                f"Rotation: "
                f"{rotation['rotation_label']}"
            ),
            (
                f"Rotation agreement: "
                f"{rotation['rotation_score']:.3f}"
                if not np.isnan(
                    rotation["rotation_score"]
                )
                else
                "Rotation agreement: N/A"
            ),
            (
                f"Median residual angle: "
                f"{rotation['mean_ang_deg']:.3f} deg"
                if not np.isnan(
                    rotation["mean_ang_deg"]
                )
                else
                "Median residual angle: N/A"
            )
        ]
    )

    for text in lines:

        cv2.putText(
            image,
            text,
            (
                x0,
                y0
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            TEXT_COLOR,
            2,
            cv2.LINE_AA
        )

        y0 += 30

    return image


# ============================================================
# MAIN
# ============================================================

def main():

    ensure_dirs()

    files = get_storm_files()

    if not files:
        raise FileNotFoundError(
            f"No {STORM_NAME} images found."
        )

    print("=" * 72)
    print("YOLO PHASE 4 INTEGRATED PIPELINE - ROTATION V2")
    print("=" * 72)

    print(
        f"Storm: {STORM_NAME.upper()}"
    )

    print(
        f"Images: {len(files)}"
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Model: {MODEL_PATH}"
    )

    print()

    model = YOLO(
        str(MODEL_PATH)
    )

    # Persistent tracker state across gaps
    previous_tracked_center = None
    previous_detection_time = None

    # Adjacent-frame scientific-analysis state
    previous_frame_gray = None
    previous_frame_time = None
    previous_frame_center = None

    rows = []

    tracked_frames = 0
    missing_frames = 0
    false_candidates_rejected = 0

    for frame_index, image_path in enumerate(
        files,
        start=1
    ):

        timestamp = (
            parse_timestamp_from_filename(
                image_path.name
            )
        )

        print(
            f"[{frame_index:02d}/{len(files)}] "
            f"{image_path.name}"
        )

        # ----------------------------------------------------
        # YOLO FULL-DISK DETECTION
        # ----------------------------------------------------

        raw_detections = detect_full_disk(
            model,
            image_path
        )

        candidates = merge_detections(
            raw_detections
        )

        selected = (
            select_tracked_detection(
                candidates,
                previous_tracked_center
            )
        )

        if selected is not None:

            center_x = (
                selected["x1"]
                +
                selected["x2"]
            ) / 2

            center_y = (
                selected["y1"]
                +
                selected["y2"]
            ) / 2

            current_center = (
                center_x,
                center_y
            )

            confidence = (
                selected[
                    "confidence"
                ]
            )

            tracked_frames += 1

            rejected = max(
                0,
                len(candidates) - 1
            )

            false_candidates_rejected += (
                rejected
            )

        else:

            current_center = None
            confidence = np.nan

            rejected = len(candidates)

            false_candidates_rejected += (
                rejected
            )

            missing_frames += 1

        current_gray = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        # ----------------------------------------------------
        # TIME SINCE PREVIOUS VALID TRACKED DETECTION
        # ----------------------------------------------------

        dt_detection = None

        if (
            current_center is not None
            and
            previous_detection_time is not None
            and
            timestamp is not None
        ):

            dt_detection = (
                timestamp -
                previous_detection_time
            ).total_seconds() / 60.0

        # ----------------------------------------------------
        # TRACK MOTION
        # ----------------------------------------------------

        motion = compute_motion(
            previous_tracked_center,
            current_center,
            dt_detection
        )

        # ----------------------------------------------------
        # ADJACENT FRAME OPTICAL FLOW
        # ----------------------------------------------------

        flow_dx = np.nan
        flow_dy = np.nan
        flow_n = 0

        if (
            previous_frame_gray is not None
            and
            previous_frame_center is not None
            and
            current_center is not None
        ):

            (
                flow_dx,
                flow_dy,
                flow_n
            ) = estimate_optical_flow(
                previous_frame_gray,
                current_gray,
                previous_frame_center,
                ANALYSIS_RADIUS
            )

        # ----------------------------------------------------
        # ROTATION V2
        # ----------------------------------------------------

        rotation = {
            "rotation_label": "Uncertain",
            "rotation_score": np.nan,
            "mean_ang_deg": np.nan,
            "ccw_fraction": np.nan,
            "cw_fraction": np.nan,
            "rot_n": 0,
            "center_dx": np.nan,
            "center_dy": np.nan,
        }

        if (
            previous_frame_gray is not None
            and
            previous_frame_center is not None
            and
            current_center is not None
        ):

            rotation = (
                estimate_rotational_flow(
                    previous_frame_gray,
                    current_gray,
                    previous_frame_center,
                    current_center,
                    ANALYSIS_RADIUS
                )
            )

        # ----------------------------------------------------
        # ADJACENT SOURCE FRAME TIME
        # ----------------------------------------------------

        frame_dt = None

        if (
            previous_frame_time is not None
            and
            timestamp is not None
        ):

            frame_dt = (
                timestamp -
                previous_frame_time
            ).total_seconds() / 60.0

        rotation_rate = np.nan

        if (
            frame_dt is not None
            and
            frame_dt > 0
            and
            not np.isnan(
                rotation[
                    "mean_ang_deg"
                ]
            )
        ):

            rotation_rate = (
                rotation[
                    "mean_ang_deg"
                ]
                /
                frame_dt
            )

        # ----------------------------------------------------
        # OVERLAY
        # ----------------------------------------------------

        overlay = draw_overlay(
            image_path=image_path,
            frame_index=frame_index,
            current_center=current_center,
            previous_center=previous_frame_center,
            confidence=confidence,
            motion=motion,
            flow_dx=flow_dx,
            flow_dy=flow_dy,
            flow_n=flow_n,
            rotation=rotation,
            dt_minutes=dt_detection
        )

        if overlay is not None:

            cv2.imwrite(
                str(
                    OVERLAY_DIR /
                    f"{frame_index:03d}.png"
                ),
                overlay
            )

        # ----------------------------------------------------
        # CSV ROW
        # ----------------------------------------------------

        rows.append(
            {
                "frame_index": frame_index,
                "filename": image_path.name,
                "timestamp": (
                    timestamp.isoformat()
                    if timestamp
                    else ""
                ),
                "tracked": (
                    current_center
                    is not None
                ),
                "candidate_count": len(candidates),
                "raw_tile_detections": len(
                    raw_detections
                ),
                "rejected_candidates": rejected,
                "yolo_confidence": (
                    confidence
                    if not np.isnan(
                        confidence
                    )
                    else ""
                ),
                "center_x": (
                    current_center[0]
                    if current_center
                    else ""
                ),
                "center_y": (
                    current_center[1]
                    if current_center
                    else ""
                ),
                "analysis_radius": (
                    ANALYSIS_RADIUS
                    if current_center
                    else ""
                ),
                "dt_minutes_since_previous_detection": (
                    dt_detection
                    if dt_detection is not None
                    else ""
                ),
                "motion_dx": motion["dx"],
                "motion_dy": motion["dy"],
                "motion_distance_px": (
                    motion[
                        "distance_px"
                    ]
                ),
                "motion_speed_px_per_min": (
                    motion[
                        "speed_px_per_min"
                    ]
                ),
                "flow_dx": flow_dx,
                "flow_dy": flow_dy,
                "flow_n": flow_n,
                "rotation_center_dx": (
                    rotation[
                        "center_dx"
                    ]
                ),
                "rotation_center_dy": (
                    rotation[
                        "center_dy"
                    ]
                ),
                "rotation_label": (
                    rotation[
                        "rotation_label"
                    ]
                ),
                "rotation_score": (
                    rotation[
                        "rotation_score"
                    ]
                ),
                "rotation_mean_ang_deg": (
                    rotation[
                        "mean_ang_deg"
                    ]
                ),
                "rotation_rate_deg_per_min": (
                    rotation_rate
                ),
                "ccw_fraction": (
                    rotation[
                        "ccw_fraction"
                    ]
                ),
                "cw_fraction": (
                    rotation[
                        "cw_fraction"
                    ]
                ),
                "rotation_points": (
                    rotation[
                        "rot_n"
                    ]
                )
            }
        )

        # ----------------------------------------------------
        # UPDATE ADJACENT FRAME STATE
        # ----------------------------------------------------

        previous_frame_gray = (
            current_gray
        )

        previous_frame_time = (
            timestamp
        )

        previous_frame_center = (
            current_center
        )

        # ----------------------------------------------------
        # UPDATE PERSISTENT TRACK STATE
        # ----------------------------------------------------

        if current_center is not None:

            previous_tracked_center = (
                current_center
            )

            previous_detection_time = (
                timestamp
            )

        if current_center is not None:

            print(
                (
                    f"   TRACKED "
                    f"({current_center[0]:.1f}, "
                    f"{current_center[1]:.1f}) "
                    f"conf={confidence:.3f} "
                    f"candidates={len(candidates)} "
                    f"rotation="
                    f"{rotation['rotation_label']}"
                )
            )

        else:

            print(
                (
                    f"   MISSING "
                    f"candidates="
                    f"{len(candidates)}"
                )
            )

    # ========================================================
    # WRITE CSV
    # ========================================================

    fieldnames = [
        "frame_index",
        "filename",
        "timestamp",
        "tracked",
        "candidate_count",
        "raw_tile_detections",
        "rejected_candidates",
        "yolo_confidence",
        "center_x",
        "center_y",
        "analysis_radius",
        "dt_minutes_since_previous_detection",
        "motion_dx",
        "motion_dy",
        "motion_distance_px",
        "motion_speed_px_per_min",
        "flow_dx",
        "flow_dy",
        "flow_n",
        "rotation_center_dx",
        "rotation_center_dy",
        "rotation_label",
        "rotation_score",
        "rotation_mean_ang_deg",
        "rotation_rate_deg_per_min",
        "ccw_fraction",
        "cw_fraction",
        "rotation_points"
    ]

    with open(
        CSV_PATH,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 72)
    print("PHASE 4 ROTATION V2 COMPLETE")
    print("=" * 72)

    print(
        f"Total frames:             "
        f"{len(files)}"
    )

    print(
        f"Tracked frames:           "
        f"{tracked_frames}"
    )

    print(
        f"Missing frames:           "
        f"{missing_frames}"
    )

    print(
        f"False candidates rejected:"
        f" {false_candidates_rejected}"
    )

    coverage = (
        tracked_frames /
        len(files) *
        100
    )

    print(
        f"Tracking coverage:        "
        f"{coverage:.1f}%"
    )

    print()

    print(
        f"CSV: {CSV_PATH}"
    )

    print(
        f"Overlays: {OVERLAY_DIR}"
    )


if __name__ == "__main__":
    main()