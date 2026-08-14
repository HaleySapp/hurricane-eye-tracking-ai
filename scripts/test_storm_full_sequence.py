from pathlib import Path
from PIL import Image, ImageDraw
from ultralytics import YOLO
import torch
import csv
import re
import sys

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = Path(
    "runs/detect/runs/tiled_yolo/eye_detector_v1/weights/best.pt"
)

IMAGE_DIR = Path("data/yolo_export/images")

TILE_SIZE = 1024
OVERLAP = 256
STRIDE = TILE_SIZE - OVERLAP

CONF_THRESHOLD = 0.25
MERGE_DISTANCE = 100

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ============================================================
# HELPERS
# ============================================================

def get_tile_positions(length):
    if length <= TILE_SIZE:
        return [0]

    positions = list(range(0, length - TILE_SIZE + 1, STRIDE))

    last_position = length - TILE_SIZE

    if positions[-1] != last_position:
        positions.append(last_position)

    return positions


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
        cx = (detection["x1"] + detection["x2"]) / 2
        cy = (detection["y1"] + detection["y2"]) / 2

        duplicate = False

        for existing in merged:
            ecx = (existing["x1"] + existing["x2"]) / 2
            ecy = (existing["y1"] + existing["y2"]) / 2

            distance = (
                (cx - ecx) ** 2 +
                (cy - ecy) ** 2
            ) ** 0.5

            if distance <= MERGE_DISTANCE:
                duplicate = True
                break

        if not duplicate:
            merged.append(detection)

    return merged


def extract_time(filename):
    match = re.search(r"_(\d{8})_(\d{4})", filename)

    if match:
        return f"{match.group(1)}_{match.group(2)}"

    return filename


# ============================================================
# FULL-DISK INFERENCE
# ============================================================

def detect_full_disk(model, image_path):

    full_image = Image.open(image_path).convert("RGB")
    width, height = full_image.size

    x_positions = get_tile_positions(width)
    y_positions = get_tile_positions(height)

    detections = []

    for tile_y in y_positions:
        for tile_x in x_positions:

            tile = full_image.crop(
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
                conf=CONF_THRESHOLD,
                device=DEVICE,
                verbose=False
            )

            result = results[0]

            if result.boxes is None:
                continue

            for box in result.boxes:

                confidence = float(box.conf[0].cpu())

                x1, y1, x2, y2 = (
                    box.xyxy[0].cpu().tolist()
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

    merged = merge_detections(detections)

    return full_image, detections, merged


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) < 2:
        print("ERROR: Please provide a storm name.")
        print()
        print("Example:")
        print(
            "python scripts/test_storm_full_sequence.py dorian"
        )
        return

    storm = sys.argv[1].lower()
    storm_display = storm.capitalize()

    output_dir = Path(
        f"results/{storm}_full_sequence"
    )

    output_csv = (
        output_dir /
        f"{storm}_full_sequence_results.csv"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    storm_images = sorted(
        IMAGE_DIR.glob(
            f"*-{storm}_full_*.png"
        ),
        key=lambda p: extract_time(p.name)
    )

    print("=" * 72)
    print(
        f"{storm_display.upper()} "
        "FULL-DISK SEQUENCE TEST"
    )
    print("=" * 72)

    print(f"Device: {DEVICE}")
    print(f"Images found: {len(storm_images)}")
    print(
        f"Confidence threshold: "
        f"{CONF_THRESHOLD}"
    )
    print()

    if not storm_images:
        print(
            f"ERROR: No {storm_display} "
            "images found."
        )
        return

    model = YOLO(
        str(MODEL_PATH)
    )

    rows = []

    total_with_detection = 0
    total_without_detection = 0
    total_multiple = 0

    for number, image_path in enumerate(
        storm_images,
        start=1
    ):

        timestamp = extract_time(
            image_path.name
        )

        print(
            f"[{number:02d}/"
            f"{len(storm_images)}] "
            f"{timestamp}",
            end=" "
        )

        full_image, raw, merged = (
            detect_full_disk(
                model,
                image_path
            )
        )

        # ----------------------------------------------------
        # SAVE VISUAL RESULT
        # ----------------------------------------------------

        output_image = (
            full_image.copy()
        )

        draw = ImageDraw.Draw(
            output_image
        )

        if merged:
            total_with_detection += 1

            if len(merged) > 1:
                total_multiple += 1

            print(
                f"-> {len(merged)} "
                "eye candidate(s)"
            )

        else:
            total_without_detection += 1

            print(
                "-> NO DETECTION"
            )

        # Store one CSV row per final detection
        if merged:

            for eye_number, detection in enumerate(
                merged,
                start=1
            ):

                x1 = detection["x1"]
                y1 = detection["y1"]
                x2 = detection["x2"]
                y2 = detection["y2"]

                confidence = (
                    detection["confidence"]
                )

                center_x = (
                    x1 + x2
                ) / 2

                center_y = (
                    y1 + y2
                ) / 2

                marker_radius = 18

                draw.ellipse(
                    [
                        center_x - marker_radius,
                        center_y - marker_radius,
                        center_x + marker_radius,
                        center_y + marker_radius
                    ],
                    outline="red",
                    width=8
                )

                draw.rectangle(
                    [
                        x1,
                        y1,
                        x2,
                        y2
                    ],
                    outline="red",
                    width=5
                )

                rows.append(
                    {
                        "filename": image_path.name,
                        "timestamp": timestamp,
                        "eye_number": eye_number,
                        "confidence": confidence,
                        "center_x": center_x,
                        "center_y": center_y,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "raw_tile_detections": len(raw),
                        "merged_detections": len(merged)
                    }
                )

        else:

            rows.append(
                {
                    "filename": image_path.name,
                    "timestamp": timestamp,
                    "eye_number": "",
                    "confidence": "",
                    "center_x": "",
                    "center_y": "",
                    "x1": "",
                    "y1": "",
                    "x2": "",
                    "y2": "",
                    "raw_tile_detections": len(raw),
                    "merged_detections": 0
                }
            )

        output_path = (
            output_dir /
            f"{timestamp}_detected.png"
        )

        output_image.save(
            output_path
        )

    # ========================================================
    # CSV
    # ========================================================

    fieldnames = [
        "filename",
        "timestamp",
        "eye_number",
        "confidence",
        "center_x",
        "center_y",
        "x1",
        "y1",
        "x2",
        "y2",
        "raw_tile_detections",
        "merged_detections"
    ]

    with open(
        output_csv,
        "w",
        newline=""
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 72)
    print("SEQUENCE COMPLETE")
    print("=" * 72)

    print(
        f"Storm:                    "
        f"{storm_display}"
    )

    print(
        f"Total images:             "
        f"{len(storm_images)}"
    )

    print(
        f"Images with detection:    "
        f"{total_with_detection}"
    )

    print(
        f"Images without detection: "
        f"{total_without_detection}"
    )

    print(
        f"Images with >1 candidate: "
        f"{total_multiple}"
    )

    detection_rate = (
        total_with_detection /
        len(storm_images) *
        100
    )

    print(
        f"Full-disk detection rate: "
        f"{detection_rate:.1f}%"
    )

    print()
    print(
        f"CSV: {output_csv}"
    )

    print(
        f"Images: {output_dir}"
    )


if __name__ == "__main__":
    main()