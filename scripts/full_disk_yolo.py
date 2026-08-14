from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import torch

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = Path(
    "runs/detect/runs/tiled_yolo/eye_detector_v1/weights/best.pt"
)

IMAGE_PATH = Path(
    "data/yolo_export/images/"
    "a109a810-milton_full_20241007_1540.png"
)

OUTPUT_DIR = Path("results/full_disk_yolo")

TILE_SIZE = 1024
OVERLAP = 256
STRIDE = TILE_SIZE - OVERLAP

CONF_THRESHOLD = 0.25

# Used when merging duplicate detections from overlapping tiles
MERGE_DISTANCE = 100

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ============================================================
# TILE POSITION HELPER
# ============================================================

def get_tile_positions(length):
    if length <= TILE_SIZE:
        return [0]

    positions = list(
        range(0, length - TILE_SIZE + 1, STRIDE)
    )

    last_position = length - TILE_SIZE

    if positions[-1] != last_position:
        positions.append(last_position)

    return positions


# ============================================================
# DUPLICATE MERGING
# ============================================================

def merge_detections(detections):
    """
    Merge detections that represent the same eye.

    Each detection is:
        {
            "x1": ...,
            "y1": ...,
            "x2": ...,
            "y2": ...,
            "confidence": ...
        }
    """

    if not detections:
        return []

    # Highest confidence first
    detections = sorted(
        detections,
        key=lambda d: d["confidence"],
        reverse=True,
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
                (cx - ecx) ** 2
                + (cy - ecy) ** 2
            ) ** 0.5

            if distance <= MERGE_DISTANCE:
                duplicate = True
                break

        if not duplicate:
            merged.append(detection)

    return merged


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("FULL-DISK TILED YOLO INFERENCE")
    print("=" * 70)

    print(f"Model: {MODEL_PATH}")
    print(f"Image: {IMAGE_PATH}")
    print(f"Device: {DEVICE}")
    print()

    model = YOLO(str(MODEL_PATH))

    full_image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    image_width, image_height = full_image.size

    print(
        f"Full image size: "
        f"{image_width} x {image_height}"
    )

    x_positions = get_tile_positions(
        image_width
    )

    y_positions = get_tile_positions(
        image_height
    )

    total_tiles = (
        len(x_positions)
        * len(y_positions)
    )

    print(
        f"Tiles to scan: {total_tiles}"
    )

    print()

    detections = []

    tile_number = 0

    # ========================================================
    # SCAN EVERY TILE
    # ========================================================

    for tile_y in y_positions:

        for tile_x in x_positions:

            tile_number += 1

            tile = full_image.crop(
                (
                    tile_x,
                    tile_y,
                    tile_x + TILE_SIZE,
                    tile_y + TILE_SIZE,
                )
            )

            results = model.predict(
                source=tile,
                imgsz=TILE_SIZE,
                conf=CONF_THRESHOLD,
                device=DEVICE,
                verbose=False,
            )

            result = results[0]

            tile_detection_count = 0

            if result.boxes is not None:

                for box in result.boxes:

                    confidence = float(
                        box.conf[0].cpu()
                    )

                    x1, y1, x2, y2 = (
                        box.xyxy[0]
                        .cpu()
                        .tolist()
                    )

                    # ----------------------------------------
                    # MAP TILE COORDINATES BACK TO FULL DISK
                    # ----------------------------------------

                    full_x1 = x1 + tile_x
                    full_y1 = y1 + tile_y

                    full_x2 = x2 + tile_x
                    full_y2 = y2 + tile_y

                    detections.append(
                        {
                            "x1": full_x1,
                            "y1": full_y1,
                            "x2": full_x2,
                            "y2": full_y2,
                            "confidence": confidence,
                            "tile_x": tile_x,
                            "tile_y": tile_y,
                        }
                    )

                    tile_detection_count += 1

            if tile_detection_count > 0:

                print(
                    f"[{tile_number:02d}/{total_tiles}] "
                    f"tile x={tile_x}, y={tile_y} "
                    f"-> {tile_detection_count} detection(s)"
                )

    # ========================================================
    # MERGE OVERLAPPING DETECTIONS
    # ========================================================

    print()
    print(
        f"Raw detections before merging: "
        f"{len(detections)}"
    )

    merged = merge_detections(
        detections
    )

    print(
        f"Final detections after merging: "
        f"{len(merged)}"
    )

    # ========================================================
    # DRAW FINAL FULL-DISK RESULT
    # ========================================================

    output_image = full_image.copy()

    draw = ImageDraw.Draw(
        output_image
    )

    for index, detection in enumerate(
        merged,
        start=1,
    ):

        x1 = detection["x1"]
        y1 = detection["y1"]
        x2 = detection["x2"]
        y2 = detection["y2"]

        confidence = detection[
            "confidence"
        ]

        center_x = (
            x1 + x2
        ) / 2

        center_y = (
            y1 + y2
        ) / 2

        # Thick red rectangle
        for offset in range(8):

            draw.rectangle(
                [
                    x1 - offset,
                    y1 - offset,
                    x2 + offset,
                    y2 + offset,
                ],
                outline="red",
            )

        label = (
            f"Eye {index} "
            f"{confidence:.2f}"
        )

        draw.text(
            (
                x1,
                max(0, y1 - 25),
            ),
            label,
            fill="red",
        )

        print()
        print(
            f"Eye {index}:"
        )

        print(
            f"  Confidence: "
            f"{confidence:.3f}"
        )

        print(
            f"  Full-disk center: "
            f"({center_x:.1f}, "
            f"{center_y:.1f})"
        )

        print(
            f"  Box: "
            f"({x1:.1f}, {y1:.1f}) "
            f"to "
            f"({x2:.1f}, {y2:.1f})"
        )

    output_path = (
        OUTPUT_DIR
        / (
            IMAGE_PATH.stem
            + "_FULL_DISK_DETECTION.png"
        )
    )

    output_image.save(
        output_path
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"Saved result to:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()
