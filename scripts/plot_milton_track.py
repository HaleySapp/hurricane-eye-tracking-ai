from pathlib import Path
from PIL import Image, ImageDraw
from datetime import datetime
import csv
import math

# ============================================================
# CONFIG
# ============================================================

TRACK_CSV = Path(
    "results/milton_full_sequence/"
    "milton_temporal_track.csv"
)

BACKGROUND_IMAGE = Path(
    "data/yolo_export/images/"
    "a109a810-milton_full_20241007_1540.png"
)

OUTPUT_DIR = Path(
    "results/milton_full_sequence"
)

OUTPUT_CSV = OUTPUT_DIR / "milton_track_displacement.csv"
OUTPUT_IMAGE = OUTPUT_DIR / "milton_track_path.png"


# ============================================================
# HELPERS
# ============================================================

def parse_timestamp(timestamp):
    return datetime.strptime(
        timestamp,
        "%Y%m%d_%H%M"
    )


def pixel_distance(x1, y1, x2, y2):
    return math.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )


# ============================================================
# MAIN
# ============================================================

def main():

    rows = []

    with open(TRACK_CSV, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:

            timestamp = row["timestamp"]
            status = row["status"]

            if (
                status == "tracked"
                and row["center_x"]
                and row["center_y"]
            ):

                x = float(row["center_x"])
                y = float(row["center_y"])

            else:
                x = None
                y = None

            rows.append(
                {
                    "timestamp": timestamp,
                    "status": status,
                    "confidence": row["confidence"],
                    "center_x": x,
                    "center_y": y,
                }
            )

    # ========================================================
    # CALCULATE DISPLACEMENT
    # ========================================================

    previous_valid = None

    output_rows = []

    tracked_points = []

    for row in rows:

        timestamp = row["timestamp"]
        current_time = parse_timestamp(timestamp)

        x = row["center_x"]
        y = row["center_y"]

        pixel_displacement = ""
        elapsed_minutes = ""

        if x is not None and y is not None:

            tracked_points.append(
                (
                    x,
                    y,
                    timestamp
                )
            )

            if previous_valid is not None:

                previous_time = previous_valid["time"]

                elapsed = (
                    current_time -
                    previous_time
                ).total_seconds() / 60.0

                displacement = pixel_distance(
                    previous_valid["x"],
                    previous_valid["y"],
                    x,
                    y
                )

                pixel_displacement = (
                    f"{displacement:.3f}"
                )

                elapsed_minutes = (
                    f"{elapsed:.1f}"
                )

            previous_valid = {
                "x": x,
                "y": y,
                "time": current_time,
            }

        output_rows.append(
            {
                "timestamp": timestamp,
                "status": row["status"],
                "confidence": row["confidence"],
                "center_x": (
                    f"{x:.3f}"
                    if x is not None
                    else ""
                ),
                "center_y": (
                    f"{y:.3f}"
                    if y is not None
                    else ""
                ),
                "pixel_displacement": pixel_displacement,
                "elapsed_minutes_since_previous_detection": (
                    elapsed_minutes
                ),
            }
        )

    # ========================================================
    # SAVE CSV
    # ========================================================

    fieldnames = [
        "timestamp",
        "status",
        "confidence",
        "center_x",
        "center_y",
        "pixel_displacement",
        "elapsed_minutes_since_previous_detection",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(output_rows)

    # ========================================================
    # DRAW TRACK
    # ========================================================

    image = Image.open(
        BACKGROUND_IMAGE
    ).convert("RGB")

    draw = ImageDraw.Draw(image)

    # Connect consecutive valid track points
    for i in range(
        1,
        len(tracked_points)
    ):

        x1, y1, _ = tracked_points[i - 1]
        x2, y2, _ = tracked_points[i]

        draw.line(
            (
                x1,
                y1,
                x2,
                y2
            ),
            fill="red",
            width=8
        )

    # Draw every detected point
    for index, (
        x,
        y,
        timestamp
    ) in enumerate(
        tracked_points,
        start=1
    ):

        radius = 12

        draw.ellipse(
            [
                x - radius,
                y - radius,
                x + radius,
                y + radius
            ],
            outline="red",
            width=5
        )

        # Label every 5th point so image is not cluttered
        if index == 1 or index % 5 == 0:

            time_label = timestamp[-4:]

            draw.text(
                (
                    x + 15,
                    y - 15
                ),
                time_label,
                fill="red"
            )

    image.save(
        OUTPUT_IMAGE
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    valid_displacements = [
        float(
            row["pixel_displacement"]
        )
        for row in output_rows
        if row["pixel_displacement"]
    ]

    print("=" * 72)
    print("MILTON TRACK DISPLACEMENT")
    print("=" * 72)

    print(
        f"Tracked points: "
        f"{len(tracked_points)}"
    )

    print(
        f"Movement intervals: "
        f"{len(valid_displacements)}"
    )

    if valid_displacements:

        print(
            f"Average pixel displacement: "
            f"{sum(valid_displacements) / len(valid_displacements):.3f}px"
        )

        print(
            f"Maximum pixel displacement: "
            f"{max(valid_displacements):.3f}px"
        )

    if tracked_points:

        start_x, start_y, start_time = (
            tracked_points[0]
        )

        end_x, end_y, end_time = (
            tracked_points[-1]
        )

        total_direct = pixel_distance(
            start_x,
            start_y,
            end_x,
            end_y
        )

        print(
            f"Start: {start_time} "
            f"({start_x:.1f}, {start_y:.1f})"
        )

        print(
            f"End:   {end_time} "
            f"({end_x:.1f}, {end_y:.1f})"
        )

        print(
            f"Net full-disk displacement: "
            f"{total_direct:.3f}px"
        )

    print()
    print(
        f"CSV:   {OUTPUT_CSV}"
    )

    print(
        f"Image: {OUTPUT_IMAGE}"
    )


if __name__ == "__main__":
    main()
