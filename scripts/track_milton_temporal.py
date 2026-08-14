from pathlib import Path
import csv
import math
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = Path(
    "results/milton_full_sequence/"
    "milton_full_sequence_results.csv"
)

OUTPUT_CSV = Path(
    "results/milton_full_sequence/"
    "milton_temporal_track.csv"
)

# A candidate farther than this from the last valid
# Milton position is rejected.
#
# Milton only moves a few pixels between these 10-minute
# frames, so 300 px is deliberately very generous while
# still rejecting the thousands-of-pixels-away false eyes.
MAX_TRACK_DISTANCE = 300.0


# ============================================================
# HELPERS
# ============================================================

def distance(x1, y1, x2, y2):
    return math.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )


def read_candidates():
    """
    Group YOLO detections by timestamp.
    """

    grouped = defaultdict(list)

    with open(INPUT_CSV, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            timestamp = row["timestamp"]

            # No detection for this frame
            if not row["center_x"]:
                grouped[timestamp]
                continue

            grouped[timestamp].append(
                {
                    "filename": row["filename"],
                    "timestamp": timestamp,
                    "eye_number": int(row["eye_number"]),
                    "confidence": float(row["confidence"]),
                    "center_x": float(row["center_x"]),
                    "center_y": float(row["center_y"]),
                    "x1": float(row["x1"]),
                    "y1": float(row["y1"]),
                    "x2": float(row["x2"]),
                    "y2": float(row["y2"]),
                }
            )

    return grouped


# ============================================================
# MAIN TRACKER
# ============================================================

def main():

    grouped = read_candidates()

    timestamps = sorted(grouped.keys())

    previous_x = None
    previous_y = None

    output_rows = []

    selected_count = 0
    missing_count = 0
    rejected_candidates = 0

    print("=" * 72)
    print("MILTON TEMPORAL TRACKER")
    print("=" * 72)
    print()

    for frame_number, timestamp in enumerate(
        timestamps,
        start=1
    ):

        candidates = grouped[timestamp]

        # ----------------------------------------------------
        # NO YOLO DETECTION
        # ----------------------------------------------------

        if not candidates:

            print(
                f"[{frame_number:02d}/{len(timestamps)}] "
                f"{timestamp} -> MISSING"
            )

            output_rows.append(
                {
                    "timestamp": timestamp,
                    "status": "missing",
                    "confidence": "",
                    "center_x": "",
                    "center_y": "",
                    "distance_from_previous": "",
                    "candidate_count": 0,
                    "rejected_candidates": 0,
                }
            )

            missing_count += 1

            # Important:
            # Do NOT reset previous position.
            # This allows Milton to be reacquired after gaps.
            continue

        # ----------------------------------------------------
        # FIRST VALID FRAME
        # ----------------------------------------------------

        if previous_x is None:

            # No history yet, so use highest-confidence candidate.
            selected = max(
                candidates,
                key=lambda c: c["confidence"]
            )

            track_distance = 0.0

        # ----------------------------------------------------
        # NORMAL TEMPORAL ASSOCIATION
        # ----------------------------------------------------

        else:

            for candidate in candidates:
                candidate["track_distance"] = distance(
                    previous_x,
                    previous_y,
                    candidate["center_x"],
                    candidate["center_y"]
                )

            selected = min(
                candidates,
                key=lambda c: c["track_distance"]
            )

            track_distance = selected["track_distance"]

            # Even the nearest detection is too far away.
            if track_distance > MAX_TRACK_DISTANCE:

                print(
                    f"[{frame_number:02d}/{len(timestamps)}] "
                    f"{timestamp} -> "
                    f"NO PLAUSIBLE TRACK CANDIDATE "
                    f"(nearest={track_distance:.1f}px)"
                )

                output_rows.append(
                    {
                        "timestamp": timestamp,
                        "status": "rejected",
                        "confidence": "",
                        "center_x": "",
                        "center_y": "",
                        "distance_from_previous": (
                            f"{track_distance:.3f}"
                        ),
                        "candidate_count": len(candidates),
                        "rejected_candidates": len(candidates),
                    }
                )

                rejected_candidates += len(candidates)
                missing_count += 1
                continue

        # ----------------------------------------------------
        # ACCEPT SELECTED MILTON DETECTION
        # ----------------------------------------------------

        rejected_this_frame = (
            len(candidates) - 1
        )

        rejected_candidates += rejected_this_frame

        previous_x = selected["center_x"]
        previous_y = selected["center_y"]

        selected_count += 1

        print(
            f"[{frame_number:02d}/{len(timestamps)}] "
            f"{timestamp} -> "
            f"Milton "
            f"({previous_x:.1f}, {previous_y:.1f}) "
            f"conf={selected['confidence']:.3f} "
            f"move={track_distance:.1f}px "
            f"candidates={len(candidates)}"
        )

        output_rows.append(
            {
                "timestamp": timestamp,
                "status": "tracked",
                "confidence": (
                    f"{selected['confidence']:.6f}"
                ),
                "center_x": (
                    f"{selected['center_x']:.3f}"
                ),
                "center_y": (
                    f"{selected['center_y']:.3f}"
                ),
                "distance_from_previous": (
                    f"{track_distance:.3f}"
                ),
                "candidate_count": len(candidates),
                "rejected_candidates": rejected_this_frame,
            }
        )

    # ========================================================
    # SAVE TRACK
    # ========================================================

    fieldnames = [
        "timestamp",
        "status",
        "confidence",
        "center_x",
        "center_y",
        "distance_from_previous",
        "candidate_count",
        "rejected_candidates",
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

    print()
    print("=" * 72)
    print("TRACKING COMPLETE")
    print("=" * 72)

    print(
        f"Frames tracked:          {selected_count}"
    )

    print(
        f"Frames missing/rejected: {missing_count}"
    )

    print(
        f"False candidates rejected: "
        f"{rejected_candidates}"
    )

    if timestamps:

        rate = (
            selected_count /
            len(timestamps)
            * 100
        )

        print(
            f"Tracking coverage:       "
            f"{rate:.1f}%"
        )

    print()
    print(
        f"Track CSV: {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()
