from pathlib import Path
import argparse
import csv
import math
from collections import defaultdict


# ============================================================
# CONFIG
# ============================================================

MAX_TRACK_DISTANCE = 300.0


# ============================================================
# HELPERS
# ============================================================

def distance(x1, y1, x2, y2):
    return math.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )


def read_candidates(input_csv):
    """
    Group YOLO detections by timestamp.
    """

    grouped = defaultdict(list)

    with open(input_csv, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            timestamp = row["timestamp"]

            # Preserve frames that had no detection.
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

    parser = argparse.ArgumentParser(
        description=(
            "Temporally track one hurricane eye through "
            "full-disk YOLO detections."
        )
    )

    parser.add_argument(
        "storm",
        help="Storm name, for example: dorian"
    )

    parser.add_argument(
        "--start-x",
        type=float,
        default=None,
        help=(
            "Optional approximate x-coordinate of the target "
            "eye in the first valid frame."
        )
    )

    parser.add_argument(
        "--start-y",
        type=float,
        default=None,
        help=(
            "Optional approximate y-coordinate of the target "
            "eye in the first valid frame."
        )
    )

    args = parser.parse_args()

    storm = args.storm.lower()
    storm_display = storm.capitalize()

    input_csv = Path(
        f"results/{storm}_full_sequence/"
        f"{storm}_full_sequence_results.csv"
    )

    output_csv = Path(
        f"results/{storm}_full_sequence/"
        f"{storm}_temporal_track.csv"
    )

    if not input_csv.exists():
        print(
            f"ERROR: Detection CSV not found:\n"
            f"{input_csv}"
        )
        return

    if (
        (args.start_x is None) !=
        (args.start_y is None)
    ):
        print(
            "ERROR: --start-x and --start-y "
            "must be supplied together."
        )
        return

    grouped = read_candidates(input_csv)

    timestamps = sorted(grouped.keys())

    previous_x = None
    previous_y = None

    output_rows = []

    selected_count = 0
    missing_count = 0
    rejected_candidates = 0

    print("=" * 72)
    print(
        f"{storm_display.upper()} "
        "TEMPORAL TRACKER"
    )
    print("=" * 72)

    print(
        f"Maximum track distance: "
        f"{MAX_TRACK_DISTANCE:.1f}px"
    )

    if args.start_x is not None:
        print(
            f"Initial target hint: "
            f"({args.start_x:.1f}, "
            f"{args.start_y:.1f})"
        )
    else:
        print(
            "Initial target hint: none "
            "(highest-confidence first detection)"
        )

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
                f"[{frame_number:02d}/"
                f"{len(timestamps)}] "
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

            # Keep the previous valid location so the target
            # can be reacquired after a detection gap.
            continue

        # ----------------------------------------------------
        # FIRST VALID FRAME
        # ----------------------------------------------------

        if previous_x is None:

            # If a target hint was supplied, initialize using
            # whichever YOLO candidate is closest to it.
            if args.start_x is not None:

                for candidate in candidates:
                    candidate["start_distance"] = distance(
                        args.start_x,
                        args.start_y,
                        candidate["center_x"],
                        candidate["center_y"]
                    )

                selected = min(
                    candidates,
                    key=lambda c: c["start_distance"]
                )

            else:

                # With no target hint, preserve the behavior
                # of the original Milton tracker.
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

            track_distance = (
                selected["track_distance"]
            )

            # Reject detections that are implausibly far from
            # the previous valid eye position.
            if track_distance > MAX_TRACK_DISTANCE:

                print(
                    f"[{frame_number:02d}/"
                    f"{len(timestamps)}] "
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
                        "distance_from_previous":
                            f"{track_distance:.3f}",
                        "candidate_count":
                            len(candidates),
                        "rejected_candidates":
                            len(candidates),
                    }
                )

                rejected_candidates += len(candidates)
                missing_count += 1

                continue

        # ----------------------------------------------------
        # ACCEPT SELECTED TARGET DETECTION
        # ----------------------------------------------------

        rejected_this_frame = (
            len(candidates) - 1
        )

        rejected_candidates += (
            rejected_this_frame
        )

        previous_x = selected["center_x"]
        previous_y = selected["center_y"]

        selected_count += 1

        print(
            f"[{frame_number:02d}/"
            f"{len(timestamps)}] "
            f"{timestamp} -> "
            f"{storm_display} "
            f"({previous_x:.1f}, "
            f"{previous_y:.1f}) "
            f"conf={selected['confidence']:.3f} "
            f"move={track_distance:.1f}px "
            f"candidates={len(candidates)}"
        )

        output_rows.append(
            {
                "timestamp": timestamp,
                "status": "tracked",
                "confidence":
                    f"{selected['confidence']:.6f}",
                "center_x":
                    f"{selected['center_x']:.3f}",
                "center_y":
                    f"{selected['center_y']:.3f}",
                "distance_from_previous":
                    f"{track_distance:.3f}",
                "candidate_count":
                    len(candidates),
                "rejected_candidates":
                    rejected_this_frame,
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
        output_csv,
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
    # SUMMARY
    # ========================================================

    print()
    print("=" * 72)
    print("TRACKING COMPLETE")
    print("=" * 72)

    print(
        f"Storm:                     "
        f"{storm_display}"
    )

    print(
        f"Frames tracked:            "
        f"{selected_count}"
    )

    print(
        f"Frames missing/rejected:   "
        f"{missing_count}"
    )

    print(
        f"False candidates rejected: "
        f"{rejected_candidates}"
    )

    if timestamps:

        rate = (
            selected_count /
            len(timestamps) *
            100
        )

        print(
            f"Tracking coverage:         "
            f"{rate:.1f}%"
        )

    print()
    print(
        f"Track CSV: {output_csv}"
    )


if __name__ == "__main__":
    main()