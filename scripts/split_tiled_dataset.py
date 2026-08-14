from pathlib import Path
import csv
import shutil
from collections import Counter

# ============================================================
# CONFIG
# ============================================================

SOURCE_DIR = Path("data/yolo_tiled")
IMAGE_DIR = SOURCE_DIR / "images"
LABEL_DIR = SOURCE_DIR / "labels"
METADATA_PATH = SOURCE_DIR / "tile_metadata.csv"

OUTPUT_DIR = Path("data/yolo_tiled_split")

TRAIN_STORMS = {
    "dorian",
    "florence",
    "ian",
    "ida",
    "iota",
    "irma",
    "lee",
}

VAL_STORMS = {
    "maria",
    "michael",
}

TEST_STORMS = {
    "milton",
}


# ============================================================
# HELPERS
# ============================================================

def get_storm_name(source_image):
    """
    Extract storm name from filenames such as:
    uuid-irma_full_20170906_1530.png
    """

    filename = source_image.lower()

    all_storms = TRAIN_STORMS | VAL_STORMS | TEST_STORMS

    for storm in all_storms:
        if f"{storm}_full_" in filename:
            return storm

    return None


def get_split(storm):
    if storm in TRAIN_STORMS:
        return "train"

    if storm in VAL_STORMS:
        return "val"

    if storm in TEST_STORMS:
        return "test"

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    # Create output directories
    for split in ["train", "val", "test"]:

        (OUTPUT_DIR / "images" / split).mkdir(
            parents=True,
            exist_ok=True
        )

        (OUTPUT_DIR / "labels" / split).mkdir(
            parents=True,
            exist_ok=True
        )

    counts = Counter()
    storm_counts = Counter()
    missing = []

    with open(METADATA_PATH, newline="") as f:

        reader = csv.DictReader(f)

        for row in reader:

            tile_filename = row["tile_file"]
            source_image = row["source_image"]

            storm = get_storm_name(source_image)

            if storm is None:
                missing.append(source_image)
                continue

            split = get_split(storm)

            image_source = IMAGE_DIR / tile_filename

            label_filename = (
                Path(tile_filename).stem + ".txt"
            )

            label_source = LABEL_DIR / label_filename

            image_destination = (
                OUTPUT_DIR
                / "images"
                / split
                / tile_filename
            )

            label_destination = (
                OUTPUT_DIR
                / "labels"
                / split
                / label_filename
            )

            shutil.copy2(
                image_source,
                image_destination
            )

            shutil.copy2(
                label_source,
                label_destination
            )

            counts[split] += 1
            storm_counts[(split, storm)] += 1

    # ========================================================
    # CREATE YOLO DATA YAML
    # ========================================================

    yaml_path = OUTPUT_DIR / "data.yaml"

    absolute_output = OUTPUT_DIR.resolve()

    with open(yaml_path, "w") as f:

        f.write(f"path: {absolute_output}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write("\n")
        f.write("names:\n")
        f.write("  0: eye\n")

    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("=" * 60)
    print("STORM-WISE SPLIT COMPLETE")
    print("=" * 60)

    print()
    print("TRAIN")
    print("-----")

    for storm in sorted(TRAIN_STORMS):
        print(
            f"{storm:12s}: "
            f"{storm_counts[('train', storm)]} tiles"
        )

    print(f"TOTAL TRAIN: {counts['train']}")

    print()
    print("VALIDATION")
    print("----------")

    for storm in sorted(VAL_STORMS):
        print(
            f"{storm:12s}: "
            f"{storm_counts[('val', storm)]} tiles"
        )

    print(f"TOTAL VAL: {counts['val']}")

    print()
    print("TEST")
    print("----")

    for storm in sorted(TEST_STORMS):
        print(
            f"{storm:12s}: "
            f"{storm_counts[('test', storm)]} tiles"
        )

    print(f"TOTAL TEST: {counts['test']}")

    print()
    print(
        "TOTAL TILES:",
        counts["train"]
        + counts["val"]
        + counts["test"]
    )

    print()
    print(f"YOLO YAML: {yaml_path}")

    if missing:

        print()
        print("WARNING: Could not identify storm for:")

        for item in sorted(set(missing)):
            print(item)

    else:
        print()
        print("All tiles assigned successfully.")


if __name__ == "__main__":
    main()
