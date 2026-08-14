from pathlib import Path
import random
import csv
from PIL import Image

# ============================================================
# CONFIG
# ============================================================

SOURCE_DIR = Path("data/yolo_export")
IMAGE_DIR = SOURCE_DIR / "images"
LABEL_DIR = SOURCE_DIR / "labels"

OUTPUT_DIR = Path("data/yolo_tiled")
OUTPUT_IMAGE_DIR = OUTPUT_DIR / "images"
OUTPUT_LABEL_DIR = OUTPUT_DIR / "labels"

TILE_SIZE = 1024
OVERLAP = 256
STRIDE = TILE_SIZE - OVERLAP

# Number of randomly selected empty/background tiles kept
# from each original image.
NEGATIVE_TILES_PER_IMAGE = 2

RANDOM_SEED = 42

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

random.seed(RANDOM_SEED)


# ============================================================
# HELPERS
# ============================================================

def yolo_to_pixels(label, image_width, image_height):
    """
    Convert:
        class x_center y_center width height
    from normalized YOLO coordinates into pixel coordinates.

    Returns:
        class_id, x1, y1, x2, y2
    """
    class_id, xc, yc, bw, bh = label

    xc *= image_width
    yc *= image_height
    bw *= image_width
    bh *= image_height

    x1 = xc - bw / 2
    y1 = yc - bh / 2
    x2 = xc + bw / 2
    y2 = yc + bh / 2

    return int(class_id), x1, y1, x2, y2


def pixels_to_yolo(class_id, x1, y1, x2, y2, tile_width, tile_height):
    """
    Convert pixel bounding box coordinates inside a tile
    back into normalized YOLO format.
    """
    box_width = x2 - x1
    box_height = y2 - y1

    xc = x1 + box_width / 2
    yc = y1 + box_height / 2

    return (
        class_id,
        xc / tile_width,
        yc / tile_height,
        box_width / tile_width,
        box_height / tile_height,
    )


def read_yolo_labels(label_path):
    labels = []

    if not label_path.exists():
        return labels

    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                print(f"WARNING: Invalid label line in {label_path}: {line}")
                continue

            class_id = int(float(parts[0]))
            xc = float(parts[1])
            yc = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])

            labels.append((class_id, xc, yc, bw, bh))

    return labels


def get_tile_positions(length):
    """
    Generate overlapping tile starting coordinates.

    Ensures that the final tile reaches the edge of the image
    even when the stride doesn't divide the image perfectly.
    """
    if length <= TILE_SIZE:
        return [0]

    positions = list(range(0, length - TILE_SIZE + 1, STRIDE))

    last_position = length - TILE_SIZE

    if positions[-1] != last_position:
        positions.append(last_position)

    return positions


def box_center_inside_tile(box, tile_x, tile_y, tile_w, tile_h):
    """
    Determines whether the center of the original object is
    inside this tile.

    This prevents the same bounding box from being included
    solely because a tiny sliver touches a neighboring tile.
    """
    _, x1, y1, x2, y2 = box

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    return (
        tile_x <= center_x < tile_x + tile_w
        and tile_y <= center_y < tile_y + tile_h
    )


def transform_box_to_tile(box, tile_x, tile_y, tile_w, tile_h):
    """
    Convert a full-image bounding box into tile coordinates.

    Bounding box is clipped to tile boundaries if necessary.
    """
    class_id, x1, y1, x2, y2 = box

    local_x1 = max(0, x1 - tile_x)
    local_y1 = max(0, y1 - tile_y)

    local_x2 = min(tile_w, x2 - tile_x)
    local_y2 = min(tile_h, y2 - tile_y)

    if local_x2 <= local_x1 or local_y2 <= local_y1:
        return None

    return pixels_to_yolo(
        class_id,
        local_x1,
        local_y1,
        local_x2,
        local_y2,
        tile_w,
        tile_h,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)

    metadata_path = OUTPUT_DIR / "tile_metadata.csv"

    image_paths = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    print(f"Found {len(image_paths)} source images.")
    print(f"Tile size: {TILE_SIZE} x {TILE_SIZE}")
    print(f"Overlap: {OVERLAP}")
    print(f"Stride: {STRIDE}")
    print()

    total_positive_tiles = 0
    total_negative_tiles = 0
    total_boxes_written = 0

    metadata_rows = []

    for image_number, image_path in enumerate(image_paths, start=1):

        label_path = LABEL_DIR / f"{image_path.stem}.txt"

        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size

        normalized_labels = read_yolo_labels(label_path)

        pixel_boxes = [
            yolo_to_pixels(label, image_width, image_height)
            for label in normalized_labels
        ]

        x_positions = get_tile_positions(image_width)
        y_positions = get_tile_positions(image_height)

        positive_tiles = []
        negative_tiles = []

        for tile_y in y_positions:
            for tile_x in x_positions:

                tile_width = min(TILE_SIZE, image_width - tile_x)
                tile_height = min(TILE_SIZE, image_height - tile_y)

                tile_labels = []

                for box in pixel_boxes:

                    if box_center_inside_tile(
                        box,
                        tile_x,
                        tile_y,
                        tile_width,
                        tile_height,
                    ):
                        transformed = transform_box_to_tile(
                            box,
                            tile_x,
                            tile_y,
                            tile_width,
                            tile_height,
                        )

                        if transformed is not None:
                            tile_labels.append(transformed)

                tile_info = {
                    "x": tile_x,
                    "y": tile_y,
                    "width": tile_width,
                    "height": tile_height,
                    "labels": tile_labels,
                }

                if tile_labels:
                    positive_tiles.append(tile_info)
                else:
                    negative_tiles.append(tile_info)

        # Keep all eye-containing tiles.
        selected_tiles = list(positive_tiles)

        # Add a small controlled number of background examples.
        number_of_negatives = min(
            NEGATIVE_TILES_PER_IMAGE,
            len(negative_tiles),
        )

        if number_of_negatives > 0:
            selected_tiles.extend(
                random.sample(negative_tiles, number_of_negatives)
            )

        # ----------------------------------------------------
        # SAVE SELECTED TILES
        # ----------------------------------------------------

        for tile_info in selected_tiles:

            tile_x = tile_info["x"]
            tile_y = tile_info["y"]
            tile_width = tile_info["width"]
            tile_height = tile_info["height"]
            tile_labels = tile_info["labels"]

            tile_name = (
                f"{image_path.stem}"
                f"__x{tile_x}_y{tile_y}"
            )

            output_image_path = OUTPUT_IMAGE_DIR / f"{tile_name}.png"
            output_label_path = OUTPUT_LABEL_DIR / f"{tile_name}.txt"

            tile = image.crop(
                (
                    tile_x,
                    tile_y,
                    tile_x + tile_width,
                    tile_y + tile_height,
                )
            )

            tile.save(output_image_path)

            with open(output_label_path, "w") as f:

                for label in tile_labels:
                    class_id, xc, yc, bw, bh = label

                    f.write(
                        f"{class_id} "
                        f"{xc:.6f} "
                        f"{yc:.6f} "
                        f"{bw:.6f} "
                        f"{bh:.6f}\n"
                    )

                    total_boxes_written += 1

            if tile_labels:
                tile_type = "positive"
                total_positive_tiles += 1
            else:
                tile_type = "negative"
                total_negative_tiles += 1

            metadata_rows.append(
                {
                    "tile_file": f"{tile_name}.png",
                    "source_image": image_path.name,
                    "source_label": label_path.name,
                    "tile_x": tile_x,
                    "tile_y": tile_y,
                    "tile_width": tile_width,
                    "tile_height": tile_height,
                    "source_width": image_width,
                    "source_height": image_height,
                    "tile_type": tile_type,
                    "num_boxes": len(tile_labels),
                }
            )

        print(
            f"[{image_number:03d}/{len(image_paths)}] "
            f"{image_path.name} | "
            f"eyes={len(pixel_boxes)} | "
            f"positive tiles={len(positive_tiles)} | "
            f"saved negatives={number_of_negatives}"
        )

    # --------------------------------------------------------
    # SAVE METADATA
    # --------------------------------------------------------

    with open(metadata_path, "w", newline="") as csvfile:

        fieldnames = [
            "tile_file",
            "source_image",
            "source_label",
            "tile_x",
            "tile_y",
            "tile_width",
            "tile_height",
            "source_width",
            "source_height",
            "tile_type",
            "num_boxes",
        ]

        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(metadata_rows)

    print()
    print("=" * 60)
    print("TILING COMPLETE")
    print("=" * 60)

    print(f"Positive tiles: {total_positive_tiles}")
    print(f"Negative tiles: {total_negative_tiles}")
    print(
        f"Total saved tiles: "
        f"{total_positive_tiles + total_negative_tiles}"
    )
    print(f"Total eye boxes written: {total_boxes_written}")

    print()
    print(f"Images:   {OUTPUT_IMAGE_DIR}")
    print(f"Labels:   {OUTPUT_LABEL_DIR}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
