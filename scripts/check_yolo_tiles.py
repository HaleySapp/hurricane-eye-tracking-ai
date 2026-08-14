from pathlib import Path
import random
from PIL import Image, ImageDraw

IMAGE_DIR = Path("data/yolo_tiled/images")
LABEL_DIR = Path("data/yolo_tiled/labels")
OUTPUT_DIR = Path("data/yolo_tiled/inspection")

NUM_SAMPLES = 20
RANDOM_SEED = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(RANDOM_SEED)

# Only select tiles that actually contain an eye label
label_files = [
    p for p in LABEL_DIR.glob("*.txt")
    if p.read_text().strip()
]

samples = random.sample(
    label_files,
    min(NUM_SAMPLES, len(label_files))
)

print(f"Creating {len(samples)} inspection images...")

for label_path in samples:

    image_path = IMAGE_DIR / f"{label_path.stem}.png"

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    width, height = image.size

    with open(label_path, "r") as f:
        for line in f:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id, xc, yc, bw, bh = map(float, parts)

            xc *= width
            yc *= height
            bw *= width
            bh *= height

            x1 = xc - bw / 2
            y1 = yc - bh / 2
            x2 = xc + bw / 2
            y2 = yc + bh / 2

            # Thick rectangle so it is easy to see
            for offset in range(4):
                draw.rectangle(
                    [
                        x1 - offset,
                        y1 - offset,
                        x2 + offset,
                        y2 + offset,
                    ],
                    outline="red",
                )

            draw.text(
                (x1, max(0, y1 - 18)),
                "eye",
                fill="red",
            )

    output_path = OUTPUT_DIR / f"{label_path.stem}_CHECK.png"
    image.save(output_path)

    print(output_path)

print()
print("DONE")
print(f"Inspection images saved to: {OUTPUT_DIR}")
