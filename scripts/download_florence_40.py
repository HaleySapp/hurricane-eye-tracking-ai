from goes2go.data import goes_nearesttime
import numpy as np
import cv2
import csv
import os
from datetime import datetime, timedelta

# =========================================================
# STORM SETTINGS
# =========================================================
STORM_NAME = "Florence"
OUT_DIR = "FLORENCE_FULL_DISK_SEQUENCE"
SATELLITE = "goes16"
PRODUCT = "ABI-L2-CMIPF"
BAND = 13

# Candidate 6 hour 40 minute window selected near a strong/visible-eye period.
# Inspect every saved frame before labeling and adjust the start time if needed.
START = datetime.strptime("2018-09-11 12:00", "%Y-%m-%d %H:%M")
FRAMES = 40
STEP_MINUTES = 10

# Use the same physical brightness-temperature range for every frame.
TEMP_MIN_K = 180.0
TEMP_MAX_K = 320.0
INVERT_GRAYSCALE = True

os.makedirs(OUT_DIR, exist_ok=True)
metadata_path = os.path.join(OUT_DIR, "metadata.csv")


def band13_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert Band 13 brightness temperatures to consistent 8-bit grayscale."""
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(
        arr,
        nan=TEMP_MAX_K,
        posinf=TEMP_MAX_K,
        neginf=TEMP_MIN_K,
    )
    arr = np.clip(arr, TEMP_MIN_K, TEMP_MAX_K)
    arr_norm = (arr - TEMP_MIN_K) / (TEMP_MAX_K - TEMP_MIN_K)
    arr_uint8 = np.round(arr_norm * 255).astype(np.uint8)

    if INVERT_GRAYSCALE:
        arr_uint8 = 255 - arr_uint8

    return arr_uint8


def main() -> None:
    rows = []

    for i in range(FRAMES):
        requested_time = START + timedelta(minutes=STEP_MINUTES * i)

        try:
            ds = goes_nearesttime(
                satellite=SATELLITE,
                product=PRODUCT,
                bands=BAND,
                attime=requested_time.strftime("%Y-%m-%d %H:%M"),
            )

            if "CMI" not in ds:
                raise KeyError("Downloaded dataset does not contain CMI.")

            arr_uint8 = band13_to_uint8(ds.CMI.values)
            timestamp = requested_time.strftime("%Y%m%d_%H%M")
            filename = f"florence_full_{timestamp}.png"
            output_path = os.path.join(OUT_DIR, filename)

            if not cv2.imwrite(output_path, arr_uint8):
                raise OSError(f"OpenCV could not save {output_path}")

            rows.append({
                "filename": filename,
                "storm": STORM_NAME,
                "requested_time_utc": requested_time.isoformat(),
                "satellite": SATELLITE,
                "product": PRODUCT,
                "band": BAND,
                "temperature_min_k": TEMP_MIN_K,
                "temperature_max_k": TEMP_MAX_K,
                "grayscale_inverted": INVERT_GRAYSCALE,
                "height": int(arr_uint8.shape[0]),
                "width": int(arr_uint8.shape[1]),
            })

            print(f"Saved {output_path}")

            # Release the xarray dataset before downloading the next full-disk frame.
            try:
                ds.close()
            except Exception:
                pass

        except Exception as exc:
            print(f"Skipping {requested_time}: {type(exc).__name__}: {exc}")

    if rows:
        with open(metadata_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
