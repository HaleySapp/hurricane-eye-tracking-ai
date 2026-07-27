from goes2go.data import goes_nearesttime
import numpy as np
import cv2
import csv
import os
from datetime import datetime, timedelta

STORMS = [
    {"storm": "Irma", "out_dir": "IRMA_FULL_DISK_SEQUENCE", "start": "2017-09-06 12:00", "satellite": "goes16"},
    {"storm": "Maria", "out_dir": "MARIA_FULL_DISK_SEQUENCE", "start": "2017-09-19 12:00", "satellite": "goes16"},
    {"storm": "Florence", "out_dir": "FLORENCE_FULL_DISK_SEQUENCE", "start": "2018-09-11 12:00", "satellite": "goes16"},
    {"storm": "Michael", "out_dir": "MICHAEL_FULL_DISK_SEQUENCE", "start": "2018-10-10 00:00", "satellite": "goes16"},
    {"storm": "Dorian", "out_dir": "DORIAN_FULL_DISK_SEQUENCE", "start": "2019-09-01 12:00", "satellite": "goes16"},
    {"storm": "Iota", "out_dir": "IOTA_FULL_DISK_SEQUENCE", "start": "2020-11-16 12:00", "satellite": "goes16"},
    {"storm": "Ida", "out_dir": "IDA_FULL_DISK_SEQUENCE", "start": "2021-08-29 00:00", "satellite": "goes16"},
    {"storm": "Ian", "out_dir": "IAN_FULL_DISK_SEQUENCE", "start": "2022-09-28 00:00", "satellite": "goes16"},
    {"storm": "Lee", "out_dir": "LEE_FULL_DISK_SEQUENCE", "start": "2023-09-08 00:00", "satellite": "goes16"},
    {"storm": "Milton", "out_dir": "MILTON_FULL_DISK_SEQUENCE", "start": "2024-10-07 12:00", "satellite": "goes16"}
]

PRODUCT = "ABI-L2-CMIPF"
BAND = 13
FRAMES = 40
STEP_MINUTES = 10
TEMP_MIN_K = 180.0
TEMP_MAX_K = 320.0
INVERT_GRAYSCALE = True


def band13_to_uint8(arr):
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=TEMP_MAX_K, posinf=TEMP_MAX_K, neginf=TEMP_MIN_K)
    arr = np.clip(arr, TEMP_MIN_K, TEMP_MAX_K)
    arr = (arr - TEMP_MIN_K) / (TEMP_MAX_K - TEMP_MIN_K)
    image = np.round(arr * 255).astype(np.uint8)
    return 255 - image if INVERT_GRAYSCALE else image


def download_storm(config):
    storm = config["storm"]
    out_dir = config["out_dir"]
    satellite = config["satellite"]
    start = datetime.strptime(config["start"], "%Y-%m-%d %H:%M")
    os.makedirs(out_dir, exist_ok=True)
    rows = []

    for i in range(FRAMES):
        t = start + timedelta(minutes=STEP_MINUTES * i)
        try:
            ds = goes_nearesttime(
                satellite=satellite,
                product=PRODUCT,
                bands=BAND,
                attime=t.strftime("%Y-%m-%d %H:%M"),
            )
            if "CMI" not in ds:
                raise KeyError("CMI variable not found")
            image = band13_to_uint8(ds.CMI.values)
            filename = f"{storm.lower()}_full_{t.strftime('%Y%m%d_%H%M')}.png"
            path = os.path.join(out_dir, filename)
            if not cv2.imwrite(path, image):
                raise OSError(f"Could not save {path}")
            rows.append({
                "filename": filename,
                "storm": storm,
                "requested_time_utc": t.isoformat(),
                "satellite": satellite,
                "product": PRODUCT,
                "band": BAND,
                "temperature_min_k": TEMP_MIN_K,
                "temperature_max_k": TEMP_MAX_K,
                "grayscale_inverted": INVERT_GRAYSCALE,
                "height": int(image.shape[0]),
                "width": int(image.shape[1]),
            })
            print(f"[{storm}] Saved {path}")
            try:
                ds.close()
            except Exception:
                pass
        except Exception as exc:
            print(f"[{storm}] Skipping {t}: {type(exc).__name__}: {exc}")

    if rows:
        metadata = os.path.join(out_dir, "metadata.csv")
        with open(metadata, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[{storm}] Saved metadata to {metadata}")


def main():
    for config in STORMS:
        download_storm(config)


if __name__ == "__main__":
    main()
