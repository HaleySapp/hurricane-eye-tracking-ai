# Hurricane Eye Tracking AI

An artificial intelligence and computer vision project for detecting and tracking hurricane eyes in GOES-16 infrared satellite imagery.

## Project Overview

This project compares a traditional computer vision hurricane eye detection pipeline with a YOLO-based object detection approach.

The existing pipeline includes:

- Image preprocessing
- Hurricane eye estimation
- Optical flow
- Motion and rotation analysis
- Coordinate conversion
- Confidence estimation
- CSV output and visualization

The new phase of the project introduces YOLO to improve hurricane eye detection while preserving the remaining tracking pipeline.

## Dataset

The dataset contains imagery from 10 Atlantic hurricanes, with approximately 40 infrared satellite images per storm.

Approximately 400 images are used in total.

The storms are separated by storm rather than randomly splitting individual images. This helps prevent similar images from the same hurricane from appearing in both training and testing datasets.

Large satellite imagery files are not stored directly in this repository.

## Project Structure

```text
hurricane-eye-tracking-ai/
├── data/
│   └── training/
├── docs/
├── models/
├── results/
├── scripts/
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Technologies

- Python
- Ultralytics YOLO
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- GOES-16 satellite imagery

## Current Progress

- [x] Downloaded imagery for 10 hurricanes
- [x] Built the original hurricane tracking pipeline
- [x] Created machine learning correction and confidence models
- [x] Created the GitHub repository
- [x] Installed Ultralytics YOLO
- [ ] Preprocess and crop hurricane imagery
- [ ] Annotate hurricane eyes
- [ ] Prepare YOLO training, validation, and test datasets
- [ ] Train the YOLO detector
- [ ] Integrate YOLO into the tracking pipeline
- [ ] Compare YOLO with the original detection method

## Research Goal

The primary goal is to determine whether a YOLO-based hurricane eye detector improves the accuracy and reliability of the existing hurricane eye tracking pipeline.