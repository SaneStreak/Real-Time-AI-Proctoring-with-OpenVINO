# AI-Powered Online Proctoring System using OpenVINO

A real-time AI-powered online proctoring prototype that combines object detection, gaze estimation, and OpenVINO inference optimization to detect suspicious examination behavior using CPU-only inference.

The project focuses on building and optimizing a computer vision pipeline suitable for edge deployment with low latency and high throughput.

---

## Overview

This project integrates two primary computer vision tasks:

- Object Detection using YOLOv8n
- Gaze Estimation using a multi-stage inference pipeline

The models were optimized using OpenVINO and INT8 post-training quantization to achieve real-time performance on commodity CPUs.

---

## Features

- Real-time object detection using YOLOv8n
- Multi-stage gaze estimation pipeline
- OpenVINO IR model conversion
- INT8 post-training quantization with NNCF
- CPU-optimized inference
- Performance benchmarking
- Failure analysis and production considerations

---

## System Architecture

```
                    Webcam
                       │
                       ▼
               Frame Acquisition
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
 Object Detection               Gaze Estimation
    (YOLOv8n)                  (4-Stage Pipeline)
        │                             │
        └──────────────┬──────────────┘
                       ▼
             Threat Analysis Logic
                       │
                       ▼
              Alert Generation
```

---

## Technology Stack

| Component | Technology |
|----------|------------|
| Language | Python |
| Computer Vision | OpenCV |
| Deep Learning | PyTorch |
| Deployment | OpenVINO Runtime |
| Object Detection | YOLOv8n |
| Optimization | NNCF |
| Model Format | OpenVINO IR |

---

## Pipeline

### Object Detection

- Webcam frame acquisition
- Letterbox preprocessing
- YOLOv8n inference
- Bounding box post-processing
- Detection of phones, books, laptops, and persons

### Gaze Estimation

The gaze estimation module follows a modular four-stage pipeline:

1. Face Detection
2. Head Pose Estimation
3. Facial Landmark Detection
4. Gaze Estimation

Each stage performs a dedicated task, allowing the pipeline to remain modular and easier to debug or replace.

---

## Performance Optimization

The deployment pipeline includes:

- Conversion from PyTorch to OpenVINO Intermediate Representation (IR)
- OpenVINO Runtime optimization
- INT8 Post-Training Quantization
- Calibration dataset generation
- Fast Bias Correction

---

## Benchmark Results

| Model | Average Latency | Throughput |
|------|----------------:|-----------:|
| PyTorch FP32 | 65.13 ms | 15.4 FPS |
| OpenVINO FP32 | 9.80 ms | 102 FPS |
| OpenVINO INT8 | **8.51 ms** | **117.5 FPS** |

### Performance Improvements

- 7.65× reduction in inference latency
- Over 7× increase in throughput
- Real-time CPU inference without dedicated GPU hardware

---

## Repository Structure

```
.
├── models/
├── object_detection/
├── gaze_estimation/
├── benchmarking/
├── utils/
├── report/
├── presentation/
└── README.md
```

---

## Future Improvements

- Multi-person gaze estimation
- Face tracking across frames
- Temporal object tracking
- Improved robustness under challenging lighting conditions
- Fine-tuning using domain-specific datasets
- Production integration with an online proctoring platform

---

## Disclaimer

This repository demonstrates the core computer vision inference pipeline and its optimization for CPU deployment. It is a research and engineering prototype and does not represent a complete production-ready online proctoring system.