# AI-Powered Online Proctoring System using OpenVINO

A real-time AI-powered online proctoring prototype that combines object detection, gaze estimation, and OpenVINO inference optimization for efficient CPU-based inference.

The project demonstrates an optimized computer vision pipeline capable of detecting prohibited objects and estimating candidate gaze direction using pretrained deep learning models accelerated with OpenVINO.

---

## Features

- Real-time object detection using YOLOv8n
- Multi-stage gaze estimation pipeline
- OpenVINO IR model conversion
- INT8 post-training quantization using NNCF
- CPU-optimized inference
- Performance benchmarking
- Failure analysis and visualization
- Automated benchmarking and plotting utilities

---

## Technology Stack

- Python
- OpenCV
- PyTorch
- OpenVINO Runtime
- OpenVINO Model Optimizer
- NNCF
- Ultralytics YOLOv8

---

## Repository Structure

```
.
├── models/                         # OpenVINO models and downloaded gaze models
├── outputs/                        # Benchmark outputs and generated visualizations
├── test_images/                    # Sample images for testing
│
├── analyze_failures.py             # Failure case analysis
├── automated_proctor.py            # Automated proctoring pipeline
├── benchmark_object_detection.py   # Object detection benchmarking
├── detect_objects.py               # YOLOv8 object detection
├── download_gaze_models.py         # Downloads OpenVINO gaze estimation models
├── export_model.py                 # Converts YOLO model to OpenVINO IR
├── gaze_demo.py                    # Gaze estimation demo
├── gaze_utils.py                   # Gaze estimation helper functions
├── generate_plots.py               # Generates benchmark graphs
├── openvino_inference.py           # OpenVINO inference wrapper
├── proctor_system.py               # Integrated proctoring pipeline
├── quantize_yolo.py                # INT8 quantization using NNCF
├── utils.py                        # Common utility functions
└── yolov8n.pt                      # YOLOv8 pretrained weights
```

---

## Pipeline

### Object Detection

- Webcam frame acquisition
- Letterbox preprocessing
- YOLOv8n inference
- Confidence filtering
- Bounding box post-processing

Detected classes include:

- Person
- Mobile Phone
- Book
- Laptop

### Gaze Estimation

The gaze estimation pipeline consists of four stages:

1. Face Detection
2. Head Pose Estimation
3. Facial Landmark Detection
4. Gaze Estimation

The modular architecture allows each model to perform a specialized task while maintaining real-time performance.

---

## Model Optimization

The deployment pipeline includes:

- PyTorch to OpenVINO IR conversion
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

## Running the Project

### Download the gaze estimation models

```bash
python download_gaze_models.py
```

### Run object detection

```bash
python detect_objects.py
```

### Run the gaze estimation demo

```bash
python gaze_demo.py
```

### Run the complete proctoring system

```bash
python proctor_system.py
```

### Benchmark object detection

```bash
python benchmark_object_detection.py
```

### Quantize the YOLO model

```bash
python quantize_yolo.py
```

---

## Future Improvements

- Multi-person gaze estimation
- Face tracking across consecutive frames
- Temporal object tracking
- Improved robustness under low-light conditions
- Fine-tuning on domain-specific proctoring datasets
- Integration with a production proctoring platform

---

## Disclaimer

This repository demonstrates the core computer vision inference pipeline and its optimization for CPU deployment. It is intended as a research and engineering prototype and is not a complete production-ready online proctoring system.