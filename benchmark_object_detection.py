# benchmark_object_detection.py
import os
import sys
import time
import psutil
import logging
import numpy as np
import cv2
from ultralytics import YOLO
from openvino_inference import OpenVINOInferenceEngine
from utils import preprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def get_system_usage():
    """Captures current process CPU and RAM memory footprints."""
    process = psutil.Process(os.getpid())
    # memory_info().rss returns bytes; convert to Megabytes (MB)
    ram_mb = process.memory_info().rss / (1024 * 1024)
    cpu_pct = psutil.cpu_percent(interval=None)
    return ram_mb, cpu_pct

def benchmark_pytorch(image_path, model_path="yolov8n.pt", iterations=100):
    print(f"\n[STARTING] PyTorch Baseline Evaluation ({model_path})...")
    
    # Track baseline resource footprint before model load
    base_ram, _ = get_system_usage()
    
    # Load native framework weights
    start_load = time.time()
    model = YOLO(model_path)
    # Force loading onto CPU to match the OpenVINO evaluation environment
    _ = model.to("cpu")
    load_time = time.time() - start_load
    
    # Warmup runs to fill graph caches
    img = cv2.imread(image_path)
    for _ in range(5):
        _ = model(img, verbose=False)
        
    post_load_ram, _ = get_system_usage()
    model_ram_overhead = post_load_ram - base_ram
    
    # Core timing evaluation loop
    latencies = []
    cpu_readings = []
    
    for _ in range(iterations):
        t0 = time.time()
        _ = model(img, verbose=False)
        t1 = time.time()
        
        latencies.append((t1 - t0) * 1000) # Convert seconds to milliseconds
        _, cpu = get_system_usage()
        cpu_readings.append(cpu)
        
    avg_latency = np.mean(latencies)
    fps = 1000 / avg_latency
    avg_cpu = np.mean(cpu_readings)
    
    return {
        "engine": "PyTorch Native (CPU)",
        "load_time_sec": load_time,
        "model_ram_mb": model_ram_overhead,
        "avg_latency_ms": avg_latency,
        "fps": fps,
        "avg_cpu_pct": avg_cpu
    }

def benchmark_openvino(image_path, model_path="yolov8n_openvino_model/yolov8n.xml", iterations=100):
    print(f"\n[STARTING] OpenVINO Optimized Evaluation ({model_path})...")
    
    base_ram, _ = get_system_usage()
    
    start_load = time.time()
    engine = OpenVINOInferenceEngine(model_xml_path=model_path, device_name="CPU")
    load_time = time.time() - start_load
    
    # Warmup and layout compilation preprocess pass
    orig_img, input_tensor = preprocess(image_path)
    for _ in range(5):
        _ = engine.infer(input_tensor)
        
    post_load_ram, _ = get_system_usage()
    model_ram_overhead = post_load_ram - base_ram
    
    latencies = []
    cpu_readings = []
    
    for _ in range(iterations):
        t0 = time.time()
        _ = engine.infer(input_tensor)
        t1 = time.time()
        
        latencies.append((t1 - t0) * 1000)
        _, cpu = get_system_usage()
        cpu_readings.append(cpu)
        
    avg_latency = np.mean(latencies)
    fps = 1000 / avg_latency
    avg_cpu = np.mean(cpu_readings)
    
    return {
        "engine": "OpenVINO IR (CPU)",
        "load_time_sec": load_time,
        "model_ram_mb": model_ram_overhead,
        "avg_latency_ms": avg_latency,
        "fps": fps,
        "avg_cpu_pct": avg_cpu
    }

def print_comparison_report(pt_res, ov_res):
    print("\n" + "="*50)
    print("         MONITOREXAM BENCHMARK REPORT          ")
    print("="*50)
    
    format_str = "{:<25} | {:<18} | {:<18}"
    print(format_str.format("Metric Parameter", pt_res["engine"], ov_res["engine"]))
    print("-"*68)
    print(format_str.format("Model Load Time", f"{pt_res['load_time_sec']:.3f} s", f"{ov_res['load_time_sec']:.3f} s"))
    print(format_str.format("RAM Memory Footprint", f"{pt_res['model_ram_mb']:.1f} MB", f"{ov_res['model_ram_mb']:.1f} MB"))
    print(format_str.format("Average Latency", f"{pt_res['avg_latency_ms']:.2f} ms", f"{ov_res['avg_latency_ms']:.2f} ms"))
    print(format_str.format("Throughput Speed", f"{pt_res['fps']:.1f} FPS", f"{ov_res['fps']:.1f} FPS"))
    print(format_str.format("Average CPU Load", f"{pt_res['avg_cpu_pct']:.1f}%", f"{ov_res['avg_cpu_pct']:.1f}%"))
    
    speedup = pt_res['avg_latency_ms'] / ov_res['avg_latency_ms']
    print("="*50)
    print(f"CONCLUSION: OpenVINO is running {speedup:.2f}x faster than standard PyTorch.")
    print("="*50 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[ERROR] Please pass an image input path to verify configurations.")
        print("Usage: python benchmark_object_detection.py test_images/test_image_1.jpg")
        sys.exit(1)
        
    test_img = sys.argv[1]
    
    # Run profiling across both frameworks
    pytorch_results = benchmark_pytorch(test_img, iterations=100)
    openvino_results = benchmark_openvino(test_img, model_path="models/yolov8n_int8_openvino_model/yolov8n_int8.xml", iterations=100)
    
    # Output unified validation grid
    print_comparison_report(pytorch_results, openvino_results)