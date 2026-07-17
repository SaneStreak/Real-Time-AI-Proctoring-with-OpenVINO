# quantize_yolo.py
import os
import cv2
import nncf
import openvino as ov
import numpy as np

def prepare_calibration_dataset(image_dir="test_images", target_size=(640, 640)):
    """Loads and preprocesses sample validation images to calibrate quantization scales."""
    calibration_data = []
    
    if not os.path.exists(image_dir):
        print(f"[ERROR] Calibration directory '{image_dir}' not found.")
        return None
        
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)]
    
    if not images:
        print(f"[WARNING] No sample images found in '{image_dir}' to run calibration.")
        return None
        
    print(f"[INIT] Loading {len(images)} calibration image assets from workspace...")
    
    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            continue
            
        # Standardize aspect ratio padding to perfectly match the target input structure
        h, w, _ = frame.shape
        max_side = max(h, w)
        padded_img = np.zeros((max_side, max_side, 3), dtype=np.uint8)
        dx = (max_side - w) // 2
        dy = (max_side - h) // 2
        padded_img[dy:dy+h, dx:dx+w] = frame
        
        # Scale to model dimensions and cast to standard tensor format
        resized = cv2.resize(padded_img, target_size)
        tensor = resized.transpose(2, 0, 1) # HWC to CHW
        tensor = np.expand_dims(tensor, axis=0).astype(np.float32) / 255.0
        calibration_data.append(tensor)
        
    return nncf.Dataset(calibration_data)

def run_int8_quantization():
    model_xml = "models/yolov8n_openvino_model/yolov8n.xml"
    model_bin = "models/yolov8n_openvino_model/yolov8n.bin"
    output_dir = "models/yolov8n_int8_openvino_model"
    
    print("\n==================================================")
    print("      MONITOREXAM INT8 QUANTIZATION ENGINE       ")
    print("==================================================")
    
    # 1. Initialize the OpenVINO Core runtime and load FP32 IR graph structures
    core = ov.Core()
    print(f"[LOADING] Fetching FP32 graph components: {model_xml}")
    ov_model = core.read_model(model=model_xml)
    
    # 2. Extract context calibration data patterns
    calibration_dataset = prepare_calibration_dataset("test_images")
    if calibration_dataset is None:
        print("[ABORT] Quantization sequence dropped due to missing target samples.")
        return
        
    # 3. Execute NNCF Post-Training Quantization compression
    print("[COMPRESSING] Running NNCF Post-Training Quantization mapping rules...")
    quantized_model = nncf.quantize(ov_model, calibration_dataset)
    
    # 4. Serialize the newly compressed whole-number model files to disk
    os.makedirs(output_dir, exist_ok=True)
    out_xml = os.path.join(output_dir, "yolov8n_int8.xml")
    
    print(f"[SAVING] Writing optimized INT8 weights architecture grid...")
    ov.serialize(quantized_model, out_xml)
    print(f"[SUCCESS] Saved production-ready INT8 model suite to: {output_dir}")
    print("==================================================\n")

if __name__ == "__main__":
    run_int8_quantization()