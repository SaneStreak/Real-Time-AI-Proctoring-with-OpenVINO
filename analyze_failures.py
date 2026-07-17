# analyze_failures.py
import os
import sys
import cv2
import json
from detect_objects import OpenVINOInferenceEngine, preprocess as obj_preprocess, postprocess as obj_postprocess
from gaze_demo import GazePipeline

def analyze_image_failures(image_path, obj_engine, gaze_pipeline):
    if not os.path.exists(image_path):
        print(f"[ERROR] Target image not found at {image_path}")
        return

    frame = cv2.imread(image_path)
    orig_h, orig_w, _ = frame.shape
    
    print(f"\n==================================================")
    print(f"DIAGNOSTIC ANALYSIS FOR: {os.path.basename(image_path)}")
    print(f"==================================================")
    
    # 1. Evaluate Object Detection Failure Boundaries
    _, obj_tensor = obj_preprocess(image_path)
    obj_raw = obj_engine.infer(obj_tensor)
    detections = obj_postprocess(obj_raw, orig_w, orig_h)
    
    print("[STAGE 1] Object Detection Check:")
    if not detections:
        print("  Status: FAILURE -> No relevant exam objects found.")
        print("  Hypothesis: Object might be heavily occluded, out of frame, or confidence dropped below 0.25.")
    else:
        print("  Status: SUCCESS -> Objects detected.")
        for d in detections:
            print(f"    - Detected '{d['class_name']}' with {d['confidence']*100:.1f}% confidence.")

    # 2. Evaluate Gaze Pipeline Failure Boundaries
    print("\n[STAGE 2] Multi-Stage Gaze Cascade Check:")
    gaze_annotated, gaze_result = gaze_pipeline.process_frame(frame)
    
    if gaze_annotated is None:
        print(f"  Status: FAILURE -> Pipeline snapped.")
        print(f"  Reason: {gaze_result}")
    else:
        print("  Status: SUCCESS -> Gaze vectors generated.")
        print(f"    - Face Confidence: {gaze_result['face_confidence']*100:.1f}%")
        print(f"    - Head Pose Angles: Yaw={gaze_result['head_pose']['yaw']:.1f}°, Pitch={gaze_result['head_pose']['pitch']:.1f}°")
        print(f"    - Gaze Direction Vector Array: {gaze_result['gaze_vector']}")
        
    print(f"==================================================\n")

if __name__ == "__main__":
    # Initialize both tracking engines targeting your Ryzen CPU
    OBJ_MODEL_PATH = "models/yolov8n_openvino_model/yolov8n.xml"
    
    obj_engine = OpenVINOInferenceEngine(model_xml_path=OBJ_MODEL_PATH, device_name="CPU")
    gaze_pipeline = GazePipeline(models_dir="models", device="CPU")
    
    # Check if a specific image argument was provided
    if len(sys.argv) > 1:
        analyze_image_failures(sys.argv[1], obj_engine, gaze_pipeline)
    else:
        print("Usage instructions:")
        print("  1. Take edge-case photos (low light, extreme angle, phone hidden under hand).")
        print("  2. Run: python analyze_failures.py test_images/your_failure_photo.jpg")