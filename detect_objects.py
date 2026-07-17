# detect_objects.py
import os
import sys
import cv2
import json
import logging
from utils import preprocess, postprocess
from openvino_inference import OpenVINOInferenceEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def annotate_and_display(original_image, execution_results):
    """Draws clear bounding boxes and labels onto the original image frame."""
    for item in execution_results:
        x, y, w, h = item["box"]
        label = item["class_name"]
        confidence = item["confidence"]
        
        # Green bounding box for suspicious objects
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        text = f"{label} ({confidence:.2f})"
        cv2.putText(original_image, text, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return original_image

def process_static_image(image_path, engine, output_dir="outputs"):
    if not os.path.exists(image_path):
        logging.error(f"Target input image not found: {image_path}")
        return

    # Preprocess
    orig_img, input_tensor = preprocess(image_path)
    orig_h, orig_w, _ = orig_img.shape
    
    # Inference
    raw_output = engine.infer(input_tensor)
    
    # Postprocess
    detections = postprocess(raw_output, orig_w, orig_h)
    
    # Console output requirement for demo
    print("\n--- DETECTION METRICS ---")
    print(json.dumps(detections, indent=2))
    print("-------------------------\n")
    
    # Render and Save
    annotated_img = annotate_and_display(orig_img, detections)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"detected_{os.path.basename(image_path)}")
    cv2.imwrite(out_path, annotated_img)
    logging.info(f"Saved annotated demo output to: {out_path}")

def run_webcam_demo(engine):
    """Fallback capability for live testing via local webcam frame capture."""
    logging.info("Starting live webcam stream. Press 'q' to exit.")
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        orig_h, orig_w, _ = frame.shape
        
        # Temporary fileless preprocessing adaptation
        input_image = cv2.resize(frame, (640, 640)).astype(dtype="float32") / 255.0
        input_tensor = np.expand_dims(input_image.transpose(2, 0, 1), axis=0)
        
        raw_output = engine.infer(input_tensor)
        detections = postprocess(raw_output, orig_w, orig_h)
        frame = annotate_and_display(frame, detections)
        
        cv2.imshow("MonitorExam - Object Detection Live POC", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Adjust paths if yours differs
    DEFAULT_MODEL_PATH = "models/yolov8n_openvino_model/yolov8n.xml"
    
    if not os.path.exists(DEFAULT_MODEL_PATH):
        print(f"[ERROR] Could not find OpenVINO IR model at {DEFAULT_MODEL_PATH}")
        print("Please verify the output path from export_model.py or update the string in detect_objects.py.")
        sys.exit(1)
        
    # Spin up engine targeting CPU for stable inference execution
    engine = OpenVINOInferenceEngine(model_xml_path=DEFAULT_MODEL_PATH, device_name="CPU")
    
    if len(sys.argv) > 1:
        target_input = sys.argv[1]
        if target_input.lower() == "--webcam":
            import numpy as np # Import locally if needed for webcam fallback
            run_webcam_demo(engine)
        else:
            process_static_image(target_input, engine)
    else:
        print("Usage instructions:")
        print("  python detect_objects.py <path_to_image.jpg>")
        print("  python detect_objects.py --webcam")