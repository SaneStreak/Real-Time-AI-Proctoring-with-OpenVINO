# to run: python proctor_system.py --webcam
import os
import sys
import cv2
import logging
import numpy as np
from openvino_inference import OpenVINOInferenceEngine
from utils import preprocess as obj_preprocess, postprocess as obj_postprocess
from gaze_demo import GazePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class UnifiedProctorSystem:
    def __init__(self, models_dir="models", device="CPU"):
        logging.info("Initializing Unified MonitorExam Proctoring Core Engine...")
        
        # Initialize Task 1: Object Detection
        obj_xml = os.path.join(models_dir, "yolov8n_openvino_model", "yolov8n.xml")
        self.obj_engine = OpenVINOInferenceEngine(model_xml_path=obj_xml, device_name=device)
        
        # Initialize Task 2: Gaze Cascade Suite (Face, Head Pose, Landmarks, Gaze)
        self.gaze_suite = GazePipeline(models_dir=models_dir, device=device)
        logging.info("All neural models loaded onto CPU framework successfully.")

        # COCO Class mappings: 67 = cell phone, 73 = book, 63 = laptop
        self.class_mapping = {67: "cell phone", 73: "book", 63: "laptop"}
        self.target_classes = list(self.class_mapping.keys())

    def process_frame(self, frame):
        """Concurrently runs tracking and gaze vector projections with clean, readable layouts."""
        h, w, _ = frame.shape

        # 1. RUN TASK 2: Gaze Estimation Pipeline
        gaze_frame, _ = self.gaze_suite.process_frame(frame)
        annotated_frame = gaze_frame if gaze_frame is not None else frame.copy()

        # 2. RUN TASK 1: Letterboxed Object Detection Preprocessing (In-Memory)
        max_side = max(h, w)
        padded_img = np.zeros((max_side, max_side, 3), dtype=np.uint8)
        dx = (max_side - w) // 2
        dy = (max_side - h) // 2
        padded_img[dy:dy+h, dx:dx+w] = frame

        input_tensor = cv2.resize(padded_img, (640, 640))
        input_tensor = input_tensor.transpose(2, 0, 1)
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32) / 255.0
        
        obj_raw_outputs = self.obj_engine.infer(input_tensor)
        
        output = np.squeeze(obj_raw_outputs)
        boxes = output[:4, :].T
        scores = output[4:, :].T
        
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        mask = (confidences > 0.25) & np.isin(class_ids, self.target_classes)
        filtered_boxes = boxes[mask]
        filtered_confs = confidences[mask]
        filtered_classes = class_ids[mask]

        # Dynamic font scaling configuration based on canvas dimensions
        font_scale = max(0.4, min(w, h) / 800.0)
        thickness = max(1, int(min(w, h) / 300.0))

        for idx in range(len(filtered_boxes)):
            box = filtered_boxes[idx]
            conf = filtered_confs[idx]
            cls_id = filtered_classes[idx]

            x_center = box[0] * (max_side / 640.0)
            y_center = box[1] * (max_side / 640.0)
            box_w    = box[2] * (max_side / 640.0)
            box_h    = box[3] * (max_side / 640.0)
            
            xmin = int(x_center - box_w / 2.0 - dx)
            ymin = int(y_center - box_h / 2.0 - dy)
            xmax = int(x_center + box_w / 2.0 - dx)
            ymax = int(y_center + box_h / 2.0 - dy)
            
            # Bound boxes strictly within frame edges
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(w, xmax), min(h, ymax)

            # Draw the clean green detection box
            cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness)
            
            # Format text label dynamically
            class_name = self.class_mapping.get(cls_id, "object")
            label = f"{class_name}: {conf*100:.1f}%"
            
            # Calculate text box dimension metrics to render a solid contrast background strip
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Ensure label placement doesn't clip off the top edge of the image window
            label_ymin = max(ymin, text_h + 10)
            
            # Draw contrast label anchor block behind text
            cv2.rectangle(annotated_frame, (xmin, label_ymin - text_h - 4), (xmin + text_w + 6, label_ymin + baseline), (0, 255, 0), cv2.FILLED)
            
            # Draw crisp text overlay in high-contrast black on top of the green block
            cv2.putText(annotated_frame, label, (xmin + 3, label_ymin - 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        return annotated_frame

def run_static_demo(image_path, system, output_dir="outputs"):
    if not os.path.exists(image_path):
        logging.error(f"Image not found at target endpoint: {image_path}")
        return
    frame = cv2.imread(image_path)
    output_frame = system.process_frame(frame)
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"unified_{os.path.basename(image_path)}")
    cv2.imwrite(out_path, output_frame)
    logging.info(f"[SUCCESS] Unified verification image saved to: {out_path}")

def run_live_webcam(system):
    logging.info("Spinning up unified real-time webcam session. Press 'q' to quit.")
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        output_frame = system.process_frame(frame)
        cv2.imshow("MonitorExam - Unified AI Proctoring Demo Core", output_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    system = UnifiedProctorSystem(models_dir="models", device="CPU")
    
    if len(sys.argv) > 1:
        target_param = sys.argv[1]
        if target_param.lower() == "--webcam":
            run_live_webcam(system)
        else:
            run_static_demo(target_param, system)
    else:
        print("Usage parameters:")
        print("  python proctor_system.py <path_to_image.jpg>")
        print("  python proctor_system.py --webcam")