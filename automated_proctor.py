# automated_proctor.py
import os
import sys
import time
import cv2
import logging
import numpy as np
from openvino_inference import OpenVINOInferenceEngine
from gaze_demo import GazePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class ProductProctorCore:
    def __init__(self, models_dir="models", device="CPU"):
        logging.info("Initializing Autonomous Product Proctoring Core Engine...")
        
        # Point directly to your optimized INT8 quantized asset model
        obj_xml = os.path.join(models_dir, "yolov8n_int8_openvino_model", "yolov8n_int8.xml")
        self.obj_engine = OpenVINOInferenceEngine(model_xml_path=obj_xml, device_name=device)
        self.gaze_suite = GazePipeline(models_dir=models_dir, device=device)
        
        # ---------------------------------------------------------
        # AUTOMATED PRODUCT METRIC THRESHOLDS
        # ---------------------------------------------------------
        self.MAX_GAZE_ANGLE_DEVIATION = 25.0  # Max degrees off-center allowed
        self.PHONE_VIOLATION_TRIGGER_SEC = 1.5  # Time phone must be seen to alert
        self.GAZE_VIOLATION_TRIGGER_SEC = 3.0   # Time gaze must drift away to alert
        
        # ---------------------------------------------------------
        # REAL-TIME SYSTEM STATE HISTORIES
        # ---------------------------------------------------------
        self.phone_seen_timestamp = None
        self.gaze_lost_timestamp = None
        self.prev_box = None
        self.alpha = 0.25 # Coordinate smoothing filter factor

    def evaluate_frame_telemetry(self, frame):
        """Processes a single frame and calculates autonomous state exceptions."""
        h, w, _ = frame.shape
        current_time = time.time()
        system_alerts = []

        # 1. RUN DETECTIONS
        # Task A: Run the 4-stage Gaze Tracking Cascade
        gaze_frame, gaze_metrics = self.gaze_suite.process_frame(frame)
        annotated_frame = gaze_frame if gaze_frame is not None else frame.copy()

        # Task B: Run the optimized INT8 Object Detector (with letterboxing math)
        max_side = max(h, w)
        padded_img = np.zeros((max_side, max_side, 3), dtype=np.uint8)
        dx, dy = (max_side - w) // 2, (max_side - h) // 2
        padded_img[dy:dy+h, dx:dx+w] = frame

        input_tensor = cv2.resize(padded_img, (640, 640)).transpose(2, 0, 1)
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32) / 255.0
        
        obj_raw_outputs = self.obj_engine.infer(input_tensor)
        output = np.squeeze(obj_raw_outputs)
        boxes, scores = output[:4, :].T, output[4:, :].T
        class_ids, confidences = np.argmax(scores, axis=1), np.max(scores, axis=1)
        
        mask = (confidences > 0.25) & (class_ids == 67) # Class 67 = Cell phone
        filtered_boxes, filtered_confs = boxes[mask], confidences[mask]

        # 2. STATE LOGIC: Threat Accumulation & Smoothing
        phone_detected = len(filtered_boxes) > 0
        
        if phone_detected:
            # Handle Box Smoothing
            best_idx = np.argmax(filtered_confs)
            box = filtered_boxes[best_idx]
            x_c, y_c = box[0] * (max_side / 640.0), box[1] * (max_side / 640.0)
            b_w, b_h = box[2] * (max_side / 640.0), box[3] * (max_side / 640.0)
            xmin, ymin = int(x_c - b_w / 2.0 - dx), int(y_c - b_h / 2.0 - dy)
            xmax, ymax = int(x_c + b_w / 2.0 - dx), int(y_c + b_h / 2.0 - dy)
            curr_box = [max(0, xmin), max(0, ymin), min(w, xmax), min(h, ymax)]

            if self.prev_box is None: self.prev_box = curr_box
            else:
                self.prev_box = [int(self.alpha * curr_box[i] + (1 - self.alpha) * self.prev_box[i]) for i in range(4)]
            
            cv2.rectangle(annotated_frame, (self.prev_box[0], self.prev_box[1]), (self.prev_box[2], self.prev_box[3]), (0, 0, 255), 2)
            
            # --- PRODUCT TRACKING GATE: Accumulate Phone Violation Time ---
            if self.phone_seen_timestamp is None:
                self.phone_seen_timestamp = current_time
            elif (current_time - self.phone_seen_timestamp) >= self.PHONE_VIOLATION_TRIGGER_SEC:
                system_alerts.append(f"[ALERT] WARNING : Cell Phone verified for {current_time - self.phone_seen_timestamp:.1f}s!")
        else:
            self.phone_seen_timestamp = None
            self.prev_box = None

        # Task C: Evaluate Autonomous Gaze Deviations
        if gaze_metrics and 'head_pose' in gaze_metrics:
            yaw = abs(gaze_metrics['head_pose']['yaw'])
            pitch = abs(gaze_metrics['head_pose']['pitch'])
            
            # Target when head rotations break the central vision corridor limits
            if yaw > self.MAX_GAZE_ANGLE_DEVIATION or pitch > self.MAX_GAZE_ANGLE_DEVIATION:
                if self.gaze_lost_timestamp is None:
                    self.gaze_lost_timestamp = current_time
                elif (current_time - self.gaze_lost_timestamp) >= self.GAZE_VIOLATION_TRIGGER_SEC:
                    system_alerts.append(f"[WARNING] SUSPICIOUS GAZE: Candidate looking off-screen ({current_time - self.gaze_lost_timestamp:.1f}s)")
            else:
                self.gaze_lost_timestamp = None
        else:
            # Face completely missing context loop handler
            if self.gaze_lost_timestamp is None:
                self.gaze_lost_timestamp = current_time
            elif (current_time - self.gaze_lost_timestamp) >= self.GAZE_VIOLATION_TRIGGER_SEC:
                system_alerts.append(f"[ALERT] CANDIDATE ABSENT: Face missing from proctor frame for {current_time - self.gaze_lost_timestamp:.1f}s!")

        # 3. OVERLAY SYSTEM ALERTS ON-SCREEN
        y_offset = 40
        for alert in system_alerts:
            cv2.putText(annotated_frame, alert, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            y_offset += 25
            
        return annotated_frame

def start_automated_session():
    system = ProductProctorCore(models_dir="models", device="CPU")
    cap = cv2.VideoCapture(0)
    
    print("\n==================================================")
    print("      MONITOREXAM AUTONOMOUS PRODUCT ENGINE      ")
    print("==================================================")
    print("[RUNNING] Monitoring stream... Press 'q' to end session.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        output_canvas = system.evaluate_frame_telemetry(frame)
        cv2.imshow("MonitorExam - Automated Product Runtime", output_canvas)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_automated_session()