# gaze_demo.py
import os
import sys
import cv2
import json
import logging
import numpy as np
from openvino_inference import OpenVINOInferenceEngine
from gaze_utils import (
    preprocess_face_detection,
    crop_face_region,
    preprocess_head_pose,
    extract_eye_crops_geometrically,
    draw_gaze_trajectory
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class GazePipeline:
    def __init__(self, models_dir="models", device="CPU"):
        """Initializes all three distinct OpenVINO models sequentially."""
        self.face_engine = OpenVINOInferenceEngine(
            os.path.join(models_dir, "face-detection-retail-0005.xml"), device
        )
        self.pose_engine = OpenVINOInferenceEngine(
            os.path.join(models_dir, "head-pose-estimation-adas-0001.xml"), device
        )
        self.landmark_engine = OpenVINOInferenceEngine(
            os.path.join(models_dir, "landmarks-regression-retail-0009.xml"), device
        )
        self.gaze_engine = OpenVINOInferenceEngine(
            os.path.join(models_dir, "gaze-estimation-adas-0002.xml"), device
        )

    def process_frame(self, frame):
        """Executes the sequential multi-stage inference pass on a single frame."""
        h, w, _ = frame.shape
        
        # --- STAGE 1: Face Detection ---
        fd_input = preprocess_face_detection(frame)
        fd_output = self.face_engine.infer(fd_input)
        
        # Parse detections from face-detection-retail [1, 1, 200, 7]
        detections = np.squeeze(fd_output)
        best_face = None
        highest_conf = 0.5  # Baseline detection threshold
        
        for idx in range(detections.shape[0]):
            conf = detections[idx, 2]
            if conf > highest_conf:
                highest_conf = conf
                best_face = detections[idx, 3:7]  # [xmin, ymin, xmax, ymax]
                
        if best_face is None:
            return None, "No face detected matching confidence bounds."
            
        face_crop, face_coords = crop_face_region(frame, best_face)
        if face_crop.size == 0:
            return None, "Empty face region sliced."
            
        # --- STAGE 2: Head Pose Estimation ---
        hp_input = preprocess_head_pose(face_crop)
        hp_output = self.pose_engine.compiled_model(hp_input)
        
        # Model returns individual dictionaries or sequence matching keys [cite: 87]
        # output layers: angle_y_fc (Yaw), angle_p_fc (Pitch), angle_r_fc (Roll)
        yaw = float(hp_output[self.pose_engine.compiled_model.output("angle_y_fc")][0][0])
        pitch = float(hp_output[self.pose_engine.compiled_model.output("angle_p_fc")][0][0])
        roll = float(hp_output[self.pose_engine.compiled_model.output("angle_r_fc")][0][0])
        head_pose_angles = np.array([yaw, pitch, roll], dtype=np.float32)
        head_pose_angles = np.expand_dims(head_pose_angles, axis=0)
        
        # --- STAGE 3: Neural Landmark Extraction ---
        from gaze_utils import preprocess_landmarks, extract_eye_crops_with_landmarks
        lm_input = preprocess_landmarks(face_crop)
        lm_output = self.landmark_engine.infer(lm_input)
        
        left_eye, right_eye, le_center, re_center = extract_eye_crops_with_landmarks(face_crop, lm_output)
        if left_eye is None or right_eye is None:
            return None, "Failed to isolate eye structures via landmarks."
            
        # --- STAGE 4: Gaze Vector Estimation ---
        # gaze-estimation-adas-0002 explicitly mandates 3 targeted inputs [cite: 87]
        gaze_feed = {
            "left_eye_image": left_eye,
            "right_eye_image": right_eye,
            "head_pose_angles": head_pose_angles
        }
        
        gaze_output = self.gaze_engine.compiled_model(gaze_feed)
        gaze_vector = np.squeeze(gaze_output[self.gaze_engine.output_layer])
        
        # Normalize vector for clean direction visualization
        norm = np.linalg.norm(gaze_vector)
        if norm > 0:
            gaze_vector = gaze_vector / norm
            
        # --- VISUALIZATION ---
        annotated_frame = frame.copy()
        draw_gaze_trajectory(annotated_frame, le_center, face_coords, gaze_vector)
        draw_gaze_trajectory(annotated_frame, re_center, face_coords, gaze_vector)
        
        metrics = {
            "face_confidence": float(highest_conf),
            "head_pose": {"yaw": yaw, "pitch": pitch, "roll": roll},
            "gaze_vector": gaze_vector.tolist()
        }
        
        return annotated_frame, metrics

def process_static_image(image_path, pipeline, output_dir="outputs"):
    if not os.path.exists(image_path):
        logging.error(f"Image not found at: {image_path}")
        return
        
    frame = cv2.imread(image_path)
    annotated, result = pipeline.process_frame(frame)
    
    if annotated is None:
        logging.warning(f"Pipeline skipped processing: {result}")
        return
        
    print("\n--- GAZE METRICS ---")
    print(json.dumps(result, indent=2))
    print("--------------------\n")
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"gaze_{os.path.basename(image_path)}")
    cv2.imwrite(out_path, annotated)
    logging.info(f"[SUCCESS] Saved gaze tracking demo visualization to: {out_path}")

def run_webcam_demo(pipeline):
    logging.info("Spinning up live webcam stream for Gaze tracking. Press 'q' to exit.")
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        annotated, _ = pipeline.process_frame(frame)
        if annotated is not None:
            cv2.imshow("MonitorExam - Multi-Stage Gaze Tracker POC", annotated)
        else:
            cv2.imshow("MonitorExam - Multi-Stage Gaze Tracker POC", frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    pipeline = GazePipeline(models_dir="models", device="CPU")
    
    if len(sys.argv) > 1:
        target_input = sys.argv[1]
        if target_input.lower() == "--webcam":
            run_webcam_demo(pipeline)
        else:
            process_static_image(target_input, pipeline)
    else:
        print("Usage instructions:")
        print("  python gaze_demo.py <path_to_image.jpg>")
        print("  python gaze_demo.py --webcam")