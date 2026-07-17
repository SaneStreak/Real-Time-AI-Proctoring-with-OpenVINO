# gaze_utils.py
import cv2
import numpy as np

def preprocess_face_detection(image, input_w=300, input_h=300):
    """Prepares raw image frames for face-detection-retail-0005 (NCHW format)."""
    h, w, _ = image.shape
    input_image = cv2.resize(image, (input_w, input_h))
    input_image = input_image.transpose(2, 0, 1) # HWC to CHW
    input_image = np.expand_dims(input_image, axis=0) # CHW to NCHW
    return input_image

def crop_face_region(image, detection_box):
    """Extracts absolute pixel area of face boundary from normalized outputs."""
    h, w, _ = image.shape
    xmin = int(max(0, detection_box[0] * w))
    ymin = int(max(0, detection_box[1] * h))
    xmax = int(min(w, detection_box[2] * w))
    ymax = int(min(h, detection_box[3] * h))
    
    face_crop = image[ymin:ymax, xmin:xmax]
    return face_crop, (xmin, ymin, xmax, ymax)

def preprocess_head_pose(face_crop, input_w=60, input_h=60):
    """Prepares face image patch for head-pose estimation network."""
    input_face = cv2.resize(face_crop, (input_w, input_h))
    input_face = input_face.transpose(2, 0, 1)
    input_face = np.expand_dims(input_face, axis=0)
    return input_face

def extract_eye_crops_geometrically(face_crop, target_size=60):
    """
    Approximates left/right pupil zones using fixed facial anatomy ratios.
    Eliminates landmark estimator layer overhead to prevent pipeline drift bugs.
    """
    fh, fw, _ = face_crop.shape
    
    # Proportional anchors based on average face topology
    # Left eye box center approximation
    le_cx, le_cy = int(fw * 0.35), int(fh * 0.40)
    # Right eye box center approximation
    re_cx, re_cy = int(fw * 0.65), int(fh * 0.40)
    
    span = int(fw * 0.12) # Approximate size of eye bounding region
    
    def crop_and_normalize(cx, cy):
        x1, y1 = max(0, cx - span), max(0, cy - span)
        x2, y2 = min(fw, cx + span), min(fh, cy + span)
        eye_patch = face_crop[y1:y2, x1:x2]
        if eye_patch.size == 0:
            return None
        eye_resized = cv2.resize(eye_patch, (target_size, target_size))
        eye_tensor = eye_resized.transpose(2, 0, 1)
        return np.expand_dims(eye_tensor, axis=0), (cx, cy)

    left_data = crop_and_normalize(le_cx, le_cy)
    right_data = crop_and_normalize(re_cx, re_cy)
    
    if left_data is None or right_data is None:
        return None, None, None, None
        
    return left_data[0], right_data[0], left_data[1], right_data[1]

def draw_gaze_trajectory(frame, eye_center_local, face_coords, gaze_vector, length=100):
    """Calculates perspective shifting vectors and overlays tracking arrows."""
    xmin, ymin, _, _ = face_coords
    
    # Convert local coordinate relative values to absolute global frame pixels
    global_x = xmin + eye_center_local[0]
    global_y = ymin + eye_center_local[1]
    
    # gaze_vector comes as [x, y, z]. Invert Y axis for screen space projection
    dx = int(gaze_vector[0] * length)
    dy = int(-gaze_vector[1] * length)
    
    endpoint = (global_x + dx, global_y + dy)
    cv2.arrowedLine(frame, (global_x, global_y), endpoint, (255, 0, 0), 3, tipLength=0.25)

def preprocess_landmarks(face_crop, input_w=48, input_h=48):
    """Prepares the cropped face patch for landmarks-regression-retail-0009."""
    input_face = cv2.resize(face_crop, (input_w, input_h))
    input_face = input_face.transpose(2, 0, 1)
    input_face = np.expand_dims(input_face, axis=0)
    return input_face

def extract_eye_crops_with_landmarks(face_crop, landmark_output, target_size=60):
    """
    Uses exact 5-point facial landmark predictions to dynamically crop 
    the eyes, completely stabilizing the gaze vector origins.
    """
    fh, fw, _ = face_crop.shape
    landmarks = np.squeeze(landmark_output)
    
    # landmarks-regression-retail-0009 outputs 10 values: 
    # [left_eye_x, left_eye_y, right_eye_x, right_eye_y, ...]
    le_cx, le_cy = int(landmarks[0] * fw), int(landmarks[1] * fh)
    re_cx, re_cy = int(landmarks[2] * fw), int(landmarks[3] * fh)
    
    # Dynamically scale eye crop window based on 15% of the actual face width
    span = int(fw * 0.15)

    def crop_and_normalize(cx, cy):
        x1, y1 = max(0, cx - span), max(0, cy - span)
        x2, y2 = min(fw, cx + span), min(fh, cy + span)
        eye_patch = face_crop[y1:y2, x1:x2]
        if eye_patch.size == 0:
            return None
        eye_resized = cv2.resize(eye_patch, (target_size, target_size))
        eye_tensor = eye_resized.transpose(2, 0, 1)
        return np.expand_dims(eye_tensor, axis=0), (cx, cy)

    left_data = crop_and_normalize(le_cx, le_cy)
    right_data = crop_and_normalize(re_cx, re_cy)
    
    if left_data is None or right_data is None:
        return None, None, None, None
        
    return left_data[0], right_data[0], left_data[1], right_data[1]