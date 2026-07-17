# utils.py
import cv2
import numpy as np

# Standard MS COCO Class names mapping
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Filtering strictly for exam relevant objects to keep output clean
TARGET_CLASSES = {
    'cell phone', 'laptop', 'book', 'keyboard', 'mouse', 'backpack', 'handbag'
}

def preprocess(image_path, input_width=640, input_height=640):
    """Loads an image and prepares its tensor layout for OpenVINO."""
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
        
    h, w, _ = original_image.shape
    
    # Resize to YOLO standard input size
    input_image = cv2.resize(original_image, (input_width, input_height))
    
    # Scale pixel values from [0, 255] to [0.0, 1.0]
    input_image = input_image.astype(np.float32) / 255.0
    
    # HWC to NCHW layout transformation
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, axis=0)
    
    return original_image, input_image

def postprocess(pred_tensor, orig_w, orig_h, conf_threshold=0.25, iou_threshold=0.45):
    """Parses raw YOLOv8 tensor, filters targets, and applies NMS."""
    # YOLOv8 output shape is typically [1, 84, 8400] -> (cx, cy, w, h, 80 class scores)
    predictions = np.squeeze(pred_tensor)
    
    # Transpose to shape [8400, 84] for easier row-by-row iteration
    predictions = predictions.T
    
    boxes = []
    confidences = []
    class_ids = []
    
    # Scale factors to map 640x640 predictions back to the original image dimensions
    x_factor = orig_w / 640.0
    y_factor = orig_h / 640.0
    
    for row in predictions:
        scores = row[4:]
        class_id = int(np.argmax(scores))
        confidence = scores[class_id]
        
        if confidence >= conf_threshold:
            class_name = COCO_CLASSES[class_id]
            # Fast filter out unrelated background objects (like 'chair' or 'dining table') 
            # unless they match our exam target list
            if class_name in TARGET_CLASSES:
                xc, yc, w, h = row[0], row[1], row[2], row[3]
                
                # Convert center coordinates to top-left corner box coordinates
                left = int((xc - w / 2) * x_factor)
                top = int((yc - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                
                boxes.append([left, top, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    # Apply Non-Maximum Suppression to clear overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, iou_threshold)
    
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            results.append({
                "box": boxes[i],
                "confidence": confidences[i],
                "class_id": class_ids[i],
                "class_name": COCO_CLASSES[class_ids[i]]
            })
            
    return results