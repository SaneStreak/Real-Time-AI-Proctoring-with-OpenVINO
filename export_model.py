import os
from ultralytics import YOLO

def export_to_openvino():
    model_name = "yolov8n.pt"
    output_dir = "models"

    print(f"[INFO] Fetching and loading {model_name}...")
    model = YOLO(model_name)

    print(f"[INFO] Exporting model to OpenVINO IR format...")

    export_path = model.export(format="openvino", dynamic=False)

    print(f"[SUCCESS] Model successfully exported to: {export_path}")
    print("[INFO] You can now move the .xml and .bin files into your 'models/ directory if preferred.")


if __name__ == "__main__":
    export_to_openvino()