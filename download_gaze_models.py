# download_gaze_models.py
import os
import urllib.request
import ssl  # 1. Import SSL handler

def download_file(url, output_path):
    if os.path.exists(output_path):
        print(f"[INFO] File already exists: {output_path}")
        return
    try:
        print(f"[DOWNLOAD] Fetching from: {url}")
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        
        # 2. Bypass SSL validation checks for the handshake stability
        context = ssl._create_unverified_context()
        
        with urllib.request.urlopen(req, context=context) as response, open(output_path, 'wb') as out_file:
            out_file.write(response.read())
        print(f"[SUCCESS] Saved to {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to download {url}. Reason: {e}")

def fetch_omz_suite():
    os.makedirs("models", exist_ok=True)
    print("[START] Beginning download sequence for Task 2 models...")
    
    base_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2"
    
    models = {
        "face-detection-retail-0005": [
            f"{base_url}/face-detection-retail-0005/FP32/face-detection-retail-0005.xml",
            f"{base_url}/face-detection-retail-0005/FP32/face-detection-retail-0005.bin"
        ],
        "head-pose-estimation-adas-0001": [
            f"{base_url}/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml",
            f"{base_url}/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.bin"
        ],
        "gaze-estimation-adas-0002": [
            f"{base_url}/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml",
            f"{base_url}/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.bin"
        ]
    }
    
    for model_name, urls in models.items():
        for url in urls:
            filename = url.split("/")[-1]
            dest_path = os.path.join("models", filename)
            download_file(url, dest_path)
            
    print("[FINISH] Execution sequence completed.")

if __name__ == "__main__":
    fetch_omz_suite()