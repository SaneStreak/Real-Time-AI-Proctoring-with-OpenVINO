# openvino_inference.py
import logging
from openvino import Core

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class OpenVINOInferenceEngine:
    def __init__(self, model_xml_path, device_name="CPU"):
        """Initializes the OpenVINO runtime core and compiles the model."""
        self.model_xml_path = model_xml_path
        self.device_name = device_name
        self.core = Core()
        self.compiled_model = None
        self.input_layer = None
        self.output_layer = None
        
        self._load_model()

    def _load_model(self):
        """Reads the IR framework and compiles it onto the specified device."""
        try:
            logging.info(f"Loading model architecture from: {self.model_xml_path}")
            model = self.core.read_model(model=self.model_xml_path)
            
            logging.info(f"Compiling model for device: {self.device_name}")
            self.compiled_model = self.core.compile_model(model=model, device_name=self.device_name)
            
            # Store input and output layer access pointers
            self.input_layer = self.compiled_model.input(0)
            self.output_layer = self.compiled_model.output(0)
            logging.info("Model loaded successfully into OpenVINO Runtime Engine.")
            
        except Exception as e:
            logging.error(f"Failed to initialize OpenVINO model: {e}")
            raise e

    def infer(self, preprocessed_tensor):
        """Runs synchronous forward inference on the prepared data tensor."""
        if self.compiled_model is None:
            raise RuntimeError("Model is not loaded or compiled.")
            
        # Run inference
        inference_result = self.compiled_model([preprocessed_tensor])
        
        # Extract data matrix from the output layer pointer
        return inference_result[self.output_layer]