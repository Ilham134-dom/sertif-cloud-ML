from tensorflow.keras.models import load_model
import os

try:
    # Try loading the first available model to test the fix
    model_path = "plcnet_model_zone1.h5"
    if os.path.exists(model_path):
        print(f"Attempting to load {model_path} with compile=False...")
        model = load_model(model_path, compile=False)
        print("Success!")
    else:
        print(f"File {model_path} not found.")
except Exception as e:
    print(f"Failed: {e}")
