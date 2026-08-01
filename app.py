import os
import io
import urllib.request
import numpy as np
from PIL import Image
import onnxruntime as ort
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Path ya Model kwenye server
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "rice_model_best.onnx")

# -----------------------------------------------------------------------------
# DIRECT DOWNLOAD URL: 
# Weka link ya direct download ya model yako hapa (mfano: Hugging Face, GitHub Release, au Drive Direct Link).
# Unaweza pia kuiweka kwenye Render Environment Variables kama MODEL_URL.
# -----------------------------------------------------------------------------
DEFAULT_MODEL_URL = "https://huggingface.co/datasets/Mbwana/rice-disease-model/resolve/main/rice_model_best.onnx"
MODEL_URL = os.environ.get("MODEL_URL", DEFAULT_MODEL_URL)

def ensure_model_exists():
    """Inakagua kama fayili ya model ipo, kama haipo inapakua kiotomatiki."""
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ Fayili ya model haijapatikana kwenye: {MODEL_PATH}")
        print(f"⏳ Inapakua model kutoka mtandaoni: {MODEL_URL} ...")
        try:
            # Pakua model na ihifadhi kwenye MODEL_PATH
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("✅ Model imefanikiwa kupakuliwa na kuhifadhiwa!")
        except Exception as e:
            print(f"❌ Imefeli kupakua model kutoka URL: {e}")

# Hakikisha model ipo kabla ya kuanza InferenceSession
ensure_model_exists()

# Initialize ONNX Model Session
session = None
input_name = None
output_name = None

if os.path.exists(MODEL_PATH):
    try:
        session = ort.InferenceSession(MODEL_PATH)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        print(f"✅ ONNX Model successfully loaded from: {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading ONNX model session: {e}")
        session = None
else:
    print(f"❌ Error: Model file still not found at {MODEL_PATH}")

# Class Names in exact training order
CLASS_NAMES = [
    'Bacterial Leaf Blight',
    'Brown Spot',
    'Healthy Rice Leaf',
    'Leaf Blast',
    'Leaf scald',
    'Sheath Blight'
]

def preprocess_image(image_bytes):
    """
    Soma picha kutoka kwenye Bytes stream moja kwa moja bila kuihifadhi kwenye diski.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224), Image.Resampling.BILINEAR)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'status': 'online',
        'model_loaded': session is not None,
        'message': 'Rice Disease Detection API is running!'
    }), 200

# -------------------------------------------------------------
# 1. ENDPOINT YA KUGONGA SERVER ISILALE (Keep-Alive Ping)
# -------------------------------------------------------------
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({
        'status': 'alive',
        'model_loaded': session is not None,
        'message': 'Server is active'
    }), 200

# -------------------------------------------------------------
# 2. ENDPOINT YA PREDICTION KWA AJILI YA MOBILE APP & WEB
# -------------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        if session is None:
            return jsonify({
                'error': 'ONNX Model session is not loaded on server. Check server logs.'
            }), 500

        # Read image in memory
        image_bytes = file.read()
        processed_img = preprocess_image(image_bytes)

        # Run ONNX Inference
        outputs = session.run([output_name], {input_name: processed_img})
        predictions = outputs[0][0]

        # Get class index and confidence score
        predicted_class_idx = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(predictions[predicted_class_idx]) * 100

        # Jibu linalorudi kwenye Mobile App au Web Client (JSON)
        return jsonify({
            'success': True,
            'class': predicted_class,
            'confidence': f"{confidence:.2f}%",
            'confidence_raw': float(predictions[predicted_class_idx])
        }), 200

    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
