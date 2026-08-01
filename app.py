import os
import io
import numpy as np
from PIL import Image
import onnxruntime as ort
from flask import Flask, request, jsonify
from flask_cors import CORS  # 1. Tumeweka Import ya CORS hapa

app = Flask(__name__)
CORS(app)  # 2. Tumewezesha CORS kwa njia zote (Endpoints zote zitakubali requests)

# Model Path - Weka fayili ya model kwenye root folder ya mradi wako
MODEL_PATH = os.path.join(os.path.dirname(__file__), "rice_model_best.onnx")

# Load ONNX Model Session
try:
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"✅ ONNX Model successfully loaded from: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading ONNX model: {e}")
    session = None

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
        'message': 'Rice Disease Detection API is running!'
    }), 200

# -------------------------------------------------------------
# 1. ENDPOINT YA KUGONGA SERVER ISILALE (Keep-Alive Ping)
# -------------------------------------------------------------
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'alive', 'message': 'Server is active'}), 200

# -------------------------------------------------------------
# 2. ENDPOINT YA PREDICTION KWA AJILI YA MOBILE APP
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
            return jsonify({'error': 'ONNX Model session is not loaded on server'}), 500

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

        # Jibu linalorudi kwenye Mobile App (JSON)
        return jsonify({
            'success': True,
            'class': predicted_class,
            'confidence': f"{confidence:.2f}%",
            'confidence_raw': float(predictions[predicted_class_idx])
        }), 200

    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    # Render inatumia PORT environment variable
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
