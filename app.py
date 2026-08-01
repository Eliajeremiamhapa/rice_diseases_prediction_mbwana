import os
import io
import numpy as np
from PIL import Image
import onnxruntime as ort
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Wezesha CORS kwa ajili ya mobile apps na web clients

# Njia salama ya kupata Path ya Model kutoka Root Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# JINA HALISI LA MODEL KAMA LILIVYO KWENYE GITHUB YAKO:
MODEL_PATH = os.path.join(BASE_DIR, "best_rice_disease_prediction.onnx")

# Jaribu ku-load ONNX Model Session
session = None
input_name = None
output_name = None

print(f"🔍 Inatafuta model kwenye path: {MODEL_PATH}")

if os.path.exists(MODEL_PATH):
    try:
        session = ort.InferenceSession(MODEL_PATH)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        print(f"✅ ONNX Model imefanikiwa kupakizwa kutoka: {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error wakati wa ku-load ONNX model: {str(e)}")
else:
    print(f"❌ Error: Fayili ya Model '{MODEL_PATH}' haijapatikana.")
    print("Fayili zilizopo kwenye folder hili la server:", os.listdir(BASE_DIR))

# Class Names kwa mpangilio wa mafunzo ya model
CLASS_NAMES = [
    'Bacterial Leaf Blight',
    'Brown Spot',
    'Healthy Rice Leaf',
    'Leaf Blast',
    'Leaf scald',
    'Sheath Blight'
]

def preprocess_image(image_bytes):
    """Soma picha kutoka memory (bytes) na iandae kwa ajili ya model"""
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

# Endpoint ya kuzuia server isilale (Ping)
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({
        'status': 'alive',
        'model_loaded': session is not None
    }), 200

# Endpoint ya kufanya prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Hakuna picha iliyochaguliwa'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Jina la picha halijapatikana'}), 400

    if session is None:
        return jsonify({
            'error': 'ONNX Model session is not loaded on server. Hakikisha fayili ipo kwenye root folder.'
        }), 500

    try:
        image_bytes = file.read()
        processed_img = preprocess_image(image_bytes)

        # Endesha ONNX Model Inference
        outputs = session.run([output_name], {input_name: processed_img})
        predictions = outputs[0][0]

        predicted_class_idx = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(predictions[predicted_class_idx]) * 100

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
