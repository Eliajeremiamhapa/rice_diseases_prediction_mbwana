import os
import numpy as np
from PIL import Image
import onnxruntime as ort
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Inaruhusu Frontend na Mobile Apps kuwasiliana na Server

# Direct Configuration za Upload Folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model File Path - Inatumia Jina Sahihi la Localhost (rice_model_best.onnx)
MODEL_NAME = "rice_model_best.onnx"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_NAME)

# Kama rice_model_best.onnx halipo, inatafuta file lolote la .onnx kwenye root folder
if not os.path.exists(MODEL_PATH):
    onnx_files = [f for f in os.listdir(BASE_DIR) if f.endswith('.onnx')]
    if onnx_files:
        MODEL_PATH = os.path.join(BASE_DIR, onnx_files[0])
        print(f"⚠️ {MODEL_NAME} halikuonekana, inatumia model mbadala: {onnx_files[0]}")

# Load ONNX Model Session
try:
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"✅ ONNX Model imefanikiwa kupakiwa kutoka: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Imefeli kupakia ONNX Model: {e}")
    session = None

# Class Names kulingana na mafunzo (Training Order) ya Localhost
CLASS_NAMES = [
    'Bacterial Leaf Blight',
    'Brown Spot',
    'Healthy Rice Leaf',
    'Leaf Blast',
    'Leaf scald',
    'Sheath Blight'
]

def preprocess_image(image_path):
    """
    Kodi halisi ya Preprocessing iliyofanya kazi kwenye Localhost:
    1. Fungua picha kwa PIL na hakikisha ipo kwenye mfumo wa RGB.
    2. Resize kwa BILINEAR interpolation kuwa (224, 224).
    3. Badilisha kuwa float32 NumPy array [0.0 - 255.0].
    4. Ongeza batch dimension -> Shape: (1, 224, 224, 3)
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224), Image.Resampling.BILINEAR)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET'])
def index():
    try:
        return render_template('index.html')
    except Exception:
        return jsonify({
            'status': 'online',
            'model_loaded': session is not None,
            'message': 'Rice Disease Detection API is running!'
        }), 200

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({
        'status': 'alive',
        'model_loaded': session is not None
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Hifadhi picha iliyopakiwa
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        if session is None:
            return jsonify({'error': 'ONNX Model session is not loaded'}), 500

        # Mchakato wa preprocess picha kutumia function ya Localhost
        processed_img = preprocess_image(filepath)

        # Run ONNX Inference
        outputs = session.run([output_name], {input_name: processed_img})
        predictions = outputs[0][0]

        # Pata class index na confidence score
        predicted_class_idx = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence_val = float(predictions[predicted_class_idx]) * 100

        # Majibu yenye muundo unaosomwa na Frontend & App yoyote
        return jsonify({
            'success': True,
            'class': predicted_class,
            'prediction': predicted_class,
            'confidence': f"{confidence_val:.2f}%",
            'confidence_num': round(confidence_val, 2),
            'image_path': f"/static/uploads/{file.filename}"
        })

    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
