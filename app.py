import os
import numpy as np
from PIL import Image
import onnxruntime as ort
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configure Upload Folder (Local disk storage kama kwenye kodi yako ya local)
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model Path kwenye Render Root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_rice_disease_prediction.onnx")

# Load ONNX Model Session
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

# Class Names in exact training order
CLASS_NAMES = [
    'Bacterial Leaf Blight',
    'Brown Spot',
    'Healthy Rice Leaf',
    'Leaf Blast',
    'Leaf scald',
    'Sheath Blight'
]

def softmax(x):
    """
    Inabadilisha Raw Logits kutoka ONNX kuwa Probabilities halisi za (0% - 100%)
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def preprocess_image(image_path):
    """
    Preprocess image using PIL (Pillow) to match Keras load_img perfectly:
    1. Open image and ensure RGB mode
    2. Resize using nearest/bicubic interpolation to (224, 224)
    3. Convert to float32 raw array [0.0, 255.0]
    4. Expand dimensions -> (1, 224, 224, 3)
    """
    img = Image.open(image_path).convert('RGB')
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

# Ping Endpoint kwa ajili ya UptimeRobot
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
        if session is None:
            return jsonify({'error': 'ONNX Model session is not loaded'}), 500

        # Save uploaded file (Muundo wako wa local uliokuwa unafanya kazi)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocess Image kutoka kwenye filepath
        processed_img = preprocess_image(filepath)

        # Run ONNX Inference
        outputs = session.run([output_name], {input_name: processed_img})
        raw_predictions = outputs[0][0]

        # Tumia Softmax kama model haina Softmax layer ndani yake
        probabilities = softmax(raw_predictions)

        # Get class index na confidence score halisi
        predicted_class_idx = int(np.argmax(probabilities))
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(probabilities[predicted_class_idx]) * 100

        # Futa picha baada ya kupredict ili kuzuia kujaza diski ya Render
        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify({
            'success': True,
            'class': predicted_class,
            'confidence': f"{confidence:.2f}%",
            'confidence_raw': float(probabilities[predicted_class_idx])
        }), 200

    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
