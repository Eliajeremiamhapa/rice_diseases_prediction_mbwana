import os
import numpy as np
from PIL import Image
import onnxruntime as ort
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Configure Upload Folder
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model Path
MODEL_PATH = r"D:\FLASK2\rice_model_best.onnx"

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

def preprocess_image(image_path):
    """
    Preprocess image using PIL (Pillow) to match Keras load_img perfectly:
    1. Open image and ensure RGB mode
    2. Resize using nearest/bicubic interpolation to (224, 224)
    3. Convert to float32 raw array [0.0, 255.0]
    4. Expand dimensions -> (1, 224, 224, 3)
    """
    # Open image using PIL (exactly like Keras load_img)
    img = Image.open(image_path).convert('RGB')
    
    # Resize with Bilinear/Bicubic matching Keras default
    img = img.resize((224, 224), Image.Resampling.BILINEAR)
    
    # Convert to NumPy float32
    img_array = np.array(img, dtype=np.float32)
    
    # Add batch dimension -> Shape: (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Save uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        if session is None:
            return jsonify({'error': 'ONNX Model session is not loaded'}), 500

        # Preprocess Image using PIL
        processed_img = preprocess_image(filepath)

        # Run ONNX Inference
        outputs = session.run([output_name], {input_name: processed_img})
        predictions = outputs[0][0]

        # Get class index and confidence score
        predicted_class_idx = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(predictions[predicted_class_idx]) * 100

        return jsonify({
            'class': predicted_class,
            'confidence': f"{confidence:.2f}%",
            'image_path': f"/static/uploads/{file.filename}"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)