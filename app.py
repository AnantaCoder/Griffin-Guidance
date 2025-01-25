import os
import logging
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = load_model('cat_dog.h5')

CLASS_NAMES = {0: 'Cat', 1: 'Dog'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(filepath):
    img = image.load_img(filepath, target_size=(64, 64))  # Target size should match model input size
    img_array = image.img_to_array(img) / 255.0  # Normalize image to [0, 1] range
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    img = preprocess_image(filepath)
    prediction = model.predict(img)
    
    # Log raw prediction values
    logging.info(f"Raw prediction values: {prediction}")

    rounded_prediction = np.round(prediction[0][0]).astype(int)

    # Determine class based on rounded prediction
    predicted_class = CLASS_NAMES[rounded_prediction]
    confidence = 100 * (prediction[0][0] if predicted_class == 'Dog' else 1 - prediction[0][0])

    logging.info(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}%")

    return render_template('index.html', prediction=predicted_class, confidence=f"{confidence:.2f}%", image_path=f"uploads/{filename}")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

