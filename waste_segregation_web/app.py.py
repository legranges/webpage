from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the trained model
model = load_model('your_model.h5')

# Waste class labels
class_labels = ['Biodegradable', 'Non-Biodegradable', 'Recyclable']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    img = Image.open(file.stream).resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    return jsonify({'prediction': f'This is {predicted_class} waste.'})

if __name__ == '__main__':
    app.run(debug=True)
