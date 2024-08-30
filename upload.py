from flask import Blueprint, request, session, redirect, url_for, render_template, jsonify, send_from_directory
import os 
from datetime import datetime
from werkzeug.utils import secure_filename
from bson.binary import Binary
from pymongo import MongoClient
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import base64  # Import base64

# Create Blueprint
upload_bp = Blueprint('upload', __name__)

# Define upload folder and allowed extensions
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
GRADCAM_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gradcam_results')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Connect to MongoDB
def localMongoDB():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["ClassifyCXR"]
    return db

db = localMongoDB()

# Load model
model = tf.keras.models.load_model("unsegefficientnetB1_fine_tuned_10epochs.h5")

def allowed_file(filename):
    return '.' in filename and '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to predict image
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    class_labels = ['covid', 'normal', 'pneumonia']
    return class_labels[class_idx]

# Function to generate Grad-CAM heatmap
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer in model.layers[model.layers.index(last_conv_layer) + 1:]:
        x = layer(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Function to save Grad-CAM image
def save_and_display_gradcam(img_path, heatmap, cam_path, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")
    cv2.imwrite(cam_path, superimposed_img)

@upload_bp.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if "email" not in session:
        
        return redirect(url_for("login"))
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify(error='No file part')
        file = request.files['file']
        if file.filename == '':
            return jsonify(error='No selected file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            # Create 'uploads' and 'gradcam_results' folders if not exist
            if not os.path.exists(UPLOAD_FOLDER):
                os.makedirs(UPLOAD_FOLDER)
            if not os.path.exists(GRADCAM_FOLDER):
                os.makedirs(GRADCAM_FOLDER)

            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            
            with open(file_path, "rb") as image_file:
                encoded_string = Binary(image_file.read())
            
            email = session["email"]
            user_record = db["register"].find_one({"email": email})
            if user_record:
                name = user_record.get("name", "Unknown")
                upload_date = datetime.now()
                db["patient_history"].insert_one(
                    {
                        "Doctor": name,
                        "image": encoded_string,
                        "image_filename": filename,
                        "upload_date": upload_date
                    }
                )
            
            # Predict image
            result = predict_image(file_path)

            # Generate Grad-CAM
            img = image.load_img(file_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            
            last_conv_layer_name = None
            for layer in model.layers[::-1]:
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer_name = layer.name
                    break
            
            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
            
            gradcam_path = os.path.join(GRADCAM_FOLDER, f"gradcam_{filename}")
            save_and_display_gradcam(file_path, heatmap, gradcam_path)

            # Remove original uploaded image to save storage
            os.remove(file_path)

            # Read Grad-CAM image and encode it to base64
            with open(gradcam_path, "rb") as f:
                gradcam_img_base64 = base64.b64encode(f.read()).decode('utf-8')

            return jsonify(result=result, gradcam_img=gradcam_img_base64)

    return render_template('upload.html')

@upload_bp.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@upload_bp.route('/gradcam_results/<filename>')
def gradcam_file(filename):
    return send_from_directory(GRADCAM_FOLDER, filename)
