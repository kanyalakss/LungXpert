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
import base64
import matplotlib.pyplot as plt
from io import BytesIO

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

# Load models
model = tf.keras.models.load_model("unsegefficientnetB1_fine_tuned_10epochs.h5")
segmentation_model = tf.keras.models.load_model("unet_model_segmentationFINETUNE224x224.h5")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to predict image
def convert_to_grayscale(img_array):
    """Convert RGB image to grayscale."""
    if img_array.ndim == 4 and img_array.shape[-1] == 3:  # Check if image has 3 channels
        img_array = cv2.cvtColor(img_array[0], cv2.COLOR_BGR2GRAY)  # Remove batch dimension for conversion
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    elif img_array.ndim == 3 and img_array.shape[-1] == 3:  # Check if image has 3 channels (no batch dimension)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    return img_array


def predict_image(img_array):
    img_array = convert_to_grayscale(img_array)  # Convert to grayscale if needed
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    class_labels = ['covid', 'normal', 'pneumonia']
    return class_labels[class_idx], predictions[0]

def segment_lung(img_array):
    img_array = convert_to_grayscale(img_array)  # Convert to grayscale if needed
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    segmentation = segmentation_model.predict(img_array)
    return (segmentation > 0.5).astype(np.uint8)

# Function to generate Grad-CAM heatmap
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    img_array = convert_to_grayscale(img_array)  # Convert to grayscale if needed
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = tf.reduce_sum(last_conv_layer_output * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# Function to overlay segmentation mask on image
def overlay_segmentation(original_img, segmentation_mask):
    overlay = original_img.copy()
    overlay[segmentation_mask == 0] = [0, 0, 0]  # Set non-lung areas to black
    return overlay

# Function to overlay Grad-CAM on image
# Function to overlay Grad-CAM on image
def overlay_gradcam(img, heatmap, alpha=0.4, mask=None):
    heatmap = np.uint8(255 * heatmap)
    jet = plt.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    
    if mask is not None:
        mask = np.expand_dims(mask, axis=-1)  # Ensure mask is 3-channel
        jet_heatmap = jet_heatmap * mask  # Apply mask to the heatmap
    
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    return superimposed_img


def load_and_preprocess_image(file_path):
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array



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

            if not os.path.exists(UPLOAD_FOLDER):
                os.makedirs(UPLOAD_FOLDER)
            if not os.path.exists(GRADCAM_FOLDER):
                os.makedirs(GRADCAM_FOLDER)

            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            img_array = load_and_preprocess_image(file_path)

            result, predictions = predict_image(img_array)

            class_labels = ['covid', 'normal', 'pneumonia']
            class_probabilities = {label: float(prob) * 100 for label, prob in zip(class_labels, predictions)}

            last_conv_layer_name = None
            for layer in model.layers[::-1]:
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer_name = layer.name
                    break
            
            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

            segmentation_mask = segment_lung(img_array)

            original_img = cv2.imread(file_path)
            original_img = cv2.resize(original_img, (224, 224))
            segmented_img = overlay_segmentation(original_img, segmentation_mask[0, :, :, 0])

            gradcam_img = overlay_gradcam(segmented_img, heatmap, mask=segmentation_mask[0, :, :, 0])

            final_img_path = os.path.join(GRADCAM_FOLDER, f"final_{filename}")
            gradcam_img.save(final_img_path)

            with open(final_img_path, "rb") as f:
                final_img_base64 = base64.b64encode(f.read()).decode('utf-8')

            # Store the original filename and final image path in the session
            session['original_filename'] = filename
            session['final_img_path'] = final_img_path

            return jsonify(result=result, final_img=final_img_base64, class_probabilities=class_probabilities)

    return render_template('upload.html')

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
import base64
import matplotlib.pyplot as plt
from io import BytesIO

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

# Load models
model = tf.keras.models.load_model("unsegefficientnetB1_fine_tuned_10epochs.h5")
segmentation_model = tf.keras.models.load_model("unet_model_segmentationFINETUNE224x224.h5")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to predict image
def convert_to_grayscale(img_array):
    """Convert RGB image to grayscale."""
    if img_array.ndim == 4 and img_array.shape[-1] == 3:  # Check if image has 3 channels
        img_array = cv2.cvtColor(img_array[0], cv2.COLOR_BGR2GRAY)  # Remove batch dimension for conversion
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    elif img_array.ndim == 3 and img_array.shape[-1] == 3:  # Check if image has 3 channels (no batch dimension)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    return img_array


def predict_image(img_array):
    img_array = convert_to_grayscale(img_array)  # Convert to grayscale if needed
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    class_labels = ['covid', 'normal', 'pneumonia']
    return class_labels[class_idx], predictions[0]

def segment_lung(img_array):
    img_array = convert_to_grayscale(img_array)  # Convert to grayscale if needed
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    segmentation = segmentation_model.predict(img_array)
    return (segmentation > 0.5).astype(np.uint8)

# Function to generate Grad-CAM heatmap
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    img_array = convert_to_grayscale(img_array)  # Convert to grayscale if needed
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = tf.reduce_sum(last_conv_layer_output * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# Function to overlay segmentation mask on image
def overlay_segmentation(original_img, segmentation_mask):
    overlay = original_img.copy()
    overlay[segmentation_mask == 0] = [0, 0, 0]  # Set non-lung areas to black
    return overlay

# Function to overlay Grad-CAM on image
# Function to overlay Grad-CAM on image
def overlay_gradcam(img, heatmap, alpha=0.4, mask=None):
    heatmap = np.uint8(255 * heatmap)
    jet = plt.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    
    if mask is not None:
        mask = np.expand_dims(mask, axis=-1)  # Ensure mask is 3-channel
        jet_heatmap = jet_heatmap * mask  # Apply mask to the heatmap
    
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    return superimposed_img


def load_and_preprocess_image(file_path):
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array



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

            if not os.path.exists(UPLOAD_FOLDER):
                os.makedirs(UPLOAD_FOLDER)
            if not os.path.exists(GRADCAM_FOLDER):
                os.makedirs(GRADCAM_FOLDER)

            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            img_array = load_and_preprocess_image(file_path)

            result, predictions = predict_image(img_array)

            class_labels = ['covid', 'normal', 'pneumonia']
            class_probabilities = {label: float(prob) * 100 for label, prob in zip(class_labels, predictions)}

            last_conv_layer_name = None
            for layer in model.layers[::-1]:
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer_name = layer.name
                    break
            
            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

            segmentation_mask = segment_lung(img_array)

            original_img = cv2.imread(file_path)
            original_img = cv2.resize(original_img, (224, 224))
            segmented_img = overlay_segmentation(original_img, segmentation_mask[0, :, :, 0])

            gradcam_img = overlay_gradcam(segmented_img, heatmap, mask=segmentation_mask[0, :, :, 0])

            final_img_path = os.path.join(GRADCAM_FOLDER, f"final_{filename}")
            gradcam_img.save(final_img_path)

            with open(final_img_path, "rb") as f:
                final_img_base64 = base64.b64encode(f.read()).decode('utf-8')

            # Store the original filename and final image path in the session
            session['original_filename'] = filename
            session['final_img_path'] = final_img_path

            return jsonify(result=result, final_img=final_img_base64, class_probabilities=class_probabilities)

    return render_template('upload.html')

@upload_bp.route('/save', methods=['POST'])
def save_data():
    if "email" not in session:
        return jsonify(error='User not logged in')

    try:
        # Retrieve the original filename and final image path from the session
        original_filename = session.get('original_filename')
        final_img_path = session.get('final_img_path')

        if not original_filename or not final_img_path:
            return jsonify(error='No image data found')

        # Read the final image
        with open(final_img_path, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode('utf-8')

        # Get patient data from the form
        patient_name = request.form.get('patientName')
        patient_id = request.form.get('patientId')
        initial_symptoms = request.form.get('initialSymptoms')

        # Validate patient data
        if not patient_name or not patient_id or not initial_symptoms:
            return jsonify(error='Incomplete patient data')

        email = session["email"]
        user_record = db["register"].find_one({"email": email})
        if user_record:
            name = user_record.get("name", "Unknown")
            upload_date = datetime.now()
            db["patient_history"].insert_one(
                {
                    "Doctor": name,
                    "image": encoded_string,
                    "image_filename": original_filename,
                    "upload_date": upload_date,
                    "patient_name": patient_name,
                    "patient_id": patient_id,
                    "initial_symptoms": initial_symptoms
                }
            )
        # Clean up the temporary files
        os.remove(final_img_path)
        if 'original_filename' in session:
            del session['original_filename']
        if 'final_img_path' in session:
            del session['final_img_path']

        return jsonify(success=True, message='Data saved successfully')

    except Exception as e:
        return jsonify(error=str(e))


@upload_bp.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@upload_bp.route('/gradcam_results/<filename>')
def gradcam_file(filename):
    return send_from_directory(GRADCAM_FOLDER, filename)


@upload_bp.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@upload_bp.route('/gradcam_results/<filename>')
def gradcam_file(filename):
    return send_from_directory(GRADCAM_FOLDER, filename)