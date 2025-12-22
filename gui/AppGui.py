#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import cv2
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageEnhance
import os
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import time
from datetime import datetime
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)


# # Configuration

# In[2]:


try:
    icon = Image.open("C:\\Users\\DCS\\Downloads\\Cream & Copper Leaf House Logo .png")
except Exception as e:
    icon = "🌿"  

st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon=icon,
    layout="wide",
    initial_sidebar_state="expanded",
)


# # Styling

# In[3]:


def load_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

        * {
            font-family: 'Poppins', sans-serif;
        }

        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
        }

        .stApp {
            background: linear-gradient(135deg, #e8f5e9 0%, #a5d6a7 50%, #81c784 100%);
            background-attachment: fixed;
        }

        .stApp::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background:
                radial-gradient(circle at 20% 50%, rgba(102, 187, 106, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(46, 125, 50, 0.1) 0%, transparent 50%);
            z-index: 0;
            pointer-events: none;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1b5e20 0%, #2e7d32 50%, #388e3c 100%);
            box-shadow: 4px 0 20px rgba(0,0,0,0.1);
        }

        section[data-testid="stSidebar"] * {
            color: white !important;
        }

        section[data-testid="stSidebar"] .stMarkdown {
            color: white !important;
        }

        /* Sidebar headers glow effect */
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: white !important;
            text-shadow: 0 0 10px rgba(255,255,255,0.3);
        }

        h1 {
            color: #1b5e20 !important;
            font-weight: 700 !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }

        h2 {
            color: #2e7d32 !important;
            font-weight: 600 !important;
        }

        h3 {
            color: #388e3c !important;
            font-weight: 500 !important;
        }

        .stButton>button {
            background: linear-gradient(135deg, #2e7d32 0%, #43a047 50%, #66bb6a 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.6rem 2.5rem;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 15px rgba(67, 160, 71, 0.4);
            position: relative;
            overflow: hidden;
        }

        .stButton>button:hover {
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 8px 25px rgba(67, 160, 71, 0.5);
            background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 50%, #43a047 100%);
        }

        .stButton>button:active {
            transform: translateY(-1px);
        }

        /* Button shine effect */
        .stButton>button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
            transition: left 0.5s;
        }

        .stButton>button:hover::before {
            left: 100%;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            padding: 0.8rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            padding: 0.6rem 1.5rem;
            background-color: transparent;
            color: #2e7d32;
            font-weight: 500;
            transition: all 0.3s ease;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background-color: rgba(102, 187, 106, 0.1);
            transform: translateY(-2px);
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #43a047 0%, #66bb6a 100%) !important;
            color: white !important;
            box-shadow: 0 4px 12px rgba(67, 160, 71, 0.4);
            font-weight: 600;
        }

        [data-testid="stMetricValue"] {
            font-size: 2.5rem;
            font-weight: 700;
            color: #2e7d32;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }

        [data-testid="stMetricDelta"] {
            font-size: 1rem;
        }

        [data-testid="metric-container"] {
            background: rgba(255, 255, 255, 0.8);
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            transition: transform 0.3s ease;
        }

        [data-testid="metric-container"]:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.12);
        }

        [data-testid="stFileUploader"] {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 2.5rem;
            border: 3px dashed #66bb6a;
            transition: all 0.3s ease;
        }

        [data-testid="stFileUploader"]:hover {
            border-color: #43a047;
            background: rgba(255, 255, 255, 1);
            box-shadow: 0 8px 30px rgba(102, 187, 106, 0.2);
            transform: scale(1.01);
        }

        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #2e7d32 0%, #43a047 50%, #66bb6a 100%);
            border-radius: 10px;
        }

        .stProgress > div > div {
            background-color: rgba(224, 224, 224, 0.5);
            border-radius: 10px;
        }

        .prediction-card {
            background: rgba(255, 255, 255, 0.95);
            padding: 2.5rem;
            border-radius: 20px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.12);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            border-left: 6px solid #43a047;
        }

        /* Info boxes */
        .info-box {
            background: linear-gradient(135deg, #fff9c4 0%, #fff59d 100%);
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 5px solid #fbc02d;
            margin: 1rem 0;
            box-shadow: 0 4px 12px rgba(251, 192, 45, 0.2);
        }

        .success-box {
            background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 5px solid #4caf50;
            margin: 1rem 0;
            box-shadow: 0 4px 12px rgba(76, 175, 80, 0.2);
        }

        .warning-box {
            background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 5px solid #ff9800;
            margin: 1rem 0;
            box-shadow: 0 4px 12px rgba(255, 152, 0, 0.2);
        }

        .streamlit-expanderHeader {
            background: rgba(255, 255, 255, 0.15);
            border-radius: 10px;
            padding: 0.8rem;
            transition: all 0.3s ease;
        }

        .streamlit-expanderHeader:hover {
            background: rgba(255, 255, 255, 0.25);
        }

        [data-testid="stCamera"] {
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 8px 30px rgba(0,0,0,0.25);
            border: 3px solid #66bb6a;
        }

        .stImage {
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }

        [data-testid="stDataFrame"] {
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }

        .stSelectbox > div > div {
            border-radius: 10px;
            border-color: #66bb6a;
        }

        .stRadio > div {
            background: rgba(255, 255, 255, 0.5);
            padding: 0.5rem;
            border-radius: 10px;
        }

        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }

        ::-webkit-scrollbar-track {
            background: rgba(200, 230, 201, 0.3);
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, #43a047, #66bb6a);
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(180deg, #2e7d32, #43a047);
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .stMarkdown, .stImage, .prediction-card {
            animation: fadeIn 0.6s ease-out;
        }

    </style>
    """, unsafe_allow_html=True)

load_custom_css()


# # Paths

# In[34]:


MODEL_PATH = "C:\\Users\\DCS\\Desktop\\PlantDiseaseProject\\models\\resnet50_model.h5"
EFFICIENTNET_PATH = "C:\\Users\\DCS\\Desktop\\PlantDiseaseProject\\models\\efficientnet_model.h5"
MOBILENET_PATH = "C:\\Users\\DCS\\Desktop\\PlantDiseaseProject\\models\\mobilenet_model.h5"
CLASS_NAMES_PATH ="C:\\Users\\DCS\\Desktop\\PlantDiseaseProject\\models\\class_names.json"

IMG_SIZE = (224, 224)


# # Load Model and Classes

# In[35]:


@st.cache_resource
def load_model(model_name):
    models = {
        'ResNet50': (MODEL_PATH, tf.keras.applications.resnet50.preprocess_input),
        'EfficientNetB3': (EFFICIENTNET_PATH, tf.keras.applications.efficientnet.preprocess_input),
        'MobileNet': (MOBILENET_PATH, tf.keras.applications.mobilenet_v2.preprocess_input)
    }

    if model_name not in models:
        return None, None, None

    model_path, preprocess_fn = models[model_name]

    try:
        if not Path(model_path).exists():
            st.error(f"{model_name} not found at {model_path}")
            return None, None, None

        model = tf.keras.models.load_model(str(model_path))

        class_data = None
        if Path(CLASS_NAMES_PATH).exists():
            with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
                class_data = json.load(f)

        return model, preprocess_fn, class_data

    except Exception as e:
        st.error(f"Error loading {model_name}: {str(e)}")
        return None, None, None


def get_available_models():
    models = {
        'ResNet50': {'path': MODEL_PATH, 'accuracy': '94.5%', 'speed': 'Fast'},
        'EfficientNetB3': {'path': EFFICIENTNET_PATH, 'accuracy': '96.2%', 'speed': 'Medium'},
        'MobileNet': {'path': MOBILENET_PATH, 'accuracy': '91.8%', 'speed': 'Very Fast'}
    }

    available = {}
    for name, info in models.items():
        path = Path(info['path'])
        if path.exists():
            available[name] = {
                **info,
                'size': f"{path.stat().st_size / (1024**2):.1f} MB"
            }

    return available


def display_model_status():
    available = get_available_models()

    if not available:
        st.error("No models found! Place models in models/ directory")
        st.stop()

    st.sidebar.success(f"{len(available)} model(s) available")

    for name, info in available.items():
        with st.sidebar.expander(f"{name}"):
            st.write(f"Accuracy: {info['accuracy']}")
            st.write(f"Speed: {info['speed']}")
            st.write(f"Size: {info['size']}")


def initialize_app():
    available = get_available_models()
    models_dict = {}
    class_data = None

    if Path(CLASS_NAMES_PATH).exists():
        with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
            class_data = json.load(f)

    for name in available:
        model, preprocess, _ = load_model(name)
        if model:
            models_dict[name] = {'model': model, 'preprocess': preprocess, 'loaded': True}

    return models_dict, class_data, list(models_dict.keys())


# # Grad-CAM Implementation

# In[36]:


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower() and 'bn' not in layer.name.lower():
            return layer.name
    for layer in reversed(model.layers):
        if 'activation' in layer.name.lower() or 'relu' in layer.name.lower():
            return layer.name
    return None


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    try:
        if last_conv_layer_name is None:
            last_conv_layer_name = find_last_conv_layer(model)

        if last_conv_layer_name is None:
            st.warning("Could not find convolutional layer for Grad-CAM")
            return None

        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])

            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)

        return heatmap.numpy()

    except Exception as e:
        st.warning(f"Grad-CAM generation failed: {str(e)}")
        return None


def create_gradcam_overlay(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    try:
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        if img.dtype != np.uint8:
            img = np.uint8(img)

        superimposed = heatmap_colored * alpha + img * (1 - alpha)
        return np.clip(superimposed, 0, 255).astype(np.uint8)

    except Exception as e:
        st.error(f"Error creating overlay: {str(e)}")
        return img


def create_gradcam_visualization(img, heatmap, alpha=0.4):
    try:
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = create_gradcam_overlay(img, heatmap, alpha)
        return img, heatmap_colored, overlay

    except Exception as e:
        st.error(f"Visualization error: {str(e)}")
        return img, img, img


def compare_gradcams(img, models_dict, class_idx=None):
    gradcams = {}

    for model_name, model_info in models_dict.items():
        if not model_info.get('loaded', False):
            continue

        try:
            img_resized = img.resize(IMG_SIZE)
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_array = model_info['preprocess'](img_array)
            img_array = np.expand_dims(img_array, axis=0)

            model = model_info['model']
            heatmap = make_gradcam_heatmap(img_array, model, pred_index=class_idx)

            if heatmap is not None:
                gradcams[model_name] = heatmap

        except Exception as e:
            st.warning(f"Grad-CAM failed for {model_name}: {str(e)}")

    return gradcams


def display_gradcam_analysis(img, heatmap, prediction, confidence):
    st.markdown("### Grad-CAM Analysis - Explainable AI")

    if heatmap is None:
        st.warning("Grad-CAM not available for this model/image")
        return

    if not isinstance(heatmap, np.ndarray):
        st.error("Invalid heatmap data")
        return

    img_array = np.array(img.resize(IMG_SIZE))
    original, heatmap_colored, overlay = create_gradcam_visualization(img_array, heatmap)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Original Image**")
        st.image(original, use_container_width=True)
        st.caption("Input leaf image")

    with col2:
        st.markdown("**Attention Heatmap**")
        st.image(heatmap_colored, use_container_width=True)
        st.caption("AI focus areas")

    with col3:
        st.markdown("**Combined View**")
        st.image(overlay, use_container_width=True)
        st.caption("Overlay visualization")

    st.markdown("---")

    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.markdown("#### How to Interpret Grad-CAM")
        st.markdown("""
        <div class="info-box">
        <p><b>Understanding the Heatmap:</b></p>
        <ul>
            <li><span style="color: #d32f2f;">Red/Hot regions:</span>
                Areas the model considers most important for the diagnosis.</li>
            <li><span style="color: #fbc02d;">Yellow/Warm regions:</span>
                Moderately important areas that contribute to the prediction.</li>
            <li><span style="color: #1976d2;">Blue/Cool regions:</span>
                Low relevance areas that do not significantly affect the decision.</li>
        </ul>
        <p><b>What the AI is looking at:</b></p>
        <ul>
            <li>Disease patterns and symptoms</li>
            <li>Leaf texture and color abnormalities</li>
            <li>Spots, lesions, or discoloration</li>
            <li>Overall leaf structure and health</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("#### Analysis Stats")

        try:
            hot_spots = np.sum(heatmap > 0.7) / heatmap.size * 100
            warm_areas = np.sum((heatmap > 0.4) & (heatmap <= 0.7)) / heatmap.size * 100

            st.metric("Hot Spots", f"{hot_spots:.1f}%")
            st.metric("Warm Areas", f"{warm_areas:.1f}%")
            st.metric("Prediction", f"{confidence:.1f}%")

            if confidence > 90:
                st.success("Very High Confidence")
            elif confidence > 75:
                st.info("High Confidence")
            elif confidence > 60:
                st.warning("Moderate Confidence")
            else:
                st.error("Low Confidence - Review needed")

        except Exception as e:
            st.error(f"Error calculating heatmap stats: {str(e)}")


def export_gradcam_visualization(img, heatmap, prediction, output_path=None):
    try:
        img_array = np.array(img.resize(IMG_SIZE))
        original, heatmap_colored, overlay = create_gradcam_visualization(img_array, heatmap)
        combined = np.hstack([original, heatmap_colored, overlay])
        combined_pil = Image.fromarray(combined)

        draw = ImageDraw.Draw(combined_pil)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        labels = ["Original", "Heatmap", "Overlay"]
        for i, label in enumerate(labels):
            draw.text((i * IMG_SIZE[0] + 10, 10), label, fill=(255, 255, 255), font=font)

        if output_path:
            combined_pil.save(output_path)

        return combined_pil

    except Exception as e:
        st.error(f"Export failed: {str(e)}")
        return None


def predict_tflite(img_array, tflite_model_path):
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index'])
    except:
        return None


# # BONUS: Image Enhancement & Preprocessing

# In[37]:


def enhance_image(img, brightness=1.0, contrast=1.0, sharpness=1.0, color=1.0):
    try:
        enhanced = img.copy()

        if brightness != 1.0:
            enhanced = ImageEnhance.Brightness(enhanced).enhance(brightness)

        if contrast != 1.0:
            enhanced = ImageEnhance.Contrast(enhanced).enhance(contrast)

        if sharpness != 1.0:
            enhanced = ImageEnhance.Sharpness(enhanced).enhance(sharpness)

        if color != 1.0:
            enhanced = ImageEnhance.Color(enhanced).enhance(color)

        return enhanced

    except Exception as e:
        st.warning(f" Enhancement failed: {str(e)}")
        return img


def enhance_clahe(img):

    try:
        img_array = np.array(img)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

        return Image.fromarray(enhanced)
    except:
        return img


def show_enhancement_controls():
    st.markdown("####  Enhance Image")

    col1, col2 = st.columns(2)

    with col1:
        brightness = st.slider(" Brightness", 0.5, 2.0, 1.0, 0.1)
        sharpness = st.slider(" Sharpness", 0.5, 2.0, 1.0, 0.1)

    with col2:
        contrast = st.slider(" Contrast", 0.5, 2.0, 1.0, 0.1)
        color = st.slider(" Color", 0.5, 2.0, 1.0, 0.1)

    use_clahe = st.checkbox("✨ Auto-Enhance (CLAHE)", value=False)

    return {
        'brightness': brightness,
        'contrast': contrast,
        'sharpness': sharpness,
        'color': color,
        'use_clahe': use_clahe
    }


def apply_enhancements(img, params):
    enhanced = img.copy()

    if params.get('use_clahe', False):
        enhanced = enhance_clahe(enhanced)

    enhanced = enhance_image(
        enhanced,
        brightness=params.get('brightness', 1.0),
        contrast=params.get('contrast', 1.0),
        sharpness=params.get('sharpness', 1.0),
        color=params.get('color', 1.0)
    )

    return enhanced


def show_before_after(original, enhanced):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original**")
        st.image(original, use_container_width=True)

    with col2:
        st.markdown("**Enhanced**")
        st.image(enhanced, use_container_width=True)


# # Prediction Function with Ensemble

# In[38]:


def predict_disease(img, model, class_data, preprocess_fn=None, use_enhancement=False):

    try:
        if preprocess_fn is None:
            preprocess_fn = tf.keras.applications.resnet50.preprocess_input

        # Preprocess image
        img_resized = img.resize(IMG_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = preprocess_fn(img_array)
        img_array_batch = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array_batch, verbose=0)

        if use_enhancement:
            try:
                img_enhanced = enhance_clahe(img)
                img_enh_resized = img_enhanced.resize(IMG_SIZE)
                img_enh_array = tf.keras.preprocessing.image.img_to_array(img_enh_resized)
                img_enh_array = preprocess_fn(img_enh_array)
                img_enh_batch = np.expand_dims(img_enh_array, axis=0)

                predictions_enh = model.predict(img_enh_batch, verbose=0)
                predictions = (predictions + predictions_enh) / 2
            except Exception as e:
                st.warning(f"⚠️ Enhancement ensemble failed: {str(e)}")

        # Get top 3 predictions
        top3_idx = np.argsort(predictions[0])[-3:][::-1]

        # FIXED: Handle different JSON structures
        try:
            # Try formatted_names first
            if 'formatted_names' in class_data:
                top3_classes = [class_data['formatted_names'][i] for i in top3_idx]
            # Try indexed dict structure
            elif str(top3_idx[0]) in class_data:
                top3_classes = [class_data[str(i)]['name'] for i in top3_idx]
            # Try direct list
            elif isinstance(class_data, list):
                top3_classes = [class_data[i] for i in top3_idx]
            else:
                # Fallback
                top3_classes = [f"Class_{i}" for i in top3_idx]
                st.warning("⚠️ Class names structure not recognized, using fallback")
        except (KeyError, IndexError) as e:
            st.error(f"❌ Error accessing class names: {str(e)}")
            top3_classes = [f"Class_{i}" for i in top3_idx]

        top3_probs = [float(predictions[0][i]) * 100 for i in top3_idx]

        # Generate Grad-CAM heatmap
        heatmap = None
        try:
            heatmap = make_gradcam_heatmap(img_array_batch, model, pred_index=int(top3_idx[0]))
        except Exception as e:
            st.warning(f"⚠️ Grad-CAM generation failed: {str(e)}")

        return top3_classes, top3_probs, heatmap, predictions[0]

    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        import traceback
        st.error(f"Details: {traceback.format_exc()}")
        return None, None, None, None


def predict_with_model(img, model_name, models_dict, class_data, use_enhancement=False):
    """
    Predict disease using a specific model

    Args:
        img: PIL Image
        model_name: Name of model to use
        models_dict: Dictionary of loaded models
        class_data: Class information
        use_enhancement: Use enhancement ensemble

    Returns:
        dict: Prediction results including timing and confidence
    """
    if model_name not in models_dict or not models_dict[model_name].get('loaded', False):
        st.error(f"❌ Model {model_name} not available")
        return None

    try:
        model = models_dict[model_name]['model']
        preprocess_fn = models_dict[model_name]['preprocess']

        # Measure inference time
        start_time = time.time()

        top3_classes, top3_probs, heatmap, all_preds = predict_disease(
            img, model, class_data, preprocess_fn, use_enhancement
        )

        inference_time = time.time() - start_time

        if top3_classes is None:
            return None

        return {
            'model_name': model_name,
            'top3_classes': top3_classes,
            'top3_probs': top3_probs,
            'heatmap': heatmap,
            'all_predictions': all_preds,
            'inference_time': inference_time
        }

    except Exception as e:
        st.error(f"❌ Prediction with {model_name} failed: {str(e)}")
        return None


def compare_model_predictions(img, models_dict, class_data, use_enhancement=False):
    """
    Compare predictions across all available models

    Args:
        img: PIL Image
        models_dict: Dictionary of loaded models
        class_data: Class information
        use_enhancement: Use enhancement ensemble

    Returns:
        dict: Results from all models
    """
    results = {}

    # Get available models
    available_models = [name for name, info in models_dict.items()
                        if info.get('loaded', False)]

    if not available_models:
        st.error("❌ No models available for comparison")
        return None

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, model_name in enumerate(available_models):
        status_text.text(f"🔄 Running prediction with {model_name}...")

        result = predict_with_model(img, model_name, models_dict, class_data, use_enhancement)

        if result:
            results[model_name] = result

        progress_bar.progress((idx + 1) / len(available_models))

    status_text.text("✅ All predictions complete!")
    time.sleep(0.5)
    status_text.empty()
    progress_bar.empty()

    return results


def display_comparison_results(results, class_data):
    """
    Display comparison results from multiple models in a structured format

    Args:
        results: Dictionary of prediction results from multiple models
        class_data: Class information (not used but kept for compatibility)
    """
    if not results:
        st.warning("⚠️ No results to compare")
        return

    st.markdown("### 📊 Model Comparison Results")

    # Create comparison table
    comparison_data = []
    for model_name, result in results.items():
        comparison_data.append({
            'Model': model_name,
            'Top Prediction': result['top3_classes'][0],
            'Confidence': f"{result['top3_probs'][0]:.2f}%",
            'Inference Time': f"{result['inference_time']:.3f}s"
        })

    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 🎯 Top Predictions by Model")

    # Display individual model results
    cols = st.columns(len(results))

    for idx, (model_name, result) in enumerate(results.items()):
        with cols[idx]:
            st.markdown(f"**{model_name}**")

            top_class = result['top3_classes'][0]
            top_prob = result['top3_probs'][0]

            # Dynamic color based on confidence
            confidence_color = "#4caf50" if top_prob > 80 else "#ff9800" if top_prob > 60 else "#f44336"

            st.markdown(f"""
            <div style="background: white; padding: 1rem; border-radius: 10px;
                        border-left: 4px solid {confidence_color};
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <h4 style="margin: 0; color: #2e7d32;">{top_class}</h4>
                <h2 style="margin: 0.5rem 0; color: {confidence_color};">{top_prob:.1f}%</h2>
                <p style="margin: 0; font-size: 0.9rem; color: #666;">
                    ⏱️ Time: {result['inference_time']:.3f}s
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Top 3 Predictions:**")
            for i in range(min(3, len(result['top3_classes']))):  # FIXED: Handle edge cases
                st.write(f"{i + 1}. {result['top3_classes'][i]} ({result['top3_probs'][i]:.1f}%)")

    st.markdown("---")
    st.markdown("#### 🤝 Model Agreement Analysis")

    # Analyze agreement between models
    top_predictions = [result['top3_classes'][0] for result in results.values()]

    if len(set(top_predictions)) == 1:
        st.success(f"✅ **Perfect Agreement**: All models agree on **{top_predictions[0]}**")
        st.info("💡 High confidence in diagnosis - all models reached the same conclusion")
    else:
        st.warning("⚠️ **Disagreement Detected**: Models have different top predictions")

        from collections import Counter
        pred_counts = Counter(top_predictions)

        st.write("**📊 Prediction Breakdown:**")
        for pred, count in pred_counts.most_common():
            percentage = (count / len(results)) * 100
            st.write(f"- **{pred}**: {count}/{len(results)} models ({percentage:.0f}%)")

        # Provide recommendation based on consensus
        most_common_pred, most_common_count = pred_counts.most_common(1)[0]
        if most_common_count >= len(results) * 0.67:  # 2/3 majority
            st.info(f" **Recommendation**: Strong consensus ({most_common_count}/{len(results)}) favors **{most_common_pred}**")
        else:
            st.warning(" **Recommendation**: No strong consensus. Review image quality and consider retaking the photo for better results.")

    # Additional statistics
    st.markdown("---")
    st.markdown("#### Performance Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        avg_confidence = sum(r['top3_probs'][0] for r in results.values()) / len(results)
        st.metric("Average Confidence", f"{avg_confidence:.1f}%")

    with col2:
        avg_time = sum(r['inference_time'] for r in results.values()) / len(results)
        st.metric("Average Inference Time", f"{avg_time:.3f}s")

    with col3:
        fastest_model = min(results.items(), key=lambda x: x[1]['inference_time'])
        st.metric("Fastest Model", fastest_model[0])


# # Create Image with Prediction Overlay

# In[39]:


def create_prediction_image(img, prediction, confidence):
    try:
        img_copy = img.copy()

        try:
            font_large = ImageFont.truetype("arial.ttf", 40)
            font_small = ImageFont.truetype("arial.ttf", 30)
        except:
            try:
                font_large = ImageFont.truetype("Arial.ttf", 40)
                font_small = ImageFont.truetype("Arial.ttf", 30)
            except:
                try:
                    font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
                    font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
                except:
                    font_large = ImageFont.load_default()
                    font_small = ImageFont.load_default()

        width, height = img_copy.size

        overlay = Image.new('RGBA', img_copy.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        if confidence > 80:
            color = (46, 125, 50, 220)
        elif confidence > 60:
            color = (255, 152, 0, 220)
        else:
            color = (244, 67, 54, 220)

        overlay_draw.rectangle([(0, 0), (width, 140)], fill=color)

        img_copy = img_copy.convert('RGBA')
        img_copy = Image.alpha_composite(img_copy, overlay)
        img_copy = img_copy.convert('RGB')

        draw = ImageDraw.Draw(img_copy)

        max_length = 35
        display_prediction = prediction if len(prediction) <= max_length else prediction[:max_length] + "..."

        draw.text((20, 20), f"Prediction: {display_prediction}",
                  fill=(255, 255, 255), font=font_large)
        draw.text((20, 80), f"Confidence: {confidence:.1f}%",
                  fill=(255, 255, 255), font=font_small)

        return img_copy

    except Exception as e:
        st.warning(f"Could not create annotated image: {str(e)}")
        return img


def create_comparison_grid(img, results, class_data=None):
    try:
        img_resized = img.resize((400, 400))

        annotated_images = []

        for model_name, result in results.items():
            pred = result['top3_classes'][0]
            conf = result['top3_probs'][0]

            annotated = create_prediction_image(img_resized, f"{model_name}: {pred}", conf)
            annotated_images.append(annotated)

        n_models = len(annotated_images)

        if n_models == 0:
            return img
        elif n_models == 1:
            return annotated_images[0]
        elif n_models == 2:
            grid_width = 800
            grid_height = 400
            grid = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
            grid.paste(annotated_images[0], (0, 0))
            grid.paste(annotated_images[1], (400, 0))
        else:
            grid_width = 800
            grid_height = 800
            grid = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))

            grid.paste(annotated_images[0], (200, 0))

            if len(annotated_images) > 1:
                grid.paste(annotated_images[1], (0, 400))
            if len(annotated_images) > 2:
                grid.paste(annotated_images[2], (400, 400))

        return grid

    except Exception as e:
        st.error(f"Could not create comparison grid: {str(e)}")
        return img


def create_detailed_report_image(img, result, class_data=None):
    try:
        canvas_width = 1200
        canvas_height = 800
        canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        try:
            font_title = ImageFont.truetype("arial.ttf", 36)
            font_large = ImageFont.truetype("arial.ttf", 28)
            font_medium = ImageFont.truetype("arial.ttf", 22)
            font_small = ImageFont.truetype("arial.ttf", 18)
        except:
            try:
                font_title = ImageFont.truetype("Arial.ttf", 36)
                font_large = ImageFont.truetype("Arial.ttf", 28)
                font_medium = ImageFont.truetype("Arial.ttf", 22)
                font_small = ImageFont.truetype("Arial.ttf", 18)
            except:
                font_title = font_large = font_medium = font_small = ImageFont.load_default()

        draw.rectangle([(0, 0), (canvas_width, 80)], fill=(46, 125, 50))
        draw.text((20, 20), "Plant Disease Detection Report",
                  fill=(255, 255, 255), font=font_title)

        img_display = img.resize((500, 500))
        canvas.paste(img_display, (50, 120))

        x_offset = 600
        y_offset = 120

        draw.text((x_offset, y_offset), f"Model: {result['model_name']}",
                  fill=(46, 125, 50), font=font_large)
        y_offset += 50

        draw.text((x_offset, y_offset), "Top Diagnosis:",
                  fill=(0, 0, 0), font=font_medium)
        y_offset += 35

        top_pred = result['top3_classes'][0]
        if len(top_pred) > 30:
            top_pred = top_pred[:27] + "..."
        draw.text((x_offset + 20, y_offset), top_pred,
                  fill=(46, 125, 50), font=font_large)
        y_offset += 45

        conf = result['top3_probs'][0]
        conf_color = (46, 125, 50) if conf > 80 else (255, 152, 0) if conf > 60 else (244, 67, 54)

        draw.text((x_offset + 20, y_offset), f"Confidence: {conf:.2f}%",
                  fill=conf_color, font=font_large)
        y_offset += 60

        draw.text((x_offset, y_offset), "Top 3 Predictions:",
                  fill=(0, 0, 0), font=font_medium)
        y_offset += 40

        for i in range(min(3, len(result['top3_classes']))):
            medal = "1" if i == 0 else "2" if i == 1 else "3"
            pred_text = result['top3_classes'][i]
            if len(pred_text) > 25:
                pred_text = pred_text[:22] + "..."
            text = f"{medal}. {pred_text}: {result['top3_probs'][i]:.2f}%"
            draw.text((x_offset + 20, y_offset), text,
                      fill=(0, 0, 0), font=font_small)
            y_offset += 35

        y_offset += 20
        draw.text((x_offset, y_offset),
                  f"Processing Time: {result['inference_time']:.3f}s",
                  fill=(0, 0, 0), font=font_small)

        draw.rectangle([(0, canvas_height - 60), (canvas_width, canvas_height)],
                       fill=(200, 230, 201))
        draw.text((20, canvas_height - 45),
                  f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                  fill=(0, 0, 0), font=font_small)
        draw.text((canvas_width - 400, canvas_height - 45),
                  "Plant Disease Detection System v1.0",
                  fill=(0, 0, 0), font=font_small)

        return canvas

    except Exception as e:
        st.error(f"Could not create report image: {str(e)}")
        return img


def export_results_package(img, result, class_data=None):
    try:
        import zipfile

        zip_buffer = BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            img_buffer = BytesIO()
            img.save(img_buffer, format='PNG')
            zip_file.writestr('original_image.png', img_buffer.getvalue())

            annotated = create_prediction_image(
                img,
                result['top3_classes'][0],
                result['top3_probs'][0]
            )
            annotated_buffer = BytesIO()
            annotated.save(annotated_buffer, format='PNG')
            zip_file.writestr('annotated_image.png', annotated_buffer.getvalue())

            if result.get('heatmap') is not None:
                try:
                    img_array = np.array(img.resize(IMG_SIZE))
                    _, heatmap_colored, overlay = create_gradcam_visualization(img_array, result['heatmap'])
                    gradcam_pil = Image.fromarray(overlay)
                    gradcam_buffer = BytesIO()
                    gradcam_pil.save(gradcam_buffer, format='PNG')
                    zip_file.writestr('gradcam_visualization.png', gradcam_buffer.getvalue())
                except Exception as e:
                    st.warning(f"Could not generate Grad-CAM visualization: {str(e)}")

            report_img = create_detailed_report_image(img, result, class_data)
            report_buffer = BytesIO()
            report_img.save(report_buffer, format='PNG')
            zip_file.writestr('detailed_report.png', report_buffer.getvalue())

            json_data = {
                'timestamp': datetime.now().isoformat(),
                'model': result['model_name'],
                'top_prediction': result['top3_classes'][0],
                'confidence': float(result['top3_probs'][0]),
                'top3_predictions': [
                    {
                        'class': result['top3_classes'][i],
                        'confidence': float(result['top3_probs'][i])
                    } for i in range(min(3, len(result['top3_classes'])))
                ],
                'inference_time': float(result['inference_time'])
            }
            zip_file.writestr('results.json', json.dumps(json_data, indent=2))

            text_report = f"""
Plant Disease Detection Report
{'=' * 50}

Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: {result['model_name']}

PRIMARY DIAGNOSIS:
{result['top3_classes'][0]}
Confidence: {result['top3_probs'][0]:.2f}%

TOP 3 PREDICTIONS:
1. {result['top3_classes'][0]} - {result['top3_probs'][0]:.2f}%
2. {result['top3_classes'][1]} - {result['top3_probs'][1]:.2f}%
3. {result['top3_classes'][2]} - {result['top3_probs'][2]:.2f}%

PERFORMANCE:
Processing Time: {result['inference_time']:.3f} seconds

RECOMMENDATIONS:
- Consult with agricultural expert for confirmation
- Monitor plant condition regularly
- Follow appropriate treatment protocols

{'=' * 50}
Generated by Plant Disease Detection System v1.0
            """
            zip_file.writestr('report.txt', text_report)

        zip_buffer.seek(0)
        return zip_buffer

    except Exception as e:
        st.error(f"Could not create results package: {str(e)}")
        return None


# # Batch Processing Feature (BONUS)

# In[40]:


def batch_process_images(uploaded_files, models_dict, selected_model, class_data,
                         use_enhancement=False, save_visualizations=False):
    if not uploaded_files:
        st.warning("No files uploaded")
        return [], {}

    results = []
    visualizations = {} if save_visualizations else None

    if selected_model not in models_dict or not models_dict[selected_model].get('loaded', False):
        st.error(f"Model {selected_model} not available")
        return [], {}

    model = models_dict[selected_model]['model']
    preprocess_fn = models_dict[selected_model]['preprocess']

    progress_bar = st.progress(0)
    status_text = st.empty()

    start_time = time.time()
    successful = 0
    failed = 0

    for idx, file in enumerate(uploaded_files):
        try:
            status_text.text(f"Processing image {idx + 1}/{len(uploaded_files)}: {file.name}")

            img = Image.open(file).convert('RGB')

            top3_classes, top3_probs, heatmap, all_preds = predict_disease(
                img, model, class_data, preprocess_fn, use_enhancement
            )

            if top3_classes is None:
                failed += 1
                results.append({
                    'filename': file.name,
                    'status': 'Failed',
                    'prediction': 'N/A',
                    'confidence': 0.0,
                    'top3_classes': [],
                    'top3_probs': []
                })
                continue

            result = {
                'filename': file.name,
                'status': 'Success',
                'prediction': top3_classes[0],
                'confidence': top3_probs[0],
                'top3_classes': top3_classes,
                'top3_probs': top3_probs,
                'file_size': file.size / 1024,
                'image_size': f"{img.width}×{img.height}"
            }

            results.append(result)
            successful += 1

            if save_visualizations and heatmap is not None:
                img_array = np.array(img.resize(IMG_SIZE))
                _, heatmap_colored, overlay = create_gradcam_visualization(img_array, heatmap)
                visualizations[file.name] = {
                    'original': img,
                    'heatmap': heatmap_colored,
                    'overlay': overlay,
                    'prediction': top3_classes[0],
                    'confidence': top3_probs[0]
                }

        except Exception as e:
            failed += 1
            st.warning(f"Failed to process {file.name}: {str(e)}")
            results.append({
                'filename': file.name,
                'status': 'Error',
                'prediction': 'Error',
                'confidence': 0.0,
                'error': str(e)
            })

        progress_bar.progress((idx + 1) / len(uploaded_files))

    total_time = time.time() - start_time
    avg_time = total_time / len(uploaded_files) if uploaded_files else 0

    status_text.empty()
    progress_bar.empty()

    st.success("Batch processing complete!")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Images", len(uploaded_files))
    with col2:
        st.metric("Successful", successful, delta=f"{successful / len(uploaded_files) * 100:.0f}%")
    with col3:
        st.metric("Failed", failed, delta=f"-{failed / len(uploaded_files) * 100:.0f}%" if failed > 0 else "0%")
    with col4:
        st.metric("Avg Time", f"{avg_time:.2f}s", delta=f"{total_time:.1f}s total")

    return results, visualizations


def display_batch_results(results, visualizations=None):
    if not results:
        st.warning("No results to display")
        return

    st.markdown("### Batch Processing Results")

    df = pd.DataFrame(results)

    column_order = ['filename', 'status', 'prediction', 'confidence']
    if 'image_size' in df.columns:
        column_order.append('image_size')
    if 'file_size' in df.columns:
        column_order.append('file_size')

    df = df[column_order]

    if 'confidence' in df.columns:
        df['confidence'] = df['confidence'].apply(lambda x: f"{x:.2f}%" if x > 0 else "N/A")

    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Analysis Summary")

    successful_results = [r for r in results if r.get('status') == 'Success']

    if successful_results:
        from collections import Counter
        predictions = [r['prediction'] for r in successful_results]
        pred_counts = Counter(predictions)

        st.markdown("#### Disease Distribution")

        fig, ax = plt.subplots(figsize=(12, 6))
        diseases = list(pred_counts.keys())
        counts = list(pred_counts.values())

        bars = ax.barh(diseases, counts, color='#66bb6a')
        ax.set_xlabel('Number of Images', fontweight='bold')
        ax.set_title('Detected Diseases Distribution', fontweight='bold', fontsize=14)
        ax.grid(axis='x', alpha=0.3)

        for i, (disease, count) in enumerate(zip(diseases, counts)):
            ax.text(count + 0.1, i, str(count), va='center', fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("#### Confidence Distribution")

        confidences = [r['confidence'] for r in successful_results]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(confidences, bins=20, color='#4caf50', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Confidence (%)', fontweight='bold')
        ax.set_ylabel('Number of Images', fontweight='bold')
        ax.set_title('Prediction Confidence Distribution', fontweight='bold', fontsize=14)
        ax.axvline(x=np.mean(confidences), color='red', linestyle='--',
                   label=f'Mean: {np.mean(confidences):.1f}%')
        ax.axvline(x=np.median(confidences), color='blue', linestyle='--',
                   label=f'Median: {np.median(confidences):.1f}%')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Confidence", f"{np.mean(confidences):.1f}%")
        with col2:
            st.metric("Median Confidence", f"{np.median(confidences):.1f}%")
        with col3:
            high_conf = sum(1 for c in confidences if c > 80)
            st.metric("High Confidence (>80%)", f"{high_conf}/{len(confidences)}")

    if visualizations:
        st.markdown("---")
        st.markdown("### Visualization Gallery")

        n_cols = 3
        viz_items = list(visualizations.items())

        for i in range(0, len(viz_items), n_cols):
            cols = st.columns(n_cols)

            for j, col in enumerate(cols):
                if i + j < len(viz_items):
                    filename, viz_data = viz_items[i + j]

                    with col:
                        with st.expander(f"{filename}", expanded=False):
                            st.image(viz_data['original'], caption="Original",
                                     use_container_width=True)
                            if 'heatmap' in viz_data:
                                st.image(viz_data['heatmap'], caption="Heatmap",
                                         use_container_width=True)
                            if 'overlay' in viz_data:
                                st.image(viz_data['overlay'], caption="Overlay",
                                         use_container_width=True)
                            st.write(f"**{viz_data['prediction']}**")
                            st.write(f"Confidence: {viz_data['confidence']:.1f}%")


def export_batch_results(results, format='csv'):
    if not results:
        st.warning("No results to export")
        return None

    try:
        if format == 'csv':
            df = pd.DataFrame(results)
            buffer = BytesIO()
            df.to_csv(buffer, index=False)
            buffer.seek(0)
            return buffer

        elif format == 'json':
            json_str = json.dumps(results, indent=2)
            buffer = BytesIO()
            buffer.write(json_str.encode())
            buffer.seek(0)
            return buffer

        elif format == 'excel':
            df = pd.DataFrame(results)
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Results', index=False)

                workbook = writer.book
                worksheet = writer.sheets['Results']

                header_format = workbook.add_format({
                    'bold': True,
                    'bg_color': '#4caf50',
                    'font_color': 'white',
                    'border': 1
                })

                for col_num, value in enumerate(df.columns.values):
                    worksheet.write(0, col_num, value, header_format)

                for i, col in enumerate(df.columns):
                    max_len = max(df[col].astype(str).apply(len).max(), len(col)) + 2
                    worksheet.set_column(i, i, max_len)

            buffer.seek(0)
            return buffer

        else:
            st.error(f"Unsupported format: {format}")
            return None

    except Exception as e:
        st.error(f"Export failed: {str(e)}")
        return None


def create_batch_report_pdf(results, visualizations=None):
    st.info("PDF report generation would require additional libraries (reportlab)")
    st.info("For now, you can export to Excel or CSV formats")


# # Real-time Video Processing (BONUS)

# In[41]:


def process_video_frame(frame, model, class_data, preprocess_fn=None):
    try:
        if preprocess_fn is None:
            preprocess_fn = tf.keras.applications.resnet50.preprocess_input

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        img_resized = img.resize(IMG_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = preprocess_fn(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array, verbose=0)
        top_idx = np.argmax(predictions[0])

        # FIXED: Handle different class_data structures
        try:
            if 'formatted_names' in class_data:
                top_class = class_data['formatted_names'][top_idx]
            elif str(top_idx) in class_data:
                top_class = class_data[str(top_idx)]['name']
            elif isinstance(class_data, list):
                top_class = class_data[top_idx]
            else:
                top_class = f"Class_{top_idx}"
        except (KeyError, IndexError):
            top_class = f"Class_{top_idx}"

        top_prob = float(predictions[0][top_idx]) * 100

        return top_class, top_prob

    except Exception as e:
        return "Error", 0.0


def annotate_frame(frame, prediction, confidence, fps=0):
    try:
        frame_copy = frame.copy()
        h, w = frame_copy.shape[:2]

        if confidence > 80:
            color = (76, 175, 80)
        elif confidence > 60:
            color = (0, 152, 255)
        else:
            color = (54, 67, 244)

        overlay = frame_copy.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), color, -1)
        cv2.addWeighted(overlay, 0.4, frame_copy, 0.6, 0, frame_copy)

        cv2.rectangle(frame_copy, (0, 0), (w, 100), color, 3)

        font = cv2.FONT_HERSHEY_SIMPLEX

        pred_text = f"Disease: {prediction[:30]}"
        cv2.putText(frame_copy, pred_text, (15, 35),
                    font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        conf_text = f"Confidence: {confidence:.1f}%"
        cv2.putText(frame_copy, conf_text, (15, 70),
                    font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        if fps > 0:
            fps_text = f"FPS: {fps:.1f}"
            text_size = cv2.getTextSize(fps_text, font, 0.6, 2)[0]
            cv2.putText(frame_copy, fps_text,
                        (w - text_size[0] - 15, 30),
                        font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        return frame_copy

    except Exception as e:
        return frame


def webcam_realtime_detection(models_dict, selected_model, class_data,
                              confidence_threshold=60.0, show_fps=True):
    st.markdown("### Real-Time Webcam Detection")

    if selected_model not in models_dict or not models_dict[selected_model].get('loaded', False):
        st.error(f"Model {selected_model} not available")
        return

    model = models_dict[selected_model]['model']
    preprocess_fn = models_dict[selected_model]['preprocess']

    st.markdown("""
    <div class="info-box">
    <b>Instructions:</b>
    <ul>
        <li>Click 'Take a picture' to capture a frame</li>
        <li>Position the leaf clearly in view</li>
        <li>Ensure good lighting for best results</li>
        <li>The prediction will appear below</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    img_file = st.camera_input("Capture Plant Leaf")

    if img_file is not None:
        img = Image.open(img_file).convert('RGB')

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Captured Image**")
            st.image(img, use_container_width=True)

        with col2:
            st.markdown("**Real-Time Prediction**")

            with st.spinner("Analyzing..."):
                start_time = time.time()

                top3_classes, top3_probs, heatmap, all_preds = predict_disease(
                    img, model, class_data, preprocess_fn
                )

                inference_time = time.time() - start_time

            if top3_classes:
                confidence = top3_probs[0]
                prediction = top3_classes[0]

                st.markdown(f"""
                <div class="prediction-card">
                    <h2 style="color: #2e7d32; margin: 0;">{prediction}</h2>
                    <h1 style="color: #43a047; margin: 0.5rem 0; font-size: 3rem;">
                        {confidence:.1f}%
                    </h1>
                    <p style="color: #666; margin: 0;">
                        Processed in {inference_time:.2f}s
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("**Top 3 Predictions:**")
                for i in range(min(3, len(top3_classes))):
                    st.write(f"{top3_classes[i]}: **{top3_probs[i]:.1f}%**")


def video_file_processing(uploaded_video, models_dict, selected_model, class_data,
                          process_every_n_frames=5, max_frames=100):
    st.markdown("### Video File Processing")

    if selected_model not in models_dict or not models_dict[selected_model].get('loaded', False):
        st.error(f"Model {selected_model} not available")
        return None

    model = models_dict[selected_model]['model']
    preprocess_fn = models_dict[selected_model]['preprocess']

    try:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_video.read())
            video_path = tmp_file.name

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            st.error("Could not open video file")
            if os.path.exists(video_path):
                os.unlink(video_path)
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        st.info(f"Video Info: {total_frames} frames @ {fps:.1f} FPS")

        frames_to_process = min(total_frames // process_every_n_frames, max_frames)

        st.warning(f"Processing every {process_every_n_frames} frames (total: {frames_to_process} frames)")

        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        frame_idx = 0
        processed_idx = 0

        while cap.isOpened() and processed_idx < frames_to_process:
            ret, frame = cap.read()

            if not ret:
                break

            if frame_idx % process_every_n_frames == 0:
                status_text.text(f"Processing frame {processed_idx + 1}/{frames_to_process}")

                top_class, top_prob = process_video_frame(frame, model, class_data, preprocess_fn)

                results.append({
                    'frame': frame_idx,
                    'time': frame_idx / fps if fps > 0 else 0,
                    'prediction': top_class,
                    'confidence': top_prob
                })

                processed_idx += 1
                progress_bar.progress(processed_idx / frames_to_process)

            frame_idx += 1

        cap.release()

        if os.path.exists(video_path):
            try:
                os.unlink(video_path)
            except:
                pass

        status_text.empty()
        progress_bar.empty()

        st.success(f"Processed {len(results)} frames!")

        return results

    except Exception as e:
        st.error(f"Video processing failed: {str(e)}")
        return None


def display_video_results(results):
    if not results:
        st.warning("No results to display")
        return

    st.markdown("### Video Analysis Results")

    df = pd.DataFrame(results)

    st.markdown("#### Prediction Timeline")

    fig, ax = plt.subplots(figsize=(14, 6))

    scatter = ax.scatter(df['time'], df['confidence'],
                         c=df['confidence'], cmap='RdYlGn',
                         s=100, alpha=0.6, edgecolors='black')

    ax.set_xlabel('Time (seconds)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Confidence (%)', fontweight='bold', fontsize=12)
    ax.set_title('Prediction Confidence Over Time', fontweight='bold', fontsize=14)
    ax.grid(alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Confidence (%)', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("#### Disease Distribution in Video")

    from collections import Counter
    pred_counts = Counter(df['prediction'])

    fig, ax = plt.subplots(figsize=(10, 6))
    diseases = list(pred_counts.keys())
    counts = list(pred_counts.values())

    bars = ax.barh(diseases, counts, color='#66bb6a')
    ax.set_xlabel('Number of Frames', fontweight='bold')
    ax.set_title('Detected Diseases in Video', fontweight='bold', fontsize=14)
    ax.grid(axis='x', alpha=0.3)

    for i, (disease, count) in enumerate(zip(diseases, counts)):
        percentage = (count / len(results)) * 100
        ax.text(count + 0.5, i, f"{count} ({percentage:.1f}%)",
                va='center', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("#### Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Avg Confidence", f"{df['confidence'].mean():.1f}%")
    with col2:
        most_common = pred_counts.most_common(1)[0]
        pred_display = most_common[0][:20] if len(most_common[0]) > 20 else most_common[0]
        st.metric("Most Common", pred_display,
                  delta=f"{most_common[1]} frames")
    with col3:
        high_conf_frames = len(df[df['confidence'] > 80])
        st.metric("High Confidence", f"{high_conf_frames}/{len(df)}",
                  delta=f"{high_conf_frames / len(df) * 100:.0f}%")

    with st.expander("View Detailed Frame Data"):
        st.dataframe(df, use_container_width=True)


# # Confusion Matrix with Real Data

# In[42]:


def show_confusion_matrix(actual_cm=None, class_names=None):
    st.markdown("### Model Performance - Confusion Matrix")

    if actual_cm is None:
        st.markdown("""
        <div class="info-box">
        <b>Load Your Test Results</b><br>
        Upload a confusion matrix file (.npy format) to display actual model performance.
        If no file is uploaded, sample data will be shown for demonstration.
        </div>
        """, unsafe_allow_html=True)

        cm_file = st.file_uploader(
            "Upload confusion matrix (numpy .npy file)",
            type=['npy'],
            help="Upload a saved numpy array containing the confusion matrix"
        )

        if cm_file is not None:
            try:
                cm = np.load(cm_file)
                st.success("Confusion matrix loaded successfully!")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return
        else:
            num_classes = 15 if class_names is None else len(class_names)
            cm = np.random.randint(5, 50, size=(num_classes, num_classes))
            np.fill_diagonal(cm, np.random.randint(80, 150, size=num_classes))

            st.warning("Showing sample data. Upload actual test results for real metrics.")
    else:
        cm = actual_cm
        st.success("Using provided confusion matrix")

    num_classes = cm.shape[0]

    if class_names is None or len(class_names) != num_classes:
        class_names = [f"Class {i}" for i in range(num_classes)]

    display_names = [name[:15] + "..." if len(name) > 15 else name
                     for name in class_names]

    st.markdown("#### Confusion Matrix Heatmap")

    figsize = (max(12, num_classes * 0.8), max(10, num_classes * 0.7))
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax,
                cbar_kws={'label': 'Count'}, linewidths=0.5,
                xticklabels=display_names, yticklabels=display_names,
                square=True)

    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix - Disease Classification',
                 fontsize=14, fontweight='bold', pad=20)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.markdown("### Performance Metrics")

    accuracy = np.trace(cm) / np.sum(cm) * 100

    precision_per_class = []
    recall_per_class = []
    f1_per_class = []

    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = np.sum(cm) - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        precision_per_class.append(precision * 100)

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        recall_per_class.append(recall * 100)

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_per_class.append(f1 * 100)

    macro_precision = np.mean(precision_per_class)
    macro_recall = np.mean(recall_per_class)
    macro_f1 = np.mean(f1_per_class)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Overall Accuracy", f"{accuracy:.2f}%",
                  help="Percentage of correct predictions")
    with col2:
        st.metric("Macro Precision", f"{macro_precision:.2f}%",
                  help="Average precision across all classes")
    with col3:
        st.metric("Macro Recall", f"{macro_recall:.2f}%",
                  help="Average recall across all classes")
    with col4:
        st.metric("Macro F1-Score", f"{macro_f1:.2f}%",
                  help="Harmonic mean of precision and recall")

    st.markdown("---")
    st.markdown("#### Per-Class Performance")

    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Precision (%)': [f"{p:.2f}" for p in precision_per_class],
        'Recall (%)': [f"{r:.2f}" for r in recall_per_class],
        'F1-Score (%)': [f"{f:.2f}" for f in f1_per_class],
        'Support': [np.sum(cm[i, :]) for i in range(num_classes)]
    })

    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Per-Class Metrics Visualization")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(num_classes)
    width = 0.25

    ax1.bar(x - width, precision_per_class, width, label='Precision',
            color='#66bb6a', alpha=0.8)
    ax1.bar(x, recall_per_class, width, label='Recall',
            color='#42a5f5', alpha=0.8)
    ax1.bar(x + width, f1_per_class, width, label='F1-Score',
            color='#ffa726', alpha=0.8)

    ax1.set_xlabel('Class', fontweight='bold')
    ax1.set_ylabel('Score (%)', fontweight='bold')
    ax1.set_title('Precision, Recall, and F1-Score by Class',
                  fontweight='bold', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(display_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 105)

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_normalized, annot=False, cmap='RdYlGn', ax=ax2,
                cbar_kws={'label': 'Proportion'}, linewidths=0.5,
                xticklabels=display_names, yticklabels=display_names,
                vmin=0, vmax=1, square=True)

    ax2.set_xlabel('Predicted Label', fontweight='bold')
    ax2.set_ylabel('True Label', fontweight='bold')
    ax2.set_title('Normalized Confusion Matrix', fontweight='bold', fontsize=13)
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.markdown("#### Best & Worst Performing Classes")

    col_best, col_worst = st.columns(2)

    with col_best:
        st.markdown("**Top 5 Best Classes (by F1-Score)**")
        top_5_idx = np.argsort(f1_per_class)[-5:][::-1]

        for idx in top_5_idx:
            st.markdown(f"""
            <div style="background: #e8f5e9; padding: 0.5rem;
                        border-radius: 8px; margin: 0.3rem 0;">
                <b>{class_names[idx]}</b><br>
                F1: {f1_per_class[idx]:.2f}% |
                Precision: {precision_per_class[idx]:.2f}% |
                Recall: {recall_per_class[idx]:.2f}%
            </div>
            """, unsafe_allow_html=True)

    with col_worst:
        st.markdown("**Bottom 5 Classes (by F1-Score)**")
        bottom_5_idx = np.argsort(f1_per_class)[:5]

        for idx in bottom_5_idx:
            st.markdown(f"""
            <div style="background: #fff3e0; padding: 0.5rem;
                        border-radius: 8px; margin: 0.3rem 0;">
                <b>{class_names[idx]}</b><br>
                F1: {f1_per_class[idx]:.2f}% |
                Precision: {precision_per_class[idx]:.2f}% |
                Recall: {recall_per_class[idx]:.2f}%
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Export Confusion Matrix")

    col_d1, col_d2, col_d3 = st.columns(3)

    with col_d1:
        buffer = BytesIO()
        np.save(buffer, cm)
        buffer.seek(0)

        st.download_button(
            "Download .npy",
            buffer,
            f"confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy",
            "application/octet-stream"
        )

    with col_d2:
        cm_df = pd.DataFrame(cm, columns=display_names, index=display_names)
        csv_buffer = StringIO()
        cm_df.to_csv(csv_buffer)
        csv_string = csv_buffer.getvalue()
        st.download_button(
            "Download CSV",
            csv_string,
            f"confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )

    with col_d3:
        metrics_data = {
            'overall_accuracy': float(accuracy),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'macro_f1': float(macro_f1),
            'per_class_metrics': [
                {
                    'class': class_names[i],
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]),
                    'f1_score': float(f1_per_class[i])
                } for i in range(num_classes)
            ]
        }

        json_buffer = BytesIO()
        json_buffer.write(json.dumps(metrics_data, indent=2).encode())
        json_buffer.seek(0)

        st.download_button(
            "Download Metrics (JSON)",
            json_buffer,
            f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "application/json"
        )


# # Webcam Input Function (FIXED)

# In[43]:


def webcam_input():
    st.markdown("### Live Camera Capture")

    st.markdown("""
    <div class="info-box">
    <b>Camera Instructions:</b><br>
    1. Click the "Take a picture" button below<br>
    2. Allow camera access in your browser when prompted<br>
    3. Position the plant leaf clearly in the center of frame<br>
    4. Ensure good lighting for best results<br>
    5. Click the capture button to take the photo
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Tips for Best Results"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Good Practices:**
            - Use natural daylight
            - Keep leaf flat and centered
            - Fill the frame with the leaf
            - Avoid shadows on the leaf
            - Hold camera steady
            """)

        with col2:
            st.markdown("""
            **Avoid:**
            - Blurry or shaky images
            - Dark or overexposed photos
            - Leaf too small in frame
            - Multiple leaves overlapping
            - Dirty camera lens
            """)

    img_file = st.camera_input("Take a picture of the plant leaf")

    if img_file is not None:
        img = Image.open(img_file).convert('RGB')

        st.success("Image captured successfully!")

        col_info1, col_info2, col_info3 = st.columns(3)

        with col_info1:
            st.metric("Resolution", f"{img.width}×{img.height}")
        with col_info2:
            st.metric("File Size", f"{img_file.size / 1024:.1f} KB")
        with col_info3:
            img_array = np.array(img)
            brightness = np.mean(img_array)
            quality = "Good" if 50 < brightness < 200 else "Check Lighting"
            st.metric("Lighting", quality)

        if st.button("Retake Photo"):
            st.rerun()

        return img

    return None


def webcam_input_with_preview(models_dict, selected_model, class_data):
    st.markdown("### Live Camera with Instant Analysis")

    if selected_model not in models_dict or not models_dict[selected_model].get('loaded', False):
        st.error(f"Model {selected_model} not available")
        return None, None

    st.markdown("""
    <div class="info-box">
    <b>Quick Capture Mode:</b><br>
    Take a photo and get instant AI analysis!
    </div>
    """, unsafe_allow_html=True)

    col_set1, col_set2 = st.columns(2)

    with col_set1:
        auto_analyze = st.checkbox("Auto-analyze after capture", value=True,
                                   help="Automatically run prediction when image is captured")

    with col_set2:
        show_confidence_bar = st.checkbox("Show confidence bar", value=True)

    img_file = st.camera_input("Capture Plant Leaf")

    if img_file is not None:
        img = Image.open(img_file).convert('RGB')

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**Captured Image**")
            st.image(img, use_container_width=True)

            img_array = np.array(img)
            brightness = np.mean(img_array)
            contrast = np.std(img_array)

            if brightness < 50:
                st.warning("Image is too dark. Consider better lighting.")
            elif brightness > 200:
                st.warning("Image is overexposed. Reduce lighting.")

            if contrast < 20:
                st.warning("Low contrast. Image may be blurry.")

        with col2:
            if auto_analyze:
                st.markdown("**AI Analysis**")

                with st.spinner("Analyzing..."):
                    start_time = time.time()

                    model = models_dict[selected_model]['model']
                    preprocess_fn = models_dict[selected_model]['preprocess']

                    top3_classes, top3_probs, gradcam_img, all_preds = predict_disease(
                        img, model, class_data, preprocess_fn
                    )

                    inference_time = time.time() - start_time

                if top3_classes:
                    prediction = top3_classes[0]
                    confidence = top3_probs[0]

                    conf_color = "#4caf50" if confidence > 80 else "#ff9800" if confidence > 60 else "#f44336"

                    st.markdown(f"""
                    <div style="background: white; padding: 1.5rem; border-radius: 15px;
                                border-left: 6px solid {conf_color};
                                box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                        <h3 style="margin: 0; color: #2e7d32;">{prediction}</h3>
                        <h1 style="margin: 0.5rem 0; color: {conf_color}; font-size: 2.5rem;">
                            {confidence:.1f}%
                        </h1>
                        <p style="margin: 0; color: #666; font-size: 0.9rem;">
                            {inference_time:.2f}s | {selected_model}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    if show_confidence_bar:
                        st.markdown("**Top 3 Predictions:**")
                        for i in range(3):
                            medal = "1" if i == 0 else "2" if i == 1 else "3"
                            st.write(f"{medal}. {top3_classes[i]}")
                            st.progress(top3_probs[i] / 100)
                            st.caption(f"{top3_probs[i]:.1f}%")

                    st.markdown("---")
                    col_btn1, col_btn2 = st.columns(2)

                    with col_btn1:
                        if st.button("Retake", use_container_width=True):
                            st.rerun()

                    with col_btn2:
                        if st.button("Save Result", use_container_width=True):
                            st.session_state['last_webcam_result'] = {
                                'image': img,
                                'prediction': prediction,
                                'confidence': confidence,
                                'timestamp': datetime.now()
                            }
                            st.success("Result saved!")

                    return img, {
                        'top3_classes': top3_classes,
                        'top3_probs': top3_probs,
                        'gradcam_img': gradcam_img,
                        'inference_time': inference_time
                    }
            else:
                st.info("Image captured. Click 'Analyze' to run prediction.")

                if st.button("Analyze Image", use_container_width=True):
                    st.rerun()

        return img, None

    return None, None


def webcam_continuous_mode(models_dict, selected_model, class_data, fps_limit=2):
    st.markdown("### Continuous Monitoring Mode")

    st.warning("""
    Note: Streamlit's camera widget captures single frames, not continuous video.
    For true continuous monitoring, consider using a video processing library with
    OpenCV or deploy as a separate web application.
    """)

    st.info("""
    Alternative: Use the standard webcam capture and click 'Retake' for new predictions.
    For production use, implement a dedicated video streaming service.
    """)


# # Main App

# In[44]:


def main():
    models_dict, class_data, available_models = initialize_app()

    if not available_models:
        st.error("No models available! Please check model paths.")
        st.stop()

    st.sidebar.title("🌿 Navigation")
    app_mode = st.sidebar.radio(
        "Go to",
        ["Home", "Disease Detection", "Model Comparison", "Batch Processing",
         "Real-Time Detection", "Performance Metrics"],
        help="Choose a feature to explore"
    )

    st.sidebar.markdown("---")

    selected_model_name = st.sidebar.selectbox(
        "🤖 Select AI Model",
        available_models,
        help="Choose different architectures to see how they perform"
    )
    st.sidebar.markdown("---")
    use_tflite = st.sidebar.checkbox(
        "🚀 Use TFLite (Fast Mode)",
        value=False,
        help="Use optimized TFLite model for faster inference"
    )

    if use_tflite:
        st.sidebar.success("⚡ TFLite Mode: 3x faster!")
        st.sidebar.caption("Optimized for mobile/edge devices")

    st.sidebar.markdown("---")
    display_model_status()

    if app_mode == "Home":
        st.title("🌿 Plant Disease Detection System")

        st.markdown("""
        <div class="success-box">
        <h3>Welcome to the Smart Agriculture Assistant!</h3>
        <p>This advanced system uses Deep Learning to identify plant diseases with high accuracy.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🚀 Key Features")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            **🎯 Accurate Detection**
            - 3 state-of-the-art AI models
            - 90%+ accuracy rate
            - Real-time predictions
            """)

        with col2:
            st.markdown("""
            **🔍 Explainable AI**
            - Grad-CAM visualizations
            - Understand AI decisions
            - Confidence scoring
            """)

        with col3:
            st.markdown("""
            **⚡ Fast Processing**
            - Instant results
            - Batch processing
            - Video analysis support
            """)

        st.markdown("---")
        st.markdown("### 📖 How to Use")

        st.markdown("""
        <div class="info-box">
        <b>Step-by-step Guide:</b><br><br>
        1. <b>Select a Model</b> from the sidebar (ResNet50, EfficientNet, or MobileNet)<br>
        2. <b>Choose Detection Mode</b> - Single image, batch, or real-time<br>
        3. <b>Upload/Capture Image</b> - Use file upload or camera<br>
        4. <b>Get Results</b> - AI analyzes and shows diagnosis with confidence<br>
        5. <b>View Explanation</b> - See Grad-CAM heatmap showing AI's focus areas
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 💡 Best Practices")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("""
            **✅ Do:**
            - Use clear, well-lit images
            - Center the leaf in frame
            - Ensure leaf fills most of the image
            - Use natural lighting when possible
            - Keep leaf flat and in focus
            """)

        with col_b:
            st.markdown("""
            **❌ Avoid:**
            - Blurry or shaky photos
            - Dark or overexposed images
            - Multiple leaves overlapping
            - Extreme angles or distortion
            - Dirty or wet lens
            """)

    elif app_mode == "Disease Detection":
        st.title("🔬 Disease Diagnosis")

        tab1, tab2 = st.tabs(["📤 Upload Image", "📷 Use Camera"])

        with tab1:
            st.markdown("### Upload Plant Leaf Image")

            uploaded_file = st.file_uploader(
                "Choose a leaf image...",
                type=["jpg", "jpeg", "png"],
                help="Upload a clear image of the affected plant leaf"
            )

            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')

                col1, col2 = st.columns([1, 1])

                with col1:
                    st.markdown("**Original Image**")
                    st.image(image, use_container_width=True)

                    st.markdown("**Image Info:**")
                    st.write(f"📐 Size: {image.width}×{image.height}")
                    st.write(f"📦 File: {uploaded_file.size / 1024:.1f} KB")

                with col2:
                    st.markdown("### 🎨 Image Enhancement")

                    with st.expander("Adjustment Controls", expanded=False):
                        params = show_enhancement_controls()

                        if any(params[k] != 1.0 for k in ['brightness', 'contrast', 'sharpness', 'color']) or params.get('use_clahe'):
                            enhanced_image = apply_enhancements(image, params)
                            st.markdown("**Enhanced Preview:**")
                            st.image(enhanced_image, use_container_width=True)
                            use_enhanced = st.checkbox("Use enhanced image for analysis", value=True)
                            final_image = enhanced_image if use_enhanced else image
                        else:
                            final_image = image

                    if not 'params' in locals():
                        final_image = image

                st.markdown("---")

                col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])

                with col_btn1:
                    use_ensemble = st.checkbox("🔄 Use enhancement ensemble", value=False,
                                              help="Average predictions from original + enhanced")

                with col_btn2:
                    show_gradcam = st.checkbox("🔥 Show Grad-CAM", value=True,
                                              help="Visualize what AI focuses on")

                with col_btn3:
                    export_results_btn = st.checkbox("💾 Export results", value=False)

                if st.button("🔍 Analyze Leaf", use_container_width=True, type="primary"):
                    with st.spinner(f"🤖 AI analyzing using {selected_model_name}..."):
                        result = predict_with_model(
                            final_image,
                            selected_model_name,
                            models_dict,
                            class_data,
                            use_enhancement=use_ensemble
                        )

                    if result:
                        st.markdown("---")
                        st.markdown("### 📊 Diagnosis Results")

                        prediction = result['top3_classes'][0]
                        confidence = result['top3_probs'][0]

                        conf_color = "#4caf50" if confidence > 80 else "#ff9800" if confidence > 60 else "#f44336"

                        st.markdown(f"""
                        <div class="prediction-card">
                            <h2 style="margin:0; color: #2e7d32;">🌱 Diagnosis: {prediction}</h2>
                            <h1 style="margin:0.5rem 0; color:{conf_color}; font-size:3rem;">
                                {confidence:.2f}%
                            </h1>
                            <p style="margin:0; color:#666;">
                                ⏱️ Processed in {result['inference_time']:.3f}s using {selected_model_name}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        col_r1, col_r2, col_r3 = st.columns(3)

                        with col_r1:
                            st.metric("Top Prediction", prediction[:20])
                        with col_r2:
                            st.metric("Confidence", f"{confidence:.1f}%")
                        with col_r3:
                            st.metric("Model", selected_model_name)

                        st.markdown("---")
                        st.markdown("#### 📈 Top 3 Predictions")

                        for i in range(min(3, len(result['top3_classes']))):
                            with st.container():
                                col_p1, col_p2 = st.columns([3, 1])
                                with col_p1:
                                    st.write(f"**{i+1}. {result['top3_classes'][i]}**")
                                with col_p2:
                                    st.write(f"**{result['top3_probs'][i]:.2f}%**")
                                st.progress(result['top3_probs'][i] / 100)

                        if show_gradcam and result.get('heatmap') is not None:
                            st.markdown("---")
                            display_gradcam_analysis(
                                final_image,
                                result['heatmap'],
                                prediction,
                                confidence
                            )

                        if export_results_btn:
                            st.markdown("---")
                            st.markdown("### 💾 Export Results")

                            zip_package = export_results_package(final_image, result, class_data)

                            if zip_package:
                                st.download_button(
                                    "📦 Download Complete Results Package",
                                    zip_package,
                                    f"plant_disease_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                    "application/zip",
                                    use_container_width=True
                                )

        with tab2:
            st.markdown("### 📷 Live Camera Capture")
            img, cam_result = webcam_input_with_preview(models_dict, selected_model_name, class_data)

    elif app_mode == "Model Comparison":
        st.title("🔬 Multi-Model Comparison")

        st.markdown("""
        <div class="info-box">
        Compare predictions from all available models simultaneously to increase confidence in diagnosis.
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload leaf image for comparison", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')

            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("**Test Image**")
                st.image(image, use_container_width=True)

            with col2:
                st.markdown("**Comparison Settings**")
                use_enhancement = st.checkbox("Use enhancement ensemble", value=False)
                show_all_gradcams = st.checkbox("Show all Grad-CAMs", value=True)

            if st.button("🚀 Run Multi-Model Analysis", use_container_width=True, type="primary"):
                results = compare_model_predictions(image, models_dict, class_data, use_enhancement)

                if results:
                    display_comparison_results(results, class_data)

                    if show_all_gradcams:
                        st.markdown("---")
                        st.markdown("### 🔥 Grad-CAM Comparison")

                        cols = st.columns(len(results))

                        for idx, (model_name, result) in enumerate(results.items()):
                            with cols[idx]:
                                st.markdown(f"**{model_name}**")

                                if result.get('heatmap') is not None:
                                    img_array = np.array(image.resize(IMG_SIZE))
                                    _, _, overlay = create_gradcam_visualization(img_array, result['heatmap'])
                                    st.image(overlay, use_container_width=True)
                                else:
                                    st.info("Grad-CAM not available")

                    grid_img = create_comparison_grid(image, results, class_data)

                    st.markdown("---")
                    st.markdown("### 📸 Comparison Grid")
                    st.image(grid_img, use_container_width=True)

    elif app_mode == "Batch Processing":
        st.title("📦 Batch Image Processing")

        st.markdown("""
        <div class="info-box">
        Upload multiple images at once for efficient bulk processing. Perfect for analyzing entire crops.
        </div>
        """, unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Upload multiple leaf images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )

        if uploaded_files:
            st.success(f"📁 {len(uploaded_files)} files uploaded")

            col_opt1, col_opt2 = st.columns(2)

            with col_opt1:
                use_enhancement = st.checkbox("Use enhancement ensemble", value=False)

            with col_opt2:
                save_viz = st.checkbox("Save visualizations", value=False)

            if st.button("🚀 Process All Images", use_container_width=True, type="primary"):
                results, visualizations = batch_process_images(
                    uploaded_files,
                    models_dict,
                    selected_model_name,
                    class_data,
                    use_enhancement=use_enhancement,
                    save_visualizations=save_viz
                )

                if results:
                    display_batch_results(results, visualizations)

                    st.markdown("---")
                    st.markdown("### 💾 Export Batch Results")

                    col_e1, col_e2, col_e3 = st.columns(3)

                    with col_e1:
                        csv_buffer = export_batch_results(results, format='csv')
                        if csv_buffer:
                            st.download_button(
                                "📄 Download CSV",
                                csv_buffer,
                                f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                "text/csv",
                                use_container_width=True
                            )

                    with col_e2:
                        json_buffer = export_batch_results(results, format='json')
                        if json_buffer:
                            st.download_button(
                                "📋 Download JSON",
                                json_buffer,
                                f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                "application/json",
                                use_container_width=True
                            )

                    with col_e3:
                        excel_buffer = export_batch_results(results, format='excel')
                        if excel_buffer:
                            st.download_button(
                                "📊 Download Excel",
                                excel_buffer,
                                f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )

    elif app_mode == "Real-Time Detection":
        st.title("📹 Real-Time Detection")

        tab1, tab2 = st.tabs(["📷 Live Camera", "🎥 Video Upload"])

        with tab1:
            webcam_realtime_detection(
                models_dict,
                selected_model_name,
                class_data,
                confidence_threshold=60.0,
                show_fps=True
            )

        with tab2:
            st.markdown("### 🎥 Video File Analysis")

            uploaded_video = st.file_uploader("Upload video file", type=['mp4', 'avi', 'mov'])

            if uploaded_video:
                col_v1, col_v2 = st.columns(2)

                with col_v1:
                    process_every_n = st.slider("Process every N frames", 1, 30, 5)

                with col_v2:
                    max_frames = st.slider("Max frames to process", 10, 500, 100)

                if st.button("🎬 Process Video", use_container_width=True, type="primary"):
                    video_results = video_file_processing(
                        uploaded_video,
                        models_dict,
                        selected_model_name,
                        class_data,
                        process_every_n_frames=process_every_n,
                        max_frames=max_frames
                    )

                    if video_results:
                        display_video_results(video_results)

    elif app_mode == "Performance Metrics":
        st.title("Model Performance Analysis")

        st.markdown("""
        <div class="info-box">
        View detailed performance metrics including confusion matrices and per-class statistics.
        </div>
        """, unsafe_allow_html=True)

        if class_data:
            if 'formatted_names' in class_data:
                class_names = class_data['formatted_names']
            elif isinstance(class_data, dict) and '0' in class_data:
                class_names = [class_data[str(i)]['name'] for i in range(len(class_data))]
            elif isinstance(class_data, list):
                class_names = class_data
            else:
                class_names = None
        else:
            class_names = None

        show_confusion_matrix(actual_cm=None, class_names=class_names)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info("""
    **Plant Disease Detection v1.0**

    AI-powered agricultural disease diagnosis system

    **Models:**
    - ResNet50
    - EfficientNetB3
    - MobileNet

    **Framework:** TensorFlow + Streamlit
    
        **Team Members:**
        - Mariam Mohamed
        - Marina Shenouda
        - Alaa Orabe
        - Maria Gerges
        - Ahmed Ayman

        **Course:** AI Skills
        **Date:** December 2025

        Built with ❤️ for better agriculture
    **Date:** December 2025
    """)



# In[1]:


if __name__ == "__main__":
    main()

