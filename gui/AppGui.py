%pip install streamlit
import streamlit as st

try:
    icon = Image.open("C:\\Users\\DCS\\Downloads\\Cream & Copper Leaf House Logo .png")
except Exception as e:
    icon = "🌿"  # Fallback

try:
    st.set_page_config(
        page_title="Plant Disease Detector",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
        'About': """
        # Plant Disease Detection System

        **Version:** 1.0
        **Framework:** TensorFlow + Streamlit
        **Models:** ResNet50, EfficientNet, MobileNet

        ---

        **Team Members:**
        - Mariam Mohamed
        - Marina Shenouda
        - Alaa Orabe
        - Maria Gerges
        - Ahmed Ayman

        **Course:** AI Skills
        **Date:** December 2025

        Built with ❤️ for better agriculture
        """
    }
    )
except Exception:
    pass
