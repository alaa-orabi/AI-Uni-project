# 🌿 EfficientNetB3 – PlantVillage Disease Classification

## 📝 Project Description
This project focuses on the classification of plant diseases using deep learning techniques. Specifically, we leverage the **EfficientNetB3** architecture to achieve high accuracy in detecting various diseases across different plant species. The system is designed to help farmers and agronomists identify plant diseases early, allowing for timely intervention and crop protection.

## 📂 Dataset Sources
The model is trained on the **PlantVillage Dataset**, which is available on Kaggle:
- [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/emmarex/plantdisease/data)

The dataset includes thousands of images across 15 different classes, covering crops like Pepper, Potato, and Tomato.

## ⚙️ Setup Instructions

### 🛠️ Dependencies
Ensure you have the following libraries installed:
```bash
pip install tensorflow torch torchvision matplotlib numpy pillow scikit-learn kagglehub tqdm streamlit opencv-python
```

### 🚀 How to Run
1.  **Model Training:**
    - **EfficientNetB3:** Navigate to `notebook/EfficientNetB3/` and run `Final_EfficientNetB3_Model_.ipynb`.
    - **MobileNetV2:** Navigate to `notebook/MobileNetV2/` and run `Final_MobileNetV2_Model_.ipynb`.
    - **ResNet50:** Navigate to `notebook/ResNet50/` and run `Final_ResNet50_Model_.ipynb`.
2.  **GUI Application:**
    - Open `gui/AppGui.ipynb` and run all cells to launch the Streamlit interface.

### 🖥️ GUI Instructions
- Launch the application from `gui/AppGui.ipynb`.
- Select the desired model (ResNet50, EfficientNetB3, or MobileNet) from the sidebar.
- Upload an image of a plant leaf or use the camera to take a photo.
- The system will provide:
    - **Predicted Class:** The name of the detected disease or "Healthy".
    - **Confidence Score:** The probability assigned by the model.
    - **Grad-CAM Analysis:** A heatmap visualization showing parts of the leaf that influenced the model's decision.

## 📊 Model Results and Performance
- **Models Used:** EfficientNetB3, ResNet50, MobileNet.
- **Accuracy:** Accomplished high validation accuracy (~96% for ResNet50).
- **Visualization:** Integrated Grad-CAM for model explainability.
- **Performance Graphs:** Plots for training/validation loss and accuracy are available within the `notebook/` directory.

## 👥 Team Members
- **Mariam Mohamed**
- **Marina Shenouda**
- **Alaa Orabe**
- **Maria Gerges**
- **Ahmed khairy**
- **Ahmed Ayman**
