# EfficientNetB3 - PlantVillage Disease Classification Documentation

## 1. Project Overview
This notebook implements a deep learning model using the **EfficientNetB3** architecture to classify plant diseases from the **PlantVillage Dataset**. The focus is on three main plants: **Pepper**, **Potato**, and **Tomato**, covering 15 different health categories (both diseased and healthy).

## 2. Dataset Information
- **Source:** PlantVillage Dataset (downloaded via `kagglehub` from `emmarex/plantdisease`).
- **Total Classes:** 15
- **Main Categories:**
  - Pepper, bell: Bacterial spot, Healthy
  - Potato: Early blight, Late blight, Healthy
  - Tomato: Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites (Two-spotted spider mite), Target Spot, YellowLeaf Curl Virus, Mosaic virus, Healthy

## 3. Data Pipeline & Preparation
### 3.1 Data Loading
The dataset is downloaded and extracted locally. Class names are formatted from raw folder names (e.g., `Tomato___Late_blight`) into human-readable labels (`Tomato - Late Blight`) using a custom utility function.

### 3.2 Data Augmentation
To improve model generalization, the following transformations are applied to the training set using `ImageDataGenerator`:
- **Rescaling:** Pixel values normalized to [0, 1].
- **Rotation:** Up to 40 degrees.
- **Shifts:** Width and height shifts (up to 20%).
- **Shear & Zoom:** Up to 20%.
- **Flip:** Horizontal flips enabled.
- **Brightness:** Range [0.8, 1.2].

The validation and test sets only undergo **rescaling**.

### 3.3 Data Splitting
- **Training Set:** 60%
- **Validation Set:** 20%
- **Test Set:** 20%

## 4. Model Architecture
The model leverages Transfer Learning:
- **Base Model:** `EfficientNetB3` (pre-trained on ImageNet).
- **Pooling:** `GlobalAveragePooling2D` to reduce spatial dimensions.
- **Dropout:** A dropout layer is included to prevent overfitting.
- **Output Layer:** A `Dense` layer with 15 units and **Softmax** activation for multi-class classification.

## 5. Training Details
- **Optimizer:** `Adam` optimizer.
- **Loss Function:** `categorical_crossentropy`.
- **Metrics:** `accuracy`.
- **Pre-processing:** Uses `tensorflow.keras.applications.efficientnet.preprocess_input`.

## 6. Visualization & Explainability (Grad-CAM)
The notebook includes an implementation of **Grad-CAM** (Gradient-weighted Class Activation Mapping). This allows the user to see which parts of the plant leaf the model is focusing on when making a prediction, providing "explainability" for the AI's decisions.

## 7. How to Use
1. **Download Dataset:** The code automatically downloads the dataset using Kaggle API credentials or `kagglehub`.
2. **Train:** Execute the training cells to fit the model to the augmented data.
3. **Evaluate:** Check the accuracy and loss plots, and verify performance on the test set.
4. **Predict:** Use the prediction utility to test on single images and view Grad-CAM overlays.
