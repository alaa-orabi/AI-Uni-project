# MobileNetV2 - PlantVillage Disease Classification Documentation

## 1. Project Overview
This notebook implements a high-performance, lightweight image classification model using the **MobileNetV2** architecture. It is designed to identify 15 types of plant diseases (and healthy states) from the **PlantVillage Dataset**, specifically targeting Pepper, Potato, and Tomato crops.

## 2. Dataset Information
- **Classes:** 15 distinct categories.
- **Plants covered:** Pepper Bell, Potato, and Tomato.
- **Disease types:** Various blights, spots, molds, and viruses.
- **Source:** Automatically fetched from Kaggle via `kagglehub`.

## 3. Preprocessing & Data Augmentation
MobileNetV2 requires specific input sizes (typically 224x224). The notebook handles:
- **Rescaling:** Converting pixel integers to floating point values.
- **Augmentation:** Applying random rotations, shifts, and flips to the training data to increase dataset variance and prevent overfitting.
- **Consistency:** Ensuring the validation and test sets are processed identically to the training set (minus the random augmentations).

## 4. Model Architecture
- **Base Model:** `MobileNetV2` with pre-trained ImageNet weights.
- **Custom Head:** 
  - `GlobalAveragePooling2D` layer.
  - `Dropout` for regularization.
  - `Dense` softmax layer for the 15 categories.
- **Advantages:** MobileNetV2 is optimized for mobile and embedded devices, making the resulting model more efficient in terms of memory and compute compared to larger architectures.

## 5. Training Strategy
- **Optimizer:** `Adam`.
- **Loss:** `categorical_crossentropy`.
- **Pre-processing Utility:** `tensorflow.keras.applications.mobilenet_v2.preprocess_input`.
- **Data Splitting:** 60/20/20 split for training, validation, and testing respectively.

## 6. Model Evaluation
The notebook produces:
- **Confusion Matrix:** To track per-class performance.
- **Plots:** Accuracy and Loss curves for training and validation cycles.
- **Grad-CAM Visualization:** Heatmaps showing the decision-making regions in the input images.

## 7. Conclusions
The MobileNetV2 model provides a balance between accuracy and efficiency. By using transfer learning, it achieves high accuracy on plant disease detection without the need for a massive specialized dataset or weeks of training time.
