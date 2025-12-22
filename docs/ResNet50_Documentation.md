# ResNet50 – PlantVillage Disease Classification

## 1. Project Overview
This project implements a plant disease classification system using the **ResNet50** architecture within the **TENSORFLOW** framework. The goal is to accurately identify various plant diseases from leaf images in the PlantVillage dataset.

---

## 2. Dataset Information
The model is trained on the **PlantVillage Dataset**, which contains images of healthy and diseased plant leaves across 15 different classes, including:
- Pepper (bell) Bacterial spot / Healthy
- Potato Early blight / Late blight / Healthy
- Tomato Bacterial spot / Early blight / Late blight / Leaf Mold / Septoria leaf spot / Spider mites / Target Spot / Yellow Leaf Curl Virus / Mosaic virus / Healthy

### Data Splitting
The dataset is split into training, validation, and test sets using `sklearn.model_selection.train_test_split`:
- **Training Set (60%)**: Used for model weight updates.
- **Validation Set (20%)**: Used for hyperparameter tuning and monitoring.
- **Test Set (20%)**: Used for final performance evaluation.

---

## 3. Data Pipeline & Preparation
The data pipeline includes rigorous image verification and preprocessing to ensure model stability.

### Image Cleaning
The script iterates through the dataset and verifies each image using the `PIL` library, skipping any corrupted or invalid files.

### Preprocessing & Augmentation
Images are processed using `torchvision.transforms`. Data augmentation is applied to the training set to improve generalization:
- **Resize**: (224, 224) pixels.
- **Random Horizontal Flip**: Adds variance to the leaf orientation.
- **Random Rotation**: ±10 degrees rotation.
- **Normalization**: Uses ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.

```python
train_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])
```

---

## 4. Model Architecture
The implementation uses a transfer learning approach with a pre-trained ResNet50 backbone.

### Core Components:
- **Base Model**: `resnet50` from `torchvision.models` with default ImageNet weights.
- **Custom Head**: The original fully connected (`fc`) layer is replaced to match the 15 classes of the PlantVillage dataset.

```python
from torchvision.models import resnet50, ResNet50_Weights

weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)

# Replace the final fully connected layer
model.fc = nn.Linear(2048, num_classes)
```

---

## 5. Training Strategy
The model is trained using standard deep learning optimization techniques.

### Training Configuration:
- **Optimizer**: `Adam` with a learning rate of `0.001`.
- **Loss Function**: `CrossEntropyLoss`.
- **Device**: Automatically detects and uses **CUDA (GPU)** if available.
- **Monitoring**: Loss and accuracy are tracked for both training and validation sets across epochs.

---

## 6. Model Evaluation
After training, the model's performance is analyzed on the unseen test set.

### Metrics Captured:
- **Accuracy**: Overall percentage of correct predictions.
- **Precision, Recall, and F1-Score**: Calculated per class to identify specific weaknesses.
- **Confusion Matrix**: Visualized using `seaborn` to show misclassifications between specific disease categories.
- **Training Curves**: Plots showing loss and accuracy trends to detect overfitting.

---

## 7. How to Use
### Prerequisites
Ensure you have the following libraries installed:
```bash
pip install torch torchvision timm scikit-learn pillow matplotlib seaborn
```

### Running the Notebook
1.  Navigate to `All/Task1.ipynb`.
2.  Set the dataset path to your PlantVillage directory.
3.  Run all cells sequentially to perform data splitting, training, and evaluation.
4.  The best model weights will be saved as `best_model.pth`.
