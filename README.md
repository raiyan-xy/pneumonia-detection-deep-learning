# Pneumonia Detection from Chest X-Rays

Built a deep learning model using TensorFlow and Keras to detect pneumonia from chest X-ray images. This project compares a custom Convolutional Neural Network (CNN) with a transfer learning approach using VGG16 to classify chest X-rays as Normal or Pneumonia. The models are evaluated using multiple performance metrics, and Grad-CAM is used to visualize which regions of the image influenced each prediction.

---

## Results

| Model | Accuracy | Recall | F1 Score | ROC AUC |
|-------|----------|--------|----------|---------|
| CNN from Scratch | 85.90% | 96.15% | 89.50% | 0.9429 |
| VGG16 Transfer Learning | 86.54% | 98.72% | 90.16% | 0.9652 |

VGG16 won on almost every metric. The biggest difference is recall — it caught 
98.72% of pneumonia cases vs 96.15% for the CNN. In medical screening, missing 
a real case is far more dangerous than a false alarm, so recall is what matters most.

---

## What I Built

- EDA on 5,863 chest X-rays including class imbalance analysis
- Custom CNN from scratch as a baseline
- VGG16 transfer learning model with frozen base layers
- Grad-CAM heatmaps showing which lung regions drove each prediction
- Full evaluation across accuracy, precision, recall, F1, confusion matrix, and ROC AUC

---

## Tech Stack

Python, TensorFlow, Keras, NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, Google Colab

---

## Dataset

**Chest X-Ray Images (Pneumonia)** by Paul Mooney (Kaggle)

- 5,863 chest X-ray images
- Two classes: Normal and Pneumonia
- Pre-split into training, validation, and test sets

---

## Key Takeaways

- Transfer learning achieved the strongest overall performance.
- Recall proved to be the most important metric for medical screening.
- Grad-CAM increased model interpretability by highlighting the lung regions influencing predictions.

---

## Disclaimer

This is an educational project. The model has not been clinically validated 
and is not intended for real world medical diagnosis.
