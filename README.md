# 🏏 Cricket Match Winner Prediction using Machine Learning

This repository contains a complete pipeline for predicting the outcome of cricket matches using machine learning models such as Random Forest, SVM, Neural Network, and LightGBM. It also compares model performance on original vs PCA-transformed datasets.

---

## 📌 Project Objective

To build a machine learning pipeline that predicts the winner of a cricket match using pre-match data, incorporating feature engineering, PCA, and multiple classification models.

---

## 📂 Folder Structure

- `data/` - Raw and processed datasets.
- `notebooks/` - Colab/Jupyter Notebooks for step-by-step development.
- `src/` - Python scripts for preprocessing, feature engineering, model training and evaluation.
- `results/` - Evaluation metrics, plots, and saved models.

---

## 🔧 Technologies Used

- Python (Pandas, NumPy, Matplotlib, Seaborn)
- Scikit-learn
- Keras (for Neural Networks)
- LightGBM
- PCA
- Colab / Jupyter

---

## 🧠 Models Used

- Support Vector Machine (SVM)
- Random Forest Classifier
- Neural Network (Keras and MLPClassifier)
- LightGBM

Each model was evaluated using:
- Accuracy
- Precision, Recall, F1-score
- ROC AUC
- Confusion Matrix
- Training Time

---

## 📉 PCA Analysis

We applied PCA for dimensionality reduction and trained models on both original and PCA-transformed datasets. Performance metrics were compared to assess the impact.

---

## 📊 Visualizations

- Confusion Matrices
- ROC Curves
- PCA Variance Explained
- Decision Trees (for interpretability)

---

## 🚀 How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/<your-username>/cricket-match-prediction-ml.git
   cd cricket-match-prediction-ml
