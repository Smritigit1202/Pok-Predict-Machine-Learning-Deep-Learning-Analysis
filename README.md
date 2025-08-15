# Pokémon Prediction - Machine Learning & Deep Learning Analysis

This project applies multiple Machine Learning and Deep Learning models on the Pokémon dataset to predict Pokémon types, legendary status, stats, and more.  
Each ML/DL model is implemented in a separate Python file for clarity.

---



## 📊 Dataset
- Source: Kaggle Pokémon Dataset → https://www.kaggle.com/abcsds/pokemon
- Example columns:
  - Name, Type 1, Type 2, Total, HP, Attack, Defense, Sp. Atk, Sp. Def, Speed, Generation, Legendary

---

## 🔧 Installation

1️⃣ Clone the repository  
 

2️⃣ Install dependencies  

requirements.txt:
pandas
numpy
scikit-learn
matplotlib
tensorflow

---

## 🚀 How to Run

Each model is in a separate file. For example:  
python linear_regression.py  
python logistic_regression.py  
python svm.py  
python decision_tree.py  
python kmeans.py  
python pca.py  
python neural_network.py  
python deep_learning.py  
python naive_bayes.py  
python knn.py  

---

## 📈 Models Implemented
Model | Task | Type  
------|------|------  
Linear Regression | Predict Total Stats | Regression  
Logistic Regression | Predict Legendary | Classification  
SVM | Predict Type 1 | Classification  
Decision Tree | Predict Type 1 | Classification  
K-Means | Cluster Pokémon | Clustering  
PCA | Reduce Dimensions for Visualization | Unsupervised  
Neural Network (Sklearn) | Predict Legendary | Classification  
Deep Learning (Keras) | Predict Legendary | Classification  
Naive Bayes | Predict Type 1 | Classification  
KNN | Predict Type 1 | Classification  

---

## 🛠 Preprocessing Notes
- Categorical columns are Label Encoded.
- Features are Standard Scaled for models that require scaling.

---

## 📌 Future Improvements
- Add hyperparameter tuning (GridSearchCV)
- Compare all models in a single script
- Save trained models using joblib or pickle

