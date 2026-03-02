# ===============================================================
#  TRAINING SCRIPT (training.py)
#  - Trains Models (LR, RF, ANN)
#  - Saves artifacts for Streamlit
#  - Generates Plots for Streamlit Dashboard
# ===============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
import joblib
import os

sns.set(style="whitegrid")

# Create folder for charts
os.makedirs("charts", exist_ok=True)

# ===============================================================
# 1. LOAD DATASET
# ===============================================================
df = pd.read_csv("diabetes.csv")
print(df.head())
print(df.info())

# ===============================================================
# 2. CLEANING
# ===============================================================
print(df.isnull().sum())
print(df.describe())

# ===============================================================
# 3. FEATURES & TARGET
# ===============================================================
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save Scaler
joblib.dump(scaler, "diabetes_scaler.pkl")
print("\nScaler Saved as diabetes_scaler.pkl")

# ===============================================================
# 4. TRAIN-TEST SPLIT
# ===============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ===============================================================
# 5. MODEL 1: LOGISTIC REGRESSION
# ===============================================================
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
joblib.dump(log_reg, "model_logistic.pkl")

y_pred_lr = log_reg.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)

print("\nLogistic Regression Accuracy:", acc_lr)

# Confusion Matrix Plot
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, cmap="Blues", fmt="d")
plt.title("Logistic Regression Confusion Matrix")
plt.savefig("charts/cm_logistic.png")
plt.close()

# ===============================================================
# 6. MODEL 2: RANDOM FOREST CLASSIFIER
# ===============================================================
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
joblib.dump(rf, "model_rf.pkl")

y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

print("\nRandom Forest Accuracy:", acc_rf)

# Confusion Matrix Plot
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, cmap="Greens", fmt="d")
plt.title("Random Forest Confusion Matrix")
plt.savefig("charts/cm_rf.png")
plt.close()

# ===============================================================
# 7. MODEL 3: ARTIFICIAL NEURAL NETWORK (ANN)
# ===============================================================
model = Sequential([
    Dense(16, activation='relu', input_dim=X_train.shape[1]),
    Dense(12, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# Save ANN Model
model.save("diabetes_ann_model.h5")
print("\nANN Model Saved as diabetes_ann_model.h5")

y_pred_ann = (model.predict(X_test) > 0.5).astype("int32")
acc_ann = accuracy_score(y_test, y_pred_ann)

print("\nANN Accuracy:", acc_ann)

# ANN Confusion Matrix
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred_ann), annot=True, cmap="Oranges", fmt="d")
plt.title("ANN Confusion Matrix")
plt.savefig("charts/cm_ann.png")
plt.close()

# ===============================================================
# 8. PCA + CLUSTERING
# ===============================================================
kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
cluster_labels = kmeans.fit_predict(X_scaled)
df["Cluster"] = cluster_labels

# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

df["PCA1"] = pca_result[:, 0]
df["PCA2"] = pca_result[:, 1]

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df, x="PCA1", y="PCA2",
    hue="Cluster", palette="viridis", s=60
)
plt.title("Patient Clusters (PCA Visualization)")
plt.savefig("charts/pca_clusters.png")
plt.close()

# ===============================================================
# 9. ANN Training Curves
# ===============================================================
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Val")
plt.title("ANN Accuracy Curve")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Val")
plt.title("ANN Loss Curve")
plt.legend()

plt.tight_layout()
plt.savefig("charts/ann_training.png")
plt.close()

# ===============================================================
# 10. SUMMARY
# ===============================================================
print("\n========================")
print("MODEL ACCURACY SUMMARY")
print("========================")
print(f"Logistic Regression : {acc_lr:.4f}")
print(f"Random Forest       : {acc_rf:.4f}")
print(f"ANN                 : {acc_ann:.4f}")

print("\nAll Charts Stored Inside /charts Folder")
print("Artifacts Generated Successfully!")
