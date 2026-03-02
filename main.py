# disease_prediction_clustering.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🩺 Disease Prediction + Patient Clustering")
st.write("Binary classification (Diabetes) + Unsupervised clustering with PCA visualization")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")
    return df

df = load_data()
st.subheader("Dataset Preview")
st.write(df.head())

# -------------------------------
# Preprocessing
# -------------------------------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# PCA Visualization
# -------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

st.subheader("PCA Visualization")
fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap="coolwarm", alpha=0.7)
legend1 = ax.legend(*scatter.legend_elements(), title="Outcome")
ax.add_artist(legend1)
st.pyplot(fig)

# -------------------------------
# Clustering (KMeans)
# -------------------------------
st.subheader("Patient Clustering (KMeans)")
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap="viridis", alpha=0.7)
legend2 = ax.legend(*scatter.legend_elements(), title="Cluster")
ax.add_artist(legend2)
st.pyplot(fig)

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# -------------------------------
# Logistic Regression
# -------------------------------
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)

# -------------------------------
# Random Forest
# -------------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

# -------------------------------
# ANN Model
# -------------------------------
ann = Sequential([
    Dense(16, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])

ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = ann.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=0)

y_pred_ann = (ann.predict(X_test) > 0.5).astype("int32")
acc_ann = accuracy_score(y_test, y_pred_ann)

# -------------------------------
# Results
# -------------------------------
st.subheader("Model Comparison")
results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "ANN"],
    "Accuracy": [acc_lr, acc_rf, acc_ann]
})
st.write(results)

# -------------------------------
# ANN Training Curve
# -------------------------------
st.subheader("ANN Training Curve")
fig, ax = plt.subplots()
ax.plot(history.history["accuracy"], label="Train Accuracy")
ax.plot(history.history["val_accuracy"], label="Val Accuracy")
ax.set_xlabel("Epochs")
ax.set_ylabel("Accuracy")
ax.legend()
st.pyplot(fig)

# -------------------------------
# Confusion Matrix (ANN)
# -------------------------------
st.subheader("Confusion Matrix (ANN)")
cm = confusion_matrix(y_test, y_pred_ann)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
st.pyplot(fig)

# -------------------------------
# Interactive Prediction
# -------------------------------
st.subheader("🔮 Try Interactive Prediction")

# Input widgets for patient data
preg = st.slider("Pregnancies", 0, 15, 2)
glucose = st.slider("Glucose", 0, 200, 120)
bp = st.slider("BloodPressure", 0, 122, 70)
skin = st.slider("SkinThickness", 0, 99, 20)
insulin = st.slider("Insulin", 0, 846, 79)
bmi = st.slider("BMI", 0.0, 67.1, 32.0)
dpf = st.slider("DiabetesPedigreeFunction", 0.0, 2.5, 0.5)
age = st.slider("Age", 21, 81, 33)

# Collect inputs
patient_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
patient_scaled = scaler.transform(patient_data)

# Predict with ANN
patient_pred_ann = (ann.predict(patient_scaled) > 0.5).astype("int32")[0][0]

# Predict with Random Forest
patient_pred_rf = rf.predict(patient_scaled)[0]

# Predict with Logistic Regression
patient_pred_lr = log_reg.predict(patient_scaled)[0]

st.write("### Predictions")
st.write(f"Logistic Regression: {'Diabetic' if patient_pred_lr==1 else 'Healthy'}")
st.write(f"Random Forest: {'Diabetic' if patient_pred_rf==1 else 'Healthy'}")
st.write(f"ANN: {'Diabetic' if patient_pred_ann==1 else 'Healthy'}")

# -------------------------------
# Visualize patient on PCA plot
# -------------------------------
patient_pca = pca.transform(patient_scaled)

fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap="coolwarm", alpha=0.5)
ax.scatter(patient_pca[:,0], patient_pca[:,1], c="yellow", edgecolors="black", s=200, label="New Patient")
ax.legend()
ax.set_title("Patient Position in PCA Space")
st.pyplot(fig)

st.success("✅ Pipeline complete: PCA visualization, clustering, and model comparison done!")
