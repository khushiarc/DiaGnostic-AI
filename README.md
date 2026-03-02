# DiaGnostic-AI
An end-to-end healthcare analytics platform that integrates Supervised Learning and Unsupervised Learning to predict diabetes risk and visualize patient health profiles in a real-time interactive dashboard.


Project Overview
This project addresses the challenge of medical data interpretation by combining three different predictive architectures with clustering techniques. It allows clinicians to not only predict the likelihood of disease but also to see how a patient compares to the broader population through dimensionality reduction.

## Key Features
Triple-Model Classification: Compares Logistic Regression, Random Forest, and a Deep Learning (ANN) model to provide a comprehensive risk assessment.

Population Clustering: Utilizes K-Means Clustering to segment patients into distinct health profiles based on physiological similarities.

PCA Visualization: Implements Principal Component Analysis to map high-dimensional medical data into a 2D space for intuitive visual analysis.

Interactive Prediction: A Streamlit-based UI that allows users to input patient vitals (Glucose, BMI, Age, etc.) and receive instant predictions and visual mapping.

Deep Learning Insights: Includes automated training curves and confusion matrices to evaluate the Artificial Neural Network's performance.

## Tech Stack
Language: Python

Web Framework: Streamlit

Machine Learning: Scikit-learn (KMeans, PCA, RandomForest, LogisticRegression)

Deep Learning: TensorFlow, Keras

Data Analysis: Pandas, NumPy

Visualization: Matplotlib, Seaborn


## Outcomes
Real-time Diagnostics: Enabled instant classification of patient data across three different model types.

Visual Patient Mapping: Developed a system to plot "New Patient" data points onto historical clusters, aiding in comparative diagnosis.

Automated Benchmarking: Integrated a comparison table to evaluate which algorithm provides the highest accuracy for the given dataset.
