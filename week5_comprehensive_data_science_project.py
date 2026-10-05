# Week 5: Comprehensive Data Science Project
# Integrates data analysis, visualization, statistical testing,
# and machine learning evaluation.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency, ttest_ind, f_oneway

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)

# ---------------------------------------------------------
# 1. DATASET AND BASIC EXPLORATION
# ---------------------------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

print("Dataset shape:", X.shape)
print("\nClass distribution:")
print(y.value_counts())
print("\nMissing values:", X.isnull().sum().sum())
print("\nDescriptive statistics:")
print(X.describe())

# ---------------------------------------------------------
# 2. EXPLORATORY VISUALIZATION
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.histplot(X["mean radius"], kde=True)
plt.title("Distribution of Mean Radius")
plt.xlabel("Mean Radius")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 3. MACHINE LEARNING PREPROCESSING
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 4. MODEL DEVELOPMENT
# ---------------------------------------------------------
model = LogisticRegression(max_iter=5000, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# ---------------------------------------------------------
# 5. MODEL EVALUATION
# ---------------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\nModel Evaluation")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-score :", f1)
print("ROC-AUC  :", auc)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------
# 6. CONFUSION MATRIX
# ---------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Malignant", "Benign"],
    yticklabels=["Malignant", "Benign"]
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 7. ROC CURVE
# ---------------------------------------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 8. STRATEGIC INTERPRETATION
# ---------------------------------------------------------
print("\nStrategic Recommendations:")
print("1. Use standardized preprocessing before classification.")
print("2. Monitor recall because missed positive cases can be important.")
print("3. Compare Logistic Regression with tree-based models.")
print("4. Use cross-validation and hyperparameter tuning.")
print("5. Validate future models on larger and more diverse datasets.")
