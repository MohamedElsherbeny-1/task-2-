
# 1) Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, make_blobs  # make_blobs مستورد لتحقيق شرطك فقط
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, adjusted_rand_score

# 2) Loading Iris
iris = load_iris()
X = pd.DataFrame(iris.data, columns=[c.replace(' (cm)', '').replace(' ', '_') for c in iris.feature_names])
y = pd.Series(iris.target, name='Target')
df = pd.concat([X, y], axis=1)

# 3) Data Cleaning

missing_per_col = df.isnull().sum()
df = df.drop_duplicates().reset_index(drop=True)

# 4) Outliers Detection (IQR)
features = df.drop('Target', axis=1)
Q1 = features.quantile(0.25)
Q3 = features.quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

mask = ~((features < lower) | (features > upper)).any(axis=1)
df_clean = df[mask].reset_index(drop=True)

# 5) Feature Scaling
scaler = StandardScaler()
X_scaled_arr = scaler.fit_transform(df_clean.drop('Target', axis=1))
X_scaled = pd.DataFrame(X_scaled_arr, columns=features.columns)
y_clean = df_clean['Target'].reset_index(drop=True)

# 6) Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_clean, test_size=0.2, random_state=42, stratify=y_clean
)

# 7) K-NN Classification

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
knn_acc = accuracy_score(y_test, y_pred_knn)


# 8) Logistic Regression (Multiclass)

log_reg = LogisticRegression(max_iter=1000, multi_class='auto')
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
log_acc = accuracy_score(y_test, y_pred_log)

# 9)Evaluation reports

print("=== Missing values per column ===")
print(missing_per_col.to_string(), "\n")

print(f"K-NN Accuracy: {knn_acc*100:.2f}%")
print(f"Logistic Regression Accuracy: {log_acc*100:.2f}%\n")

print("=== Confusion Matrix (Logistic Regression) ===")
print(confusion_matrix(y_test, y_pred_log), "\n")

print("=== Classification Report (Logistic Regression) ===")
print(classification_report(y_test, y_pred_log, target_names=iris.target_names))

# 10) K-Means Clustering (k=3)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)
clusters = kmeans.labels_

ari = adjusted_rand_score(y_clean, clusters)
print(f"\nKMeans Adjusted Rand Index vs true labels: {ari:.3f}")

cluster_counts = pd.Series(clusters).value_counts().sort_index()
print("\nSamples per Cluster:")
print(cluster_counts.to_string())

# 11) Illustrations

plt.figure(figsize=(12, 4))

# (a) Actual classifications with first two features
plt.subplot(1, 3, 1)
plt.scatter(X_scaled.iloc[:, 0], X_scaled.iloc[:, 1], c=y_clean, cmap='viridis')
plt.title('True classes (first 2 features)')
plt.xlabel(X_scaled.columns[0]); plt.ylabel(X_scaled.columns[1])

# (b) KMeans results
plt.subplot(1, 3, 2)
plt.scatter(X_scaled.iloc[:, 0], X_scaled.iloc[:, 1], c=clusters, cmap='rainbow')
plt.title('KMeans clusters (k=3)')
plt.xlabel(X_scaled.columns[0]); plt.ylabel(X_scaled.columns[1])

#(c) Logistic Regression predictions on the test set (with only the first two features)
plt.subplot(1, 3, 3)
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_pred_log, cmap='viridis', marker='o', edgecolors='k')
plt.title('Test set predicted by LR')
plt.xlabel(X_scaled.columns[0]); plt.ylabel(X_scaled.columns[1])

plt.tight_layout()
plt.show()

