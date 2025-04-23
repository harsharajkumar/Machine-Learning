# 📘 K-Nearest Neighbors (KNN) Classifier from Scratch in Python
This repository demonstrates a simple yet complete implementation of the K-Nearest Neighbors (KNN) algorithm using only numpy and Python's built-in libraries. The KNN algorithm is a non-parametric, lazy learning algorithm used for classification and regression. This project focuses on classification.

# ❓ What is KNN?
K-Nearest Neighbors (KNN) is a supervised machine learning algorithm that can be used for both classification and regression tasks. It stores the entire training dataset and predicts the output for a new sample based on a majority vote (classification) or average (regression) of its k nearest neighbors.

# ⚙️ How It Works
Choose the number k of neighbors.

For each test instance:

Calculate the Euclidean distance from the test instance to all training data.

Pick the k closest data points.

For classification, return the most frequent class label among the k neighbors.

No actual training is involved (hence "lazy learning").

🧠 Code Explanation
🔹 Euclidean Distance Function
``` python
def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2)**2))
    return distance
```

Calculates the straight-line distance between two points x1 and x2 in N-dimensional space.

This is the most common distance metric used in KNN.

# 🔹 KNN Class
🔸 Constructor
```
python
def __init__(self, k=3):
    self.k = k
```
Initializes the classifier with a specified number of neighbors k (default is 3).

🔸 fit Method
``` python
def fit(self, X, y):
    self.X_train = X
    self.y_train = y
```
Simply stores the training data. No training is done at this stage because KNN is a lazy learner.

🔸 Predict Method
``` python
def predict(self, X):
    predictions = [self._predict(x) for x in X]
    return predictions
```

Takes a list or array of test samples and returns predictions.

Internally calls _predict(x) on each sample.

🔸 _predict Method
```python
def _predict(self, x):
    distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
    k_indices = np.argsort(distances)[:self.k]
    k_nearest_labels = [self.y_train[i] for i in k_indices]
    most_common = Counter(k_nearest_labels).most_common()
    return most_common[0][0]
```

Calculate distances from the test sample to all training points.

Sort and select the k closest points using np.argsort.

Retrieve the labels of these points.

Use Counter to find the most common label among the neighbors.

Return this label as the predicted class.

# 📌 Example Usage
``` python
import numpy as np
from knn import KNN

# Sample dataset (features and labels)
X_train = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]])
y_train = np.array([0, 0, 0, 1, 1, 1])

# Test data
X_test = np.array([[5, 5]])

# Initialize and use KNN
clf = KNN(k=3)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("Predicted class:", predictions)
```

# 💾 Installation
You only need numpy to run this project:
```
bash
pip install numpy
```
