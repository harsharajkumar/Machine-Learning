# 🧠 Logistic Regression from Scratch using NumPy

This repository contains a beginner-friendly implementation of **Logistic Regression**, a fundamental classification algorithm in machine learning, built entirely from scratch using **NumPy**.

## 📚 What is Logistic Regression?

**Logistic Regression** is a supervised learning algorithm used for **binary classification**. It estimates the probability that a data point belongs to a certain class using the **sigmoid function**. If the probability is greater than 0.5, we classify the point as class `1`, otherwise as class `0`.

---

## 🚀 Features

- Implements logistic regression using just NumPy
- Predicts binary class labels
- Trains the model using **gradient descent**
- Uses **sigmoid activation** to output probabilities
- Fully customizable learning rate and iterations

---

## 🛠️ Project Structure

### `logistic_regression.py`

This file contains the core logic of our Logistic Regression model.

#### 🔧 Initialization

```python
def __init__(self, lr=0.001, n_iters=1000):
    self.lr = lr                  # Learning rate for gradient descent
    self.n_iters = n_iters        # Number of training iterations
    self.weight = None            # Placeholder for weights
    self.bias = None              # Placeholder for bias
```

lr: Controls how much we update the model's weights at each step.

n_iters: Number of times the model loops through the entire dataset during training.

# 🧠 The Sigmoid Function
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

This activation function maps any real-valued number to a value between 0 and 1.

Used to interpret model output as probabilities.

# 🏋️ Model Training (fit method)
``` python
def fit(self, X, y):
    n_samples, n_features = X.shape
    self.weight = np.zeros(n_features)
    self.bias = 0
```

Initializes weights and bias to zero.

``` python
for _ in range(self.n_iters):
    linear_model = np.dot(X, self.weight) + self.bias
    predictions = sigmoid(linear_model)
```
Applies a linear transformation on input features.

Uses the sigmoid function to convert output into probabilities.

```python
dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
db = (1 / n_samples) * np.sum(predictions - y)

self.weight -= self.lr * dw
self.bias -= self.lr * db
```
Calculates gradients for weights and bias using binary cross-entropy loss derivative.

Updates the model using gradient descent.

# 📊 Predicting Probabilities (predict_proba method)
``` python
def predict_proba(self, X):
    linear_model = np.dot(X, self.weight) + self.bias
    return sigmoid(linear_model)
```
Returns the predicted probabilities for each input sample.

# 🧾 Predicting Class Labels (predict method)
```python
def predict(self, X):
    probabilities = self.predict_proba(X)
    return [1 if i > 0.5 else 0 for i in probabilities]
```
Classifies outputs as 1 if probability > 0.5, otherwise 0.

# 🧪 Example Usage
Here’s how you can use the model on synthetic data:

``` python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from logistic_regression import LogisticRegression
import numpy as np

# Generate sample binary classification data
X, y = make_classification(n_samples=1000, n_features=2, n_classes=2, random_state=123)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize and train the model
model = LogisticRegression(lr=0.1, n_iters=1000)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate accuracy
accuracy = np.mean(predictions == y_test)
print("Accuracy:", accuracy)
```
# 📊 Sample Output
makefile
Accuracy: 0.89
✅ To-Do / Improvements
 Add multi-class support using One-vs-Rest

 Add regularization (L1/L2)

 Visualize decision boundaries

 Implement early stopping

# 🤝 Contributing
Contributions are welcome! Fork the repo, make your changes, and open a pull request.

# 🙌 Acknowledgements
Built for educational purposes and a deeper understanding of how logistic regression works under the hood.

Inspired by classic ML courses and real-world math.

