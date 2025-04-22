# 📘 **Linear Regression from Scratch using NumPy**

This project demonstrates how the **Linear Regression** algorithm works under the hood, using only **NumPy** for matrix operations. It helps you develop a strong foundation in how gradient descent optimizes a cost function to train a machine learning model.

---

## 🔧 **Class: `LinearRegression`**

A custom-built class that mimics the functionality of linear regression in libraries like `scikit-learn`. This approach helps deepen your understanding of how the algorithm learns by iteratively updating weights using gradient descent.

---

## ✅ **Code Explanation**

### 1. 🛠️ **Initialization: `__init__` method**

The constructor initializes key hyperparameters:

- **`lr`**: Learning rate — controls the step size during gradient descent.
- **`n_iters`**: Number of iterations — how many times weights and bias will be updated.

```python
def __init__(self, lr=0.001, n_iters=1000):
    self.lr = lr
    self.n_iters = n_iters
    self.weight = None
    self.bias = None

## **'2. 📉 Training: fit method'**
🔹 Input:
X: Feature matrix (shape: [n_samples, n_features])

y: Target values (shape: [n_samples])

🔹 Steps:
Initialize weights and bias:
self.weight = np.zeros(n_features)
self.bias = 0

Forward Pass and Gradient Calculation:
y_pred = np.dot(X, self.weight) + self.bias
dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
db = (1 / n_samples) * np.sum(y_pred - y)

Gradient Descent (Weight Update):
self.weight -= self.lr * dw
self.bias -= self.lr * db
This loop runs for n_iters steps to minimize the Mean Squared Error (MSE).

3. 🎯 Prediction: predict method
Returns predictions based on learned weights and bias:
def predict(self, X):
    return np.dot(X, self.weight) + self.bias
🧠 Concept Behind the Algorithm
Linear Regression finds the best-fit line by minimizing the difference between predicted values and actual values — typically using Mean Squared Error (MSE).
This implementation uses Gradient Descent to update the model's parameters step-by-step toward the optimal solution.

🧪 Example Usage
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# Create a dummy dataset
X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression(lr=0.01, n_iters=1000)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Visualize
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, predictions, color='red', label='Predicted')
plt.legend()
plt.show()
🧾 Key Takeaways
✅ Built completely from scratch using NumPy

🔍 Offers a clear view of how gradient descent works in linear regression

⚙️ Code is modular, clean, and extendable

🧠 Great for learning core ML concepts like optimization and error minimization

