# **📘 Linear Regression from Scratch using NumPy**
This project implements Linear Regression from the ground up using only NumPy, demonstrating how gradient descent optimizes the cost function to train a machine learning model. Perfect for understanding the core mechanics behind one of the most fundamental ML algorithms.

🚀 Features
✅ Pure NumPy Implementation – No reliance on high-level ML libraries
✅ Custom Gradient Descent – Learn how optimization works step-by-step
✅ Modular & Clean Code – Easy to extend and adapt
✅ Visualization Support – Works seamlessly with matplotlib for plotting

# 🛠️ Class: LinearRegression
A custom-built class that replicates the functionality of scikit-learn's LinearRegression, with full control over the training process.

🔧 Initialization
```python
def __init__(self, lr=0.001, n_iters=1000):
    self.lr = lr          # Learning rate (step size for gradient descent)
    self.n_iters = n_iters # Number of training iterations
    self.weight = None     # Model weights (coefficients)
    self.bias = None       # Bias term (intercept)
```


# 📊 Training (fit method)
Input:

X → Feature matrix of shape (n_samples, n_features)

y → Target values of shape (n_samples,)

Steps:

Initialize weights & bias (zeros by default).

Gradient Descent Loop:

Compute predictions:

``` python
y_pred = np.dot(X, self.weight) + self.bias
```

# Calculate gradients (MSE derivatives):

```python
dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))  # Weight gradient
db = (1 / n_samples) * np.sum(y_pred - y)         # Bias gradient
```

# Update parameters:

``` python
self.weight -= self.lr * dw
self.bias -= self.lr * db
```

# 🔮 Prediction (predict method)
Returns predictions for new data:

``` python
def predict(self, X):
    return np.dot(X, self.weight) + self.bias
```

# 🧠 How It Works
Linear Regression minimizes the Mean Squared Error (MSE) between predictions and true values by iteratively adjusting weights via gradient descent

# � Example Usage

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression(lr=0.01, n_iters=1000)
model.fit(X_train, y_train)

# Predict and visualize
predictions = model.predict(X_test)
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, predictions, color='red', linewidth=2, label='Predicted')
plt.legend()
plt.show()
```

Output:
Linear Regression Fit (Replace with real plot)
<img width="583" alt="image" src="https://github.com/user-attachments/assets/17cf0b81-3a27-4d00-bd88-405dedd41617" />


# 📚 Key Takeaways
🔹 Understand Gradient Descent – See how weights update to minimize error.
🔹 No Black Box – Full transparency into the algorithm’s workings.
🔹 Foundational ML – Builds intuition for more complex models.

