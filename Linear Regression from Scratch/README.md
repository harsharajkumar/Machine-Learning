📘 Linear Regression from Scratch using NumPy
This implementation demonstrates how the basic Linear Regression algorithm works under the hood, using only NumPy for matrix operations.

🔧 Class: LinearRegression
A custom-built class that mimics the functionality of linear regression in libraries like scikit-learn. This helps you understand how gradient descent optimizes the cost function.

✅ Code Explanation
1. Initialization: __init__ method
python
Copy
Edit
def __init__(self, lr=0.001, n_iters=1000):
lr: Learning rate — controls how big a step the model takes during each iteration of gradient descent.

n_iters: Number of iterations — how many times the model will update its weights.

self.weight and self.bias: Initialized later in fit().

2. Training: fit method
python
Copy
Edit
def fit(self, X, y):
Input:

X: Feature matrix (shape: [n_samples, n_features])

y: Target values (shape: [n_samples])

Steps:

Initialize weights and bias:

python
Copy
Edit
self.weight = np.zeros(n_features)
self.bias = 0
Forward Pass & Gradient Descent:

Predicted values:

python
Copy
Edit
y_pred = np.dot(X, self.weight) + self.bias
Gradients:

python
Copy
Edit
dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
db = (1 / n_samples) * np.sum(y_pred - y)
Weight update rule (Gradient Descent):

python
Copy
Edit
self.weight -= self.lr * dw
self.bias -= self.lr * db
The loop runs for n_iters steps to minimize the Mean Squared Error (MSE) cost function.

3. Prediction: predict method
python
Copy
Edit
def predict(self, X):
Takes in a matrix of new features X

Returns the predicted values using the learned weight and bias

python
Copy
Edit
return np.dot(X, self.weight) + self.bias
🧠 Concept Behind the Algorithm
Linear regression tries to find the best-fit line through the data by minimizing the difference between predicted values and actual values — commonly using Mean Squared Error (MSE). This implementation uses gradient descent to update the weights and bias iteratively.

🧪 Example Usage
python
Copy
Edit
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
No external ML libraries used — built from scratch using NumPy.

Helps build an intuitive understanding of how gradient descent optimizes linear regression.

Clean, modular, and easy to extend.
