# 🚀 Understanding Gradient Descent from Scratch (with Python Example)

Gradient Descent is the backbone of many machine learning algorithms — from linear regression to deep learning.  
In this post, we’ll **build a simple gradient descent implementation from scratch in Python** to understand how it optimizes parameters step by step.

---

## 💡 What is Gradient Descent?

Gradient Descent is an optimization algorithm used to minimize a **cost function** by iteratively adjusting parameters (like weights and bias in regression).  

Think of it as trying to walk down a hill in fog — at every step, you look around, find the direction of steepest descent, and take a small step downward.

In our case:
- We’re minimizing the **Mean Squared Error (MSE)** between predicted and actual values.
- Parameters to update: **weights (w)** and **bias (b)**.

---

## 🧮 The Math Behind It

For **Linear Regression**,  
\[
\hat{y} = Xw + b
\]

Our **cost function (MSE)** is:
\[
J(w, b) = \frac{1}{N} \sum (y - \hat{y})^2
\]

To minimize \( J \), we find its gradient (partial derivatives):

\[
\frac{\partial J}{\partial w} = -\frac{2}{N}\sum (y - \hat{y})X
\]
\[
\frac{\partial J}{\partial b} = -\frac{2}{N}\sum (y - \hat{y})
\]

And then update weights:
\[
w = w - \alpha \frac{\partial J}{\partial w}
\]
\[
b = b - \alpha \frac{\partial J}{\partial b}
\]

Where \( \alpha \) is the **learning rate** — how big a step we take each time.

---

## 🧑‍💻 Let’s Code It!

Here’s the full implementation:

```python
import numpy as np
from sklearn.datasets import make_regression

# Generate synthetic linear data
X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)

# Hyperparameters
alpha = 0.01  # learning rate
epochs = range(100)
weights = 1
bias = 0

# To track progress
error_mat = []
weights_mat = []
bias_mat = []

def weight_bias_derivative(alpha, X, Y, weights, bias):
    N = len(X)
    Y_pred = X * weights + bias
    error = float(np.sum((Y - Y_pred)**2) / N)
    
    # Gradients
    dw = (-2 / N) * np.sum((Y - Y_pred) * X)
    db = (-2 / N) * np.sum(Y - Y_pred)
    
    # Update parameters
    weights = weights - alpha * dw
    bias = bias - alpha * db
    
    return error, weights, bias

# Run gradient descent
for i in epochs:
    error, weights, bias = weight_bias_derivative(alpha, X, y, weights, bias)
    error_mat.append(round(error, 3))
    weights_mat.append(round(weights, 3))
    bias_mat.append(round(bias, 3))

# Print best result
i = error_mat.index(min(error_mat))
print(f"Best epoch: {i}, Error: {error_mat[i]}, Weights: {weights_mat[i]}, Bias: {bias_mat[i]}")
