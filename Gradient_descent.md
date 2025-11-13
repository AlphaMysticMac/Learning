# 🚀 Understanding Gradient Descent from Scratch (with Python Example)

Gradient Descent is one of the most important algorithms in machine learning — it powers everything from **Linear Regression** to **Neural Networks**.  
In this post, we’ll build **Gradient Descent from scratch in Python**, understand how it works mathematically, and visualize how the error reduces over time.

---

## 💡 What is Gradient Descent?

Gradient Descent is an **optimization algorithm** that minimizes a *cost function* by iteratively updating parameters like **weights (w)** and **bias (b)**.

Imagine you’re standing on a hill, blindfolded, trying to reach the lowest point.  
You take small steps in the direction where the ground slopes downward — that’s essentially what Gradient Descent does!

In our case:
- We’re minimizing the **Mean Squared Error (MSE)** between predicted and actual values.
- The parameters being optimized are **w (weights)** and **b (bias)**.

---

## 🧮 The Math Behind It

Let’s take **Linear Regression** as our example model.

### 1. Hypothesis (Model)
We predict the output \( \hat{y} \) using:

$$
\hat{y} = wX + b
$$

---

### 2. Cost Function (Mean Squared Error)
Our goal is to minimize the error between predicted and actual values:

$$
J(w, b) = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

where:  
- \( N \) = number of samples  
- \( y_i \) = actual (true) value  
- \( \hat{y}_i \) = predicted value  

---

### 3. Gradient Calculation
To minimize \( J(w, b) \), we compute partial derivatives (gradients):

$$
\frac{\partial J}{\partial w} = -\frac{2}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i) X_i
$$

$$
\frac{\partial J}{\partial b} = -\frac{2}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)
$$

These gradients tell us the *direction of steepest ascent* —  
so we move in the **opposite direction** to minimize the cost.

---

### 4. Parameter Update Rule
We update the weights and bias iteratively using the gradient descent update rule:

$$
w := w - \alpha \frac{\partial J}{\partial w}
$$

$$
b := b - \alpha \frac{\partial J}{\partial b}
$$

where \( alpha \) is the **learning rate**, which controls how large a step we take during each update.


## 🧑‍💻 Implementing Gradient Descent in Python

Here’s the full implementation using NumPy.

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

# Track progress
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

# Best result
i = error_mat.index(min(error_mat))
print(f"Best epoch: {i}, Error: {error_mat[i]}, Weights: {weights_mat[i]}, Bias: {bias_mat[i]}")

```


## 🧩 Step-by-Step Breakdown

### 1. Generate Data  
Use `make_regression()` to create simple linear data with some noise.

---

### 2. Initialize Parameters  
Start with arbitrary values for **weights** and **bias**.

---

### 3. Predict  
Calculate predictions using the hypothesis equation:  
\( \hat{y} = wX + b \)

---

### 4. Compute Error  
Find how far predictions are from actual values using the **Mean Squared Error (MSE)**.

---

### 5. Calculate Gradients  
Determine how much to change \( w \) and \( b \) based on the computed gradients.

---

### 6. Update Parameters  
Adjust \( w \) and \( b \) using the **learning rate** \( \alpha \):  
\( w := w - \alpha \frac{\partial J}{\partial w} \),  
\( b := b - \alpha \frac{\partial J}{\partial b} \)

---

### 7. Repeat  
Continue updating until the **error stabilizes** or reaches a **minimum**.

