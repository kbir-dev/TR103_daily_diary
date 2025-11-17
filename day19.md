# Day 19: Introduction to Linear Regression

## Today's Topics
- Concepts of linear regression
- Simple linear regression
- Multiple linear regression
- Model evaluation metrics

## Learning Journal

### Introduction to Linear Regression

Linear regression is a fundamental predictive modeling technique that models the relationship between a dependent variable and one or more independent variables.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
```

### Simple Linear Regression

Simple linear regression models the relationship between two variables using a straight line: y = mx + b

```python
# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Independent variable
y = 2.5 * X + 5 + np.random.randn(100, 1) * 2  # Dependent variable with noise

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Model parameters
print(f"Slope (coefficient): {model.coef_[0][0]:.4f}")
print(f"Intercept: {model.intercept_[0]:.4f}")
print(f"R² Score: {r2_score(y, y_pred):.4f}")

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5, label='Actual data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Simple Linear Regression')
plt.legend()
```

### Multiple Linear Regression

Multiple linear regression extends simple regression to multiple independent variables.

```python
# Generate data with multiple features
np.random.seed(42)
n_samples = 200
X1 = np.random.rand(n_samples) * 10  # Feature 1
X2 = np.random.rand(n_samples) * 5   # Feature 2
X3 = np.random.rand(n_samples) * 8   # Feature 3

# Target variable
y = 3*X1 + 2*X2 - 1.5*X3 + 10 + np.random.randn(n_samples) * 3

# Create DataFrame
df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'y': y})

# Split data
X = df[['X1', 'X2', 'X3']]
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
print(f"\nMultiple Linear Regression Results:")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
```

### Model Evaluation

```python
# Visualize predictions vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')

# Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
```

## Reflections

Linear regression is a powerful yet simple technique for predictive modeling. Understanding the relationship between variables through regression coefficients provides valuable insights. The R² score, RMSE, and MAE metrics help evaluate model performance. Residual plots are crucial for checking model assumptions.

## Resources
- [Scikit-learn Linear Regression](https://scikit-learn.org/stable/modules/linear_model.html)
- [An Introduction to Statistical Learning](https://www.statlearning.com/)
