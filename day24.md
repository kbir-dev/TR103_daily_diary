# Day 24: Linear Regression Practice


## Topics
- Housing price prediction
- Feature engineering
- Model diagnostics
- Business interpretation

## Journal

Applied linear regression to predict house prices using the Boston Housing dataset. Focused on:
- Handling multicollinearity
- Checking linearity assumptions
- Interpreting coefficients
- Evaluating business impact

Key steps:
```python
from sklearn.datasets import fetch_openml
boston = fetch_openml(name='boston', version=1)

# Feature engineering
X = boston.data[['RM', 'LSTAT', 'PTRATIO']]
y = boston.target

# Model building
model = LinearRegression()
model.fit(X, y)

# Interpretation
print(f"Each additional room increases price by ${model.coef_[0]*1000:.0f}")
```

## Reflections
Real-world regression requires careful feature selection and assumption checking. Business context transforms statistical outputs into actionable insights.

## Resources
- [Boston Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html)
