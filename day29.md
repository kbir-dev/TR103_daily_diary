# Day 29: Feature Engineering

## Topics
- Feature creation
- Transformation
- Selection
- Dimensionality reduction

## Journal

Advanced feature engineering techniques:
1. Polynomial features
2. Binning
3. Interaction terms
4. Target encoding
5. PCA

```python
from sklearn.preprocessing import PolynomialFeatures

# Create polynomial features
poly = PolynomialFeatures(degree=2, interaction_only=True)
X_poly = poly.fit_transform(X)

# Feature selection
from sklearn.feature_selection import SelectKBest, f_regression
selector = SelectKBest(f_regression, k=10)
X_selected = selector.fit_transform(X_poly, y)
```

## Reflections
Feature engineering often improves models more than algorithm selection. Domain knowledge drives effective feature creation. Dimensionality reduction balances information retention and model complexity.

## Resources
- [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)
