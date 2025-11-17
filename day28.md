# Day 28: Model Validation Techniques
## Topics
- Cross-validation
- Hyperparameter tuning
- Bias-variance tradeoff
- Learning curves

## Journal

Practiced robust validation techniques:
1. K-Fold CV
2. Stratified sampling
3. Time series splits
4. GridSearch for tuning

```python
from sklearn.model_selection import GridSearchCV

# Parameter grid
param_grid = {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}

# Grid search
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X, y)

print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.4f}")
```

## Reflections
Proper validation prevents overfitting. Time series requires special handling to avoid data leakage. Learning curves diagnose under/overfitting.

## Resources
- [Scikit-learn Model Selection](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection)
