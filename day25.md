# Day 25: Logistic Regression Practice


## Topics
- Customer churn prediction
- Handling imbalanced data
- Feature importance
- Business impact

## Journal

Built a churn prediction model using logistic regression. Addressed class imbalance with SMOTE. Focused on precision-recall tradeoff for business optimization.

```python
from imblearn.over_sampling import SMOTE

# Handle imbalance
smote = SMOTE()
X_res, y_res = smote.fit_resample(X_train, y_train)

# Train model
model = LogisticRegression()
model.fit(X_res, y_res)

# Feature importance
importance = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_[0]})
print(importance.sort_values('Coefficient', ascending=False))
```

## Reflections
For churn, recall is often prioritized to capture at-risk customers. Cost-benefit analysis determines optimal decision threshold. Feature importance guides intervention strategies.

## Resources
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/stable/)
