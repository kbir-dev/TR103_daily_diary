# Day 20: Logistic Regression

## Topics
- Binary classification
- Logistic function
- Model interpretation
- Evaluation metrics

## Journal

Learned logistic regression for classification problems. Unlike linear regression, it predicts probabilities using the sigmoid function. Key metrics: accuracy, precision, recall, F1-score, ROC-AUC.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Sample data
X, y = make_classification(n_samples=1000, n_features=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
print(classification_report(y_test, model.predict(X_test)))
print(f"ROC-AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:,1]):.4f}")
```

## Reflections
Logistic regression is foundational for classification tasks. The coefficients represent log-odds ratios, providing interpretability. ROC curve helps visualize trade-off between sensitivity and specificity.

## Resources
- [Scikit-learn Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
