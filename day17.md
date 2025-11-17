# Day 17: Hypothesis Testing

## Today's Topics
- Introduction to hypothesis testing
- Null and alternative hypotheses
- p-values and significance levels
- Types of statistical tests

## Learning Journal

### Introduction to Hypothesis Testing

Today I explored hypothesis testing, a fundamental concept in inferential statistics that allows us to make decisions about populations based on sample data.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set the style for our plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
```

### Null and Alternative Hypotheses

The null hypothesis (H₀) typically represents "no effect" or "no difference," while the alternative hypothesis (H₁) represents the claim we're testing.

Example: Testing if a new teaching method improves test scores
- H₀: The new teaching method has no effect on test scores (μ = μ₀)
- H₁: The new teaching method improves test scores (μ > μ₀)

### p-values and Significance Levels

The p-value is the probability of observing our sample results (or more extreme) if the null hypothesis is true. We typically reject H₀ if p < α (significance level, commonly 0.05).

```python
# Example: One-sample t-test
# Test if sample mean is significantly different from population mean

# Sample data: Student test scores with new teaching method
np.random.seed(42)
new_method_scores = np.random.normal(75, 15, 30)  # Sample mean around 75

# Null hypothesis: Population mean = 70 (traditional method mean)
population_mean = 70

# Perform one-sample t-test
t_stat, p_value = stats.ttest_1samp(new_method_scores, population_mean)

print(f"One-sample t-test results:")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Interpret results
alpha = 0.05
if p_value < alpha:
    print(f"Reject null hypothesis (p={p_value:.4f} < {alpha})")
    print("The new teaching method has a significant effect on test scores.")
else:
    print(f"Fail to reject null hypothesis (p={p_value:.4f} >= {alpha})")
    print("There is insufficient evidence that the new teaching method affects test scores.")
```

### Types of Statistical Tests

#### 1. t-tests

```python
# Two-sample t-test: Compare means of two independent groups
# Example: Compare test scores between two teaching methods

# Generate sample data
method_A = np.random.normal(70, 15, 35)  # Traditional method
method_B = np.random.normal(78, 15, 40)  # New method

# Perform two-sample t-test
t_stat, p_value = stats.ttest_ind(method_A, method_B)

print(f"\nTwo-sample t-test results:")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Visualize the comparison
plt.figure(figsize=(10, 6))
sns.boxplot(data=[method_A, method_B])
plt.xticks([0, 1], ['Traditional Method', 'New Method'])
plt.ylabel('Test Score')
plt.title('Comparison of Test Scores Between Teaching Methods')
```

#### 2. ANOVA (Analysis of Variance)

```python
# One-way ANOVA: Compare means of three or more groups
# Example: Compare test scores across three teaching methods

# Generate sample data
method_A = np.random.normal(70, 15, 30)  # Traditional method
method_B = np.random.normal(78, 15, 30)  # New method 1
method_C = np.random.normal(75, 15, 30)  # New method 2

# Perform one-way ANOVA
f_stat, p_value = stats.f_oneway(method_A, method_B, method_C)

print(f"\nOne-way ANOVA results:")
print(f"F-statistic: {f_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Create a DataFrame for visualization
df_methods = pd.DataFrame({
    'score': np.concatenate([method_A, method_B, method_C]),
    'method': ['A'] * len(method_A) + ['B'] * len(method_B) + ['C'] * len(method_C)
})

# Visualize the comparison
plt.figure(figsize=(10, 6))
sns.boxplot(x='method', y='score', data=df_methods)
plt.xlabel('Teaching Method')
plt.ylabel('Test Score')
plt.title('Comparison of Test Scores Across Three Teaching Methods')
```

#### 3. Chi-Square Test

```python
# Chi-square test of independence
# Example: Test if student performance is independent of teaching method

# Create a contingency table
contingency_table = pd.DataFrame({
    'Pass': [40, 55, 45],
    'Fail': [20, 15, 25]
}, index=['Method A', 'Method B', 'Method C'])

print("\nContingency Table:")
print(contingency_table)

# Perform chi-square test
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print(f"\nChi-square test results:")
print(f"Chi-square statistic: {chi2:.4f}")
print(f"p-value: {p:.4f}")
print(f"Degrees of freedom: {dof}")
```

## Reflections

Today's exploration of hypothesis testing has equipped me with powerful tools for making statistical inferences. Understanding the process of formulating hypotheses, selecting appropriate tests, and interpreting p-values is crucial for data-driven decision making.

I found the concept of statistical significance particularly important - it provides a framework for determining when observed differences are likely real effects versus random chance. However, I also learned that statistical significance doesn't necessarily imply practical significance.

The different types of tests (t-tests, ANOVA, chi-square) each serve specific purposes depending on the data structure and research question. Choosing the right test is as important as correctly interpreting the results.

## Resources
- [StatQuest: Hypothesis Testing](https://www.youtube.com/watch?v=0oc49DyA3hU)
- [SciPy Documentation - Statistical Functions](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Khan Academy - Hypothesis Testing](https://www.khanacademy.org/math/statistics-probability/significance-tests-one-sample)
