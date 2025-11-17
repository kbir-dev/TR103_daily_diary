# Day 18: T-test, ANOVA, and Normality Testing

## Today's Topics
- Detailed exploration of t-tests
- ANOVA and post-hoc tests
- Normality testing
- Parametric vs. non-parametric tests

## Learning Journal

### T-tests in Depth

Today I explored different types of t-tests and their applications in statistical analysis.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set the style for our plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

# Generate sample data
np.random.seed(42)
group1 = np.random.normal(loc=50, scale=10, size=30)  # Control group
group2 = np.random.normal(loc=55, scale=10, size=30)  # Treatment group
paired_before = np.random.normal(loc=50, scale=10, size=25)  # Before treatment
paired_after = paired_before + np.random.normal(loc=5, scale=3, size=25)  # After treatment
```

#### Independent Samples T-test

```python
# Independent samples t-test (two-sample t-test)
t_stat, p_value = stats.ttest_ind(group1, group2)
print(f"Independent samples t-test: t={t_stat:.4f}, p={p_value:.4f}")

# Visualize the comparison
plt.figure(figsize=(10, 6))
sns.boxplot(data=[group1, group2])
plt.xticks([0, 1], ['Control Group', 'Treatment Group'])
plt.ylabel('Score')
plt.title('Comparison of Control and Treatment Groups')
```

#### Paired Samples T-test

```python
# Paired samples t-test
t_stat, p_value = stats.ttest_rel(paired_before, paired_after)
print(f"Paired samples t-test: t={t_stat:.4f}, p={p_value:.4f}")

# Visualize paired data
plt.figure(figsize=(10, 6))
plt.plot([1, 2], [paired_before, paired_after], 'o-', alpha=0.3)
plt.boxplot([paired_before, paired_after])
plt.xticks([1, 2], ['Before', 'After'])
plt.ylabel('Score')
plt.title('Before and After Treatment (Paired Data)')
```

### ANOVA and Post-hoc Tests

ANOVA (Analysis of Variance) is used to compare means across three or more groups.

```python
# Generate data for three groups
group_a = np.random.normal(loc=50, scale=10, size=30)
group_b = np.random.normal(loc=55, scale=10, size=30)
group_c = np.random.normal(loc=60, scale=10, size=30)

# One-way ANOVA
f_stat, p_value = stats.f_oneway(group_a, group_b, group_c)
print(f"One-way ANOVA: F={f_stat:.4f}, p={p_value:.4f}")

# Prepare data for post-hoc tests
all_data = np.concatenate([group_a, group_b, group_c])
group_labels = ['A'] * len(group_a) + ['B'] * len(group_b) + ['C'] * len(group_c)

# Create a DataFrame
anova_df = pd.DataFrame({'score': all_data, 'group': group_labels})

# Post-hoc test: Tukey's HSD
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey = pairwise_tukeyhsd(anova_df['score'], anova_df['group'], alpha=0.05)
print("\nTukey's HSD post-hoc test:")
print(tukey)

# Visualize the groups
plt.figure(figsize=(10, 6))
sns.boxplot(x='group', y='score', data=anova_df)
plt.title('Comparison of Three Groups')
plt.xlabel('Group')
plt.ylabel('Score')
```

### Normality Testing

Before applying parametric tests like t-tests and ANOVA, we should check if the data follows a normal distribution.

```python
# Generate data with different distributions
normal_data = np.random.normal(loc=50, scale=10, size=100)
skewed_data = np.random.exponential(scale=10, size=100)

# Shapiro-Wilk test for normality
shapiro_normal = stats.shapiro(normal_data)
shapiro_skewed = stats.shapiro(skewed_data)

print("\nShapiro-Wilk test for normality:")
print(f"Normal data: W={shapiro_normal[0]:.4f}, p={shapiro_normal[1]:.4f}")
print(f"Skewed data: W={shapiro_skewed[0]:.4f}, p={shapiro_skewed[1]:.4f}")

# Q-Q plots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
stats.probplot(normal_data, dist="norm", plot=plt)
plt.title('Q-Q Plot: Normal Data')

plt.subplot(1, 2, 2)
stats.probplot(skewed_data, dist="norm", plot=plt)
plt.title('Q-Q Plot: Skewed Data')

plt.tight_layout()
```

### Parametric vs. Non-parametric Tests

When data doesn't meet the normality assumption, non-parametric tests are more appropriate.

```python
# Mann-Whitney U test (non-parametric alternative to independent t-test)
u_stat, p_value = stats.mannwhitneyu(group1, group2)
print(f"\nMann-Whitney U test: U={u_stat:.4f}, p={p_value:.4f}")

# Wilcoxon signed-rank test (non-parametric alternative to paired t-test)
w_stat, p_value = stats.wilcoxon(paired_before, paired_after)
print(f"Wilcoxon signed-rank test: W={w_stat:.4f}, p={p_value:.4f}")

# Kruskal-Wallis H test (non-parametric alternative to one-way ANOVA)
h_stat, p_value = stats.kruskal(group_a, group_b, group_c)
print(f"Kruskal-Wallis H test: H={h_stat:.4f}, p={p_value:.4f}")
```

## Reflections

Today's exploration of statistical tests has deepened my understanding of how to compare groups and analyze differences. The t-test is versatile for comparing two groups, while ANOVA extends this to multiple groups. Post-hoc tests like Tukey's HSD help identify which specific groups differ when ANOVA indicates significant differences.

I've learned the importance of checking assumptions, particularly normality, before applying parametric tests. The Shapiro-Wilk test and Q-Q plots provide formal and visual methods to assess normality. When assumptions are violated, non-parametric alternatives like the Mann-Whitney U test, Wilcoxon signed-rank test, and Kruskal-Wallis test offer robust alternatives.

Understanding when to use each test and how to interpret the results is crucial for drawing valid conclusions from data. This knowledge forms a solid foundation for more advanced statistical analyses and machine learning techniques.

## Resources
- [SciPy Documentation - Statistical Functions](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Statsmodels Documentation - Multiple Comparison Procedures](https://www.statsmodels.org/stable/generated/statsmodels.stats.multicomp.pairwise_tukeyhsd.html)
- [An Introduction to Statistical Learning](https://www.statlearning.com/)
