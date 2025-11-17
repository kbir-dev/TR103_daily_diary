# Day 16: Introduction to Statistics

## Today's Topics
- Fundamentals of statistics
- Measures of central tendency
- Measures of dispersion
- Normal distribution
- Statistical visualization

## Learning Journal

### Introduction to Statistics

Today I began exploring statistics, which forms the foundation of data science and machine learning. Statistics provides the tools to summarize, analyze, and draw conclusions from data.

I started by setting up the necessary libraries:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Set the style for our plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
```

### Measures of Central Tendency

I learned about the three main measures of central tendency: mean, median, and mode:

```python
# Create a sample dataset
data = [15, 21, 24, 28, 30, 32, 35, 36, 40, 45, 45, 47, 56, 65, 88]
print("Sample data:", data)

# Calculate mean
mean = np.mean(data)
print("\nMean:", mean)

# Calculate median
median = np.median(data)
print("Median:", median)

# Calculate mode
from scipy import stats
mode = stats.mode(data)[0][0]  # mode returns a ModeResult object
print("Mode:", mode)

# Create a DataFrame for easier analysis
df = pd.DataFrame({'values': data})
print("\nBasic statistics:")
print(df.describe())
```

### Comparing Central Tendency Measures

I explored how different measures of central tendency behave with different data distributions:

```python
# Create datasets with different distributions
# Symmetric data
symmetric = [10, 12, 14, 15, 16, 18, 20]

# Skewed data (right-skewed)
right_skewed = [10, 12, 14, 15, 16, 18, 50]

# Skewed data (left-skewed)
left_skewed = [5, 20, 22, 24, 25, 26, 28]

# Data with outliers
with_outliers = [10, 12, 14, 15, 16, 18, 20, 100]

# Calculate central tendency measures for each dataset
datasets = {
    'Symmetric': symmetric,
    'Right-skewed': right_skewed,
    'Left-skewed': left_skewed,
    'With outliers': with_outliers
}

for name, dataset in datasets.items():
    mean = np.mean(dataset)
    median = np.median(dataset)
    mode_result = stats.mode(dataset)
    mode = mode_result[0][0]
    
    print(f"\n{name} data:")
    print(f"Data: {dataset}")
    print(f"Mean: {mean:.2f}")
    print(f"Median: {median:.2f}")
    print(f"Mode: {mode}")
    
    # Visualize the distribution with central tendency markers
    plt.figure(figsize=(10, 6))
    sns.histplot(dataset, kde=True, bins=10)
    plt.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.2f}')
    plt.axvline(median, color='green', linestyle='--', label=f'Median: {median:.2f}')
    plt.axvline(mode, color='blue', linestyle='--', label=f'Mode: {mode}')
    plt.title(f'Distribution of {name} Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    # plt.savefig(f'{name.lower().replace(" ", "_")}_distribution.png')
    # plt.show()
```

### Measures of Dispersion

Next, I explored measures of dispersion, which describe how spread out the data is:

```python
# Calculate dispersion measures for our sample data
range_val = max(data) - min(data)
variance = np.var(data, ddof=1)  # ddof=1 for sample variance
std_dev = np.std(data, ddof=1)   # ddof=1 for sample standard deviation
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
iqr = q3 - q1

print("\nMeasures of Dispersion:")
print(f"Range: {range_val}")
print(f"Variance: {variance:.2f}")
print(f"Standard Deviation: {std_dev:.2f}")
print(f"Interquartile Range (IQR): {iqr}")
print(f"Q1 (25th percentile): {q1}")
print(f"Q3 (75th percentile): {q3}")

# Calculate coefficient of variation (CV)
cv = (std_dev / mean) * 100
print(f"Coefficient of Variation: {cv:.2f}%")
```

### Comparing Dispersion Across Datasets

I compared dispersion measures across different datasets:

```python
# Calculate dispersion measures for each dataset
for name, dataset in datasets.items():
    range_val = max(dataset) - min(dataset)
    variance = np.var(dataset, ddof=1)
    std_dev = np.std(dataset, ddof=1)
    q1 = np.percentile(dataset, 25)
    q3 = np.percentile(dataset, 75)
    iqr = q3 - q1
    mean = np.mean(dataset)
    cv = (std_dev / mean) * 100
    
    print(f"\nDispersion measures for {name} data:")
    print(f"Range: {range_val}")
    print(f"Variance: {variance:.2f}")
    print(f"Standard Deviation: {std_dev:.2f}")
    print(f"IQR: {iqr}")
    print(f"Coefficient of Variation: {cv:.2f}%")
    
    # Create a box plot to visualize dispersion
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=dataset)
    plt.title(f'Box Plot of {name} Data')
    plt.ylabel('Value')
    # plt.savefig(f'{name.lower().replace(" ", "_")}_boxplot.png')
    # plt.show()
```

### Normal Distribution

I learned about the normal distribution, which is a fundamental concept in statistics:

```python
# Generate data from a normal distribution
np.random.seed(42)
normal_data = np.random.normal(loc=50, scale=10, size=1000)

# Calculate statistics
mean = np.mean(normal_data)
median = np.median(normal_data)
std_dev = np.std(normal_data)

print("\nNormal Distribution Statistics:")
print(f"Mean: {mean:.2f}")
print(f"Median: {median:.2f}")
print(f"Standard Deviation: {std_dev:.2f}")

# Visualize the normal distribution
plt.figure(figsize=(12, 6))

# Histogram with KDE
plt.subplot(1, 2, 1)
sns.histplot(normal_data, kde=True, bins=30)
plt.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.2f}')
plt.axvline(mean + std_dev, color='green', linestyle='--', label=f'Mean + SD: {mean + std_dev:.2f}')
plt.axvline(mean - std_dev, color='green', linestyle='--', label=f'Mean - SD: {mean - std_dev:.2f}')
plt.title('Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()

# Q-Q plot to check normality
plt.subplot(1, 2, 2)
stats.probplot(normal_data, dist="norm", plot=plt)
plt.title('Q-Q Plot')

plt.tight_layout()
# plt.savefig('normal_distribution.png')
# plt.show()
```

### Standard Normal Distribution and Z-scores

I explored the standard normal distribution and how to calculate z-scores:

```python
# Calculate z-scores for our normal data
z_scores = stats.zscore(normal_data)

print("\nZ-scores statistics:")
print(f"Mean of z-scores: {np.mean(z_scores):.4f}")
print(f"Standard deviation of z-scores: {np.std(z_scores):.4f}")

# Visualize z-scores
plt.figure(figsize=(10, 6))
sns.histplot(z_scores, kde=True, bins=30)
plt.axvline(0, color='red', linestyle='--', label='Mean (0)')
plt.axvline(1, color='green', linestyle='--', label='1 SD')
plt.axvline(-1, color='green', linestyle='--', label='-1 SD')
plt.axvline(2, color='blue', linestyle='--', label='2 SD')
plt.axvline(-2, color='blue', linestyle='--', label='-2 SD')
plt.title('Standard Normal Distribution (Z-scores)')
plt.xlabel('Z-score')
plt.ylabel('Frequency')
plt.legend()
# plt.savefig('z_scores.png')
# plt.show()

# Calculate percentages within standard deviations
within_1sd = np.mean((z_scores >= -1) & (z_scores <= 1)) * 100
within_2sd = np.mean((z_scores >= -2) & (z_scores <= 2)) * 100
within_3sd = np.mean((z_scores >= -3) & (z_scores <= 3)) * 100

print(f"\nPercentage within 1 standard deviation: {within_1sd:.2f}%")
print(f"Percentage within 2 standard deviations: {within_2sd:.2f}%")
print(f"Percentage within 3 standard deviations: {within_3sd:.2f}%")

# Theoretical values for normal distribution
print("\nTheoretical percentages for normal distribution:")
print("Within 1 standard deviation: 68.27%")
print("Within 2 standard deviations: 95.45%")
print("Within 3 standard deviations: 99.73%")
```

### Statistical Visualization

I explored various ways to visualize statistical data:

```python
# Generate a more complex dataset
np.random.seed(42)
age = np.random.normal(35, 10, 100).clip(18, 65).astype(int)
income = np.random.lognormal(10.5, 0.5, 100).astype(int)
education = np.random.normal(16, 3, 100).clip(10, 22).astype(int)
experience = (age - education - 5).clip(0, 40).astype(int)  # Approximate work experience

# Create a DataFrame
df = pd.DataFrame({
    'age': age,
    'income': income,
    'education': education,
    'experience': experience
})

print("\nComplex dataset statistics:")
print(df.describe())

# Create a figure with multiple plots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Histogram
sns.histplot(df['income'], kde=True, bins=20, ax=axes[0, 0])
axes[0, 0].set_title('Income Distribution')
axes[0, 0].set_xlabel('Income')
axes[0, 0].set_ylabel('Frequency')

# Box plot
sns.boxplot(data=df[['age', 'education', 'experience']], ax=axes[0, 1])
axes[0, 1].set_title('Box Plots of Age, Education, and Experience')
axes[0, 1].set_ylabel('Years')

# Scatter plot with regression line
sns.regplot(x='experience', y='income', data=df, ax=axes[1, 0])
axes[1, 0].set_title('Income vs. Experience')
axes[1, 0].set_xlabel('Years of Experience')
axes[1, 0].set_ylabel('Income')

# Correlation heatmap
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=axes[1, 1])
axes[1, 1].set_title('Correlation Matrix')

plt.tight_layout()
# plt.savefig('statistical_visualizations.png')
# plt.show()
```

### Practical Exercise: Analyzing Student Performance Data

I applied what I learned to analyze a student performance dataset:

```python
# Create a student performance dataset
np.random.seed(42)

# Generate data for 200 students
n_students = 200

# Student characteristics
gender = np.random.choice(['Male', 'Female'], n_students)
grade_level = np.random.choice([9, 10, 11, 12], n_students)
hours_studied = np.random.normal(10, 5, n_students).clip(1, 25).round(1)
sleep_hours = np.random.normal(7, 1.5, n_students).clip(4, 10).round(1)
absences = np.random.poisson(3, n_students).clip(0, 15)

# Create base scores with some correlation to study hours and sleep
math_base = 50 + 2 * hours_studied + sleep_hours + np.random.normal(0, 10, n_students)
science_base = 45 + 2.5 * hours_studied + 0.5 * sleep_hours + np.random.normal(0, 12, n_students)
english_base = 55 + 1.5 * hours_studied + 1.5 * sleep_hours + np.random.normal(0, 11, n_students)

# Adjust scores based on absences (negative effect)
math_scores = (math_base - absences * 1.5).clip(0, 100).round()
science_scores = (science_base - absences * 1.2).clip(0, 100).round()
english_scores = (english_base - absences * 1.0).clip(0, 100).round()

# Calculate overall score
overall_scores = ((math_scores + science_scores + english_scores) / 3).round()

# Create the DataFrame
students_df = pd.DataFrame({
    'student_id': range(1, n_students + 1),
    'gender': gender,
    'grade_level': grade_level,
    'hours_studied': hours_studied,
    'sleep_hours': sleep_hours,
    'absences': absences,
    'math_score': math_scores,
    'science_score': science_scores,
    'english_score': english_scores,
    'overall_score': overall_scores
})

print("\nStudent performance dataset:")
print(students_df.head())

# Basic statistics
print("\nBasic statistics for student performance:")
print(students_df.describe())

# Analysis 1: Central tendency and dispersion of scores
score_columns = ['math_score', 'science_score', 'english_score', 'overall_score']
score_stats = students_df[score_columns].agg(['mean', 'median', 'std', 'min', 'max'])
print("\nScore statistics:")
print(score_stats)

# Analysis 2: Score distributions
plt.figure(figsize=(14, 8))

for i, subject in enumerate(score_columns):
    plt.subplot(2, 2, i+1)
    sns.histplot(students_df[subject], kde=True, bins=15)
    plt.axvline(students_df[subject].mean(), color='red', linestyle='--', 
                label=f'Mean: {students_df[subject].mean():.1f}')
    plt.axvline(students_df[subject].median(), color='green', linestyle='--', 
                label=f'Median: {students_df[subject].median():.1f}')
    plt.title(f'Distribution of {subject.replace("_", " ").title()}')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend()

plt.tight_layout()
# plt.savefig('score_distributions.png')
# plt.show()

# Analysis 3: Correlation between variables
correlation_matrix = students_df.drop('student_id', axis=1).corr()
print("\nCorrelation matrix:")
print(correlation_matrix.round(2))

# Visualize correlations
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Student Variables')
# plt.savefig('correlation_heatmap.png')
# plt.show()

# Analysis 4: Gender differences in performance
gender_stats = students_df.groupby('gender')[score_columns].agg(['mean', 'std'])
print("\nPerformance statistics by gender:")
print(gender_stats)

# Visualize gender differences
plt.figure(figsize=(12, 6))
for i, subject in enumerate(score_columns):
    plt.subplot(1, 4, i+1)
    sns.boxplot(x='gender', y=subject, data=students_df)
    plt.title(f'{subject.replace("_", " ").title()} by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Score')

plt.tight_layout()
# plt.savefig('gender_performance.png')
# plt.show()

# Analysis 5: Effect of study hours on performance
plt.figure(figsize=(12, 8))
for i, subject in enumerate(score_columns[:3]):  # Exclude overall_score
    plt.subplot(1, 3, i+1)
    sns.regplot(x='hours_studied', y=subject, data=students_df, scatter_kws={'alpha':0.5})
    plt.title(f'Study Hours vs. {subject.replace("_", " ").title()}')
    plt.xlabel('Hours Studied')
    plt.ylabel('Score')

plt.tight_layout()
# plt.savefig('study_hours_effect.png')
# plt.show()

# Analysis 6: Z-scores for identifying outliers
for subject in score_columns:
    students_df[f'{subject}_zscore'] = stats.zscore(students_df[subject])

# Find students with extreme scores (|z| > 2)
outliers = students_df[
    (np.abs(students_df['overall_score_zscore']) > 2) |
    (np.abs(students_df['math_score_zscore']) > 2) |
    (np.abs(students_df['science_score_zscore']) > 2) |
    (np.abs(students_df['english_score_zscore']) > 2)
]

print(f"\nNumber of students with extreme scores: {len(outliers)}")
print("Sample of students with extreme scores:")
print(outliers[['student_id', 'math_score', 'science_score', 'english_score', 'overall_score']].head())

# Analysis 7: Normal probability plot for overall scores
plt.figure(figsize=(10, 6))
stats.probplot(students_df['overall_score'], dist="norm", plot=plt)
plt.title('Q-Q Plot of Overall Scores')
# plt.savefig('qq_plot.png')
# plt.show()

# Analysis 8: Calculate percentiles and interpret scores
percentiles = [10, 25, 50, 75, 90]
score_percentiles = students_df[score_columns].quantile(q=[p/100 for p in percentiles])
print("\nScore percentiles:")
print(score_percentiles)

# Function to assign letter grades based on percentiles
def assign_letter_grade(score, subject):
    thresholds = score_percentiles[subject].values
    if score < thresholds[0]:  # Below 10th percentile
        return 'F'
    elif score < thresholds[1]:  # 10th to 25th percentile
        return 'D'
    elif score < thresholds[2]:  # 25th to 50th percentile
        return 'C'
    elif score < thresholds[3]:  # 50th to 75th percentile
        return 'B'
    elif score < thresholds[4]:  # 75th to 90th percentile
        return 'A'
    else:  # Above 90th percentile
        return 'A+'

# Assign letter grades
for subject in score_columns:
    students_df[f'{subject}_grade'] = students_df[subject].apply(
        lambda x: assign_letter_grade(x, subject)
    )

# Count letter grades
grade_counts = {
    subject: students_df[f'{subject}_grade'].value_counts().sort_index()
    for subject in score_columns
}

print("\nLetter grade distribution:")
for subject, counts in grade_counts.items():
    print(f"\n{subject.replace('_', ' ').title()}:")
    print(counts)

# Visualize grade distribution
plt.figure(figsize=(14, 8))
for i, subject in enumerate(score_columns):
    plt.subplot(2, 2, i+1)
    sns.countplot(x=f'{subject}_grade', data=students_df, order=['F', 'D', 'C', 'B', 'A', 'A+'])
    plt.title(f'Grade Distribution - {subject.replace("_", " ").title()}')
    plt.xlabel('Letter Grade')
    plt.ylabel('Count')

plt.tight_layout()
# plt.savefig('grade_distribution.png')
# plt.show()
```

## Reflections

Today's exploration of statistics has provided me with a solid foundation for understanding and analyzing data. Statistics is truly the language of data science, offering tools to summarize, interpret, and draw conclusions from data.

I found the measures of central tendency (mean, median, mode) particularly interesting, especially how they behave differently with different data distributions. The mean is sensitive to outliers, making the median a more robust measure for skewed data. Understanding when to use each measure is crucial for accurate data interpretation.

The measures of dispersion (range, variance, standard deviation, IQR) provide critical context about the spread of data. Without understanding dispersion, central tendency measures can be misleading. I particularly appreciated how the coefficient of variation allows for comparing dispersion across datasets with different scales.

The normal distribution is a fundamental concept that appears frequently in natural phenomena and statistical methods. Understanding its properties, such as the 68-95-99.7 rule (the percentages of data within 1, 2, and 3 standard deviations), provides a framework for interpreting data and identifying outliers.

Statistical visualization techniques like histograms, box plots, scatter plots, and Q-Q plots are powerful tools for exploring data distributions and relationships. These visualizations make abstract statistical concepts more concrete and intuitive.

The student performance exercise demonstrated how these statistical concepts can be applied to real-world data analysis. By calculating central tendency measures, dispersion, correlations, and z-scores, I was able to gain insights into factors affecting student performance and identify patterns that might not be apparent from the raw data.

## Questions to Explore
- How do statistical measures behave with very large datasets or datasets with complex structures?
- What are the best practices for handling outliers in different types of analyses?
- How can I use statistical tests to validate assumptions about data distributions?
- What are the limitations of classical statistical methods, and when should I consider non-parametric alternatives?

## Resources
- [Khan Academy - Statistics and Probability](https://www.khanacademy.org/math/statistics-probability)
- [SciPy Documentation - Statistical Functions](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Seaborn Documentation - Statistical Data Visualization](https://seaborn.pydata.org/tutorial/statistical_visualization.html)
- [Statistics for Data Science - O'Reilly](https://www.oreilly.com/library/view/statistics-for-data/9781788290678/)
- [Think Stats: Exploratory Data Analysis in Python](https://greenteapress.com/wp/think-stats-2e/)
