# Day 8: Frequency Distribution and Descriptive Statistics

## Today's Topics
- Frequency distribution analysis
- Descriptive statistics in Python
- Data visualization for distributions
- Statistical measures and their interpretation

## Learning Journal

### Frequency Distribution Analysis

Today I focused on frequency distribution analysis, which is a fundamental technique for understanding the distribution of data. A frequency distribution shows how often each value (or range of values) occurs in a dataset.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for our plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

# Create a sample dataset
np.random.seed(42)
ages = np.random.normal(35, 10, 200).astype(int)  # 200 ages centered around 35
ages = np.clip(ages, 18, 70)  # Clip to reasonable age range

# Create a DataFrame
df = pd.DataFrame({'Age': ages})
print("Sample of age data:")
print(df.head())
```

#### Calculating Frequency Distributions

I learned several ways to calculate frequency distributions:

```python
# Method 1: Using value_counts()
freq_counts = df['Age'].value_counts()
print("\nFrequency counts using value_counts():")
print(freq_counts.sort_index().head())

# Method 2: Using groupby
freq_groupby = df.groupby('Age').size()
print("\nFrequency counts using groupby():")
print(freq_groupby.sort_index().head())

# Method 3: Using pd.cut() for binning
age_bins = [15, 25, 35, 45, 55, 65, 75]
age_labels = ['16-25', '26-35', '36-45', '46-55', '56-65', '66-75']
df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)

age_group_freq = df['Age_Group'].value_counts()
print("\nFrequency counts by age group:")
print(age_group_freq.sort_index())

# Calculate relative frequencies (percentages)
rel_freq = df['Age_Group'].value_counts(normalize=True) * 100
print("\nRelative frequency (%):")
print(rel_freq.sort_index())
```

#### Visualizing Frequency Distributions

I explored different ways to visualize frequency distributions:

```python
# Create a figure with multiple plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Histogram
axes[0, 0].hist(df['Age'], bins=15, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Histogram of Ages')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')

# 2. Bar plot of value counts
freq_counts.sort_index().plot(kind='bar', ax=axes[0, 1], color='skyblue', edgecolor='black')
axes[0, 1].set_title('Bar Plot of Age Frequencies')
axes[0, 1].set_xlabel('Age')
axes[0, 1].set_ylabel('Frequency')

# 3. Bar plot of age groups
age_group_freq.sort_index().plot(kind='bar', ax=axes[1, 0], color='lightgreen', edgecolor='black')
axes[1, 0].set_title('Frequency by Age Group')
axes[1, 0].set_xlabel('Age Group')
axes[1, 0].set_ylabel('Frequency')

# 4. KDE plot (Kernel Density Estimate)
sns.kdeplot(df['Age'], ax=axes[1, 1], fill=True, color='salmon')
axes[1, 1].set_title('Kernel Density Estimate of Ages')
axes[1, 1].set_xlabel('Age')
axes[1, 1].set_ylabel('Density')

plt.tight_layout()
# plt.savefig('frequency_distributions.png')
# plt.show()
```

### Descriptive Statistics

Next, I explored descriptive statistics, which provide summary measures of the central tendency, dispersion, and shape of a dataset:

```python
# Create a more comprehensive dataset
np.random.seed(42)
data = {
    'Age': np.random.normal(35, 10, 100).astype(int),
    'Income': np.random.normal(60000, 15000, 100),
    'Years_Experience': np.random.normal(10, 5, 100),
    'Satisfaction': np.random.uniform(1, 10, 100),
    'Department': np.random.choice(['HR', 'IT', 'Finance', 'Marketing', 'Operations'], 100)
}

# Clip values to reasonable ranges
data['Age'] = np.clip(data['Age'], 22, 65)
data['Years_Experience'] = np.clip(data['Years_Experience'], 0, 40)
data['Satisfaction'] = np.round(data['Satisfaction'], 1)

# Create DataFrame
employee_df = pd.DataFrame(data)
print("\nEmployee dataset sample:")
print(employee_df.head())
```

#### Basic Descriptive Statistics

```python
# Basic descriptive statistics
desc_stats = employee_df.describe()
print("\nBasic descriptive statistics:")
print(desc_stats)

# Include categorical variables
desc_stats_all = employee_df.describe(include='all')
print("\nDescriptive statistics including categorical variables:")
print(desc_stats_all)

# Calculate specific statistics
print("\nSpecific statistics for Age:")
print(f"Mean: {employee_df['Age'].mean():.2f}")
print(f"Median: {employee_df['Age'].median():.2f}")
print(f"Mode: {employee_df['Age'].mode()[0]}")
print(f"Standard Deviation: {employee_df['Age'].std():.2f}")
print(f"Variance: {employee_df['Age'].var():.2f}")
print(f"Minimum: {employee_df['Age'].min()}")
print(f"Maximum: {employee_df['Age'].max()}")
print(f"Range: {employee_df['Age'].max() - employee_df['Age'].min()}")
print(f"25th Percentile: {employee_df['Age'].quantile(0.25)}")
print(f"75th Percentile: {employee_df['Age'].quantile(0.75)}")
print(f"Interquartile Range (IQR): {employee_df['Age'].quantile(0.75) - employee_df['Age'].quantile(0.25)}")
```

#### Measures of Central Tendency

I explored the three main measures of central tendency:

```python
# Create a skewed dataset to demonstrate differences in central tendency measures
skewed_data = np.concatenate([np.random.normal(50, 10, 95), np.random.normal(150, 20, 5)])
skewed_df = pd.DataFrame({'Values': skewed_data})

# Calculate central tendency measures
mean_val = skewed_df['Values'].mean()
median_val = skewed_df['Values'].median()
mode_val = skewed_df['Values'].mode()[0]

print("\nCentral tendency measures for skewed data:")
print(f"Mean: {mean_val:.2f}")
print(f"Median: {median_val:.2f}")
print(f"Mode: {mode_val:.2f}")

# Visualize the skewed distribution with central tendency markers
plt.figure(figsize=(10, 6))
sns.histplot(skewed_df['Values'], bins=20, kde=True)
plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
plt.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
plt.axvline(mode_val, color='blue', linestyle='--', label=f'Mode: {mode_val:.2f}')
plt.title('Skewed Distribution with Central Tendency Measures')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.legend()
# plt.savefig('central_tendency.png')
# plt.show()
```

#### Measures of Dispersion

I studied measures of dispersion, which describe how spread out the data is:

```python
# Calculate dispersion measures for different variables
dispersion = pd.DataFrame({
    'Standard Deviation': employee_df.std(),
    'Variance': employee_df.var(),
    'Range': employee_df.max() - employee_df.min(),
    'IQR': employee_df.quantile(0.75) - employee_df.quantile(0.25)
})

print("\nMeasures of dispersion:")
print(dispersion)

# Coefficient of Variation (CV) - standardized measure of dispersion
cv = employee_df.std() / employee_df.mean() * 100
print("\nCoefficient of Variation (%):")
print(cv)
```

#### Measures of Shape

I learned about measures that describe the shape of a distribution:

```python
# Calculate skewness and kurtosis
from scipy import stats

shape_measures = pd.DataFrame({
    'Skewness': employee_df.skew(),
    'Kurtosis': employee_df.kurtosis()
})

print("\nMeasures of shape:")
print(shape_measures)

# Interpret skewness
def interpret_skewness(skew_value):
    if skew_value < -1:
        return "Highly negatively skewed"
    elif -1 <= skew_value < -0.5:
        return "Moderately negatively skewed"
    elif -0.5 <= skew_value < 0:
        return "Approximately symmetric with slight negative skew"
    elif 0 <= skew_value < 0.5:
        return "Approximately symmetric with slight positive skew"
    elif 0.5 <= skew_value < 1:
        return "Moderately positively skewed"
    else:
        return "Highly positively skewed"

# Interpret kurtosis
def interpret_kurtosis(kurt_value):
    if kurt_value < -1:
        return "Very platykurtic (flat distribution)"
    elif -1 <= kurt_value < 0:
        return "Platykurtic (flatter than normal)"
    elif 0 <= kurt_value < 1:
        return "Mesokurtic (similar to normal)"
    else:
        return "Leptokurtic (more peaked than normal)"

# Print interpretations
for col in employee_df.select_dtypes(include=[np.number]).columns:
    skew_val = employee_df[col].skew()
    kurt_val = employee_df[col].kurtosis()
    print(f"\n{col}:")
    print(f"  Skewness: {skew_val:.2f} - {interpret_skewness(skew_val)}")
    print(f"  Kurtosis: {kurt_val:.2f} - {interpret_kurtosis(kurt_val)}")
```

### Visualizing Descriptive Statistics

I explored various ways to visualize descriptive statistics:

```python
# Create a figure with multiple plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Box plot
sns.boxplot(data=employee_df[['Age', 'Years_Experience']], ax=axes[0, 0])
axes[0, 0].set_title('Box Plot of Age and Years of Experience')
axes[0, 0].set_ylabel('Years')

# 2. Violin plot
sns.violinplot(x='Department', y='Income', data=employee_df, ax=axes[0, 1])
axes[0, 1].set_title('Violin Plot of Income by Department')
axes[0, 1].set_ylabel('Income')
axes[0, 1].tick_params(axis='x', rotation=45)

# 3. Scatter plot with regression line
sns.regplot(x='Years_Experience', y='Income', data=employee_df, ax=axes[1, 0])
axes[1, 0].set_title('Relationship between Experience and Income')
axes[1, 0].set_xlabel('Years of Experience')
axes[1, 0].set_ylabel('Income')

# 4. Heatmap of correlation matrix
corr_matrix = employee_df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=axes[1, 1])
axes[1, 1].set_title('Correlation Matrix')

plt.tight_layout()
# plt.savefig('descriptive_statistics_visualizations.png')
# plt.show()
```

### Practical Exercise: Sales Data Analysis

I applied what I learned to analyze a sales dataset:

```python
# Create a sales dataset
np.random.seed(42)
months = pd.date_range(start='2024-01-01', periods=12, freq='M')
products = ['Product A', 'Product B', 'Product C', 'Product D']

sales_data = []
for month in months:
    for product in products:
        # Generate sales with seasonal patterns
        base_sales = np.random.normal(1000, 200)
        seasonal_factor = 1 + 0.3 * np.sin((month.month - 1) * np.pi / 6)  # Seasonal pattern
        product_factor = {'Product A': 1.2, 'Product B': 0.9, 'Product C': 1.0, 'Product D': 0.8}[product]
        
        sales = base_sales * seasonal_factor * product_factor
        units = int(sales / np.random.uniform(20, 50))
        
        sales_data.append({
            'Date': month,
            'Product': product,
            'Sales': round(sales, 2),
            'Units': units,
            'Region': np.random.choice(['North', 'South', 'East', 'West'])
        })

sales_df = pd.DataFrame(sales_data)
print("\nSales dataset sample:")
print(sales_df.head())

# 1. Overall sales statistics
print("\nOverall sales statistics:")
print(sales_df['Sales'].describe())

# 2. Sales distribution by product
product_stats = sales_df.groupby('Product')['Sales'].describe()
print("\nSales statistics by product:")
print(product_stats)

# 3. Sales distribution by region
region_stats = sales_df.groupby('Region')['Sales'].describe()
print("\nSales statistics by region:")
print(region_stats)

# 4. Monthly sales trends
monthly_sales = sales_df.groupby(sales_df['Date'].dt.strftime('%Y-%m'))['Sales'].sum()
print("\nMonthly sales trends:")
print(monthly_sales)

# 5. Visualize sales distributions
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
sns.boxplot(x='Product', y='Sales', data=sales_df)
plt.title('Sales Distribution by Product')
plt.xticks(rotation=45)

plt.subplot(2, 2, 2)
sns.boxplot(x='Region', y='Sales', data=sales_df)
plt.title('Sales Distribution by Region')

plt.subplot(2, 2, 3)
sns.histplot(sales_df['Sales'], bins=15, kde=True)
plt.title('Overall Sales Distribution')
plt.xlabel('Sales Amount')

plt.subplot(2, 2, 4)
monthly_sales.plot(kind='bar', color='skyblue')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)

plt.tight_layout()
# plt.savefig('sales_analysis.png')
# plt.show()

# 6. Calculate frequency distribution of sales by range
sales_bins = [0, 500, 750, 1000, 1250, 1500, 2000]
sales_labels = ['0-500', '501-750', '751-1000', '1001-1250', '1251-1500', '1501+']
sales_df['Sales_Range'] = pd.cut(sales_df['Sales'], bins=sales_bins, labels=sales_labels)

sales_range_freq = sales_df['Sales_Range'].value_counts().sort_index()
sales_range_pct = sales_df['Sales_Range'].value_counts(normalize=True).sort_index() * 100

print("\nSales frequency distribution by range:")
print(pd.DataFrame({
    'Frequency': sales_range_freq,
    'Percentage': sales_range_pct.round(2)
}))

# 7. Identify outliers using IQR method
Q1 = sales_df['Sales'].quantile(0.25)
Q3 = sales_df['Sales'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = sales_df[(sales_df['Sales'] < lower_bound) | (sales_df['Sales'] > upper_bound)]
print(f"\nNumber of outliers detected: {len(outliers)}")
if len(outliers) > 0:
    print("Outlier details:")
    print(outliers)
```

## Reflections

Today's exploration of frequency distributions and descriptive statistics has been incredibly valuable for my data analysis toolkit. These fundamental techniques provide the foundation for understanding and interpreting data before moving on to more advanced analyses.

I found the different measures of central tendency (mean, median, mode) particularly interesting, especially how they behave differently with skewed distributions. The mean is sensitive to outliers, while the median is more robust, which makes the median a better measure for skewed data or data with outliers.

The measures of dispersion (standard deviation, variance, range, IQR) provide important context about the spread of the data. I learned that the coefficient of variation is especially useful for comparing the variability of different variables measured on different scales.

The visualization techniques for frequency distributions and descriptive statistics were eye-opening. Box plots, histograms, and violin plots each provide unique insights into the distribution of data. The correlation heatmap is also a powerful tool for quickly identifying relationships between variables.

The sales data analysis exercise demonstrated how these techniques can be applied to real-world business scenarios. By analyzing the distribution of sales across different products, regions, and time periods, I was able to gain insights that could inform business decisions.

## Questions to Explore
- How do descriptive statistics change when dealing with very large datasets?
- What are the best practices for handling outliers in different types of analyses?
- How can I use descriptive statistics to identify potential data quality issues?
- What are the limitations of traditional descriptive statistics for non-normal distributions?

## Resources
- [Pandas Documentation - Descriptive Statistics](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html)
- [SciPy Documentation - Statistical Functions](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Seaborn Documentation - Statistical Data Visualization](https://seaborn.pydata.org/tutorial/statistical_visualization.html)
- [Towards Data Science - Understanding Descriptive Statistics](https://towardsdatascience.com/understanding-descriptive-statistics-c9c2b0641291)
