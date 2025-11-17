# Day 12: Grouping Numeric Data in Python

## Today's Topics
- Techniques for grouping numeric data
- Binning and discretization
- Quantile-based grouping
- Custom grouping strategies

## Learning Journal

### Introduction to Grouping Numeric Data

Today I focused on techniques for grouping numeric data in Python. Grouping continuous numeric data into discrete categories or bins is a common preprocessing step in data analysis and visualization. It helps in simplifying complex data and identifying patterns that might not be apparent in the raw data.

I started by setting up the necessary libraries and creating a sample dataset:

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
data = {
    'age': np.random.normal(40, 10, 200).clip(18, 80).astype(int),
    'income': np.random.lognormal(10.5, 0.5, 200).astype(int),
    'education_years': np.random.randint(8, 22, 200),
    'satisfaction_score': np.random.uniform(1, 10, 200).round(1),
    'purchase_amount': np.random.exponential(500, 200).round(2)
}

df = pd.DataFrame(data)

print("Sample data:")
print(df.head())

# Basic statistics
print("\nBasic statistics:")
print(df.describe())
```

### Equal-Width Binning

I learned how to create bins of equal width using the `pd.cut()` function:

```python
# Equal-width binning for age
age_bins = [18, 30, 40, 50, 60, 80]
df['age_group'] = pd.cut(df['age'], bins=age_bins)

print("\nAge groups using equal-width binning:")
print(df['age_group'].value_counts().sort_index())

# Visualize the distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='age_group', data=df)
plt.title('Distribution of Age Groups')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.xticks(rotation=45)
# plt.savefig('age_groups_equal_width.png')
# plt.show()

# Equal-width binning with custom labels
age_labels = ['Young Adult', 'Early 30s', 'Early 40s', 'Early 50s', 'Senior']
df['age_category'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)

print("\nAge categories with custom labels:")
print(df['age_category'].value_counts().sort_index())
```

### Equal-Frequency Binning (Quantile-Based)

Next, I explored quantile-based binning using the `pd.qcut()` function:

```python
# Equal-frequency binning (quartiles) for income
df['income_quartile'] = pd.qcut(df['income'], q=4)

print("\nIncome quartiles:")
print(df['income_quartile'].value_counts().sort_index())

# Equal-frequency binning with custom labels
income_labels = ['Low', 'Medium-Low', 'Medium-High', 'High']
df['income_level'] = pd.qcut(df['income'], q=4, labels=income_labels)

print("\nIncome levels with custom labels:")
print(df['income_level'].value_counts().sort_index())

# Visualize the distribution
plt.figure(figsize=(10, 6))
sns.boxplot(x='income_level', y='income', data=df)
plt.title('Income Distribution by Level')
plt.xlabel('Income Level')
plt.ylabel('Income')
# plt.savefig('income_quartiles.png')
# plt.show()
```

### Custom Binning Logic

I learned how to implement custom binning logic for more specific requirements:

```python
# Custom binning for satisfaction score
def categorize_satisfaction(score):
    if score < 3:
        return 'Dissatisfied'
    elif score < 7:
        return 'Neutral'
    else:
        return 'Satisfied'

df['satisfaction_category'] = df['satisfaction_score'].apply(categorize_satisfaction)

print("\nSatisfaction categories using custom logic:")
print(df['satisfaction_category'].value_counts().sort_index())

# Custom binning using numpy's digitize
purchase_bins = [0, 100, 500, 1000, float('inf')]
purchase_labels = ['Small', 'Medium', 'Large', 'Very Large']

df['purchase_category'] = pd.cut(df['purchase_amount'], bins=purchase_bins, labels=purchase_labels)

print("\nPurchase categories:")
print(df['purchase_category'].value_counts().sort_index())
```

### Binning with Statistical Measures

I explored binning based on statistical measures like mean and standard deviation:

```python
# Binning based on mean and standard deviation
mean = df['education_years'].mean()
std = df['education_years'].std()

def education_level(years):
    if years < mean - std:
        return 'Below Average'
    elif years > mean + std:
        return 'Above Average'
    else:
        return 'Average'

df['education_level'] = df['education_years'].apply(education_level)

print("\nEducation levels based on mean and standard deviation:")
print(df['education_level'].value_counts().sort_index())

# Visualize the distribution
plt.figure(figsize=(10, 6))
sns.boxplot(x='education_level', y='education_years', data=df)
plt.title('Education Years by Level')
plt.xlabel('Education Level')
plt.ylabel('Years of Education')
# plt.savefig('education_levels.png')
# plt.show()
```

### Binning for Visualization

I learned how to use binning to create more effective visualizations:

```python
# Create a scatter plot with color-coded points based on binning
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    df['age'], 
    df['income'], 
    c=df['education_level'].astype('category').cat.codes, 
    alpha=0.6, 
    cmap='viridis'
)

plt.title('Income vs. Age, Color-Coded by Education Level')
plt.xlabel('Age')
plt.ylabel('Income')
plt.colorbar(scatter, label='Education Level', ticks=[0, 1, 2])
plt.clim(-0.5, 2.5)
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${int(x):,}"))
# plt.savefig('income_age_education.png')
# plt.show()

# Create a heatmap of average purchase amount by age and income groups
heatmap_data = df.pivot_table(
    values='purchase_amount',
    index='age_category',
    columns='income_level',
    aggfunc='mean'
)

plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlGnBu')
plt.title('Average Purchase Amount by Age and Income Level')
plt.tight_layout()
# plt.savefig('purchase_heatmap.png')
# plt.show()
```

### Analyzing Grouped Data

I explored techniques for analyzing data after grouping:

```python
# Group statistics by age category
age_stats = df.groupby('age_category').agg({
    'income': ['mean', 'median', 'std'],
    'education_years': ['mean', 'median'],
    'satisfaction_score': ['mean', 'median'],
    'purchase_amount': ['mean', 'count']
})

print("\nStatistics by age category:")
print(age_stats)

# Group statistics by income level
income_stats = df.groupby('income_level').agg({
    'age': ['mean', 'median'],
    'education_years': ['mean', 'median'],
    'satisfaction_score': ['mean', 'median'],
    'purchase_amount': ['mean', 'sum']
})

print("\nStatistics by income level:")
print(income_stats)

# Cross-tabulation of education level and satisfaction category
education_satisfaction = pd.crosstab(
    df['education_level'], 
    df['satisfaction_category'],
    normalize='index'  # Convert to proportions
) * 100  # Convert to percentages

print("\nPercentage of satisfaction categories by education level:")
print(education_satisfaction.round(1))
```

### Practical Exercise: Customer Segmentation

I applied what I learned to a customer segmentation scenario:

```python
# Create a more comprehensive customer dataset
np.random.seed(42)

# Generate 1000 customer records
n_customers = 1000

# Age: normal distribution centered at 35
age = np.random.normal(35, 12, n_customers).clip(18, 80).astype(int)

# Income: lognormal distribution
income = np.random.lognormal(10.5, 0.6, n_customers).astype(int)

# Years as customer: exponential distribution
years_as_customer = np.random.exponential(5, n_customers).clip(0.1, 20).round(1)

# Number of purchases: poisson distribution
num_purchases = np.random.poisson(8, n_customers)

# Average purchase value: gamma distribution
avg_purchase = np.random.gamma(shape=5, scale=50, size=n_customers).round(2)

# Customer support calls: poisson with lower lambda
support_calls = np.random.poisson(2, n_customers)

# Create the DataFrame
customers = pd.DataFrame({
    'customer_id': range(1001, 1001 + n_customers),
    'age': age,
    'income': income,
    'years_as_customer': years_as_customer,
    'num_purchases': num_purchases,
    'avg_purchase': avg_purchase,
    'support_calls': support_calls
})

# Calculate total spend
customers['total_spend'] = customers['num_purchases'] * customers['avg_purchase']

print("\nCustomer dataset sample:")
print(customers.head())
print("\nCustomer dataset statistics:")
print(customers.describe())

# Step 1: Create age segments
age_bins = [18, 25, 35, 50, 65, 80]
age_labels = ['18-24', '25-34', '35-49', '50-64', '65+']
customers['age_segment'] = pd.cut(customers['age'], bins=age_bins, labels=age_labels)

# Step 2: Create income segments using quantiles
income_labels = ['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High']
customers['income_segment'] = pd.qcut(customers['income'], q=5, labels=income_labels)

# Step 3: Create customer tenure segments
tenure_bins = [0, 1, 3, 5, 10, 20]
tenure_labels = ['New', '1-3 years', '3-5 years', '5-10 years', '10+ years']
customers['tenure_segment'] = pd.cut(customers['years_as_customer'], bins=tenure_bins, labels=tenure_labels)

# Step 4: Create purchase frequency segments
purchase_bins = [0, 3, 8, 15, float('inf')]
purchase_labels = ['Low', 'Medium', 'High', 'Very High']
customers['purchase_frequency'] = pd.cut(customers['num_purchases'], bins=purchase_bins, labels=purchase_labels)

# Step 5: Create average purchase value segments
avg_purchase_labels = ['Low', 'Medium', 'High', 'Premium']
customers['purchase_value_segment'] = pd.qcut(customers['avg_purchase'], q=4, labels=avg_purchase_labels)

# Step 6: Create total spend segments
spend_labels = ['Low', 'Medium', 'High', 'Very High', 'Top']
customers['spend_segment'] = pd.qcut(customers['total_spend'], q=5, labels=spend_labels)

# Step 7: Create support intensity segments
def support_intensity(calls):
    if calls == 0:
        return 'None'
    elif calls <= 2:
        return 'Low'
    elif calls <= 5:
        return 'Medium'
    else:
        return 'High'

customers['support_intensity'] = customers['support_calls'].apply(support_intensity)

print("\nCustomer segments created:")
for col in ['age_segment', 'income_segment', 'tenure_segment', 
            'purchase_frequency', 'purchase_value_segment', 
            'spend_segment', 'support_intensity']:
    print(f"\n{col} distribution:")
    print(customers[col].value_counts().sort_index())

# Step 8: Create customer value segments based on RFM (Recency, Frequency, Monetary)
# For this example, we'll use frequency (num_purchases) and monetary (total_spend)
# We'll create a simple 2x2 matrix

# Convert to quintiles (1-5)
purchase_quintiles = pd.qcut(customers['num_purchases'], q=5, labels=False) + 1
spend_quintiles = pd.qcut(customers['total_spend'], q=5, labels=False) + 1

# Calculate RFM score (just FM in this case)
customers['fm_score'] = purchase_quintiles + spend_quintiles

# Create customer value segment
def customer_value(score):
    if score <= 4:
        return 'Low Value'
    elif score <= 7:
        return 'Medium Value'
    else:
        return 'High Value'

customers['customer_value'] = customers['fm_score'].apply(customer_value)

print("\nCustomer value segment distribution:")
print(customers['customer_value'].value_counts().sort_index())

# Step 9: Analyze segments
# Average metrics by customer value segment
value_analysis = customers.groupby('customer_value').agg({
    'age': 'mean',
    'income': 'mean',
    'years_as_customer': 'mean',
    'num_purchases': 'mean',
    'avg_purchase': 'mean',
    'total_spend': 'mean',
    'support_calls': 'mean',
    'customer_id': 'count'
}).rename(columns={'customer_id': 'count'})

print("\nMetrics by customer value segment:")
print(value_analysis)

# Step 10: Cross-tabulation analysis
# Age segment vs Customer value
age_value = pd.crosstab(
    customers['age_segment'], 
    customers['customer_value'],
    normalize='index'
) * 100

print("\nPercentage of customer value segments by age segment:")
print(age_value.round(1))

# Income segment vs Customer value
income_value = pd.crosstab(
    customers['income_segment'], 
    customers['customer_value'],
    normalize='index'
) * 100

print("\nPercentage of customer value segments by income segment:")
print(income_value.round(1))

# Tenure segment vs Customer value
tenure_value = pd.crosstab(
    customers['tenure_segment'], 
    customers['customer_value'],
    normalize='index'
) * 100

print("\nPercentage of customer value segments by tenure segment:")
print(tenure_value.round(1))

# Step 11: Visualize key relationships
# Customer value distribution
plt.figure(figsize=(10, 6))
customers['customer_value'].value_counts().sort_index().plot(kind='bar', color=['#FF9999', '#66B2FF', '#99FF99'])
plt.title('Distribution of Customer Value Segments')
plt.xlabel('Customer Value')
plt.ylabel('Number of Customers')
# plt.savefig('customer_value_distribution.png')
# plt.show()

# Average total spend by age and income segments
spend_by_age_income = customers.pivot_table(
    values='total_spend',
    index='age_segment',
    columns='income_segment',
    aggfunc='mean'
)

plt.figure(figsize=(12, 8))
sns.heatmap(spend_by_age_income, annot=True, fmt='.0f', cmap='YlGnBu')
plt.title('Average Total Spend by Age and Income Segments')
plt.tight_layout()
# plt.savefig('spend_by_age_income.png')
# plt.show()

# Customer value by tenure and purchase frequency
value_by_tenure_frequency = pd.crosstab(
    [customers['tenure_segment'], customers['purchase_frequency']],
    customers['customer_value']
)

print("\nCustomer value segments by tenure and purchase frequency:")
print(value_by_tenure_frequency)
```

## Reflections

Today's exploration of techniques for grouping numeric data has significantly enhanced my data preprocessing and analysis toolkit. The ability to transform continuous numeric data into meaningful categories is essential for simplifying complex data, identifying patterns, and communicating insights effectively.

I found the different binning approaches particularly useful for different scenarios. Equal-width binning (`pd.cut()`) is great when the actual values and ranges are important, such as age groups or price ranges. Equal-frequency binning (`pd.qcut()`) is more useful when the distribution is skewed and you want to ensure a balanced number of observations in each bin, like for income quartiles.

Custom binning logic provides the most flexibility, allowing for domain-specific categorization that might not follow a simple mathematical pattern. This is particularly valuable when working with data that has natural breakpoints or when business rules dictate specific groupings.

The customer segmentation exercise demonstrated how these techniques can be applied to create meaningful customer segments based on various attributes. By categorizing customers along multiple dimensions (age, income, tenure, purchase behavior), we can develop a more nuanced understanding of the customer base and tailor strategies accordingly.

I also learned that effective visualization of grouped data can reveal patterns and relationships that might not be apparent in the raw data. Techniques like heatmaps, color-coded scatter plots, and cross-tabulations provide powerful ways to explore relationships between different categorical variables.

## Questions to Explore
- How do different binning strategies affect the results of statistical analyses and machine learning models?
- What are the best practices for determining the optimal number of bins for different types of data?
- How can I incorporate domain knowledge more effectively into the binning process?
- What are the most effective ways to visualize relationships between multiple grouped variables?

## Resources
- [Pandas Documentation - Discretization and Quantizing](https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html#discretization-and-quantizing)
- [Pandas Documentation - pd.cut](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.cut.html)
- [Pandas Documentation - pd.qcut](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.qcut.html)
- [Towards Data Science - Data Binning in Python](https://towardsdatascience.com/data-binning-in-python-8d8d6f5c8c3)
- [Python for Data Analysis, 2nd Edition - Chapter 7](https://wesmckinney.com/book/)
