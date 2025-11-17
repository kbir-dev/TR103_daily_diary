# Day 6: Handling Missing Values and If-Else Statements in Python

## Today's Topics
- Identifying and handling missing values in Pandas
- If-else statements in Python
- Advanced techniques for conditional operations
- Applying conditions to DataFrames

## Learning Journal

### Handling Missing Values in Pandas

Today I focused on techniques for identifying and handling missing values in Pandas, which is a critical skill for data cleaning and preparation.

I started by creating a DataFrame with intentional missing values:

```python
import pandas as pd
import numpy as np

# Create a DataFrame with missing values
data = {
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, 4, 5],
    'C': [1, 2, 3, np.nan, np.nan],
    'D': [1, np.nan, np.nan, np.nan, 5]
}

df = pd.DataFrame(data)
print("DataFrame with missing values:")
print(df)
```

#### Identifying Missing Values

I learned various methods to identify missing values:

```python
# Check for missing values
print("\nMissing values by column:")
print(df.isna().sum())

# Check for missing values by row
print("\nNumber of missing values in each row:")
print(df.isna().sum(axis=1))

# Identify which values are missing
print("\nBoolean mask of missing values:")
print(df.isna())

# Find rows with any missing values
rows_with_na = df[df.isna().any(axis=1)]
print("\nRows with any missing values:")
print(rows_with_na)

# Find rows with all missing values
rows_all_na = df[df.isna().all(axis=1)]
print("\nRows with all missing values:")
print(rows_all_na)

# Find columns with any missing values
cols_with_na = df.columns[df.isna().any()]
print("\nColumns with missing values:", list(cols_with_na))
```

#### Handling Missing Values

I explored different strategies for handling missing values:

```python
# 1. Dropping missing values
# Drop rows with any missing values
df_dropped = df.dropna()
print("\nDataFrame after dropping rows with missing values:")
print(df_dropped)

# Drop rows where all values are missing
df_dropped_all = df.dropna(how='all')
print("\nDataFrame after dropping rows where all values are missing:")
print(df_dropped_all)

# Drop columns with any missing values
df_dropped_cols = df.dropna(axis=1)
print("\nDataFrame after dropping columns with missing values:")
print(df_dropped_cols)

# 2. Filling missing values
# Fill with a constant value
df_filled = df.fillna(0)
print("\nDataFrame with missing values filled with 0:")
print(df_filled)

# Fill with different values for each column
fill_values = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
df_filled_dict = df.fillna(fill_values)
print("\nDataFrame with missing values filled with different values by column:")
print(df_filled_dict)

# Fill with the mean of each column
df_filled_mean = df.fillna(df.mean())
print("\nDataFrame with missing values filled with column means:")
print(df_filled_mean)

# Fill forward (use previous value)
df_filled_ffill = df.fillna(method='ffill')
print("\nDataFrame with missing values filled forward:")
print(df_filled_ffill)

# Fill backward (use next value)
df_filled_bfill = df.fillna(method='bfill')
print("\nDataFrame with missing values filled backward:")
print(df_filled_bfill)

# 3. Interpolation
# Linear interpolation
df_interp = df.interpolate()
print("\nDataFrame with linear interpolation:")
print(df_interp)
```

### If-Else Statements in Python

Next, I studied if-else statements in Python, which are fundamental for implementing conditional logic:

```python
# Basic if-else statement
x = 10
if x > 5:
    print("\nX is greater than 5")
else:
    print("\nX is not greater than 5")

# If-elif-else statement
y = 7
if y > 10:
    print("Y is greater than 10")
elif y > 5:
    print("Y is greater than 5 but not greater than 10")
else:
    print("Y is not greater than 5")

# Nested if statements
z = 15
if z > 0:
    print("Z is positive")
    if z % 2 == 0:
        print("Z is even")
    else:
        print("Z is odd")
else:
    print("Z is not positive")
```

### Advanced Conditional Techniques

I explored more advanced conditional techniques in Python:

```python
# Conditional expressions (ternary operator)
age = 20
status = "Adult" if age >= 18 else "Minor"
print(f"\nStatus: {status}")

# Multiple conditions with and/or
temperature = 25
weather = "sunny"

if temperature > 20 and weather == "sunny":
    print("It's a nice day!")
elif temperature > 20 or weather == "sunny":
    print("It's an okay day.")
else:
    print("It's not a good day.")

# Using in for membership tests
fruit = "apple"
if fruit in ["apple", "banana", "orange"]:
    print(f"{fruit} is in the list of fruits")

# Using not for negation
is_weekend = False
if not is_weekend:
    print("It's a weekday")
else:
    print("It's the weekend")
```

### Applying Conditions to DataFrames

I learned how to apply conditional logic to DataFrames:

```python
# Create a sample DataFrame
sales_data = pd.DataFrame({
    'Product': ['A', 'B', 'C', 'D', 'E'],
    'Category': ['Electronics', 'Clothing', 'Electronics', 'Home', 'Clothing'],
    'Price': [100, 45, 200, 30, 80],
    'Stock': [15, 25, 5, 40, 10]
})

print("\nSales data:")
print(sales_data)

# 1. Using .loc with conditions
# Add a 'Status' column based on stock levels
sales_data.loc[sales_data['Stock'] < 10, 'Status'] = 'Low'
sales_data.loc[(sales_data['Stock'] >= 10) & (sales_data['Stock'] < 30), 'Status'] = 'Medium'
sales_data.loc[sales_data['Stock'] >= 30, 'Status'] = 'High'

print("\nSales data with status:")
print(sales_data)

# 2. Using np.where for simple conditions
# Add a 'Price_Category' column
sales_data['Price_Category'] = np.where(sales_data['Price'] > 50, 'High', 'Low')
print("\nSales data with price category:")
print(sales_data)

# 3. Using np.select for multiple conditions
conditions = [
    (sales_data['Category'] == 'Electronics') & (sales_data['Price'] > 150),
    (sales_data['Category'] == 'Electronics') & (sales_data['Price'] <= 150),
    (sales_data['Category'] == 'Clothing'),
    (sales_data['Category'] == 'Home')
]

choices = ['Premium Electronics', 'Standard Electronics', 'Apparel', 'Household']

sales_data['Product_Type'] = np.select(conditions, choices, default='Other')
print("\nSales data with product type:")
print(sales_data)
```

### Using Apply with Lambda Functions

I explored how to use the `apply` method with lambda functions for more complex conditional operations:

```python
# Define a function to calculate discount based on category and price
def calculate_discount(row):
    if row['Category'] == 'Electronics':
        if row['Price'] > 150:
            return 0.1  # 10% discount
        else:
            return 0.05  # 5% discount
    elif row['Category'] == 'Clothing':
        return 0.15  # 15% discount
    else:
        return 0.08  # 8% discount

# Apply the function to each row
sales_data['Discount'] = sales_data.apply(calculate_discount, axis=1)
sales_data['Discount_Amount'] = sales_data['Price'] * sales_data['Discount']
sales_data['Final_Price'] = sales_data['Price'] - sales_data['Discount_Amount']

print("\nSales data with discounts:")
print(sales_data)

# Using lambda for simple calculations
sales_data['Stock_Value'] = sales_data.apply(lambda row: row['Price'] * row['Stock'], axis=1)
print("\nSales data with stock value:")
print(sales_data)
```

### Practical Exercise: Customer Segmentation

I applied what I learned to a practical customer segmentation scenario:

```python
# Create a customer dataset
np.random.seed(42)
customers = pd.DataFrame({
    'CustomerID': range(1, 21),
    'Age': np.random.randint(18, 70, 20),
    'Income': np.random.randint(20000, 100000, 20),
    'Years_as_Customer': np.random.randint(1, 15, 20),
    'Purchases_Last_Year': np.random.randint(0, 50, 20),
    'Average_Purchase': np.random.randint(10, 200, 20)
})

# Introduce some missing values
customers.loc[3, 'Income'] = np.nan
customers.loc[7, 'Purchases_Last_Year'] = np.nan
customers.loc[12, 'Average_Purchase'] = np.nan
customers.loc[15, 'Years_as_Customer'] = np.nan

print("\nCustomer data with missing values:")
print(customers)

# Handle missing values
# Fill missing Income with median
customers['Income'] = customers['Income'].fillna(customers['Income'].median())

# Fill other missing values with mean
for col in ['Years_as_Customer', 'Purchases_Last_Year', 'Average_Purchase']:
    customers[col] = customers[col].fillna(customers[col].mean())

print("\nCustomer data after handling missing values:")
print(customers)

# Calculate total spend
customers['Total_Spend'] = customers['Purchases_Last_Year'] * customers['Average_Purchase']

# Customer segmentation based on multiple criteria
conditions = [
    (customers['Total_Spend'] > 5000) & (customers['Years_as_Customer'] >= 5),
    (customers['Total_Spend'] > 5000) & (customers['Years_as_Customer'] < 5),
    (customers['Total_Spend'] > 2000) & (customers['Total_Spend'] <= 5000),
    (customers['Total_Spend'] <= 2000) & (customers['Purchases_Last_Year'] >= 10),
    (customers['Total_Spend'] <= 2000) & (customers['Purchases_Last_Year'] < 10)
]

segments = ['Premium Loyal', 'Premium New', 'Mid-tier', 'Low-value Active', 'Low-value Inactive']

customers['Segment'] = np.select(conditions, segments, default='Other')

print("\nCustomer data with segmentation:")
print(customers)

# Calculate segment statistics
segment_stats = customers.groupby('Segment').agg({
    'CustomerID': 'count',
    'Age': 'mean',
    'Income': 'mean',
    'Years_as_Customer': 'mean',
    'Total_Spend': 'mean'
})

print("\nSegment statistics:")
print(segment_stats)
```

## Reflections

Today's learning about handling missing values and if-else statements has been incredibly valuable. Missing values are a common challenge in real-world data, and knowing various strategies to identify and handle them is essential for data preparation.

The if-else statements and conditional logic in Python provide powerful tools for implementing business rules and making decisions based on data. I particularly appreciated learning about the different ways to apply conditions to DataFrames, from simple boolean indexing to more complex techniques like `np.select` and `apply` with custom functions.

I found that choosing the right strategy for handling missing values depends on the context and the nature of the data. Sometimes dropping missing values is appropriate, while in other cases, imputation techniques like filling with mean/median or interpolation are more suitable.

The customer segmentation exercise was a great way to apply both missing value handling and conditional logic to a practical business scenario. It showed how these techniques can be combined to derive meaningful insights from data.

## Questions to Explore
- What are the implications of different missing value handling strategies on statistical analyses?
- How can I automate the process of selecting the best imputation method for different types of data?
- What are the performance considerations when using `apply` with lambda functions versus vectorized operations?
- How can I implement more complex decision trees or rule-based systems using Python's conditional logic?

## Resources
- [Pandas Documentation - Working with Missing Data](https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html)
- [Pandas Documentation - Conditional Selection](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#boolean-indexing)
- [Python Documentation - if Statements](https://docs.python.org/3/tutorial/controlflow.html#if-statements)
- [Python Data Science Handbook - Data Cleaning and Preparation](https://jakevdp.github.io/PythonDataScienceHandbook/03.04-missing-values.html)
