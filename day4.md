# Day 4: Loc and Iloc for Data Selection in Pandas

## Today's Topics
- Understanding loc and iloc indexers
- Label-based vs. position-based selection
- Advanced data selection techniques
- Boolean indexing with loc

## Learning Journal

### Understanding Loc and Iloc Indexers

Today I explored two powerful indexing methods in Pandas: `loc` and `iloc`. These methods provide precise control over data selection in DataFrames.

I started by creating a sample DataFrame to work with:

```python
import pandas as pd
import numpy as np

# Create a sample DataFrame with custom indices
data = {
    'Name': ['John', 'Sarah', 'Mike', 'Lisa', 'Tom'],
    'Age': [28, 34, 29, 42, 35],
    'City': ['New York', 'Boston', 'New York', 'Chicago', 'Boston'],
    'Department': ['IT', 'HR', 'Finance', 'IT', 'Marketing'],
    'Salary': [65000, 72000, 59000, 81000, 76000]
}

df = pd.DataFrame(data)
# Set a custom index
df.set_index('Name', inplace=True)
print("DataFrame with 'Name' as index:")
print(df)
```

### Loc: Label-Based Selection

The `loc` indexer is used for label-based selection, which means we use the actual row and column labels to select data:

```python
# Select a single row by label
john_data = df.loc['John']
print("\nJohn's data:")
print(john_data)

# Select multiple rows by labels
selected_employees = df.loc[['Sarah', 'Lisa']]
print("\nData for Sarah and Lisa:")
print(selected_employees)

# Select specific rows and columns
sarah_age_city = df.loc['Sarah', ['Age', 'City']]
print("\nSarah's age and city:")
print(sarah_age_city)

# Select a range of rows
first_three = df.loc['John':'Mike']
print("\nData from John to Mike (inclusive):")
print(first_three)
```

I learned that `loc` is inclusive of the end label when using slices, which differs from Python's usual slicing behavior.

### Iloc: Position-Based Selection

The `iloc` indexer is used for integer-based selection, which means we use the position (0-based) of rows and columns:

```python
# Select a single row by position
first_row = df.iloc[0]
print("\nFirst row:")
print(first_row)

# Select multiple rows by positions
rows_1_and_3 = df.iloc[[1, 3]]
print("\nSecond and fourth rows:")
print(rows_1_and_3)

# Select specific rows and columns by position
subset = df.iloc[0:2, 1:3]
print("\nSubset (first two rows, second and third columns):")
print(subset)

# Select all rows but only specific columns
all_rows_some_cols = df.iloc[:, [0, 2]]
print("\nAll rows, first and third columns:")
print(all_rows_some_cols)
```

### Combining Loc with Boolean Filtering

One of the most powerful features of `loc` is its ability to work with boolean masks:

```python
# Filter rows where Age > 30 using loc
older_than_30 = df.loc[df['Age'] > 30]
print("\nEmployees older than 30:")
print(older_than_30)

# Filter rows where City is 'Boston' and select specific columns
boston_it_hr = df.loc[df['City'] == 'Boston', ['Department', 'Salary']]
print("\nDepartment and Salary for employees from Boston:")
print(boston_it_hr)

# Complex condition with loc
high_salary_it = df.loc[(df['Salary'] > 70000) & (df['Department'] == 'IT'), :]
print("\nIT employees with high salary:")
print(high_salary_it)
```

### Advanced Selection Techniques

I also explored some advanced selection techniques:

```python
# Reset index to make 'Name' a regular column again
df_reset = df.reset_index()
print("\nDataFrame with reset index:")
print(df_reset)

# Set multiple columns as index
df_multi = df_reset.set_index(['Department', 'City'])
print("\nDataFrame with multi-level index:")
print(df_multi)

# Select using multi-level index
it_employees = df_multi.loc['IT']
print("\nAll IT employees:")
print(it_employees)

# Select a specific combination of multi-level index
it_ny = df_multi.loc[('IT', 'New York')]
print("\nIT employees from New York:")
print(it_ny)
```

### Using Loc and Iloc for Data Modification

I learned that `loc` and `iloc` can also be used to modify data:

```python
# Create a copy to avoid modifying the original DataFrame
df_copy = df.copy()

# Modify a single value using loc
df_copy.loc['John', 'Salary'] = 68000
print("\nJohn's salary updated:")
print(df_copy.loc['John'])

# Modify multiple values using iloc
df_copy.iloc[1:3, 2] = 'Remote'  # Change City for the 2nd and 3rd employees
print("\nUpdated DataFrame with remote employees:")
print(df_copy)

# Conditional modification
df_copy.loc[df_copy['Age'] > 35, 'Department'] = 'Senior ' + df_copy.loc[df_copy['Age'] > 35, 'Department']
print("\nUpdated departments for senior employees:")
print(df_copy)
```

## Practical Exercise

I applied what I learned to solve a practical data analysis task:

```python
# Create a more complex DataFrame
np.random.seed(42)
dates = pd.date_range('20250101', periods=10)
complex_df = pd.DataFrame({
    'Date': dates,
    'Product': np.random.choice(['A', 'B', 'C'], 10),
    'Region': np.random.choice(['North', 'South', 'East', 'West'], 10),
    'Sales': np.random.randint(100, 1000, 10),
    'Units': np.random.randint(10, 100, 10)
})

complex_df.set_index('Date', inplace=True)
print("\nSales DataFrame:")
print(complex_df)

# Find the highest sales day for each product using loc
for product in complex_df['Product'].unique():
    product_data = complex_df.loc[complex_df['Product'] == product]
    max_sales_idx = product_data['Sales'].idxmax()
    print(f"\nHighest sales for Product {product}:")
    print(complex_df.loc[max_sales_idx])

# Calculate average sales by region using loc and groupby
region_avg = complex_df.loc[:, ['Region', 'Sales']].groupby('Region').mean()
print("\nAverage sales by region:")
print(region_avg)
```

## Reflections

Today's exploration of `loc` and `iloc` has significantly enhanced my data manipulation capabilities in Pandas. The ability to precisely select and modify data based on labels or positions gives me tremendous flexibility when working with DataFrames.

I found that `loc` is generally more intuitive when working with labeled data, especially when combined with boolean filtering. On the other hand, `iloc` is invaluable when I need to work with data based on its position, similar to how I would access elements in a list or array.

Understanding the difference between label-based and position-based indexing is crucial, especially when dealing with custom indices or when performing slicing operations. The inclusive nature of `loc` slicing (including the end label) versus the exclusive nature of `iloc` slicing (excluding the end position) is an important distinction to remember.

## Questions to Explore
- How does the performance of `loc` compare to `iloc` for very large DataFrames?
- What are the best practices for working with multi-level indices?
- How can I optimize complex selection operations for better readability and performance?

## Resources
- [Pandas Documentation - Selection by Label](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#selection-by-label)
- [Pandas Documentation - Selection by Position](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#selection-by-position)
- [Pandas Documentation - Advanced Indexing](https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html)
- [Python for Data Analysis, 2nd Edition - Chapter 5](https://wesmckinney.com/book/)
