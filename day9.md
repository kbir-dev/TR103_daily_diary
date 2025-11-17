# Day 9: Merging, Binding, and Appending DataFrames

## Today's Topics
- Different types of joins in Pandas: inner, outer, left, and right
- Binding DataFrames (vertical and horizontal concatenation)
- Appending data to DataFrames
- Practical applications of data merging techniques

## Learning Journal

### Introduction to Data Merging

Today I explored various techniques for combining DataFrames in Pandas. These operations are essential when working with data from multiple sources or when data is split across different files or tables.

I started by creating sample DataFrames to work with:

```python
import pandas as pd
import numpy as np

# Create sample DataFrames for employees and departments
employees = pd.DataFrame({
    'employee_id': [101, 102, 103, 104, 105],
    'name': ['John Smith', 'Jane Doe', 'Robert Johnson', 'Lisa Wong', 'Michael Brown'],
    'department_id': [1, 2, 3, 2, 4],
    'salary': [60000, 75000, 65000, 70000, 80000]
})

departments = pd.DataFrame({
    'department_id': [1, 2, 3, 5],
    'department_name': ['HR', 'IT', 'Finance', 'Marketing'],
    'location': ['New York', 'San Francisco', 'Chicago', 'Boston']
})

print("Employees DataFrame:")
print(employees)
print("\nDepartments DataFrame:")
print(departments)
```

### Types of Joins in Pandas

I learned about the different types of joins (merges) available in Pandas:

#### Inner Join

An inner join returns only the rows where there is a match in both DataFrames:

```python
# Inner join (default)
inner_join = pd.merge(employees, departments, on='department_id')
print("\nInner Join Result:")
print(inner_join)
```

#### Left Join

A left join returns all rows from the left DataFrame and matching rows from the right DataFrame:

```python
# Left join
left_join = pd.merge(employees, departments, on='department_id', how='left')
print("\nLeft Join Result:")
print(left_join)
```

#### Right Join

A right join returns all rows from the right DataFrame and matching rows from the left DataFrame:

```python
# Right join
right_join = pd.merge(employees, departments, on='department_id', how='right')
print("\nRight Join Result:")
print(right_join)
```

#### Outer Join

An outer join returns all rows when there is a match in either DataFrame:

```python
# Outer join
outer_join = pd.merge(employees, departments, on='department_id', how='outer')
print("\nOuter Join Result:")
print(outer_join)
```

### Handling Non-Matching Column Names

I learned how to merge DataFrames when the key columns have different names:

```python
# Create DataFrames with different column names
orders = pd.DataFrame({
    'order_id': [1001, 1002, 1003, 1004, 1005],
    'customer_id': [1, 2, 3, 1, 2],
    'product': ['Laptop', 'Phone', 'Tablet', 'Monitor', 'Keyboard'],
    'quantity': [1, 2, 1, 3, 2]
})

customers = pd.DataFrame({
    'id': [1, 2, 3, 4],
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'city': ['New York', 'Los Angeles', 'Chicago', 'Houston']
})

print("\nOrders DataFrame:")
print(orders)
print("\nCustomers DataFrame:")
print(customers)

# Merge with different column names
orders_customers = pd.merge(orders, customers, left_on='customer_id', right_on='id')
print("\nMerge with different column names:")
print(orders_customers)
```

### Merging on Multiple Columns

I practiced merging DataFrames based on multiple columns:

```python
# Create DataFrames for sales data
sales_2024 = pd.DataFrame({
    'product_id': [101, 102, 103, 101, 102],
    'region': ['North', 'North', 'South', 'South', 'East'],
    'sales_2024': [10000, 15000, 12000, 8000, 9000]
})

sales_2025 = pd.DataFrame({
    'product_id': [101, 102, 103, 101, 104],
    'region': ['North', 'North', 'South', 'East', 'West'],
    'sales_2025': [12000, 18000, 14000, 10000, 11000]
})

print("\nSales 2024 DataFrame:")
print(sales_2024)
print("\nSales 2025 DataFrame:")
print(sales_2025)

# Merge on multiple columns
sales_comparison = pd.merge(sales_2024, sales_2025, on=['product_id', 'region'], how='outer')
print("\nMerge on multiple columns:")
print(sales_comparison)
```

### Binding DataFrames (Concatenation)

Next, I explored how to concatenate DataFrames, both vertically and horizontally:

#### Vertical Concatenation (Appending)

```python
# Create sample DataFrames for vertical concatenation
df1 = pd.DataFrame({
    'A': ['A0', 'A1', 'A2'],
    'B': ['B0', 'B1', 'B2'],
    'C': ['C0', 'C1', 'C2']
})

df2 = pd.DataFrame({
    'A': ['A3', 'A4', 'A5'],
    'B': ['B3', 'B4', 'B5'],
    'C': ['C3', 'C4', 'C5']
})

print("\nDataFrame 1:")
print(df1)
print("\nDataFrame 2:")
print(df2)

# Vertical concatenation (appending)
result_vertical = pd.concat([df1, df2])
print("\nVertical Concatenation Result:")
print(result_vertical)

# Reset index after concatenation
result_vertical_reset = pd.concat([df1, df2], ignore_index=True)
print("\nVertical Concatenation with Reset Index:")
print(result_vertical_reset)
```

#### Horizontal Concatenation (Binding)

```python
# Create sample DataFrames for horizontal concatenation
df3 = pd.DataFrame({
    'A': ['A0', 'A1', 'A2'],
    'B': ['B0', 'B1', 'B2']
})

df4 = pd.DataFrame({
    'C': ['C0', 'C1', 'C2'],
    'D': ['D0', 'D1', 'D2']
})

print("\nDataFrame 3:")
print(df3)
print("\nDataFrame 4:")
print(df4)

# Horizontal concatenation (binding)
result_horizontal = pd.concat([df3, df4], axis=1)
print("\nHorizontal Concatenation Result:")
print(result_horizontal)
```

### Handling Different Column Sets

I learned how to handle cases where DataFrames have different columns:

```python
# Create DataFrames with different column sets
df5 = pd.DataFrame({
    'A': ['A0', 'A1', 'A2'],
    'B': ['B0', 'B1', 'B2'],
    'C': ['C0', 'C1', 'C2']
})

df6 = pd.DataFrame({
    'B': ['B3', 'B4', 'B5'],
    'C': ['C3', 'C4', 'C5'],
    'D': ['D3', 'D4', 'D5']
})

print("\nDataFrame 5:")
print(df5)
print("\nDataFrame 6:")
print(df6)

# Concatenate with different columns (outer join)
result_outer = pd.concat([df5, df6], ignore_index=True)
print("\nConcatenation with Different Columns (Outer Join):")
print(result_outer)

# Concatenate with different columns (inner join)
result_inner = pd.concat([df5, df6], ignore_index=True, join='inner')
print("\nConcatenation with Different Columns (Inner Join):")
print(result_inner)
```

### Appending Data to DataFrames

I explored different ways to append data to existing DataFrames:

```python
# Create a base DataFrame
base_df = pd.DataFrame({
    'product': ['A', 'B', 'C'],
    'price': [10, 20, 30]
})

print("\nBase DataFrame:")
print(base_df)

# Append a single row (Series)
new_row = pd.Series({'product': 'D', 'price': 40})
# Note: append is deprecated, using concat instead
updated_df = pd.concat([base_df, pd.DataFrame([new_row])], ignore_index=True)
print("\nDataFrame with Appended Row:")
print(updated_df)

# Append multiple rows (DataFrame)
new_rows = pd.DataFrame({
    'product': ['E', 'F'],
    'price': [50, 60]
})
updated_df2 = pd.concat([updated_df, new_rows], ignore_index=True)
print("\nDataFrame with Multiple Appended Rows:")
print(updated_df2)
```

### Practical Exercise: Sales Data Analysis

I applied what I learned to a practical sales data analysis scenario:

```python
# Create DataFrames for sales data from different regions
np.random.seed(42)

# Function to generate sales data
def generate_sales_data(region, num_records):
    products = ['Laptop', 'Phone', 'Tablet', 'Monitor', 'Keyboard', 'Mouse']
    customers = [f'Customer_{i}' for i in range(1, 11)]
    
    data = {
        'date': pd.date_range(start='2025-01-01', periods=num_records),
        'product': np.random.choice(products, num_records),
        'customer': np.random.choice(customers, num_records),
        'quantity': np.random.randint(1, 10, num_records),
        'unit_price': np.random.uniform(50, 1000, num_records).round(2),
        'region': region
    }
    
    df = pd.DataFrame(data)
    df['total_price'] = df['quantity'] * df['unit_price']
    return df

# Generate sales data for different regions
north_sales = generate_sales_data('North', 50)
south_sales = generate_sales_data('South', 40)
east_sales = generate_sales_data('East', 30)
west_sales = generate_sales_data('West', 20)

print("\nNorth Region Sales (Sample):")
print(north_sales.head(3))

# 1. Combine all sales data
all_sales = pd.concat([north_sales, south_sales, east_sales, west_sales], ignore_index=True)
print(f"\nCombined Sales Data Shape: {all_sales.shape}")
print("Combined Sales Data (Sample):")
print(all_sales.head(3))

# 2. Create a product catalog
product_catalog = pd.DataFrame({
    'product': ['Laptop', 'Phone', 'Tablet', 'Monitor', 'Keyboard', 'Mouse'],
    'category': ['Computers', 'Mobile', 'Mobile', 'Accessories', 'Accessories', 'Accessories'],
    'supplier': ['Supplier A', 'Supplier B', 'Supplier A', 'Supplier C', 'Supplier D', 'Supplier D'],
    'stock_level': [120, 200, 150, 100, 300, 400]
})

print("\nProduct Catalog:")
print(product_catalog)

# 3. Create a customer database
customer_data = pd.DataFrame({
    'customer': [f'Customer_{i}' for i in range(1, 11)],
    'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], 10),
    'segment': np.random.choice(['Consumer', 'Business', 'Government'], 10),
    'account_manager': np.random.choice(['Manager A', 'Manager B', 'Manager C'], 10)
})

print("\nCustomer Database:")
print(customer_data)

# 4. Merge sales data with product information
sales_with_products = pd.merge(all_sales, product_catalog, on='product')
print("\nSales Data with Product Information (Sample):")
print(sales_with_products.head(3))

# 5. Merge sales data with customer information
complete_sales = pd.merge(sales_with_products, customer_data, on='customer')
print("\nComplete Sales Data (Sample):")
print(complete_sales.head(3))

# 6. Analyze sales by product category and region
category_region_sales = complete_sales.groupby(['category', 'region'])['total_price'].sum().reset_index()
print("\nSales by Product Category and Region:")
print(category_region_sales)

# 7. Analyze sales by customer segment and product
segment_product_sales = complete_sales.groupby(['segment', 'product'])['total_price'].sum().reset_index()
print("\nSales by Customer Segment and Product:")
print(segment_product_sales)

# 8. Create a pivot table of sales by region and product
region_product_pivot = pd.pivot_table(
    complete_sales, 
    values='total_price',
    index='region',
    columns='product',
    aggfunc='sum',
    fill_value=0
)

print("\nPivot Table of Sales by Region and Product:")
print(region_product_pivot)

# 9. Calculate the percentage of sales by product category
category_sales = complete_sales.groupby('category')['total_price'].sum()
total_sales = category_sales.sum()
category_percentage = (category_sales / total_sales * 100).round(2)

print("\nPercentage of Sales by Product Category:")
print(category_percentage)

# 10. Find top customers by sales amount
top_customers = complete_sales.groupby('customer')['total_price'].sum().sort_values(ascending=False).head(5)
print("\nTop 5 Customers by Sales Amount:")
print(top_customers)
```

## Reflections

Today's exploration of merging, binding, and appending DataFrames has significantly enhanced my data manipulation capabilities in Pandas. These techniques are essential for integrating data from different sources and preparing it for analysis.

I found the different types of joins (inner, left, right, outer) particularly useful, as they provide flexibility in how to combine DataFrames based on the specific requirements of the analysis. The inner join is great for finding matches between datasets, while the outer join is useful for getting a complete view of all data.

Concatenation (both vertical and horizontal) is a powerful technique for combining DataFrames with similar structures. The ability to handle different column sets and reset indices makes it versatile for various data integration scenarios.

The practical exercise demonstrated how these techniques can be applied to real-world business scenarios. By merging sales data with product and customer information, I was able to create a comprehensive dataset that enabled multidimensional analysis of sales performance.

I also learned that while the `append()` method is still available in Pandas, it's deprecated and `pd.concat()` is the recommended approach for adding rows to a DataFrame. This is good to know for maintaining up-to-date code practices.

## Questions to Explore
- How do these merging operations scale with very large DataFrames?
- What are the best practices for optimizing memory usage when merging large datasets?
- How can I handle more complex merging scenarios, such as fuzzy matching or merging based on date ranges?
- What are the performance implications of different join types?

## Resources
- [Pandas Documentation - Merge, Join, and Concatenate](https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html)
- [Pandas Documentation - Merge Function](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.merge.html)
- [Pandas Documentation - Concat Function](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html)
- [Python for Data Analysis, 2nd Edition - Chapter 8](https://wesmckinney.com/book/)
