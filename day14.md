# Day 14: Reshape Functions in Python

## Today's Topics
- Reshaping data in Python
- Pivot, melt, stack, and unstack operations
- Wide vs. long format data
- Practical applications of reshape functions

## Learning Journal

### Introduction to Data Reshaping

Today I focused on reshape functions in Python, which are essential tools for transforming data between different formats. Data reshaping is a critical skill for data preparation and analysis, as different analyses and visualizations require data in specific formats.

I started by setting up the necessary libraries:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for our plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
```

### Understanding Wide vs. Long Format

I began by exploring the concepts of wide and long format data:

```python
# Create a sample dataset in wide format
wide_data = pd.DataFrame({
    'student_id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'math_score': [85, 92, 78, 88, 95],
    'science_score': [92, 85, 79, 94, 88],
    'history_score': [88, 79, 85, 90, 92]
})

print("Data in wide format:")
print(wide_data)

# Convert to long format using melt
long_data = pd.melt(
    wide_data, 
    id_vars=['student_id', 'name'],
    value_vars=['math_score', 'science_score', 'history_score'],
    var_name='subject',
    value_name='score'
)

print("\nData in long format:")
print(long_data)

# Clean up the subject names by removing '_score'
long_data['subject'] = long_data['subject'].str.replace('_score', '')

print("\nData in long format with clean subject names:")
print(long_data)
```

### Pivot Function: Long to Wide Format

Next, I learned how to use the `pivot` function to convert data from long to wide format:

```python
# Create a sample dataset in long format
sales_long = pd.DataFrame({
    'date': pd.date_range(start='2025-01-01', periods=12, freq='M'),
    'region': ['North', 'South', 'East', 'West'] * 3,
    'product': ['A', 'A', 'B', 'B'] * 3,
    'sales': np.random.randint(100, 1000, 12)
})

print("\nSales data in long format:")
print(sales_long)

# Convert to wide format using pivot
sales_wide = sales_long.pivot(
    index='date',
    columns='region',
    values='sales'
)

print("\nSales data pivoted by region:")
print(sales_wide)

# Multiple column pivot
sales_wide_multi = sales_long.pivot(
    index='date',
    columns=['region', 'product'],
    values='sales'
)

print("\nSales data with multi-level columns:")
print(sales_wide_multi)
```

### Pivot Table: Aggregation with Reshape

I explored the `pivot_table` function, which combines reshaping with aggregation:

```python
# Create a more complex sales dataset
np.random.seed(42)
dates = pd.date_range(start='2025-01-01', periods=100)
regions = ['North', 'South', 'East', 'West']
products = ['A', 'B', 'C']

complex_sales = []
for _ in range(200):
    date = np.random.choice(dates)
    region = np.random.choice(regions)
    product = np.random.choice(products)
    units = np.random.randint(1, 50)
    price_per_unit = np.random.uniform(10, 100)
    
    complex_sales.append({
        'date': date,
        'region': region,
        'product': product,
        'units': units,
        'price_per_unit': price_per_unit,
        'revenue': units * price_per_unit
    })

sales_df = pd.DataFrame(complex_sales)
sales_df['month'] = sales_df['date'].dt.strftime('%Y-%m')

print("\nComplex sales data sample:")
print(sales_df.head())

# Create a pivot table of total revenue by region and product
revenue_pivot = pd.pivot_table(
    sales_df,
    values='revenue',
    index='region',
    columns='product',
    aggfunc='sum'
)

print("\nRevenue by region and product:")
print(revenue_pivot)

# Create a pivot table with multiple aggregation functions
multi_agg_pivot = pd.pivot_table(
    sales_df,
    values=['revenue', 'units'],
    index='region',
    columns='product',
    aggfunc={'revenue': 'sum', 'units': 'mean'}
)

print("\nPivot table with multiple aggregation functions:")
print(multi_agg_pivot)

# Create a pivot table with multiple indices
monthly_pivot = pd.pivot_table(
    sales_df,
    values='revenue',
    index=['month', 'region'],
    columns='product',
    aggfunc='sum',
    fill_value=0
)

print("\nMonthly revenue by region and product:")
print(monthly_pivot)
```

### Melt Function: Wide to Long Format

I explored the `melt` function in more detail, which is used to convert data from wide to long format:

```python
# Create a sample dataset in wide format
weather_wide = pd.DataFrame({
    'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
    'temp_jan': [32, 65, 25, 55, 65],
    'temp_apr': [55, 70, 50, 75, 80],
    'temp_jul': [85, 85, 80, 95, 105],
    'temp_oct': [60, 75, 55, 80, 85]
})

print("\nWeather data in wide format:")
print(weather_wide)

# Convert to long format using melt
weather_long = pd.melt(
    weather_wide,
    id_vars=['city'],
    value_vars=['temp_jan', 'temp_apr', 'temp_jul', 'temp_oct'],
    var_name='month',
    value_name='temperature'
)

print("\nWeather data in long format:")
print(weather_long)

# Clean up the month names
weather_long['month'] = weather_long['month'].str.replace('temp_', '')

print("\nWeather data with clean month names:")
print(weather_long)

# Visualize the data
plt.figure(figsize=(12, 6))
sns.lineplot(data=weather_long, x='month', y='temperature', hue='city', marker='o')
plt.title('Temperature by City and Month')
plt.ylabel('Temperature (°F)')
# plt.savefig('city_temperatures.png')
# plt.show()
```

### Stack and Unstack Functions

I learned about the `stack` and `unstack` functions, which provide another way to reshape data:

```python
# Create a multi-index DataFrame
multi_index_df = pd.DataFrame(
    np.random.randn(6, 3),
    index=pd.MultiIndex.from_product([['A', 'B'], [1, 2, 3]], names=['letter', 'number']),
    columns=['X', 'Y', 'Z']
)

print("\nMulti-index DataFrame:")
print(multi_index_df)

# Stack: pivot columns to rows
stacked = multi_index_df.stack()
print("\nStacked DataFrame:")
print(stacked)

# Unstack: pivot rows to columns
unstacked = stacked.unstack()
print("\nUnstacked DataFrame (back to original):")
print(unstacked)

# Unstack at different levels
unstacked_level1 = stacked.unstack(level=1)  # Unstack 'number'
print("\nUnstacked at level 1 (number):")
print(unstacked_level1)

unstacked_level0 = stacked.unstack(level=0)  # Unstack 'letter'
print("\nUnstacked at level 0 (letter):")
print(unstacked_level0)
```

### Combining Multiple Reshape Operations

I explored how to combine multiple reshape operations for more complex transformations:

```python
# Create a sample sales dataset with hierarchical structure
sales_data = pd.DataFrame({
    'year': [2023, 2023, 2023, 2023, 2024, 2024, 2024, 2024],
    'quarter': [1, 2, 3, 4, 1, 2, 3, 4],
    'region': ['East', 'East', 'West', 'West', 'East', 'East', 'West', 'West'],
    'product': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
    'sales': [100, 150, 200, 250, 120, 170, 220, 270]
})

print("\nSales data with hierarchical structure:")
print(sales_data)

# Step 1: Create a pivot table with year and quarter as index, region and product as columns
pivot1 = sales_data.pivot_table(
    values='sales',
    index=['year', 'quarter'],
    columns=['region', 'product'],
    aggfunc='sum'
)

print("\nStep 1 - Pivot table:")
print(pivot1)

# Step 2: Unstack the quarter level to create a flatter structure
pivot2 = pivot1.unstack(level='quarter')
print("\nStep 2 - Unstacked quarter level:")
print(pivot2)

# Step 3: Stack the region level to create a different view
pivot3 = pivot1.stack(level='region')
print("\nStep 3 - Stacked region level:")
print(pivot3)

# Alternative approach: Use pivot_table with multiple indices and columns
pivot_alt = sales_data.pivot_table(
    values='sales',
    index=['year'],
    columns=['quarter', 'region', 'product'],
    aggfunc='sum'
)

print("\nAlternative approach - Complex pivot table:")
print(pivot_alt)
```

### Practical Exercise: Sales Analysis with Reshape Functions

I applied what I learned to analyze a more comprehensive sales dataset:

```python
# Create a detailed sales dataset
np.random.seed(42)

# Generate dates for a year
dates = pd.date_range(start='2025-01-01', end='2025-12-31', freq='D')

# Products with categories and price ranges
products = {
    'Laptop': {'category': 'Electronics', 'price_range': (800, 1500)},
    'Desktop': {'category': 'Electronics', 'price_range': (1000, 2000)},
    'Tablet': {'category': 'Electronics', 'price_range': (300, 700)},
    'Phone': {'category': 'Electronics', 'price_range': (500, 1200)},
    'Monitor': {'category': 'Accessories', 'price_range': (200, 500)},
    'Keyboard': {'category': 'Accessories', 'price_range': (50, 150)},
    'Mouse': {'category': 'Accessories', 'price_range': (20, 80)},
    'Headphones': {'category': 'Accessories', 'price_range': (100, 300)}
}

# Regions and channels
regions = ['North', 'South', 'East', 'West']
channels = ['Online', 'Retail', 'Distributor']

# Generate sales records
sales_records = []
for _ in range(5000):
    date = np.random.choice(dates)
    product_name = np.random.choice(list(products.keys()))
    product_info = products[product_name]
    category = product_info['category']
    region = np.random.choice(regions)
    channel = np.random.choice(channels)
    
    # Add seasonality to certain products
    season_factor = 1.0
    if product_name in ['Laptop', 'Desktop'] and date.month in [8, 9]:  # Back to school
        season_factor = 1.5
    elif product_name in ['Phone', 'Tablet'] and date.month in [11, 12]:  # Holiday season
        season_factor = 1.8
    
    # Calculate units and price
    units = np.random.randint(1, 10)
    unit_price = np.random.uniform(*product_info['price_range']) * season_factor
    revenue = units * unit_price
    
    # Create the record
    record = {
        'date': date,
        'product': product_name,
        'category': category,
        'region': region,
        'channel': channel,
        'units': units,
        'unit_price': round(unit_price, 2),
        'revenue': round(revenue, 2)
    }
    
    sales_records.append(record)

# Create the DataFrame
detailed_sales = pd.DataFrame(sales_records)

# Add time-based columns
detailed_sales['year'] = detailed_sales['date'].dt.year
detailed_sales['quarter'] = 'Q' + detailed_sales['date'].dt.quarter.astype(str)
detailed_sales['month'] = detailed_sales['date'].dt.strftime('%b')
detailed_sales['week'] = detailed_sales['date'].dt.isocalendar().week

print("Detailed sales dataset sample:")
print(detailed_sales.head())
print(f"Total records: {len(detailed_sales)}")

# Analysis 1: Monthly sales by product category
monthly_category_sales = pd.pivot_table(
    detailed_sales,
    values='revenue',
    index='month',
    columns='category',
    aggfunc='sum'
)

# Reorder months chronologically
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_category_sales = monthly_category_sales.reindex(month_order)

print("\nMonthly sales by product category:")
print(monthly_category_sales)

# Visualize monthly sales by category
plt.figure(figsize=(12, 6))
monthly_category_sales.plot(kind='bar', stacked=True)
plt.title('Monthly Sales by Product Category')
plt.xlabel('Month')
plt.ylabel('Revenue')
plt.legend(title='Category')
plt.xticks(rotation=45)
# plt.savefig('monthly_category_sales.png')
# plt.show()

# Analysis 2: Sales by region and channel
region_channel_sales = pd.pivot_table(
    detailed_sales,
    values='revenue',
    index='region',
    columns='channel',
    aggfunc='sum',
    margins=True,
    margins_name='Total'
)

print("\nSales by region and channel:")
print(region_channel_sales)

# Analysis 3: Product performance across regions
product_region_sales = pd.pivot_table(
    detailed_sales,
    values='revenue',
    index='product',
    columns='region',
    aggfunc='sum'
)

print("\nProduct performance across regions:")
print(product_region_sales)

# Analysis 4: Quarterly sales trends by category
quarterly_category_sales = pd.pivot_table(
    detailed_sales,
    values='revenue',
    index=['year', 'quarter'],
    columns='category',
    aggfunc='sum'
)

print("\nQuarterly sales by category:")
print(quarterly_category_sales)

# Analysis 5: Channel performance by product
channel_product_sales = pd.pivot_table(
    detailed_sales,
    values=['revenue', 'units'],
    index='channel',
    columns='product',
    aggfunc={'revenue': 'sum', 'units': 'sum'}
)

print("\nChannel performance by product:")
print(channel_product_sales)

# Analysis 6: Convert to long format for visualization
sales_by_category_channel = pd.pivot_table(
    detailed_sales,
    values='revenue',
    index='category',
    columns='channel',
    aggfunc='sum'
)

# Melt the pivot table to long format
sales_long = pd.melt(
    sales_by_category_channel.reset_index(),
    id_vars='category',
    value_vars=channels,
    var_name='channel',
    value_name='revenue'
)

print("\nSales by category and channel (long format):")
print(sales_long)

# Visualize with a grouped bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x='category', y='revenue', hue='channel', data=sales_long)
plt.title('Revenue by Category and Channel')
plt.xlabel('Category')
plt.ylabel('Revenue')
plt.xticks(rotation=45)
plt.legend(title='Channel')
# plt.savefig('category_channel_revenue.png')
# plt.show()

# Analysis 7: Create a heatmap of product performance by region
plt.figure(figsize=(12, 8))
sns.heatmap(product_region_sales, annot=True, fmt='.0f', cmap='YlGnBu')
plt.title('Product Revenue by Region')
plt.tight_layout()
# plt.savefig('product_region_heatmap.png')
# plt.show()

# Analysis 8: Melt and stack for detailed time series analysis
# First, create a pivot table of monthly sales by product
monthly_product_sales = pd.pivot_table(
    detailed_sales,
    values='revenue',
    index=['year', 'month'],
    columns='product',
    aggfunc='sum'
)

print("\nMonthly sales by product:")
print(monthly_product_sales)

# Melt to long format for time series visualization
monthly_product_long = monthly_product_sales.reset_index()
monthly_product_long = pd.melt(
    monthly_product_long,
    id_vars=['year', 'month'],
    value_vars=list(products.keys()),
    var_name='product',
    value_name='revenue'
)

# Add category information
product_to_category = {product: info['category'] for product, info in products.items()}
monthly_product_long['category'] = monthly_product_long['product'].map(product_to_category)

# Create a combined year-month field for proper ordering
monthly_product_long['year_month'] = monthly_product_long['year'].astype(str) + '-' + monthly_product_long['month']

print("\nMonthly product sales in long format:")
print(monthly_product_long.head())

# Visualize time series by category
plt.figure(figsize=(14, 8))
sns.lineplot(
    data=monthly_product_long,
    x='year_month',
    y='revenue',
    hue='category',
    style='category',
    markers=True,
    dashes=False
)
plt.title('Monthly Revenue by Product Category')
plt.xlabel('Month')
plt.ylabel('Revenue')
plt.xticks(rotation=45)
# plt.savefig('monthly_category_trend.png')
# plt.show()
```

## Reflections

Today's exploration of reshape functions in Python has significantly enhanced my data manipulation capabilities. The ability to transform data between wide and long formats is essential for both analysis and visualization, as different techniques require data in specific structures.

I found the `pivot` and `pivot_table` functions particularly useful for creating summary views of data. The `pivot_table` function is especially powerful as it combines reshaping with aggregation, allowing for complex analyses in a single operation. The ability to specify multiple index and column levels, as well as different aggregation functions for different values, provides tremendous flexibility.

The `melt` function is invaluable for converting data from wide to long format, which is often necessary for visualization with libraries like Seaborn that prefer "tidy" data. I appreciated how the `var_name` and `value_name` parameters make it easy to create meaningful column names in the resulting long-format DataFrame.

The `stack` and `unstack` functions provide another approach to reshaping data, particularly when working with multi-index DataFrames. These functions are especially useful for more complex hierarchical data structures where you need fine-grained control over which levels to pivot.

The practical exercise demonstrated how these reshape functions can be applied to real-world data analysis scenarios. By transforming the sales data into different formats, I was able to analyze performance across multiple dimensions (time, product, region, channel) and create visualizations that revealed patterns and trends that might not have been apparent in the original data structure.

## Questions to Explore
- How do reshape operations scale with very large datasets?
- What are the best practices for optimizing memory usage when reshaping data?
- How can I handle more complex reshaping scenarios, such as those involving hierarchical data with multiple levels?
- What are the most effective ways to visualize data after reshaping?

## Resources
- [Pandas Documentation - Reshaping and Pivot Tables](https://pandas.pydata.org/pandas-docs/stable/user_guide/reshaping.html)
- [Pandas Documentation - Pivot Function](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.pivot.html)
- [Pandas Documentation - Melt Function](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.melt.html)
- [Pandas Documentation - Stack and Unstack](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.stack.html)
- [Python for Data Analysis, 2nd Edition - Chapter 7](https://wesmckinney.com/book/)
