# Day 10: Pivot Tables and Grouping Functions in Python

## Today's Topics
- Creating and using pivot tables in Python
- Groupby operations for data aggregation
- Advanced grouping techniques
- Practical applications of pivot tables and grouping

## Learning Journal

### Introduction to Pivot Tables

Today I explored pivot tables in Python, which are powerful tools for summarizing and analyzing data. Pivot tables allow us to reorganize and aggregate data, similar to pivot tables in Excel but with more flexibility and power.

I started by creating a sample dataset to work with:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for our plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

# Create a sample sales dataset
np.random.seed(42)

# Generate data
data = {
    'Date': pd.date_range(start='2025-01-01', periods=100),
    'Product': np.random.choice(['Laptop', 'Tablet', 'Phone', 'Desktop'], 100),
    'Category': np.random.choice(['Electronics', 'Accessories'], 100),
    'Region': np.random.choice(['North', 'South', 'East', 'West'], 100),
    'Sales_Rep': np.random.choice(['John', 'Mary', 'Robert', 'Lisa', 'David'], 100),
    'Units': np.random.randint(1, 10, 100),
    'Unit_Price': np.random.uniform(100, 1000, 100).round(2)
}

# Create DataFrame
sales_df = pd.DataFrame(data)

# Add a Total_Sales column
sales_df['Total_Sales'] = sales_df['Units'] * sales_df['Unit_Price']

# Add a Month column for easier grouping
sales_df['Month'] = sales_df['Date'].dt.strftime('%B')
sales_df['Quarter'] = 'Q' + sales_df['Date'].dt.quarter.astype(str)

print("Sample of sales data:")
print(sales_df.head())
```

### Basic Pivot Tables

I learned how to create basic pivot tables using the `pivot_table()` function:

```python
# Basic pivot table: Sales by Region and Product
region_product_pivot = pd.pivot_table(
    sales_df,
    values='Total_Sales',
    index='Region',
    columns='Product',
    aggfunc='sum'
)

print("\nPivot Table: Total Sales by Region and Product")
print(region_product_pivot)

# Pivot table with multiple values
multi_value_pivot = pd.pivot_table(
    sales_df,
    values=['Total_Sales', 'Units'],
    index='Region',
    columns='Product',
    aggfunc='sum'
)

print("\nPivot Table with Multiple Values:")
print(multi_value_pivot)
```

### Advanced Pivot Table Features

I explored more advanced pivot table features:

```python
# Pivot table with multiple aggregation functions
multi_agg_pivot = pd.pivot_table(
    sales_df,
    values='Total_Sales',
    index='Region',
    columns='Product',
    aggfunc=['sum', 'mean', 'count']
)

print("\nPivot Table with Multiple Aggregation Functions:")
print(multi_agg_pivot)

# Pivot table with multiple indices
multi_index_pivot = pd.pivot_table(
    sales_df,
    values='Total_Sales',
    index=['Region', 'Category'],
    columns=['Product'],
    aggfunc='sum'
)

print("\nPivot Table with Multiple Indices:")
print(multi_index_pivot)

# Pivot table with multiple columns
multi_column_pivot = pd.pivot_table(
    sales_df,
    values='Total_Sales',
    index='Region',
    columns=['Quarter', 'Product'],
    aggfunc='sum'
)

print("\nPivot Table with Multiple Column Levels:")
print(multi_column_pivot)

# Adding totals to pivot table
pivot_with_margins = pd.pivot_table(
    sales_df,
    values='Total_Sales',
    index='Region',
    columns='Product',
    aggfunc='sum',
    margins=True,
    margins_name='Total'
)

print("\nPivot Table with Margins (Totals):")
print(pivot_with_margins)

# Filling missing values in pivot table
pivot_filled = pd.pivot_table(
    sales_df,
    values='Total_Sales',
    index='Region',
    columns='Product',
    aggfunc='sum',
    fill_value=0
)

print("\nPivot Table with Missing Values Filled:")
print(pivot_filled)
```

### Visualizing Pivot Tables

I learned how to visualize pivot tables for better insights:

```python
# Create a heatmap of the pivot table
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_filled, annot=True, fmt='.0f', cmap='YlGnBu')
plt.title('Total Sales by Region and Product')
plt.tight_layout()
# plt.savefig('pivot_heatmap.png')
# plt.show()

# Create a stacked bar chart from pivot table
pivot_filled.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('Total Sales by Region and Product')
plt.xlabel('Region')
plt.ylabel('Total Sales')
plt.legend(title='Product')
plt.tight_layout()
# plt.savefig('pivot_stacked_bar.png')
# plt.show()
```

### Introduction to Groupby Operations

Next, I explored the powerful `groupby()` function in Pandas, which allows for flexible data aggregation:

```python
# Basic groupby with single column
region_sales = sales_df.groupby('Region')['Total_Sales'].sum()
print("\nTotal Sales by Region:")
print(region_sales)

# Groupby with multiple columns
region_product_sales = sales_df.groupby(['Region', 'Product'])['Total_Sales'].sum()
print("\nTotal Sales by Region and Product:")
print(region_product_sales)

# Converting groupby result to DataFrame
region_product_df = region_product_sales.reset_index()
print("\nGroupby Result as DataFrame:")
print(region_product_df)
```

### Multiple Aggregations with Groupby

I learned how to perform multiple aggregations in a single groupby operation:

```python
# Multiple aggregations on a single column
region_stats = sales_df.groupby('Region')['Total_Sales'].agg(['sum', 'mean', 'count', 'min', 'max'])
print("\nMultiple Statistics by Region:")
print(region_stats)

# Multiple aggregations on different columns
multi_agg = sales_df.groupby('Region').agg({
    'Total_Sales': ['sum', 'mean'],
    'Units': ['sum', 'mean', 'max'],
    'Unit_Price': ['mean', 'min', 'max']
})

print("\nMultiple Aggregations on Different Columns:")
print(multi_agg)

# Custom aggregation functions
def range_calc(x):
    return x.max() - x.min()

custom_agg = sales_df.groupby('Region').agg({
    'Total_Sales': ['sum', range_calc],
    'Units': 'sum',
    'Unit_Price': lambda x: x.mean().round(2)
})

print("\nCustom Aggregation Functions:")
print(custom_agg)
```

### Groupby with Transformation and Filtering

I explored how to use groupby for transformation and filtering operations:

```python
# Transform: Calculate percentage of total sales
total_sales = sales_df['Total_Sales'].sum()
sales_df['Pct_of_Total'] = sales_df['Total_Sales'] / total_sales * 100

# Transform: Calculate percentage within each region
region_totals = sales_df.groupby('Region')['Total_Sales'].transform('sum')
sales_df['Pct_of_Region'] = sales_df['Total_Sales'] / region_totals * 100

print("\nSales DataFrame with Percentage Columns:")
print(sales_df[['Region', 'Product', 'Total_Sales', 'Pct_of_Total', 'Pct_of_Region']].head())

# Filter: Find regions with total sales above average
region_avg_sales = sales_df.groupby('Region')['Total_Sales'].mean()
high_performing_regions = region_avg_sales[region_avg_sales > region_avg_sales.mean()]
print("\nRegions with Above-Average Sales:")
print(high_performing_regions)

# Filter: Find products that sell well in all regions
product_region_count = sales_df.groupby('Product')['Region'].nunique()
products_in_all_regions = product_region_count[product_region_count == sales_df['Region'].nunique()]
print("\nProducts Sold in All Regions:")
print(products_in_all_regions)
```

### Advanced Groupby Techniques

I learned some advanced groupby techniques:

```python
# Groupby with hierarchical index
hierarchical = sales_df.groupby(['Quarter', 'Month', 'Region'])['Total_Sales'].sum()
print("\nHierarchical Groupby Result:")
print(hierarchical)

# Accessing specific groups
q1_data = hierarchical.loc['Q1']
print("\nQ1 Sales Data:")
print(q1_data)

# Groupby with categorical data
sales_df['Region_Cat'] = pd.Categorical(sales_df['Region'], 
                                       categories=['North', 'East', 'South', 'West'],
                                       ordered=True)
ordered_region_sales = sales_df.groupby('Region_Cat')['Total_Sales'].sum()
print("\nSales by Region (Ordered Categorically):")
print(ordered_region_sales)

# Groupby with time-based data
time_sales = sales_df.groupby(pd.Grouper(key='Date', freq='M'))['Total_Sales'].sum()
print("\nMonthly Sales:")
print(time_sales)
```

### Combining Pivot Tables and Groupby

I explored how pivot tables and groupby operations can be combined for powerful analyses:

```python
# First use groupby to calculate statistics
product_stats = sales_df.groupby(['Product', 'Category']).agg({
    'Total_Sales': ['sum', 'mean'],
    'Units': 'sum'
}).reset_index()

# Then create a pivot table from the result
product_pivot = pd.pivot_table(
    product_stats,
    values=('Total_Sales', 'sum'),
    index='Category',
    columns='Product'
)

print("\nPivot Table from Groupby Result:")
print(product_pivot)
```

### Practical Exercise: Sales Performance Analysis

I applied what I learned to analyze sales performance across different dimensions:

```python
# Create a more detailed sales dataset
np.random.seed(42)

# Generate dates for the whole year
dates = pd.date_range(start='2025-01-01', end='2025-12-31', freq='D')

# Create a list to store the data
sales_records = []

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

# Sales representatives with regions
sales_reps = {
    'John': 'North',
    'Mary': 'South',
    'Robert': 'East',
    'Lisa': 'West',
    'David': 'North',
    'Sarah': 'South',
    'Michael': 'East',
    'Emily': 'West'
}

# Generate sales records
for date in dates:
    # Generate more sales for weekdays, fewer for weekends
    num_sales = np.random.randint(5, 15) if date.weekday() < 5 else np.random.randint(2, 8)
    
    for _ in range(num_sales):
        product = np.random.choice(list(products.keys()))
        product_info = products[product]
        
        rep = np.random.choice(list(sales_reps.keys()))
        region = sales_reps[rep]
        
        # Add seasonality to certain products
        season_factor = 1.0
        if product in ['Laptop', 'Desktop'] and date.month in [8, 9]:  # Back to school
            season_factor = 1.5
        elif product in ['Phone', 'Tablet'] and date.month in [11, 12]:  # Holiday season
            season_factor = 1.8
        
        # Calculate units and price
        units = np.random.randint(1, 5)
        unit_price = np.random.uniform(*product_info['price_range']) * season_factor
        
        # Create the record
        record = {
            'Date': date,
            'Product': product,
            'Category': product_info['category'],
            'Region': region,
            'Sales_Rep': rep,
            'Units': units,
            'Unit_Price': round(unit_price, 2),
            'Total_Sales': round(units * unit_price, 2)
        }
        
        sales_records.append(record)

# Create the detailed sales DataFrame
detailed_sales = pd.DataFrame(sales_records)

# Add time-based columns
detailed_sales['Year'] = detailed_sales['Date'].dt.year
detailed_sales['Quarter'] = 'Q' + detailed_sales['Date'].dt.quarter.astype(str)
detailed_sales['Month'] = detailed_sales['Date'].dt.strftime('%B')
detailed_sales['Week'] = detailed_sales['Date'].dt.isocalendar().week
detailed_sales['Day_of_Week'] = detailed_sales['Date'].dt.day_name()

print("Detailed Sales Dataset Sample:")
print(detailed_sales.head())
print(f"Total Records: {len(detailed_sales)}")

# Analysis 1: Monthly Sales Trend
monthly_sales = detailed_sales.groupby(detailed_sales['Date'].dt.strftime('%Y-%m'))['Total_Sales'].sum()
print("\nMonthly Sales Trend:")
print(monthly_sales)

# Plot monthly sales trend
plt.figure(figsize=(12, 6))
monthly_sales.plot(kind='bar', color='skyblue')
plt.title('Monthly Sales Trend (2025)')
plt.xlabel('Month')
plt.ylabel('Total Sales ($)')
plt.xticks(rotation=45)
plt.tight_layout()
# plt.savefig('monthly_sales_trend.png')
# plt.show()

# Analysis 2: Product Category Performance by Quarter
category_quarter_pivot = pd.pivot_table(
    detailed_sales,
    values='Total_Sales',
    index='Category',
    columns='Quarter',
    aggfunc='sum',
    margins=True,
    margins_name='Total'
)

print("\nProduct Category Performance by Quarter:")
print(category_quarter_pivot)

# Analysis 3: Sales Representative Performance
rep_performance = detailed_sales.groupby('Sales_Rep').agg({
    'Total_Sales': 'sum',
    'Units': 'sum',
    'Date': pd.Series.nunique  # Number of days with sales
}).rename(columns={'Date': 'Active_Days'})

# Calculate average sales per day
rep_performance['Avg_Sales_Per_Day'] = rep_performance['Total_Sales'] / rep_performance['Active_Days']

# Sort by total sales
rep_performance = rep_performance.sort_values('Total_Sales', ascending=False)

print("\nSales Representative Performance:")
print(rep_performance)

# Analysis 4: Day of Week Analysis
day_of_week_sales = detailed_sales.groupby('Day_of_Week')['Total_Sales'].sum()

# Reorder days of week
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_of_week_sales = day_of_week_sales.reindex(day_order)

print("\nSales by Day of Week:")
print(day_of_week_sales)

# Analysis 5: Product Performance Heatmap
product_region_pivot = pd.pivot_table(
    detailed_sales,
    values='Total_Sales',
    index='Product',
    columns='Region',
    aggfunc='sum',
    fill_value=0
)

print("\nProduct Performance by Region:")
print(product_region_pivot)

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(product_region_pivot, annot=True, fmt='.0f', cmap='YlGnBu')
plt.title('Product Performance by Region')
plt.tight_layout()
# plt.savefig('product_region_heatmap.png')
# plt.show()

# Analysis 6: Top Products by Month
top_products_by_month = detailed_sales.groupby(['Month', 'Product'])['Total_Sales'].sum().reset_index()
top_products_by_month = top_products_by_month.sort_values(['Month', 'Total_Sales'], ascending=[True, False])

# Get the top product for each month
top_product_per_month = top_products_by_month.groupby('Month').first()
print("\nTop Selling Product by Month:")
print(top_product_per_month[['Product', 'Total_Sales']])

# Analysis 7: Sales Rep Performance by Product Category
rep_category_pivot = pd.pivot_table(
    detailed_sales,
    values='Total_Sales',
    index='Sales_Rep',
    columns='Category',
    aggfunc='sum',
    fill_value=0
)

# Calculate percentage of sales by category for each rep
rep_category_pct = rep_category_pivot.div(rep_category_pivot.sum(axis=1), axis=0) * 100

print("\nSales Rep Performance by Product Category (%):")
print(rep_category_pct.round(1))
```

## Reflections

Today's exploration of pivot tables and groupby operations has significantly enhanced my data analysis capabilities in Python. These tools provide powerful ways to summarize, aggregate, and transform data, making it easier to extract meaningful insights.

I found pivot tables particularly useful for creating cross-tabulations and summarizing data across multiple dimensions. The ability to specify different aggregation functions, handle missing values, and include totals makes pivot tables in Pandas even more powerful than their Excel counterparts.

The groupby operation is incredibly versatile, allowing for everything from simple aggregations to complex transformations and filtering. I especially appreciated the flexibility of being able to apply different aggregation functions to different columns and the ability to use custom aggregation functions.

The practical exercise demonstrated how these techniques can be applied to real-world business scenarios. By analyzing sales data across different dimensions (time, product, region, sales rep), I was able to identify trends, top performers, and seasonal patterns that would be valuable for business decision-making.

I also learned that combining pivot tables and groupby operations can lead to even more powerful analyses. For example, using groupby to calculate complex statistics and then using pivot_table to reshape the results for better visualization.

## Questions to Explore
- How do pivot tables and groupby operations scale with very large datasets?
- What are the best practices for optimizing memory usage when working with large pivot tables?
- How can I incorporate more advanced statistical analyses into groupby operations?
- What are the most effective ways to visualize pivot table results for different types of analyses?

## Resources
- [Pandas Documentation - Pivot Tables](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.pivot_table.html)
- [Pandas Documentation - GroupBy](https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html)
- [Python for Data Analysis, 2nd Edition - Chapter 10](https://wesmckinney.com/book/)
- [Towards Data Science - Advanced Pivot Table Techniques in Python](https://towardsdatascience.com/advanced-pivot-table-techniques-in-python-3f5d32539d1c)
