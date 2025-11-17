# Day 5: Boolean Filtering and Appending in Pandas

## Today's Topics
- Advanced Boolean filtering techniques
- Appending and combining DataFrames
- Chaining filtering operations
- Working with missing values in filters

## Learning Journal

### Advanced Boolean Filtering Techniques

Today I expanded my knowledge of boolean filtering in Pandas, exploring more complex scenarios and techniques. Boolean filtering is a powerful way to extract specific subsets of data based on logical conditions.

I started with a more complex DataFrame to work with:

```python
import pandas as pd
import numpy as np

# Create a sample DataFrame
np.random.seed(42)
data = {
    'Product': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B', 'C', 'A'],
    'Category': ['Electronics', 'Clothing', 'Electronics', 'Home', 'Clothing', 
                'Home', 'Electronics', 'Clothing', 'Home', 'Electronics'],
    'Price': np.random.randint(10, 100, 10),
    'Stock': np.random.randint(0, 50, 10),
    'Rating': np.round(np.random.uniform(1, 5, 10), 1),
    'OnSale': np.random.choice([True, False], 10)
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)
```

### Complex Boolean Filtering

I practiced creating more complex boolean filters by combining multiple conditions:

```python
# Find electronics with high ratings (> 4.0)
high_rated_electronics = df[(df['Category'] == 'Electronics') & (df['Rating'] > 4.0)]
print("\nHigh-rated electronics:")
print(high_rated_electronics)

# Find products that are either on sale OR have low stock (< 10)
sale_or_low_stock = df[(df['OnSale'] == True) | (df['Stock'] < 10)]
print("\nProducts on sale or with low stock:")
print(sale_or_low_stock)

# Find clothing items that are NOT on sale
clothing_not_on_sale = df[(df['Category'] == 'Clothing') & (df['OnSale'] == False)]
print("\nClothing items not on sale:")
print(clothing_not_on_sale)
```

### Using Boolean Masks as Variables

I learned that storing boolean masks as variables can make complex filtering more readable:

```python
# Create boolean masks
is_electronic = df['Category'] == 'Electronics'
is_on_sale = df['OnSale'] == True
has_good_rating = df['Rating'] >= 4.0
is_in_stock = df['Stock'] > 0

# Combine masks for complex filtering
good_deals = df[is_electronic & is_on_sale & has_good_rating & is_in_stock]
print("\nGood electronic deals (on sale, well-rated, in stock):")
print(good_deals)
```

### Working with Missing Values in Filters

I explored how to handle missing values (NaN) in boolean filters:

```python
# Create a DataFrame with some missing values
df_with_na = df.copy()
df_with_na.loc[2, 'Rating'] = np.nan
df_with_na.loc[5, 'Price'] = np.nan
df_with_na.loc[8, 'Stock'] = np.nan
print("\nDataFrame with missing values:")
print(df_with_na)

# Filter rows with missing values
rows_with_na = df_with_na[df_with_na.isna().any(axis=1)]
print("\nRows with any missing values:")
print(rows_with_na)

# Filter rows without missing values
complete_rows = df_with_na.dropna()
print("\nRows without any missing values:")
print(complete_rows)

# Filter based on a column that might have NaN values
# Note: comparison with NaN always returns False
high_rating = df_with_na[df_with_na['Rating'] > 4.0]
print("\nProducts with rating > 4.0 (NaN values excluded):")
print(high_rating)

# To include NaN values in a filter, use .isna()
missing_or_high_rating = df_with_na[(df_with_na['Rating'] > 4.0) | (df_with_na['Rating'].isna())]
print("\nProducts with rating > 4.0 OR missing rating:")
print(missing_or_high_rating)
```

### Appending DataFrames

I then learned about appending and combining DataFrames, which is useful for merging data from different sources:

```python
# Create two separate DataFrames
df1 = pd.DataFrame({
    'Product': ['D', 'E', 'F'],
    'Category': ['Electronics', 'Home', 'Clothing'],
    'Price': [85, 45, 60],
    'Stock': [12, 8, 20],
    'Rating': [4.2, 3.8, 4.5],
    'OnSale': [False, True, False]
})

df2 = pd.DataFrame({
    'Product': ['G', 'H'],
    'Category': ['Electronics', 'Home'],
    'Price': [120, 30],
    'Stock': [5, 15],
    'Rating': [4.8, 3.2],
    'OnSale': [True, False]
})

print("\nFirst additional DataFrame:")
print(df1)
print("\nSecond additional DataFrame:")
print(df2)

# Append df1 and df2 to the original df
# Note: pd.concat is the modern way to do this (append is deprecated)
combined_df = pd.concat([df, df1, df2], ignore_index=True)
print("\nCombined DataFrame:")
print(combined_df)
```

### Using Append Method (Legacy Approach)

I also learned about the older `append` method, which is now deprecated but still appears in many existing codebases:

```python
# Using the append method (deprecated but still useful to know)
# This is equivalent to pd.concat([df, df1])
appended_df = df.append(df1, ignore_index=True)
print("\nAppended DataFrame (using .append()):")
print(appended_df)

# Append multiple DataFrames
multi_append = df.append([df1, df2], ignore_index=True)
print("\nMultiple appended DataFrames:")
print(multi_append)
```

### Filtering After Appending

I practiced applying boolean filters to the combined DataFrame:

```python
# Find all electronics in the combined DataFrame
all_electronics = combined_df[combined_df['Category'] == 'Electronics']
print("\nAll electronics products:")
print(all_electronics)

# Find products with high stock levels across all categories
high_stock = combined_df[combined_df['Stock'] > 20]
print("\nProducts with high stock levels:")
print(high_stock)
```

### Practical Exercise: Sales Analysis

I applied what I learned to a practical sales analysis scenario:

```python
# Create a sales DataFrame
np.random.seed(42)
dates = pd.date_range('20250101', periods=15)
sales_data = pd.DataFrame({
    'Date': dates,
    'Product': np.random.choice(['A', 'B', 'C', 'D', 'E'], 15),
    'Units': np.random.randint(5, 50, 15),
    'Revenue': np.random.randint(100, 1000, 15),
    'Promotion': np.random.choice([True, False], 15)
})

print("\nSales Data:")
print(sales_data)

# Filter for high-revenue days
high_revenue = sales_data[sales_data['Revenue'] > 500]
print("\nHigh revenue days:")
print(high_revenue)

# Filter for days with promotions and calculate average revenue
promo_days = sales_data[sales_data['Promotion'] == True]
avg_promo_revenue = promo_days['Revenue'].mean()
print(f"\nAverage revenue on promotion days: ${avg_promo_revenue:.2f}")

# Filter for days without promotions and calculate average revenue
non_promo_days = sales_data[sales_data['Promotion'] == False]
avg_non_promo_revenue = non_promo_days['Revenue'].mean()
print(f"Average revenue on non-promotion days: ${avg_non_promo_revenue:.2f}")

# Calculate revenue lift from promotions
revenue_lift = ((avg_promo_revenue - avg_non_promo_revenue) / avg_non_promo_revenue) * 100
print(f"Promotion revenue lift: {revenue_lift:.1f}%")

# Find the best-selling product by total units
product_units = sales_data.groupby('Product')['Units'].sum()
best_product = product_units.idxmax()
print(f"\nBest-selling product: {best_product} with {product_units[best_product]} units sold")

# Create a new DataFrame with sales from the first week
first_week = sales_data[sales_data['Date'] < '2025-01-08']

# Create a new DataFrame with sales from the second week
second_week = sales_data[sales_data['Date'] >= '2025-01-08']

# Append a 'Week' column to each DataFrame
first_week['Week'] = 'Week 1'
second_week['Week'] = 'Week 2'

# Combine the DataFrames
weekly_sales = pd.concat([first_week, second_week])
print("\nSales data with week information:")
print(weekly_sales)

# Calculate average revenue by week
avg_by_week = weekly_sales.groupby('Week')['Revenue'].mean()
print("\nAverage revenue by week:")
print(avg_by_week)
```

## Reflections

Today's exploration of advanced boolean filtering and DataFrame appending has significantly expanded my data manipulation toolkit. The ability to create complex filters by combining boolean conditions allows for precise data selection, while appending provides a way to combine data from different sources.

I found that storing boolean masks as variables makes complex filtering operations more readable and maintainable. This approach also makes it easier to reuse filtering conditions across different operations.

Working with missing values in filters requires special attention, as comparisons with NaN values behave differently than regular comparisons. The `isna()` and `notna()` methods are essential tools for handling these cases.

For appending DataFrames, I learned that `pd.concat()` is the modern approach, offering more flexibility and clearer semantics than the deprecated `append()` method.

## Questions to Explore
- How can I optimize complex boolean filtering operations for large DataFrames?
- What are the performance implications of different approaches to handling missing values?
- How do `pd.concat()` and `append()` compare in terms of performance for large DataFrames?
- What are the best practices for maintaining data integrity when combining DataFrames from different sources?

## Resources
- [Pandas Documentation - Boolean Indexing](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#boolean-indexing)
- [Pandas Documentation - Concatenating Objects](https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html#concatenating-objects)
- [Pandas Documentation - Working with Missing Data](https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html)
- [Python for Data Analysis, 2nd Edition - Chapter 7](https://wesmckinney.com/book/)
