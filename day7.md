# Day 7: Advanced If-Else Statements and Removing Duplicates

## Today's Topics
- Advanced if-else statement techniques
- Identifying and removing duplicates in Pandas
- Working with duplicate data
- Practical applications of deduplication

## Learning Journal

### Advanced If-Else Statement Techniques

Today I explored advanced techniques for using if-else statements in Python, particularly in the context of data analysis.

#### Nested If-Else with Multiple Conditions

```python
import pandas as pd
import numpy as np

# Complex decision-making with nested if-else
def classify_product(category, price, stock):
    if category == 'Electronics':
        if price > 500:
            if stock > 10:
                return 'Premium Electronics - Well Stocked'
            else:
                return 'Premium Electronics - Limited Stock'
        else:
            if stock > 20:
                return 'Standard Electronics - Well Stocked'
            else:
                return 'Standard Electronics - Limited Stock'
    elif category == 'Clothing':
        if price > 50:
            return 'High-end Apparel'
        else:
            return 'Budget Apparel'
    else:
        return 'Other Category'

# Test the function
print(classify_product('Electronics', 600, 15))  # Premium Electronics - Well Stocked
print(classify_product('Electronics', 600, 5))   # Premium Electronics - Limited Stock
print(classify_product('Electronics', 300, 25))  # Standard Electronics - Well Stocked
print(classify_product('Clothing', 75, 10))      # High-end Apparel
```

#### Using Dictionaries Instead of Multiple If-Else

I learned that dictionaries can be a cleaner alternative to multiple if-else statements:

```python
# Using dictionaries instead of multiple if-else
def get_tax_rate_if_else(state):
    if state == 'CA':
        return 0.0725
    elif state == 'NY':
        return 0.045
    elif state == 'TX':
        return 0.0625
    elif state == 'FL':
        return 0.06
    else:
        return 0.05  # Default rate

# Same logic using a dictionary
tax_rates = {
    'CA': 0.0725,
    'NY': 0.045,
    'TX': 0.0625,
    'FL': 0.06
}

def get_tax_rate_dict(state):
    return tax_rates.get(state, 0.05)  # Default 0.05 if state not found

# Test both functions
states = ['CA', 'NY', 'TX', 'FL', 'WA']
print("\nTax rates using if-else:")
for state in states:
    print(f"{state}: {get_tax_rate_if_else(state)}")

print("\nTax rates using dictionary:")
for state in states:
    print(f"{state}: {get_tax_rate_dict(state)}")
```

#### Using If-Else in List Comprehensions

I discovered how to use conditional logic within list comprehensions:

```python
# Regular if-else
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_odd = []
for num in numbers:
    if num % 2 == 0:
        even_odd.append('Even')
    else:
        even_odd.append('Odd')
print("\nEven/Odd using for loop:", even_odd)

# Using if-else in list comprehension
even_odd_comp = ['Even' if num % 2 == 0 else 'Odd' for num in numbers]
print("Even/Odd using list comprehension:", even_odd_comp)

# More complex example with multiple conditions
grades = [85, 92, 78, 65, 98, 72]
letter_grades = ['A' if g >= 90 else 'B' if g >= 80 else 'C' if g >= 70 else 'D' if g >= 60 else 'F' for g in grades]
print("\nLetter grades:", letter_grades)
```

#### Conditional Function Arguments

I learned how to use conditional logic when passing arguments to functions:

```python
# Conditional function arguments
def analyze_data(data, include_outliers=True):
    if include_outliers:
        print("Analyzing all data points")
        return data.mean(), data.std()
    else:
        print("Analyzing data without outliers")
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        filter_data = data[(data >= q1 - 1.5 * iqr) & (data <= q3 + 1.5 * iqr)]
        return filter_data.mean(), filter_data.std()

# Generate sample data with outliers
np.random.seed(42)
data = pd.Series(np.random.normal(100, 15, 100).tolist() + [200, 210, 50, 30])

# Analyze with and without outliers
with_outliers = analyze_data(data)
without_outliers = analyze_data(data, include_outliers=False)

print(f"With outliers: Mean = {with_outliers[0]:.2f}, Std = {with_outliers[1]:.2f}")
print(f"Without outliers: Mean = {without_outliers[0]:.2f}, Std = {without_outliers[1]:.2f}")
```

### Identifying and Removing Duplicates in Pandas

Next, I focused on techniques for identifying and handling duplicate data in Pandas:

```python
# Create a DataFrame with duplicate rows
data = {
    'Customer_ID': [101, 102, 101, 103, 104, 102, 105],
    'Name': ['John Smith', 'Jane Doe', 'John Smith', 'Bob Johnson', 'Alice Brown', 'Jane Doe', 'Charlie Davis'],
    'Email': ['john@example.com', 'jane@example.com', 'john@example.com', 'bob@example.com', 'alice@example.com', 'jane@example.com', 'charlie@example.com'],
    'Purchase_Amount': [150, 200, 150, 75, 300, 200, 125]
}

df = pd.DataFrame(data)
print("\nOriginal DataFrame with duplicates:")
print(df)
```

#### Identifying Duplicates

```python
# Check for duplicate rows
duplicates = df.duplicated()
print("\nDuplicate rows (boolean mask):")
print(duplicates)

# Find duplicate rows
duplicate_rows = df[df.duplicated()]
print("\nDuplicate rows:")
print(duplicate_rows)

# Find all duplicates (including first occurrence)
all_duplicates = df[df.duplicated(keep=False)]
print("\nAll duplicate rows (including first occurrences):")
print(all_duplicates)

# Check for duplicates based on specific columns
email_duplicates = df[df.duplicated(subset=['Email'], keep=False)]
print("\nRows with duplicate emails:")
print(email_duplicates)
```

#### Removing Duplicates

```python
# Remove duplicate rows (keep first occurrence)
df_no_dupes = df.drop_duplicates()
print("\nDataFrame with duplicates removed (keeping first occurrence):")
print(df_no_dupes)

# Remove duplicates based on specific columns
df_unique_customers = df.drop_duplicates(subset=['Customer_ID'])
print("\nDataFrame with unique Customer_IDs (keeping first occurrence):")
print(df_unique_customers)

# Remove duplicates keeping last occurrence
df_last = df.drop_duplicates(keep='last')
print("\nDataFrame with duplicates removed (keeping last occurrence):")
print(df_last)

# Remove duplicates ignoring all occurrences
df_no_all = df.drop_duplicates(keep=False)
print("\nDataFrame with all instances of duplicates removed:")
print(df_no_all)
```

#### Working with Partial Duplicates

I explored how to handle cases where only some columns have duplicate values:

```python
# Create a DataFrame with partial duplicates
orders = pd.DataFrame({
    'Order_ID': [1001, 1002, 1003, 1004, 1005],
    'Customer_ID': [101, 101, 102, 103, 101],
    'Product': ['Laptop', 'Mouse', 'Monitor', 'Keyboard', 'Headphones'],
    'Quantity': [1, 2, 1, 3, 1],
    'Price': [1200, 25, 300, 50, 80]
})

print("\nCustomer orders:")
print(orders)

# Find customers with multiple orders
customer_order_counts = orders['Customer_ID'].value_counts()
print("\nNumber of orders per customer:")
print(customer_order_counts)

multiple_orders = customer_order_counts[customer_order_counts > 1].index
customers_with_multiple_orders = orders[orders['Customer_ID'].isin(multiple_orders)]
print("\nCustomers with multiple orders:")
print(customers_with_multiple_orders)

# Calculate total spent per customer
customer_totals = orders.groupby('Customer_ID').agg({
    'Order_ID': 'count',
    'Price': 'sum'
}).rename(columns={'Order_ID': 'Number_of_Orders', 'Price': 'Total_Spent'})

print("\nCustomer order summary:")
print(customer_totals)
```

### Practical Exercise: Customer Data Deduplication

I applied what I learned to a practical customer data deduplication scenario:

```python
# Create a messy customer database with various duplicates
customer_data = pd.DataFrame({
    'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'First_Name': ['John', 'Jane', 'John', 'Robert', 'Sarah', 'Jane', 'Michael', 'John', 'Sarah', 'Robert'],
    'Last_Name': ['Smith', 'Doe', 'Smith', 'Johnson', 'Williams', 'Doe', 'Brown', 'Smith', 'Williams', 'JOHNSON'],
    'Email': ['john.smith@example.com', 'jane.doe@example.com', 'johnsmith@example.com', 
              'r.johnson@example.com', 'sarah.w@example.com', 'jane.doe@example.com',
              'michael.b@example.com', 'john.smith@example.com', 's.williams@example.com', 'r.johnson@example.com'],
    'Phone': ['555-1234', '555-5678', '555-1234', '555-9012', '555-3456', 
              '555-5678', '555-7890', '555-1235', '555-3456', '555-9012'],
    'Address': ['123 Main St', '456 Oak Ave', '123 Main St', '789 Pine Rd', '101 Maple Dr',
                '456 Oak Avenue', '202 Elm Blvd', '123 Main Street', '101 Maple Drive', '789 Pine Road']
})

print("\nOriginal messy customer database:")
print(customer_data)

# Step 1: Standardize data for better duplicate detection
# Standardize names (convert to lowercase)
customer_data['First_Name'] = customer_data['First_Name'].str.lower()
customer_data['Last_Name'] = customer_data['Last_Name'].str.lower()

# Standardize addresses (convert to lowercase)
customer_data['Address'] = customer_data['Address'].str.lower()

# Replace common variations
customer_data['Address'] = customer_data['Address'].str.replace(' street', ' st', regex=False)
customer_data['Address'] = customer_data['Address'].str.replace(' avenue', ' ave', regex=False)
customer_data['Address'] = customer_data['Address'].str.replace(' road', ' rd', regex=False)
customer_data['Address'] = customer_data['Address'].str.replace(' drive', ' dr', regex=False)
customer_data['Address'] = customer_data['Address'].str.replace(' boulevard', ' blvd', regex=False)

print("\nStandardized customer data:")
print(customer_data)

# Step 2: Find exact duplicates
exact_dupes = customer_data[customer_data.duplicated(keep=False)]
print("\nExact duplicates after standardization:")
print(exact_dupes)

# Step 3: Find potential duplicates based on email
email_dupes = customer_data[customer_data.duplicated(subset=['Email'], keep=False)]
print("\nPotential duplicates based on email:")
print(email_dupes)

# Step 4: Find potential duplicates based on name and phone
name_phone_dupes = customer_data[customer_data.duplicated(subset=['First_Name', 'Last_Name', 'Phone'], keep=False)]
print("\nPotential duplicates based on name and phone:")
print(name_phone_dupes)

# Step 5: Find potential duplicates based on name and address
name_address_dupes = customer_data[customer_data.duplicated(subset=['First_Name', 'Last_Name', 'Address'], keep=False)]
print("\nPotential duplicates based on name and address:")
print(name_address_dupes)

# Step 6: Create a clean customer database
# For this example, we'll deduplicate based on email as the primary key
clean_customers = customer_data.drop_duplicates(subset=['Email'])
print("\nClean customer database (deduplicated by email):")
print(clean_customers)

# Step 7: Create a report of merged records
# Group by email and aggregate other fields
merged_records = customer_data.groupby('Email').agg({
    'ID': lambda x: ', '.join(x.astype(str)),
    'First_Name': 'first',
    'Last_Name': 'first',
    'Phone': lambda x: ', '.join(x.unique()),
    'Address': lambda x: ', '.join(x.unique())
}).reset_index()

print("\nMerged customer records:")
print(merged_records)

# Step 8: Calculate deduplication statistics
original_count = len(customer_data)
deduplicated_count = len(clean_customers)
duplicate_count = original_count - deduplicated_count
duplicate_percentage = (duplicate_count / original_count) * 100

print(f"\nDeduplication Statistics:")
print(f"Original records: {original_count}")
print(f"Unique records: {deduplicated_count}")
print(f"Duplicate records: {duplicate_count}")
print(f"Duplicate percentage: {duplicate_percentage:.2f}%")
```

## Reflections

Today's exploration of advanced if-else techniques and duplicate handling has been incredibly valuable for my data analysis toolkit. The ability to write clean, efficient conditional logic is essential for implementing business rules and making data-driven decisions.

I found the dictionary-based approach to be a cleaner alternative to nested if-else statements in many cases, especially when mapping inputs to outputs. List comprehensions with conditional logic are also powerful for concise data transformations.

Handling duplicates in data is a critical skill for data cleaning and preparation. I learned that duplication can occur at different levels (exact duplicates, partial duplicates, or fuzzy duplicates), and Pandas provides flexible tools for identifying and removing them. The choice of which duplicates to keep (first, last, or none) depends on the specific business context.

The customer deduplication exercise was particularly insightful, as it demonstrated a real-world scenario where standardization and multiple criteria are needed to properly identify and merge duplicate records.

## Questions to Explore
- How can I implement more sophisticated fuzzy matching algorithms for identifying near-duplicate records?
- What are the performance implications of different deduplication strategies for very large datasets?
- How can I automate the process of identifying the best columns to use for deduplication?
- What are the best practices for maintaining data integrity when merging duplicate records?

## Resources
- [Pandas Documentation - Duplicate Data](https://pandas.pydata.org/pandas-docs/stable/user_guide/duplicates.html)
- [Python Documentation - Control Flow Tools](https://docs.python.org/3/tutorial/controlflow.html)
- [Real Python - Conditional Statements in Python](https://realpython.com/python-conditional-statements/)
- [Data Cleaning with Python and Pandas](https://towardsdatascience.com/data-cleaning-with-python-and-pandas-detecting-missing-values-3e9c6ebcf78b)
