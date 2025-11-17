# Day 11: SQL Queries Using Python

## Today's Topics
- Introduction to SQL in Python
- SQLite and pandas integration
- Executing SQL queries on DataFrames
- Practical applications of SQL in Python

## Learning Journal

### Introduction to SQL in Python

Today I explored how to use SQL queries within Python, which combines the power of SQL for data querying with Python's flexibility and rich ecosystem. There are several ways to work with SQL in Python:

1. Using SQLite directly with the `sqlite3` module
2. Using pandas with the `read_sql` function
3. Using SQLAlchemy as an ORM (Object-Relational Mapper)
4. Using the `pandasql` library to run SQL queries directly on pandas DataFrames

I started by setting up the necessary libraries:

```python
import pandas as pd
import numpy as np
import sqlite3
from pandasql import sqldf
import matplotlib.pyplot as plt
import seaborn as sns

# For SQL queries on pandas DataFrames
pysqldf = lambda q: sqldf(q, globals())

# Set the style for our plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
```

### Working with SQLite in Python

First, I learned how to work with SQLite databases directly:

```python
# Create an in-memory SQLite database
conn = sqlite3.connect(':memory:')
cursor = conn.cursor()

# Create tables
cursor.execute('''
CREATE TABLE employees (
    employee_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department_id INTEGER,
    salary REAL,
    hire_date TEXT
)
''')

cursor.execute('''
CREATE TABLE departments (
    department_id INTEGER PRIMARY KEY,
    department_name TEXT NOT NULL,
    location TEXT
)
''')

# Insert data into departments
departments_data = [
    (1, 'HR', 'New York'),
    (2, 'IT', 'San Francisco'),
    (3, 'Finance', 'Chicago'),
    (4, 'Marketing', 'Boston'),
    (5, 'Operations', 'Seattle')
]

cursor.executemany('''
INSERT INTO departments VALUES (?, ?, ?)
''', departments_data)

# Insert data into employees
employees_data = [
    (101, 'John Smith', 1, 60000, '2020-06-15'),
    (102, 'Jane Doe', 2, 75000, '2019-03-20'),
    (103, 'Robert Johnson', 3, 65000, '2021-01-10'),
    (104, 'Lisa Wong', 2, 70000, '2018-11-05'),
    (105, 'Michael Brown', 4, 80000, '2020-09-30'),
    (106, 'Sarah Davis', 1, 55000, '2022-02-15'),
    (107, 'David Miller', 3, 68000, '2019-07-22'),
    (108, 'Emily Wilson', 4, 72000, '2021-05-18'),
    (109, 'James Taylor', 2, 85000, '2017-08-12'),
    (110, 'Jennifer Garcia', 5, 67000, '2020-11-03')
]

cursor.executemany('''
INSERT INTO employees VALUES (?, ?, ?, ?, ?)
''', employees_data)

# Commit the changes
conn.commit()

# Execute a simple query
cursor.execute('SELECT * FROM employees LIMIT 5')
print("Sample employee data:")
for row in cursor.fetchall():
    print(row)

# Execute a join query
cursor.execute('''
SELECT e.name, e.salary, d.department_name, d.location
FROM employees e
JOIN departments d ON e.department_id = d.department_id
ORDER BY e.salary DESC
''')

print("\nEmployees with department information:")
for row in cursor.fetchall():
    print(row)
```

### Using pandas with SQLite

Next, I learned how to use pandas with SQLite for more convenient data handling:

```python
# Read data from SQLite into pandas DataFrames
employees_df = pd.read_sql('SELECT * FROM employees', conn)
departments_df = pd.read_sql('SELECT * FROM departments', conn)

print("\nEmployees DataFrame:")
print(employees_df.head())

print("\nDepartments DataFrame:")
print(departments_df)

# Execute a SQL query and load results into a DataFrame
joined_data = pd.read_sql('''
SELECT e.name, e.salary, d.department_name, d.location
FROM employees e
JOIN departments d ON e.department_id = d.department_id
ORDER BY e.salary DESC
''', conn)

print("\nJoined Data:")
print(joined_data)

# Calculate aggregate statistics using SQL
stats_by_dept = pd.read_sql('''
SELECT d.department_name,
       COUNT(e.employee_id) as employee_count,
       AVG(e.salary) as avg_salary,
       MIN(e.salary) as min_salary,
       MAX(e.salary) as max_salary
FROM employees e
JOIN departments d ON e.department_id = d.department_id
GROUP BY d.department_name
ORDER BY avg_salary DESC
''', conn)

print("\nStatistics by Department:")
print(stats_by_dept)
```

### Writing DataFrames to SQLite

I learned how to write pandas DataFrames back to a SQLite database:

```python
# Create a new DataFrame
projects_df = pd.DataFrame({
    'project_id': [1, 2, 3, 4, 5],
    'project_name': ['Website Redesign', 'Mobile App', 'Database Migration', 'Cloud Integration', 'AI Implementation'],
    'department_id': [2, 2, 3, 2, 4],
    'budget': [50000, 75000, 60000, 90000, 120000],
    'start_date': ['2025-01-15', '2025-02-01', '2025-01-10', '2025-03-01', '2025-04-01']
})

print("\nProjects DataFrame:")
print(projects_df)

# Write the DataFrame to a new table in the SQLite database
projects_df.to_sql('projects', conn, if_exists='replace', index=False)

# Verify the data was written correctly
projects_from_db = pd.read_sql('SELECT * FROM projects', conn)
print("\nProjects from database:")
print(projects_from_db)
```

### Using pandasql for SQL Queries on DataFrames

I explored how to use the `pandasql` library to execute SQL queries directly on pandas DataFrames:

```python
# Create a more complex employees DataFrame
employees_extended = pd.DataFrame({
    'employee_id': range(101, 121),
    'name': [f'Employee {i}' for i in range(1, 21)],
    'department_id': np.random.choice([1, 2, 3, 4, 5], 20),
    'salary': np.random.randint(50000, 100000, 20),
    'hire_date': pd.date_range(start='2018-01-01', periods=20, freq='M').strftime('%Y-%m-%d').tolist(),
    'performance_score': np.random.randint(1, 6, 20),
    'bonus_eligible': np.random.choice([True, False], 20)
})

print("\nExtended Employees DataFrame:")
print(employees_extended.head())

# Simple query using pandasql
query1 = """
SELECT name, salary, performance_score
FROM employees_extended
WHERE salary > 80000
ORDER BY salary DESC
"""

high_salary_employees = pysqldf(query1)
print("\nHigh Salary Employees:")
print(high_salary_employees)

# More complex query with aggregation
query2 = """
SELECT department_id,
       COUNT(*) as employee_count,
       AVG(salary) as avg_salary,
       AVG(performance_score) as avg_performance
FROM employees_extended
GROUP BY department_id
HAVING COUNT(*) > 2
ORDER BY avg_salary DESC
"""

dept_stats = pysqldf(query2)
print("\nDepartment Statistics:")
print(dept_stats)

# Join query using pandasql
query3 = """
SELECT e.name, e.salary, d.department_name
FROM employees_extended e
JOIN departments_df d ON e.department_id = d.department_id
WHERE e.performance_score >= 4
ORDER BY e.salary DESC
"""

high_performers = pysqldf(query3)
print("\nHigh Performing Employees:")
print(high_performers)
```

### Advanced SQL Queries in Python

I practiced writing more advanced SQL queries:

```python
# Create a sales dataset
dates = pd.date_range(start='2025-01-01', periods=100)
products = ['Laptop', 'Tablet', 'Phone', 'Desktop', 'Monitor']
regions = ['North', 'South', 'East', 'West']

sales_data = []
for i in range(500):
    date = np.random.choice(dates)
    product = np.random.choice(products)
    region = np.random.choice(regions)
    units = np.random.randint(1, 10)
    price_per_unit = np.random.uniform(100, 1000)
    
    sales_data.append({
        'sale_id': i + 1,
        'date': date.strftime('%Y-%m-%d'),
        'product': product,
        'region': region,
        'units': units,
        'price_per_unit': round(price_per_unit, 2),
        'total_amount': round(units * price_per_unit, 2)
    })

sales_df = pd.DataFrame(sales_data)

# Write to SQLite
sales_df.to_sql('sales', conn, if_exists='replace', index=False)

print("\nSales Data Sample:")
print(sales_df.head())

# Window functions
query4 = """
SELECT s.date, s.product, s.total_amount,
       SUM(s.total_amount) OVER (PARTITION BY s.product ORDER BY s.date) as running_total,
       AVG(s.total_amount) OVER (PARTITION BY s.product) as product_avg
FROM sales s
ORDER BY s.product, s.date
LIMIT 10
"""

window_results = pd.read_sql(query4, conn)
print("\nWindow Function Results:")
print(window_results)

# Common Table Expressions (CTE)
query5 = """
WITH ProductTotals AS (
    SELECT product, SUM(total_amount) as total_sales
    FROM sales
    GROUP BY product
),
RegionTotals AS (
    SELECT region, SUM(total_amount) as total_sales
    FROM sales
    GROUP BY region
)
SELECT p.product, p.total_sales as product_sales,
       r.region, r.total_sales as region_sales
FROM ProductTotals p
CROSS JOIN RegionTotals r
ORDER BY p.total_sales DESC, r.total_sales DESC
LIMIT 10
"""

cte_results = pd.read_sql(query5, conn)
print("\nCTE Results:")
print(cte_results)

# Subqueries
query6 = """
SELECT s.product, s.region, SUM(s.total_amount) as region_product_sales,
       (SELECT SUM(total_amount) FROM sales WHERE product = s.product) as product_total_sales,
       (SUM(s.total_amount) / (SELECT SUM(total_amount) FROM sales WHERE product = s.product)) * 100 as percentage
FROM sales s
GROUP BY s.product, s.region
ORDER BY s.product, percentage DESC
"""

subquery_results = pd.read_sql(query6, conn)
print("\nSubquery Results:")
print(subquery_results.head(10))
```

### Practical Exercise: Sales Analysis with SQL

I applied what I learned to analyze sales data using SQL queries:

```python
# Create a more comprehensive sales database

# Create customers table
customers = pd.DataFrame({
    'customer_id': range(1, 21),
    'customer_name': [f'Customer {i}' for i in range(1, 21)],
    'segment': np.random.choice(['Consumer', 'Corporate', 'Home Office'], 20),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 20)
})

customers.to_sql('customers', conn, if_exists='replace', index=False)

# Create products table
products = pd.DataFrame({
    'product_id': range(1, 11),
    'product_name': ['Laptop', 'Desktop', 'Tablet', 'Phone', 'Monitor', 'Keyboard', 'Mouse', 'Printer', 'Scanner', 'Headphones'],
    'category': ['Computers', 'Computers', 'Electronics', 'Electronics', 'Accessories', 'Accessories', 'Accessories', 'Office', 'Office', 'Accessories'],
    'base_price': [1200, 1500, 600, 800, 300, 80, 50, 250, 200, 100]
})

products.to_sql('products', conn, if_exists='replace', index=False)

# Generate detailed sales data
np.random.seed(42)
sales_records = []

for i in range(1000):
    customer_id = np.random.randint(1, 21)
    product_id = np.random.randint(1, 11)
    date = pd.Timestamp('2025-01-01') + pd.Timedelta(days=np.random.randint(0, 365))
    quantity = np.random.randint(1, 6)
    discount = np.random.choice([0, 0.1, 0.2, 0.3], p=[0.7, 0.1, 0.1, 0.1])
    
    # Get base price from products
    base_price = products.loc[products['product_id'] == product_id, 'base_price'].values[0]
    
    # Calculate final price
    unit_price = base_price * (1 - discount)
    total_amount = unit_price * quantity
    
    sales_records.append({
        'order_id': i + 1,
        'customer_id': customer_id,
        'product_id': product_id,
        'order_date': date.strftime('%Y-%m-%d'),
        'quantity': quantity,
        'unit_price': round(unit_price, 2),
        'discount': discount,
        'total_amount': round(total_amount, 2)
    })

detailed_sales = pd.DataFrame(sales_records)
detailed_sales.to_sql('detailed_sales', conn, if_exists='replace', index=False)

print("\nDetailed Sales Database Created")
print(f"Customers: {len(customers)} records")
print(f"Products: {len(products)} records")
print(f"Sales: {len(detailed_sales)} records")

# Analysis 1: Monthly sales trend
query_monthly = """
SELECT strftime('%Y-%m', order_date) as month,
       SUM(total_amount) as monthly_sales
FROM detailed_sales
GROUP BY month
ORDER BY month
"""

monthly_sales = pd.read_sql(query_monthly, conn)
print("\nMonthly Sales Trend:")
print(monthly_sales)

# Analysis 2: Top selling products
query_top_products = """
SELECT p.product_name,
       SUM(s.quantity) as units_sold,
       SUM(s.total_amount) as total_sales,
       COUNT(DISTINCT s.customer_id) as unique_customers
FROM detailed_sales s
JOIN products p ON s.product_id = p.product_id
GROUP BY p.product_name
ORDER BY total_sales DESC
"""

top_products = pd.read_sql(query_top_products, conn)
print("\nTop Selling Products:")
print(top_products)

# Analysis 3: Sales by customer segment
query_segments = """
SELECT c.segment,
       COUNT(s.order_id) as order_count,
       SUM(s.total_amount) as total_sales,
       AVG(s.total_amount) as avg_order_value
FROM detailed_sales s
JOIN customers c ON s.customer_id = c.customer_id
GROUP BY c.segment
ORDER BY total_sales DESC
"""

segment_sales = pd.read_sql(query_segments, conn)
print("\nSales by Customer Segment:")
print(segment_sales)

# Analysis 4: Product category analysis
query_categories = """
SELECT p.category,
       COUNT(s.order_id) as order_count,
       SUM(s.quantity) as units_sold,
       SUM(s.total_amount) as total_sales,
       AVG(s.discount) as avg_discount
FROM detailed_sales s
JOIN products p ON s.product_id = p.product_id
GROUP BY p.category
ORDER BY total_sales DESC
"""

category_analysis = pd.read_sql(query_categories, conn)
print("\nProduct Category Analysis:")
print(category_analysis)

# Analysis 5: Customer purchase frequency
query_frequency = """
WITH customer_orders AS (
    SELECT customer_id, COUNT(DISTINCT order_id) as order_count
    FROM detailed_sales
    GROUP BY customer_id
)
SELECT order_count as purchase_frequency,
       COUNT(customer_id) as customer_count
FROM customer_orders
GROUP BY order_count
ORDER BY order_count
"""

purchase_frequency = pd.read_sql(query_frequency, conn)
print("\nCustomer Purchase Frequency:")
print(purchase_frequency)

# Analysis 6: Discount impact analysis
query_discount = """
SELECT 
    CASE 
        WHEN discount = 0 THEN 'No Discount'
        WHEN discount = 0.1 THEN '10% Discount'
        WHEN discount = 0.2 THEN '20% Discount'
        WHEN discount = 0.3 THEN '30% Discount'
        ELSE 'Other'
    END as discount_tier,
    COUNT(order_id) as order_count,
    SUM(quantity) as units_sold,
    SUM(total_amount) as total_sales,
    AVG(quantity) as avg_quantity_per_order
FROM detailed_sales
GROUP BY discount_tier
ORDER BY discount
"""

discount_analysis = pd.read_sql(query_discount, conn)
print("\nDiscount Impact Analysis:")
print(discount_analysis)

# Analysis 7: Regional performance with product breakdown
query_regional = """
WITH regional_sales AS (
    SELECT c.region,
           SUM(s.total_amount) as region_total
    FROM detailed_sales s
    JOIN customers c ON s.customer_id = c.customer_id
    GROUP BY c.region
)
SELECT c.region,
       p.category,
       SUM(s.total_amount) as category_sales,
       (SUM(s.total_amount) / r.region_total) * 100 as percentage_of_region
FROM detailed_sales s
JOIN customers c ON s.customer_id = c.customer_id
JOIN products p ON s.product_id = p.product_id
JOIN regional_sales r ON c.region = r.region
GROUP BY c.region, p.category
ORDER BY c.region, percentage_of_region DESC
"""

regional_performance = pd.read_sql(query_regional, conn)
print("\nRegional Performance with Product Breakdown:")
print(regional_performance)

# Close the connection
conn.close()
```

## Reflections

Today's exploration of SQL in Python has significantly enhanced my data analysis capabilities. The ability to combine the power of SQL for data querying with Python's rich ecosystem provides tremendous flexibility for data manipulation and analysis.

I found the integration between pandas and SQLite particularly useful. The `read_sql` function makes it easy to execute SQL queries and load the results directly into pandas DataFrames, while the `to_sql` method allows for seamless writing of DataFrames back to a database.

The `pandasql` library is also a powerful tool, enabling SQL queries directly on pandas DataFrames without the need for a separate database. This is especially useful for quick ad-hoc analyses or when working with multiple DataFrames.

Advanced SQL features like window functions, common table expressions (CTEs), and subqueries provide powerful tools for complex data analysis. These features allow for sophisticated calculations and data transformations that would be more cumbersome to implement using only pandas operations.

The practical exercise demonstrated how SQL can be used for comprehensive sales analysis. By creating a relational database with customers, products, and sales tables, I was able to perform various analyses including sales trends, product performance, customer segmentation, and discount impact analysis.

## Questions to Explore
- How does the performance of SQL queries in Python compare to equivalent pandas operations for large datasets?
- What are the best practices for optimizing SQL queries in Python?
- How can I integrate SQL with other Python libraries for more advanced analytics (e.g., machine learning)?
- What are the trade-offs between using a full-fledged database system versus in-memory solutions like SQLite?

## Resources
- [SQLite Documentation](https://www.sqlite.org/docs.html)
- [pandas Documentation - SQL](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#sql-queries)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [pandasql Documentation](https://github.com/yhat/pandasql)
- [Python for Data Analysis, 2nd Edition - Chapter 8](https://wesmckinney.com/book/)
