# Day 3: Data Frame Filtering in Python

## Today's Topics
- Concepts of DataFrame filtering
- Boolean filtering
- Filtering with conditions
- Basic data selection techniques

## Learning Journal

### DataFrame Filtering Concepts

Today I focused on filtering data in Pandas DataFrames, which is a fundamental skill for data analysis. Filtering allows us to extract specific subsets of data based on conditions, making it easier to focus on relevant information.

I started by creating a sample DataFrame to work with:

```python
import pandas as pd
import numpy as np

# Create a sample DataFrame
data = {
    'Name': ['John', 'Sarah', 'Mike', 'Lisa', 'Tom', 'Emma', 'David', 'Anna'],
    'Age': [28, 34, 29, 42, 35, 31, 37, 25],
    'City': ['New York', 'Boston', 'New York', 'Chicago', 'Boston', 'Chicago', 'New York', 'Boston'],
    'Salary': [65000, 72000, 59000, 81000, 76000, 68000, 90000, 62000],
    'Experience': [3, 6, 2, 9, 7, 4, 8, 1]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)
```

### Boolean Filtering

One of the most powerful aspects of Pandas is the ability to use boolean expressions for filtering:

```python
# Filter employees from New York
ny_employees = df[df['City'] == 'New York']
print("\nEmployees from New York:")
print(ny_employees)

# Filter employees with age greater than 30
senior_employees = df[df['Age'] > 30]
print("\nEmployees older than 30:")
print(senior_employees)

# Filter employees with salary between 60000 and 75000
mid_salary = df[(df['Salary'] >= 60000) & (df['Salary'] <= 75000)]
print("\nEmployees with salary between 60000 and 75000:")
print(mid_salary)
```

I learned that the key to boolean filtering is the use of comparison operators (`==`, `>`, `<`, `>=`, `<=`, `!=`) to create boolean masks, which are then applied to the DataFrame.

### Combining Multiple Conditions

I practiced combining multiple filtering conditions using logical operators:

```python
# AND condition: Employees from Boston with more than 5 years of experience
boston_experienced = df[(df['City'] == 'Boston') & (df['Experience'] > 5)]
print("\nExperienced employees from Boston:")
print(boston_experienced)

# OR condition: Employees from Chicago OR with salary > 80000
chicago_or_high_salary = df[(df['City'] == 'Chicago') | (df['Salary'] > 80000)]
print("\nEmployees from Chicago OR with high salary:")
print(chicago_or_high_salary)

# NOT condition: Employees not from New York
not_ny = df[df['City'] != 'New York']
print("\nEmployees not from New York:")
print(not_ny)
```

### Using .query() Method

I discovered the `.query()` method, which provides a more readable syntax for filtering:

```python
# Using query method
young_chicago = df.query('Age < 35 and City == "Chicago"')
print("\nYoung employees from Chicago:")
print(young_chicago)

# More complex query
high_exp_or_salary = df.query('Experience > 5 or Salary > 70000')
print("\nHighly experienced or well-paid employees:")
print(high_exp_or_salary)
```

### Using .isin() for Multiple Values

When I needed to filter based on multiple possible values, I found the `.isin()` method very useful:

```python
# Filter employees from either Boston or Chicago
east_coast = df[df['City'].isin(['Boston', 'Chicago'])]
print("\nEmployees from Boston or Chicago:")
print(east_coast)

# Filter employees NOT in a specific age group
not_middle_age = df[~df['Age'].isin(range(30, 36))]
print("\nEmployees not in their early 30s:")
print(not_middle_age)
```

### String Filtering

I also practiced filtering based on string patterns:

```python
# Filter names that start with 'J'
j_names = df[df['Name'].str.startswith('J')]
print("\nEmployees whose names start with J:")
print(j_names)

# Filter names that contain 'a'
contains_a = df[df['Name'].str.contains('a')]
print("\nEmployees whose names contain 'a':")
print(contains_a)
```

## Practical Exercise

I combined various filtering techniques to answer a business question:

```python
# Business question: Find high-value employees (high experience, good salary)
# These are employees with above-average experience AND above-average salary
avg_experience = df['Experience'].mean()
avg_salary = df['Salary'].mean()

high_value_employees = df[(df['Experience'] > avg_experience) & 
                          (df['Salary'] > avg_salary)]

print("\nHigh-value employees (above average in both experience and salary):")
print(high_value_employees)

# Calculate what percentage of each city's employees are "high-value"
city_counts = df.groupby('City').size()
high_value_by_city = high_value_employees.groupby('City').size()

percentage = (high_value_by_city / city_counts * 100).fillna(0)
print("\nPercentage of high-value employees by city:")
print(percentage)
```

## Reflections

Today's learning about DataFrame filtering has been extremely valuable. I can now extract specific subsets of data based on various conditions, which is essential for data analysis and exploration. The boolean filtering approach in Pandas is intuitive and powerful, allowing for complex queries to be expressed clearly.

I found that combining multiple conditions requires careful attention to parentheses and the use of `&` and `|` operators instead of the Python keywords `and` and `or` when working with DataFrame boolean masks.

Tomorrow, I'll be diving deeper into more advanced filtering techniques using `loc` and `iloc`, which will give me even more precise control over data selection.

## Questions to Explore
- How does filtering performance scale with very large DataFrames?
- What are the best practices for complex filtering operations?
- How can I optimize my filtering code for better readability and performance?

## Resources
- [Pandas Documentation - Indexing and Selecting Data](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html)
- [10 Minutes to Pandas - Selection](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html#selection)
- [Python for Data Analysis, 2nd Edition - Chapter 5](https://wesmckinney.com/book/)
