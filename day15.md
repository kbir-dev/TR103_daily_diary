# Day 15: Additional Functions in Python

## Today's Topics
- Advanced Python functions
- Lambda functions
- Map, filter, and reduce
- List comprehensions and generator expressions
- Decorators and context managers

## Learning Journal

### Introduction to Advanced Python Functions

Today I explored advanced function concepts in Python that are particularly useful for data analysis and manipulation. These techniques help write more concise, readable, and efficient code.

I started by setting up the necessary libraries:

```python
import pandas as pd
import numpy as np
from functools import reduce
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for our plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
```

### Lambda Functions

I began by exploring lambda functions, which are small anonymous functions defined with the `lambda` keyword:

```python
# Basic lambda function
square = lambda x: x**2
print("Square of 5:", square(5))

# Lambda with multiple arguments
multiply = lambda x, y: x * y
print("5 × 7 =", multiply(5, 7))

# Lambda with conditional expression
is_even = lambda x: True if x % 2 == 0 else False
print("Is 6 even?", is_even(6))
print("Is 7 even?", is_even(7))

# Using lambda with sorting
students = [
    {'name': 'Alice', 'grade': 85},
    {'name': 'Bob', 'grade': 92},
    {'name': 'Charlie', 'grade': 78},
    {'name': 'David', 'grade': 95}
]

# Sort by grade
students_sorted = sorted(students, key=lambda student: student['grade'], reverse=True)
print("\nStudents sorted by grade:")
for student in students_sorted:
    print(f"{student['name']}: {student['grade']}")
```

### Map Function

Next, I learned about the `map` function, which applies a function to each item in an iterable:

```python
# Basic map example
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
print("\nSquared numbers:", squared)

# Map with multiple iterables
list1 = [1, 2, 3]
list2 = [4, 5, 6]
added = list(map(lambda x, y: x + y, list1, list2))
print("Element-wise addition:", added)

# Map with a regular function
def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

temperatures_c = [0, 10, 20, 30, 40]
temperatures_f = list(map(celsius_to_fahrenheit, temperatures_c))
print("\nTemperatures in Fahrenheit:", temperatures_f)

# Map with a list of strings
names = ['alice', 'bob', 'charlie', 'david']
capitalized = list(map(str.capitalize, names))
print("Capitalized names:", capitalized)

# Map with pandas Series
series = pd.Series([1, 2, 3, 4, 5])
squared_series = series.map(lambda x: x**2)
print("\nSquared pandas Series:")
print(squared_series)
```

### Filter Function

I explored the `filter` function, which constructs an iterator from elements that satisfy a condition:

```python
# Basic filter example
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
print("\nEven numbers:", even_numbers)

# Filter with a regular function
def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

numbers = range(1, 30)
primes = list(filter(is_prime, numbers))
print("Prime numbers up to 30:", primes)

# Filter with strings
words = ['apple', 'banana', 'cherry', 'date', 'elderberry', 'fig']
long_words = list(filter(lambda word: len(word) > 5, words))
print("\nWords longer than 5 characters:", long_words)

# Filter with pandas DataFrame
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'age': [25, 30, 35, 40, 45],
    'city': ['New York', 'Boston', 'Chicago', 'Boston', 'Seattle']
})

boston_residents = df[df['city'] == 'Boston']
print("\nBoston residents:")
print(boston_residents)
```

### Reduce Function

I learned about the `reduce` function from the `functools` module, which applies a function cumulatively to the items of an iterable:

```python
# Basic reduce example
numbers = [1, 2, 3, 4, 5]
product = reduce(lambda x, y: x * y, numbers)
print("\nProduct of numbers:", product)

# Reduce for finding maximum
max_value = reduce(lambda x, y: x if x > y else y, numbers)
print("Maximum value:", max_value)

# More complex reduce example
sentences = [
    "Python is a programming language",
    "It is easy to learn",
    "It has many useful libraries"
]

# Count total words across all sentences
word_count = reduce(lambda count, sentence: count + len(sentence.split()), sentences, 0)
print("\nTotal word count:", word_count)
```

### List Comprehensions and Generator Expressions

I explored list comprehensions and generator expressions, which provide a concise way to create lists and generators:

```python
# Basic list comprehension
numbers = [1, 2, 3, 4, 5]
squared = [x**2 for x in numbers]
print("\nSquared numbers using list comprehension:", squared)

# List comprehension with condition
even_squares = [x**2 for x in numbers if x % 2 == 0]
print("Squares of even numbers:", even_squares)

# Nested list comprehension
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [x for row in matrix for x in row]
print("Flattened matrix:", flattened)

# List comprehension vs. map and filter
# Using map and filter
even_squares_map_filter = list(map(lambda x: x**2, filter(lambda x: x % 2 == 0, numbers)))
print("\nSquares of even numbers using map and filter:", even_squares_map_filter)

# Generator expression
squared_gen = (x**2 for x in numbers)
print("Generator object:", squared_gen)
print("Values from generator:", list(squared_gen))

# Memory efficiency of generator expressions
def get_memory_usage(obj):
    import sys
    return sys.getsizeof(obj)

large_list = [i for i in range(10000)]
large_gen = (i for i in range(10000))

print(f"\nMemory usage of list: {get_memory_usage(large_list)} bytes")
print(f"Memory usage of generator: {get_memory_usage(large_gen)} bytes")
```

### Decorators

I learned about decorators, which are a powerful way to modify or extend the behavior of functions:

```python
# Basic decorator
def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to run")
        return result
    return wrapper

@timer_decorator
def slow_function(n):
    """A deliberately slow function for demonstration"""
    result = 0
    for i in range(n):
        result += i
    return result

print("\nRunning slow function:")
result = slow_function(1000000)
print(f"Result: {result}")

# Decorator with arguments
def repeat(n):
    def decorator(func):
        def wrapper(*args, **kwargs):
            results = []
            for _ in range(n):
                results.append(func(*args, **kwargs))
            return results
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    return f"Hello, {name}!"

print("\nRepeated greetings:", greet("World"))

# Preserving function metadata with functools.wraps
from functools import wraps

def logger(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling function: {func.__name__}")
        result = func(*args, **kwargs)
        print(f"Function {func.__name__} returned: {result}")
        return result
    return wrapper

@logger
def add(a, b):
    """Add two numbers and return the result."""
    return a + b

print("\nUsing logger decorator:")
sum_result = add(3, 5)
print("Function name:", add.__name__)
print("Function docstring:", add.__doc__)
```

### Context Managers

I explored context managers, which provide a way to allocate and release resources precisely when needed:

```python
# Basic context manager using 'with'
print("\nUsing file context manager:")
with open('temp_file.txt', 'w') as f:
    f.write('Hello, World!')
print("File written and closed automatically")

# Custom context manager using class
class Timer:
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        print(f"Elapsed time: {self.elapsed:.4f} seconds")

print("\nUsing custom Timer context manager:")
with Timer():
    # Some time-consuming operation
    sum([i**2 for i in range(1000000)])

# Context manager using contextlib
from contextlib import contextmanager

@contextmanager
def temporary_attribute(obj, attr_name, attr_value):
    original_value = getattr(obj, attr_name, None)
    setattr(obj, attr_name, attr_value)
    try:
        yield
    finally:
        if original_value is None:
            delattr(obj, attr_name)
        else:
            setattr(obj, attr_name, original_value)

class Example:
    def __init__(self):
        self.attribute = "original"

example = Example()
print(f"\nBefore: {example.attribute}")

with temporary_attribute(example, 'attribute', 'temporary'):
    print(f"During: {example.attribute}")

print(f"After: {example.attribute}")
```

### Practical Exercise: Data Analysis with Advanced Functions

I applied what I learned to a data analysis scenario:

```python
# Create a sample dataset
np.random.seed(42)
data = {
    'id': range(1, 101),
    'age': np.random.randint(18, 65, 100),
    'income': np.random.normal(50000, 15000, 100).astype(int),
    'education_years': np.random.randint(12, 22, 100),
    'expenses': np.random.normal(30000, 10000, 100).astype(int),
    'satisfaction': np.random.uniform(1, 10, 100).round(1)
}

df = pd.DataFrame(data)
print("\nSample dataset:")
print(df.head())

# 1. Use lambda and apply to calculate savings
df['savings'] = df.apply(lambda row: row['income'] - row['expenses'], axis=1)
print("\nDataFrame with savings column:")
print(df[['income', 'expenses', 'savings']].head())

# 2. Use map to categorize age groups
def age_category(age):
    if age < 25:
        return 'Young'
    elif age < 40:
        return 'Adult'
    elif age < 55:
        return 'Middle-aged'
    else:
        return 'Senior'

df['age_group'] = df['age'].map(age_category)
print("\nAge group distribution:")
print(df['age_group'].value_counts())

# 3. Use filter to find high earners with low satisfaction
high_earners_low_satisfaction = df[
    (df['income'] > df['income'].quantile(0.75)) & 
    (df['satisfaction'] < df['satisfaction'].quantile(0.25))
]
print("\nHigh earners with low satisfaction:")
print(high_earners_low_satisfaction[['income', 'satisfaction']])

# 4. Use list comprehension to create a list of dictionaries with selected data
selected_data = [
    {'id': row['id'], 'savings_ratio': row['savings'] / row['income']}
    for _, row in df.iterrows() if row['savings'] > 0
]
print("\nSavings ratio for positive savers:")
print(selected_data[:5])  # Show first 5 entries

# 5. Use reduce to find the average savings ratio
total_ratio = reduce(lambda acc, item: acc + item['savings_ratio'], selected_data, 0)
avg_ratio = total_ratio / len(selected_data)
print(f"\nAverage savings ratio: {avg_ratio:.2f}")

# 6. Create a decorator for timing DataFrame operations
def time_operation(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Operation '{func.__name__}' took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@time_operation
def calculate_statistics(dataframe):
    stats = {}
    # Calculate various statistics
    stats['mean_income'] = dataframe['income'].mean()
    stats['median_income'] = dataframe['income'].median()
    stats['income_std'] = dataframe['income'].std()
    stats['savings_corr'] = dataframe['income'].corr(dataframe['savings'])
    stats['education_income_corr'] = dataframe['education_years'].corr(dataframe['income'])
    stats['age_satisfaction_corr'] = dataframe['age'].corr(dataframe['satisfaction'])
    
    # Group by age_group and calculate mean values
    stats['age_group_means'] = dataframe.groupby('age_group').mean()
    
    return stats

print("\nCalculating statistics:")
statistics = calculate_statistics(df)

print("\nStatistics results:")
for key, value in statistics.items():
    if key != 'age_group_means':
        print(f"{key}: {value:.4f}")

print("\nMean values by age group:")
print(statistics['age_group_means'][['income', 'expenses', 'savings', 'satisfaction']])

# 7. Use a context manager for a temporary DataFrame transformation
@contextmanager
def temporary_log_transform(dataframe, columns):
    """Temporarily apply log transformation to specified columns"""
    original_data = {}
    for col in columns:
        original_data[col] = dataframe[col].copy()
        # Apply log transform, handling zeros and negative values
        dataframe[col] = np.log1p(dataframe[col] - dataframe[col].min() + 1)
    
    try:
        yield dataframe
    finally:
        # Restore original data
        for col in columns:
            dataframe[col] = original_data[col]

# Use the context manager to analyze log-transformed data
print("\nAnalyzing log-transformed data:")
with temporary_log_transform(df, ['income', 'expenses']):
    print("Correlation matrix with log-transformed income and expenses:")
    print(df[['income', 'expenses', 'savings', 'satisfaction']].corr())

print("\nOriginal data restored:")
print(df[['income', 'expenses']].head())

# 8. Visualize the data using functions we've learned
plt.figure(figsize=(12, 8))

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Income vs. Expenses colored by Age Group
sns.scatterplot(
    data=df, 
    x='income', 
    y='expenses', 
    hue='age_group', 
    alpha=0.7,
    ax=axes[0, 0]
)
axes[0, 0].set_title('Income vs. Expenses by Age Group')

# Plot 2: Savings Distribution
sns.histplot(
    data=df,
    x='savings',
    kde=True,
    ax=axes[0, 1]
)
axes[0, 1].set_title('Savings Distribution')

# Plot 3: Education Years vs. Income
sns.boxplot(
    data=df,
    x='education_years',
    y='income',
    ax=axes[1, 0]
)
axes[1, 0].set_title('Income by Education Years')

# Plot 4: Satisfaction by Age Group
sns.barplot(
    data=df,
    x='age_group',
    y='satisfaction',
    ax=axes[1, 1]
)
axes[1, 1].set_title('Average Satisfaction by Age Group')

plt.tight_layout()
# plt.savefig('data_analysis_plots.png')
# plt.show()
```

## Reflections

Today's exploration of advanced Python functions has significantly enhanced my programming toolkit. These techniques allow for more concise, readable, and efficient code, which is particularly valuable for data analysis and manipulation tasks.

Lambda functions provide a way to create small, anonymous functions inline, which is especially useful for operations like sorting or filtering where a full function definition would be overkill. They're perfect for simple operations that don't require multiple statements or complex logic.

The `map`, `filter`, and `reduce` functions offer functional programming paradigms that can make code more expressive and focused on what needs to be done rather than how to do it. While list comprehensions often provide a more Pythonic alternative to `map` and `filter`, understanding all these approaches gives me flexibility in choosing the most appropriate tool for each situation.

List comprehensions and generator expressions are powerful features that allow for concise creation of lists and iterators. I particularly appreciated the memory efficiency of generator expressions for large datasets, as they evaluate items on-demand rather than creating the entire collection in memory at once.

Decorators are an incredibly powerful feature that allows for modifying or extending the behavior of functions without changing their code. They're perfect for cross-cutting concerns like timing, logging, or validation. The ability to stack decorators and create decorators that accept arguments provides tremendous flexibility.

Context managers offer a clean way to handle resource allocation and release, ensuring that cleanup code is executed even if exceptions occur. They're not just for file operations but can be used for any situation where setup and teardown actions are needed, such as database connections, temporary settings, or performance measurements.

The practical exercise demonstrated how these advanced function concepts can be applied to real-world data analysis tasks. By combining these techniques, I was able to create more efficient and readable code for data transformation, analysis, and visualization.

## Questions to Explore
- How do these functional programming techniques compare in terms of performance for large datasets?
- What are the best practices for error handling in decorators and context managers?
- How can I combine these techniques with parallel processing for even more efficient data operations?
- What are the most common use cases for decorators and context managers in data science workflows?

## Resources
- [Python Documentation - Functional Programming HOWTO](https://docs.python.org/3/howto/functional.html)
- [Python Documentation - Decorators](https://docs.python.org/3/glossary.html#term-decorator)
- [Python Documentation - Context Managers](https://docs.python.org/3/reference/datamodel.html#context-managers)
- [Real Python - Decorators in Python](https://realpython.com/primer-on-python-decorators/)
- [Python for Data Analysis, 2nd Edition - Chapter 3](https://wesmckinney.com/book/)
