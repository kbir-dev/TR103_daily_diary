# Day 2: Python Packages for Data Science - Pandas & NumPy

## Today's Topics
- Introduction to Python packages
- NumPy: Numerical Python
- Pandas: Data manipulation and analysis

## Learning Journal

### Python Packages for Data Science

Today I explored the essential Python packages for data science. Python's ecosystem is rich with libraries that extend its functionality, particularly for data analysis and scientific computing.

### NumPy: Numerical Python

NumPy (Numerical Python) is the foundation for scientific computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with mathematical functions to operate on these data structures.

I installed NumPy using pip:
```
pip install numpy
```

#### Key NumPy Concepts I Learned:

1. **Arrays**: The fundamental data structure in NumPy
```python
import numpy as np

# Creating arrays
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.zeros((3, 3))  # 3x3 array of zeros
arr3 = np.ones((2, 4))   # 2x4 array of ones
arr4 = np.arange(10)     # array from 0 to 9
arr5 = np.linspace(0, 1, 5)  # 5 evenly spaced values from 0 to 1

print("1D array:", arr1)
print("Array of zeros:\n", arr2)
print("Array of ones:\n", arr3)
print("Range array:", arr4)
print("Linearly spaced array:", arr5)
```

2. **Array Operations**: Vectorized operations for efficient computation
```python
# Element-wise operations
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("a + b =", a + b)
print("a * b =", a * b)
print("a ** 2 =", a ** 2)

# Statistical operations
print("Mean of a:", a.mean())
print("Sum of b:", b.sum())
print("Max of a:", a.max())
```

3. **Array Reshaping and Indexing**:
```python
arr = np.arange(12)
print("Original array:", arr)

# Reshape to 3x4 matrix
reshaped = arr.reshape(3, 4)
print("Reshaped to 3x4:\n", reshaped)

# Indexing and slicing
print("First row:", reshaped[0])
print("First column:", reshaped[:, 0])
print("Submatrix:\n", reshaped[0:2, 1:3])
```

### Pandas: Data Manipulation and Analysis

Pandas is built on top of NumPy and provides data structures and functions designed for data manipulation and analysis. It's particularly well-suited for working with tabular data.

I installed Pandas using pip:
```
pip install pandas
```

#### Key Pandas Concepts I Learned:

1. **Series**: One-dimensional labeled array
```python
import pandas as pd

# Creating a Series
s = pd.Series([1, 3, 5, 7, 9], index=['a', 'b', 'c', 'd', 'e'])
print("Pandas Series:\n", s)
print("Value at index 'c':", s['c'])
```

2. **DataFrame**: Two-dimensional labeled data structure
```python
# Creating a DataFrame from a dictionary
data = {
    'Name': ['John', 'Anna', 'Peter', 'Linda'],
    'Age': [28, 34, 29, 42],
    'City': ['New York', 'Paris', 'Berlin', 'London']
}

df = pd.DataFrame(data)
print("DataFrame:\n", df)

# Basic information about the DataFrame
print("\nDataFrame info:")
print(df.info())
print("\nDataFrame description:")
print(df.describe())
```

3. **Reading Data**: Loading data from various file formats
```python
# Reading data from CSV (example - would need an actual file)
# df = pd.read_csv('data.csv')

# Creating a sample DataFrame and saving it
sample_df = pd.DataFrame({
    'A': range(1, 6),
    'B': ['x', 'y', 'z', 'x', 'y'],
    'C': np.random.randn(5)
})

print("Sample DataFrame:\n", sample_df)
# sample_df.to_csv('sample_data.csv', index=False)
```

## Practical Exercise

I combined NumPy and Pandas to analyze some synthetic data:

```python
# Generate synthetic data
np.random.seed(42)  # for reproducibility
dates = pd.date_range('20250101', periods=6)
numeric_data = np.random.randn(6, 4)

# Create DataFrame
df = pd.DataFrame(numeric_data, index=dates, columns=list('ABCD'))
print("Time series DataFrame:\n", df)

# Basic analysis
print("\nFirst 2 rows:\n", df.head(2))
print("\nBasic statistics:\n", df.describe())
print("\nTransposed DataFrame:\n", df.T)
```

## Reflections

Today was incredibly productive. NumPy and Pandas are powerful libraries that form the backbone of data science in Python. I can already see how these tools will be essential for the more advanced topics coming up in the curriculum.

The syntax for both libraries is intuitive once you understand the core concepts. I particularly appreciate how Pandas makes it easy to work with tabular data, which will be crucial for real-world data analysis tasks.

## Questions to Explore
- How do NumPy arrays compare to Python lists in terms of performance?
- What are the memory optimization techniques in Pandas for large datasets?
- How can I effectively combine NumPy and Pandas for complex data transformations?

## Resources
- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [10 Minutes to Pandas Tutorial](https://pandas.pydata.org/docs/user_guide/10min.html)
- [Python Data Science Handbook - NumPy Chapter](https://jakevdp.github.io/PythonDataScienceHandbook/02.00-introduction-to-numpy.html)
