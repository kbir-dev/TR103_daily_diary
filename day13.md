# Day 13: Text Functions and Data Cleaning in Python

## Today's Topics
- Data cleaning with text functions
- Built-in string functions in Python
- Regular expressions for text processing
- Practical text data cleaning techniques

## Learning Journal

### Introduction to Text Data Cleaning

Today I focused on text functions and data cleaning techniques in Python. Text data often requires extensive preprocessing before it can be effectively analyzed. Python offers a rich set of tools for manipulating and cleaning text data, from built-in string methods to more powerful regular expressions.

I started by setting up the necessary libraries:

```python
import pandas as pd
import numpy as np
import re
import string
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for our plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
```

### Built-in String Functions in Python

I explored Python's built-in string methods, which provide a wide range of functionality for text manipulation:

```python
# Sample text
text = "  Python is a powerful programming language. PYTHON is widely used in Data Science!  "

# Basic string operations
print("Original text:", text)
print("Length:", len(text))
print("Uppercase:", text.upper())
print("Lowercase:", text.lower())
print("Title case:", text.title())
print("Stripped:", text.strip())

# Finding and counting
print("\nFinding and counting:")
print("Contains 'Python':", 'Python' in text)
print("Count of 'Python':", text.count('Python'))
print("Find 'Python':", text.find('Python'))  # Returns index of first occurrence
print("Find 'Java':", text.find('Java'))  # Returns -1 if not found

# Splitting and joining
words = text.split()
print("\nSplit into words:", words)
print("Join words back:", ' '.join(words))

# Replace
print("\nReplace 'Python' with 'R':", text.replace('Python', 'R'))

# Check string properties
sample_strings = ["123", "abc", "abc123", "  ", "Hello!"]
for s in sample_strings:
    print(f"\nString: '{s}'")
    print(f"Is alphanumeric? {s.isalnum()}")
    print(f"Is alphabetic? {s.isalpha()}")
    print(f"Is numeric? {s.isdigit()}")
    print(f"Is lowercase? {s.islower()}")
    print(f"Is uppercase? {s.isupper()}")
    print(f"Is whitespace? {s.isspace()}")
```

### Working with Text Data in Pandas

Next, I learned how to apply string functions to text data in pandas DataFrames:

```python
# Create a DataFrame with messy text data
data = {
    'name': ['John SMITH', '  jane doe', 'Robert Johnson  ', 'LISA   wong', 'michael brown'],
    'email': ['john.smith@example.com', 'jane.DOE@example.COM', 'robert_johnson@gmail.com', 'lisa.wong@company.org', 'michael@brown.net'],
    'address': ['123 Main St., New York, NY', '456 Oak Ave, San Francisco, CA', '789 Pine Rd, Chicago, IL', '101 Maple Dr., Boston, MA', '202 Elm Blvd, Seattle, WA'],
    'date': ['2025-01-15', '01/20/2025', '2025.02.10', '03-15-2025', '2025/04/05'],
    'id': ['ID-001', 'id_002', 'ID_003', 'id-004', 'ID005']
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# String methods in pandas
print("\nNames in title case:")
print(df['name'].str.title())

# Strip whitespace
df['name'] = df['name'].str.strip()
print("\nNames after stripping whitespace:")
print(df['name'])

# Convert to lowercase
df['email'] = df['email'].str.lower()
print("\nEmails in lowercase:")
print(df['email'])

# Extract domain from email
df['email_domain'] = df['email'].str.split('@').str[1]
print("\nExtracted email domains:")
print(df['email_domain'])

# Extract state from address
df['state'] = df['address'].str.extract(r', ([A-Z]{2})')
print("\nExtracted states:")
print(df['state'])

# Standardize date format
def standardize_date(date_str):
    # Check different formats and convert to YYYY-MM-DD
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return date_str  # Already in YYYY-MM-DD format
    elif re.match(r'^\d{2}/\d{2}/\d{4}$', date_str):
        month, day, year = date_str.split('/')
        return f"{year}-{month}-{day}"
    elif re.match(r'^\d{4}\.\d{2}\.\d{2}$', date_str):
        year, month, day = date_str.split('.')
        return f"{year}-{month}-{day}"
    elif re.match(r'^\d{2}-\d{2}-\d{4}$', date_str):
        month, day, year = date_str.split('-')
        return f"{year}-{month}-{day}"
    elif re.match(r'^\d{4}/\d{2}/\d{2}$', date_str):
        year, month, day = date_str.split('/')
        return f"{year}-{month}-{day}"
    else:
        return date_str  # Return as is if format not recognized

df['standardized_date'] = df['date'].apply(standardize_date)
print("\nStandardized dates:")
print(df['standardized_date'])

# Standardize ID format
def standardize_id(id_str):
    # Remove any non-alphanumeric characters and convert to uppercase
    clean_id = re.sub(r'[^A-Za-z0-9]', '', id_str).upper()
    # Ensure format is ID followed by 3 digits
    match = re.match(r'ID(\d{3})', clean_id)
    if match:
        return f"ID-{match.group(1)}"
    else:
        return id_str  # Return as is if format not recognized

df['standardized_id'] = df['id'].apply(standardize_id)
print("\nStandardized IDs:")
print(df['standardized_id'])
```

### Regular Expressions for Text Processing

I explored regular expressions, which provide powerful pattern matching capabilities for text processing:

```python
# Sample text for regex examples
text_samples = [
    "Contact us at info@example.com or support@company.org",
    "Phone: (555) 123-4567 or 555-987-6543",
    "Product codes: ABC-123, XYZ-456, and LMN-789",
    "Dates: 2025-01-15, 01/20/2025, and 2025.02.10",
    "URLs: https://www.example.com, http://company.org, and www.website.net"
]

# Extract email addresses
print("\nExtracted email addresses:")
for text in text_samples:
    emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    if emails:
        print(f"From '{text[:30]}...': {emails}")

# Extract phone numbers
print("\nExtracted phone numbers:")
for text in text_samples:
    phones = re.findall(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
    if phones:
        print(f"From '{text[:30]}...': {phones}")

# Extract product codes
print("\nExtracted product codes:")
for text in text_samples:
    codes = re.findall(r'[A-Z]{3}-\d{3}', text)
    if codes:
        print(f"From '{text[:30]}...': {codes}")

# Extract dates
print("\nExtracted dates:")
for text in text_samples:
    dates = re.findall(r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{4}\.\d{2}\.\d{2}', text)
    if dates:
        print(f"From '{text[:30]}...': {dates}")

# Extract URLs
print("\nExtracted URLs:")
for text in text_samples:
    urls = re.findall(r'https?://[^\s]+|www\.[^\s]+', text)
    if urls:
        print(f"From '{text[:30]}...': {urls}")
```

### Advanced Text Cleaning Techniques

I learned more advanced techniques for cleaning and preprocessing text data:

```python
# Create a DataFrame with more complex text data
complex_data = {
    'product_description': [
        'High-quality LAPTOP with 16GB RAM and 512GB SSD',
        'Smartphone, 6.5" display, 128GB storage',
        'Wireless headphones - Noise cancelling (Black)',
        'Tablet w/ 10.2" Retina display & 64GB',
        'Smart watch: fitness tracker, heart rate monitor'
    ],
    'customer_review': [
        "This product is amazing!! I love it so much!!!",
        "It's okay, but not worth the price. Disappointed.",
        "Works great for me, but the battery life could be better.",
        "DON'T BUY THIS! It broke after just 2 weeks of use.",
        "Good product overall. Shipping was fast and packaging was good."
    ],
    'tags': [
        'electronics, laptop, computer, high-performance',
        'phone, smartphone, mobile, electronics',
        'audio, headphones, wireless, noise-cancelling',
        'tablet, electronics, portable, touchscreen',
        'wearable, smartwatch, fitness, health'
    ]
}

complex_df = pd.DataFrame(complex_data)
print("Complex text data:")
print(complex_df)

# Function to clean and standardize product descriptions
def clean_product_description(text):
    # Convert to lowercase
    text = text.lower()
    # Replace special characters with spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text

complex_df['clean_description'] = complex_df['product_description'].apply(clean_product_description)
print("\nCleaned product descriptions:")
print(complex_df['clean_description'])

# Function to extract key specifications
def extract_specs(text):
    specs = {}
    # Extract storage
    storage_match = re.search(r'(\d+)GB', text, re.IGNORECASE)
    if storage_match:
        specs['storage'] = storage_match.group(1) + 'GB'
    
    # Extract display size
    display_match = re.search(r'(\d+\.?\d?)[""]', text)
    if display_match:
        specs['display'] = display_match.group(1) + '"'
    
    # Extract RAM
    ram_match = re.search(r'(\d+)GB RAM', text, re.IGNORECASE)
    if ram_match:
        specs['ram'] = ram_match.group(1) + 'GB'
    
    return specs

complex_df['specifications'] = complex_df['product_description'].apply(extract_specs)
print("\nExtracted specifications:")
print(complex_df['specifications'])

# Function to analyze sentiment in reviews
def analyze_sentiment(text):
    # Very simple sentiment analysis based on keywords
    positive_words = ['amazing', 'love', 'great', 'good', 'fast']
    negative_words = ['disappointed', 'don\'t buy', 'broke', 'not worth']
    
    text_lower = text.lower()
    
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        return 'Positive'
    elif negative_count > positive_count:
        return 'Negative'
    else:
        return 'Neutral'

complex_df['sentiment'] = complex_df['customer_review'].apply(analyze_sentiment)
print("\nSentiment analysis:")
print(complex_df[['customer_review', 'sentiment']])

# Function to standardize and split tags
def process_tags(tags_str):
    # Convert to lowercase, split by comma, and strip whitespace
    tags_list = [tag.strip().lower() for tag in tags_str.split(',')]
    # Remove duplicates while preserving order
    unique_tags = []
    for tag in tags_list:
        if tag not in unique_tags:
            unique_tags.append(tag)
    return unique_tags

complex_df['processed_tags'] = complex_df['tags'].apply(process_tags)
print("\nProcessed tags:")
print(complex_df['processed_tags'])
```

### Practical Exercise: Cleaning a Customer Dataset

I applied what I learned to clean a messy customer dataset:

```python
# Create a messy customer dataset
messy_data = {
    'customer_name': [
        'john smith', 'JANE DOE', 'Robert  Johnson', 'lisa  WONG', 'Michael Brown  '
    ],
    'email': [
        'john.smith@example.com', 'jane.DOE@example.COM', 'robert_johnson@gmail.com', 
        'lisa.wong@company.org', 'michael@brown.net'
    ],
    'phone': [
        '(555) 123-4567', '555.987.6543', '555 321 7890', '(555)456-7890', '555-789-0123'
    ],
    'address': [
        '123 Main St., New York, NY 10001', '456 Oak Ave, San Francisco, CA 94102', 
        '789 Pine Rd, Chicago, IL 60601', '101 Maple Dr., Boston, MA 02108', 
        '202 Elm Blvd, Seattle, WA 98101'
    ],
    'purchase_date': [
        '2025-01-15', '01/20/2025', '2025.02.10', '03-15-2025', '2025/04/05'
    ],
    'purchase_amount': [
        '$1,234.56', '2,345.67', '$3456.78', '4,567', '$ 5,678.90'
    ],
    'product_id': [
        'PROD-001', 'prod_002', 'PROD/003', 'prod-004', 'PROD 005'
    ],
    'notes': [
        'Customer requested express shipping', 'Returned due to wrong size', 
        'Frequent buyer - VIP status', 'First-time customer', 'Has pending complaint'
    ]
}

messy_df = pd.DataFrame(messy_data)
print("Original messy customer data:")
print(messy_df.head())

# Step 1: Clean customer names
def clean_name(name):
    # Strip whitespace, convert to title case, and remove extra spaces
    name = name.strip().title()
    name = re.sub(r'\s+', ' ', name)
    return name

messy_df['customer_name'] = messy_df['customer_name'].apply(clean_name)

# Step 2: Standardize email addresses
messy_df['email'] = messy_df['email'].str.lower()

# Step 3: Standardize phone numbers to (XXX) XXX-XXXX format
def standardize_phone(phone):
    # Extract only the digits
    digits = re.sub(r'\D', '', phone)
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    else:
        return phone  # Return as is if not 10 digits

messy_df['phone'] = messy_df['phone'].apply(standardize_phone)

# Step 4: Extract components from address
def parse_address(address):
    # Extract street, city, state, and zip
    match = re.match(r'(.*?),\s*(.*?),\s*([A-Z]{2})\s*(\d{5})?', address)
    if match:
        street = match.group(1).strip()
        city = match.group(2).strip()
        state = match.group(3)
        zip_code = match.group(4) if match.group(4) else ''
        return pd.Series([street, city, state, zip_code])
    else:
        return pd.Series([address, '', '', ''])

address_components = messy_df['address'].apply(parse_address)
messy_df[['street', 'city', 'state', 'zip']] = address_components

# Step 5: Standardize purchase dates to YYYY-MM-DD
messy_df['purchase_date'] = messy_df['purchase_date'].apply(standardize_date)

# Step 6: Clean purchase amounts
def clean_amount(amount):
    # Remove currency symbols and commas, then convert to float
    amount = re.sub(r'[$,]', '', amount)
    try:
        return float(amount)
    except ValueError:
        return None

messy_df['purchase_amount'] = messy_df['purchase_amount'].apply(clean_amount)

# Step 7: Standardize product IDs
def standardize_product_id(product_id):
    # Remove any non-alphanumeric characters and convert to uppercase
    clean_id = re.sub(r'[^A-Za-z0-9]', '', product_id).upper()
    return f"PROD{clean_id[-3:]}"

messy_df['product_id'] = messy_df['product_id'].apply(standardize_product_id)

# Step 8: Extract keywords from notes
def extract_keywords(notes):
    # Define keywords to look for
    keywords = {
        'shipping': ['shipping', 'delivery', 'express'],
        'return': ['return', 'refund', 'exchange', 'wrong'],
        'loyalty': ['vip', 'frequent', 'loyal', 'regular'],
        'new': ['first', 'new'],
        'issue': ['complaint', 'issue', 'problem']
    }
    
    found_keywords = []
    notes_lower = notes.lower()
    
    for category, words in keywords.items():
        if any(word in notes_lower for word in words):
            found_keywords.append(category)
    
    return found_keywords

messy_df['keywords'] = messy_df['notes'].apply(extract_keywords)

# Display the cleaned data
print("\nCleaned customer data:")
print(messy_df.head())

# Create a summary of the cleaning process
print("\nData Cleaning Summary:")
print(f"1. Standardized {len(messy_df)} customer names to title case")
print(f"2. Converted {len(messy_df)} email addresses to lowercase")
print(f"3. Standardized {len(messy_df)} phone numbers to (XXX) XXX-XXXX format")
print(f"4. Extracted street, city, state, and zip from {len(messy_df)} addresses")
print(f"5. Standardized {len(messy_df)} dates to YYYY-MM-DD format")
print(f"6. Converted {len(messy_df)} purchase amounts to numeric values")
print(f"7. Standardized {len(messy_df)} product IDs to PRODXXX format")
print(f"8. Extracted keywords from {len(messy_df)} customer notes")

# Analyze the extracted keywords
all_keywords = [keyword for keywords_list in messy_df['keywords'] for keyword in keywords_list]
keyword_counts = Counter(all_keywords)

print("\nKeyword frequency in customer notes:")
for keyword, count in keyword_counts.items():
    print(f"{keyword}: {count}")

# Visualize the keyword distribution
plt.figure(figsize=(10, 6))
plt.bar(keyword_counts.keys(), keyword_counts.values(), color='skyblue')
plt.title('Frequency of Keywords in Customer Notes')
plt.xlabel('Keyword')
plt.ylabel('Count')
# plt.savefig('keyword_distribution.png')
# plt.show()
```

## Reflections

Today's exploration of text functions and data cleaning techniques has significantly enhanced my data preprocessing capabilities. Text data is often messy and inconsistent, requiring careful cleaning and standardization before it can be effectively analyzed.

I found Python's built-in string methods to be incredibly useful for basic text manipulation tasks like changing case, removing whitespace, and splitting strings. These methods are simple yet powerful, and they form the foundation of text data cleaning.

Regular expressions provide a more advanced and flexible way to work with text patterns. While the syntax can be complex, regular expressions are invaluable for extracting specific information from text, validating formats, and performing complex replacements. I particularly appreciated their power in standardizing inconsistent data formats like dates, phone numbers, and IDs.

The pandas string methods make it easy to apply text operations to entire columns of data, which is essential for efficient data cleaning. The ability to chain these methods together allows for concise and readable code.

The practical exercise demonstrated how these techniques can be applied to clean a messy customer dataset. By standardizing names, emails, phone numbers, and other fields, I was able to transform inconsistent raw data into a clean, structured format suitable for analysis. The extraction of components from complex fields like addresses and the identification of keywords in unstructured notes showed how text processing can uncover valuable information hidden in the data.

## Questions to Explore
- How can I optimize text cleaning operations for very large datasets?
- What are the best practices for handling multilingual text data?
- How can I incorporate more advanced natural language processing techniques into my data cleaning workflow?
- What are the most effective ways to validate and verify the results of text data cleaning?

## Resources
- [Python Documentation - String Methods](https://docs.python.org/3/library/stdtypes.html#string-methods)
- [Python Documentation - Regular Expressions](https://docs.python.org/3/library/re.html)
- [Pandas Documentation - Working with Text Data](https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html)
- [Regular Expression 101](https://regex101.com/) - Tool for testing and debugging regular expressions
- [Python for Data Analysis, 2nd Edition - Chapter 7](https://wesmckinney.com/book/)
