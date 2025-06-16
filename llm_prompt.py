from langchain.prompts import PromptTemplate

GENERATION_PROMPT = PromptTemplate.from_template("""
TASK: Generate exactly {num_records} unique synthetic CSV records

DATA CONTEXT:
{context}


STRICT REQUIREMENTS:
1. OUTPUT EXACTLY {num_records} data rows (+ 1 header row)
2. Each record must be completely unique - NO DUPLICATES
3. Start immediately with CSV header, then data rows
4. NO explanations, comments, or extra text
5. Maintain realistic data relationships and distributions

GENERATION RULES:
- Numeric columns: Stay within observed ranges, maintain distributions
- Categorical columns: Use variety from observed categories
- ID columns: Generate unique identifiers (increment, UUID-style, etc.)
- Date columns: Generate dates within observed ranges
- Text columns: Create realistic variations with similar patterns
- Email columns: Generate unique, realistic email addresses
- Name columns: Use diverse, realistic names (no repetition)

QUALITY CONSTRAINTS:
- Ensure referential integrity between related fields
- Maintain data consistency (e.g., state matches city)
- Generate realistic combinations
- Avoid obvious patterns or sequences that repeat

OUTPUT FORMAT:
[HEADER ROW]
[DATA ROW 1]
[DATA ROW 2]
...
[DATA ROW {num_records}]

BEGIN CSV OUTPUT:

""")

# Task: Generate synthetic records based on data.csv

# Number of records to generate: {num_records}

# Context from data.csv:
# {context}


# Instructions:
# 1. Strictly maintain:
#    - All column names and order from data.csv
#    - Data types and formats of each column
#    - Value distributions and ranges
#    - Relationships between columns

# 2. For each column type:
#    - Numeric: Maintain similar statistical distribution
#    - Categorical: Use same categories with similar frequencies
#    - Text: Generate similar format/length text
#    - Dates: Keep within same date ranges
#    - IDs: Generate new unique values

# 3. Output format:
#    - CSV with header row identical to data.csv
#    - {num_records} data rows
#    - No additional explanations, just the raw CSV data
#     - not even the first line, directly give only csv table headings

# Synthetic Data:

CLASSIFY_PROMPT = PromptTemplate.from_template("""
TASK: Classify dataset columns for PII (Personally Identifiable Information)

DATASET CONTEXT:
{context}

CLASSIFICATION RULES:
Mark as TRUE if column contains:
- Direct identifiers: Full names, email addresses, phone numbers, SSN, passport numbers
- Physical identifiers: Home addresses, precise GPS coordinates
- Financial identifiers: Credit card numbers, bank account numbers
- Medical identifiers: Patient IDs, medical record numbers
- Biometric data: Fingerprints, facial recognition data
- Government IDs: Driver's license numbers, tax IDs

Mark as FALSE if column contains:
- Anonymous/hashed identifiers that cannot identify individuals
- Aggregated/statistical data
- Product codes, order numbers (unless linked to individuals)
- Generic categories (product types, status codes)
- Business identifiers (company names, department codes)
- Postal/ZIP codes alone (without other identifying info)

SPECIAL CASES:
- Age + Location combinations: Consider TRUE if highly specific
- User IDs: FALSE if anonymous/hashed, TRUE if contain personal info
- Timestamps: Generally FALSE unless they reveal personal patterns

                                               
give no explanation

OUTPUT FORMAT (one line per column, alphabetically sorted):
column_name - TRUE
column_name - FALSE

ANALYSIS:
""")                                             




# Context: {context}
# Task: Analyze the provided dataset's columns and classify whether each contains PII (Personally Identifiable Information) or sensitive data. Output the results as a .txt file where each line follows the format:
# column_name - TRUE/FALSE

# Instructions:
# PII Definition:

# Mark a column as TRUE if it contains any of the following:

# Direct identifiers: Name, Email, Phone Number, SSN, Address, ID Number, etc.

# Indirect identifiers: Age + ZIP Code, Gender + Birthdate, etc. (if combinable to identify a person).

# Sensitive data: Medical Records, Financial Info, Passwords, GPS Coordinates.

# Mark as FALSE for non-sensitive data (e.g., Product_ID, Order_Total, anonymized Hashed_Data).

# Output Rules:

# One line per column: column_name - TRUE or column_name - FALSE.

# Sort alphabetically by column name.

# Save the output as a .txt file (or return as plaintext).
                                            
# GIVE only the PII, nothing else, no explanation
                                               
# Example Input:

# csv
# user_id, name, email, credit_score, postal_code, device_type  
# Example Output:

# txt
# credit_score - FALSE  
# device_type - FALSE  
# email - TRUE  
# name - TRUE  
# postal_code - TRUE  # Can be PII when combined with other data  
# user_id - FALSE    # Assuming it’s an anonymous UUID  
# end
# """)