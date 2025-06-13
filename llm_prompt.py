from langchain.prompts import PromptTemplate

GENERATION_PROMPT = PromptTemplate.from_template("""
Task: Generate synthetic records based on data.csv

Number of records to generate: {num_records}

Context from data.csv:
{context}


Instructions:
1. Strictly maintain:
   - All column names and order from data.csv
   - Data types and formats of each column
   - Value distributions and ranges
   - Relationships between columns

2. For each column type:
   - Numeric: Maintain similar statistical distribution
   - Categorical: Use same categories with similar frequencies
   - Text: Generate similar format/length text
   - Dates: Keep within same date ranges
   - IDs: Generate new unique values

3. Output format:
   - CSV with header row identical to data.csv
   - {num_records} data rows
   - No additional explanations, just the raw CSV data
    - not even the first line, directly give only csv table headings

Synthetic Data:
""")

CLASSIFY_PROMPT = PromptTemplate.from_template("""
Context: {context}
Task: Analyze the provided dataset's columns and classify whether each contains PII (Personally Identifiable Information) or sensitive data. Output the results as a .txt file where each line follows the format:
column_name - TRUE/FALSE

Instructions:
PII Definition:

Mark a column as TRUE if it contains any of the following:

Direct identifiers: Name, Email, Phone Number, SSN, Address, ID Number, etc.

Indirect identifiers: Age + ZIP Code, Gender + Birthdate, etc. (if combinable to identify a person).

Sensitive data: Medical Records, Financial Info, Passwords, GPS Coordinates.

Mark as FALSE for non-sensitive data (e.g., Product_ID, Order_Total, anonymized Hashed_Data).

Output Rules:

One line per column: column_name - TRUE or column_name - FALSE.

Sort alphabetically by column name.

Save the output as a .txt file (or return as plaintext).
                                            
GIVE only the PII, nothing else, no explanation

Example Input:

csv
user_id, name, email, credit_score, postal_code, device_type  
Example Output:

txt
credit_score - FALSE  
device_type - FALSE  
email - TRUE  
name - TRUE  
postal_code - TRUE  # Can be PII when combined with other data  
user_id - FALSE    # Assuming it’s an anonymous UUID  

""")