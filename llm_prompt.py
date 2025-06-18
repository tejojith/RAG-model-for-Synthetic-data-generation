from langchain.prompts import PromptTemplate
from typing import Dict

CLASSIFY_PROMPT = PromptTemplate.from_template("""
Analyze the dataset columns for PII (Personally Identifiable Information).

Context: {context}

Instructions:
- Mark TRUE for: names, emails, phone numbers, addresses, SSN, medical records, financial info
- Mark FALSE for: anonymous IDs, product codes, aggregated data, generic categories
- Format: column_name - TRUE/FALSE (one per line)
- Sort alphabetically

Output only the analysis, no explanations:
""")                                             


def create_enhanced_generation_prompt(num_records: int, context: str) -> str:
        """Create a more structured and specific prompt"""
        
        prompt_template = """You are a synthetic data generator. Generate exactly {num_records} unique CSV records.

STRICT REQUIREMENTS:
1. Generate EXACTLY {num_records} records - no more, no less
2. Each record must be completely unique (no duplicates)
3. Output ONLY the CSV data with headers - no explanations or extra text
4. Start directly with the CSV headers

CSV Structure from context:
{context}

DATA GENERATION RULES:
1. Maintain realistic data distributions
2. Ensure referential integrity between related columns
3. Generate diverse, non-repeating values
4. Follow original data patterns and formats
5. Use appropriate data types for each column

Output Format: Pure CSV only
"""
        
        return prompt_template.format(
            num_records=num_records,
            context=context[:1000],  # Limit context to avoid token limits
        )
