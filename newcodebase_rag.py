import json
import hashlib
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
import pandas as pd
import csv
import os
import io
import re
from concurrent.futures import ThreadPoolExecutor
import time


from llm_prompt import create_enhanced_generation_prompt, CLASSIFY_PROMPT

class CodebaseRag:
    def __init__(self, project_path, db_path):
        self.project_path = project_path
        self.db_path = db_path
        self.embed_model = "nomic-embed-text"  # or mxbai-embed-large for code
        self.embedding = OllamaEmbeddings(model=self.embed_model)
        self.vector_db = None
        self.llm = None
        self.schema_cache = {}
        self.generated_records_cache = set()

    def initialize_llm(self):
        if not self.llm:
            self.llm = OllamaLLM(
                model = "llama3",
                temperature = 0.7, # for more variety
                top_k = 40,
                top_p = 0.9,
                repeat_penalty = 1.2, #higher penalty to avoid repetition
                num_predict= 2048, #limit output lenght
                num_ctx = 4096 #context window
            )

    def save_PII(self, answer):
        new  = answer[answer.index(":")+3:] 
        output_format = "txt"
        output_file = f"PII.{output_format}"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"{new}\n{'-'*40}\n")
            
    def save_to_file(self, num_records, result):
        try:
            # Clean the result
            clean_result = result[result.index(":")+3:] if ":" in result else result
        except:
            clean_result = result
            
        output_file = f"synthetic_data_{num_records}_records.csv"
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            f.write(clean_result)
        print(f"Data saved to {output_file}")


    def create_embeddings_and_store(self):
        """Optimized embedding creation with better file handling"""
        relevant_extensions = ['.py', '.sql', '.txt', '.json', '.html', '.md', '.csv']
        documents = []
        
        # Use ThreadPoolExecutor for parallel file loading
        def load_file(file_path):
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                return loader.load()
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                return []

        file_paths = []
        for root, _, files in os.walk(self.project_path):
            for file in files:
                if any(file.endswith(ext) for ext in relevant_extensions):
                    file_paths.append(os.path.join(root, file))

        # Parallel loading
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = executor.map(load_file, file_paths)
            for result in results:
                documents.extend(result)
        
        if not documents:
            print("No documents found!")
            return
            
        # Optimized text splitting
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Increased for better context
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", ",", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        
        print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        
        # Create FAISS index with better parameters
        self.vector_db = FAISS.from_documents(
            documents=chunks,
            embedding=self.embedding,
        )
        
        # Save the index
        self.vector_db.save_local(self.db_path)
        print(f"Vector database saved to {self.db_path}")

    def load_vector_db(self):
        self.vector_db = FAISS.load_local(
            folder_path=self.db_path,
            embeddings=self.embedding,
            allow_dangerous_deserialization=True  # Only if you trust the source
        )


    #implement validating CSV here

    def _validate_and_clean_csv_output(self, output: str, expected_records: int) -> tuple[str, int]:
        """Validate and clean the generated CSV output"""
        try:
            # Clean the output - remove any non-CSV content
            lines = output.strip().split('\n')
            csv_lines = []
            
            # Find the start of CSV data (header line)
            start_idx = 0
            for i, line in enumerate(lines):
                if ',' in line and not line.startswith('#') and not line.startswith('//'):
                    start_idx = i
                    break
            
            # Extract CSV lines
            for line in lines[start_idx:]:
                line = line.strip()
                if line and ',' in line:
                    csv_lines.append(line)
            
            if len(csv_lines) < 2:  # Need at least header + 1 data row
                return "", 0
                
            # Validate CSV structure
            header = csv_lines[0]
            data_rows = csv_lines[1:]
            
            # Remove duplicates while preserving order
            seen_rows = set()
            unique_rows = []
            for row in data_rows:
                row_hash = hashlib.md5(row.encode()).hexdigest()
                if row_hash not in seen_rows:
                    seen_rows.add(row_hash)
                    unique_rows.append(row)
            
            # Reconstruct CSV
            final_csv = header + '\n' + '\n'.join(unique_rows)
            return final_csv, len(unique_rows)
            
        except Exception as e:
            print(f"Error validating CSV output: {e}")
            return "", 0

    def generate_batch_with_retry(self, num_records: int, context: str, max_retries: int = 10) -> str:
        """Generate synthetic data with retry logic for exact count"""
        self.initialize_llm()
        
        total_generated = ""
        remaining_records = num_records
        attempt = 0
        
        while remaining_records > 0 and attempt < max_retries:
            attempt += 1
            print(f"Generation attempt {attempt}, remaining records: {remaining_records}")
            
            # Create prompt for remaining records
            prompt = create_enhanced_generation_prompt(remaining_records, context)
            
            # Generate data
            start_time = time.time()
            result = self.llm(prompt)
            generation_time = time.time() - start_time
            print(f"LLM generation took {generation_time:.2f} seconds")
            
            # Validate and clean output
            clean_csv, actual_count = self._validate_and_clean_csv_output(result, remaining_records)
            
            if actual_count > 0:
                if total_generated == "":
                    total_generated = clean_csv
                else:
                    data_lines = clean_csv.split('\n')[1:]  # Skip header
                    total_generated += '\n' + '\n'.join(data_lines)
                
                remaining_records -= actual_count
                print(f"Generated {actual_count} records, {remaining_records} remaining")
            else:
                print(f"Attempt {attempt} failed to generate valid records")
        return total_generated


    def query_rag_system(self):
        """Enhanced RAG querying with better performance"""
        if not self.vector_db:
            self.load_vector_db()
            
        # Create retriever with optimized parameters
        retriever = self.vector_db.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance for diversity
            search_kwargs={
                "k": 8,  # Retrieve more documents
                "fetch_k": 20,  # Fetch more candidates
                "lambda_mult": 0.7  # Diversity parameter
            }
        )
        
        self.initialize_llm()

        pii_prompt = CLASSIFY_PROMPT

        # PII Analysis
        print("🔍 Analyzing for PII...")
        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": pii_prompt,
                "document_variable_name": "context"
            }
        )
        
        pii_result = qa("Identify all PII columns")
        print("📋 PII Analysis completed")
        print(pii_result["result"])
        self.save_PII(pii_result["result"])

        # Get CSV context and schema
        csv_docs = retriever.get_relevant_documents("CSV data structure columns schema")
        context = "\n".join([doc.page_content for doc in csv_docs])


        # Interactive synthetic data generation
        while True:
            try:
                num_input = input("\n🎯 How many synthetic records? (or 'exit'): ").strip()
                if num_input.lower() in ["exit", "quit", "q"]:
                    break
                    
                num_records = int(num_input)
                if num_records <= 0:
                    print("❌ Please enter a positive number")
                    continue
                    
                if num_records > 10000:
                    confirm = input(f"⚠️  Generating {num_records} records may take time. Continue? (y/n): ")
                    if confirm.lower() != 'y':
                        continue
                
                print(f"🚀 Generating {num_records} synthetic records...")
                start_time = time.time()
                
                # Generate with retry logic
                result = self.generate_batch_with_retry(num_records, context)
                print(result)
                
                total_time = time.time() - start_time
                print(f"⏱️  Total generation time: {total_time:.2f} seconds")
                
                if result:
                    print("✅ Generation completed!")
                    self.save_to_file(num_records = num_records, result = result)
                else:
                    print("❌ Failed to generate synthetic data")

                self.save_to_file(num_records = num_records, result = result)

                    
            except ValueError:
                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n⏹️  Generation interrupted by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
