import os
from langchain_community.vectorstores import FAISS
#from langchain_community.embeddings import OllamaEmbeddings
#from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
#from :class:`~langchain_ollama import OllamaLLM`
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
import json
import hashlib
from typing import List, Dict, Any
import pandas as pd


from multiprocessing import Pool
from langchain_community.document_loaders import (TextLoader, PythonLoader, 
                                                        JSONLoader, BSHTMLLoader)


from llm_prompt import GENERATION_PROMPT, CLASSIFY_PROMPT

import csv
import io
import re



class CodebaseRAG:
    def __init__(self, project_path, db_path):
        self.project_path = project_path
        self.db_path = db_path
        self.embed_model = "nomic-embed-text"  # or mxbai-embed-large for code
        self.embedding = OllamaEmbeddings(model=self.embed_model)
        self.vector_db = None
        self.llm = None
        self.schema_cache = {}
        self.generated_records_cache = set()  # Track generated records to avoid duplicates


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

    def _extract_csv_schema(self, csv_content: str) -> Dict[str, Any]:
        """Extract and analyze CSV schema for better synthetic data generation"""
        try:
            # Parse CSV to understand structure
            df = pd.read_csv(io.StringIO(csv_content))
            
            schema = {
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'sample_values': {},
                'value_ranges': {},
                'unique_counts': {},
                'null_counts': df.isnull().sum().to_dict()
            }
            
            for col in df.columns:
                # Get sample values (non-null)
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    schema['sample_values'][col] = non_null_values.sample(
                        min(10, len(non_null_values))
                    ).tolist()
                
                # Get value ranges for numeric columns
                if df[col].dtype in ['int64', 'float64']:
                    schema['value_ranges'][col] = {
                        'min': df[col].min(),
                        'max': df[col].max(),
                        'mean': df[col].mean(),
                        'std': df[col].std()
                    }
                
                # Get unique value counts
                schema['unique_counts'][col] = df[col].nunique()
            
            return schema
            
        except Exception as e:
            print(f"Error extracting schema: {e}")
            return {}


    def save_PII(self, answer):
        new  = answer[answer.index(":")+3:] 
        output_format = "txt"
        output_file = f"PII.{output_format}"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"{new}\n{'-'*40}\n")
            
    def save_to_file(self, num_records, result):
        new  = result[result.index(":")+3:]
        output_file = f"synthetic_data_{num_records}_records.csv"
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            f.write(new)
        print(f"Data saved to {output_file}")
        
    def get_loader(file_path):
        if file_path.endswith('.py'):
            return PythonLoader(file_path)
        elif file_path.endswith('.json'):
            return JSONLoader(file_path)
        elif file_path.endswith('.html'):
            return BSHTMLLoader(file_path)
        else:
            return TextLoader(file_path)

    def create_embeddings_and_store(self):
        # Load documents (optimized version)
        relevant_extensions = ['.py', '.sql', '.txt', '.json', '.html', '.md','.csv']
        documents = []
        for root, _, files in os.walk(self.project_path):
            for file in files:
                if any(file.endswith(ext) for ext in relevant_extensions):
                    try:
                        loader = TextLoader(os.path.join(root, file))
                        documents.extend(loader.load())
                    except Exception as e:
                        print(f"Error loading {file}: {e}")
        
        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False
        )
        chunks = splitter.split_documents(documents)
        
        # Create FAISS index with optimized parameters
        self.vector_db = FAISS.from_documents(
            documents=chunks,
            embedding=self.embedding,
            distance_strategy="METRIC_INNER_PRODUCT"  # Faster than L2 for many cases
        )
        
        # Save the index
        self.vector_db.save_local(self.db_path)



    def load_vector_db(self):
        self.vector_db = FAISS.load_local(
            folder_path=self.db_path,
            embeddings=self.embedding,
            allow_dangerous_deserialization=True  # Only if you trust the source
        )
    

    def query_rag_system(self):
        if not self.vector_db:
            self.load_vector_db()
        retriever = self.vector_db.as_retriever()
        
        llm = OllamaLLM(model="llama3",
                        temperature=0.1,  # Less randomness
                        top_k=10,  # Faster sampling
                        repeat_penalty=1.1  # Prevent repetition
                        )  # Use any: mistral, wizardcoder, codellama
        
        #first checking for PII
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": CLASSIFY_PROMPT,
                "document_variable_name": "context"  # Make sure this matches your prompt
            }
        )

        print("Got the files")
        print("Checking for PII")

        ques = "Give me all the PII and only that"
        result = qa(ques)
        answer = result["result"]
        print("Here are list of PII, also saved in the file - PII.txt ")
        print(answer)
        self.save_PII(answer)

        # Create a custom chain that can handle our parameters
        def generate_synthetic_data(query_dict):
            # Get the sample data context
            sample_docs = retriever.get_relevant_documents("data.csv structure")
            context = "\n".join([doc.page_content for doc in sample_docs])
            
            # Format the prompt with all required variables
            formatted_prompt = GENERATION_PROMPT.format(
                num_records=query_dict["num_records"],
                context=context
            )
            
            # Generate the synthetic data
            return llm(formatted_prompt)    

        while True:
            num_records = input("\n🔍 How many synthetic records do you want to generate? (or type 'exit'): ")
            if num_records.lower() in ["exit", "quit"]:
                break
            
            num_records = int(num_records)
            if num_records <= 0:
                print("Please enter a positive number")
                continue

            result = generate_synthetic_data({"num_records": str(num_records)})
            
            # Clean and save the output
            print("\nGenerated Synthetic Data:")
            print(result)
            
            # Improved CSV saving
            self.save_to_file(num_records, result)
