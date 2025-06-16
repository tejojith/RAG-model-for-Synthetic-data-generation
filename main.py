from codebase_rag import CodebaseRAG
from rag_config import check_for_file

PROJECT_PATH, DB_PATH = check_for_file()

rag = CodebaseRAG(PROJECT_PATH, DB_PATH)

rag.create_embeddings_and_store()   
rag.query_rag_system()