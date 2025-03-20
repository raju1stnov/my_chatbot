import os
import ast
from git import Repo
from pathlib import Path
from py2neo import Graph, Node, Relationship
from sentence_transformers import SentenceTransformer
import chromadb

# Configuration
REPOS = [
    "https://github.com/raju1stnov/my_chatbot.git",
    "https://github.com/raju1stnov/my-rag-metrics-demo.git",
    "https://github.com/raju1stnov/document_parser.git",
    "https://github.com/raju1stnov/document_store.git"
]

# Initialize components
graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def process_repository(repo_url):
    # Clone repo
    repo_name = repo_url.split('/')[-1].replace('.git','')
    Repo.clone_from(repo_url, f"repos/{repo_name}")
    
    # Process files
    for root, _, files in os.walk(f"repos/{repo_name}"):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix == '.md':
                process_markdown(file_path, repo_name)
            elif file_path.suffix == '.py':
                process_python(file_path, repo_name)

def process_markdown(file_path, repo_name):
    with open(file_path) as f:
        content = f.read()
    
    # Create Knowledge Graph nodes
    doc_node = Node("Document", 
                   name=file_path.name,
                   content=content[:1000],
                   repo=repo_name)
    graph.merge(doc_node, "Document", "name")
    
    # Vector DB ingestion
    chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
    embeddings = model.encode(chunks)
    
    collection = chroma_client.get_or_create_collection(name=repo_name)
    collection.add(
        embeddings=embeddings.tolist(),
        documents=chunks,
        ids=[f"{file_path.name}_{i}" for i in range(len(chunks))]
    )

def process_python(file_path, repo_name):
    with open(file_path) as f:
        tree = ast.parse(f.read())
    
    # Extract classes and functions
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_node = Node("Class", 
                             name=node.name,
                             file=file_path.name)
            graph.merge(class_node, "Class", "name")
            
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    func_node = Node("Function",
                                    name=item.name,
                                    docstring=ast.get_docitem))
                    graph.merge(func_node, "Function", "name")
                    graph.create(Relationship(class_node, "CONTAINS", func_node))

def main():
    for repo_url in REPOS:
        process_repository(repo_url)

if __name__ == "__main__":
    main()