import chromadb
from .embedding import get_local_embedding
from .utils import chunk_text
import os

# Tạo ChromaDB client
chroma_client = chromadb.PersistentClient(path="db")
collection = chroma_client.get_or_create_collection("story")

def build_vectorstore(file_path: str):
    """Load file điều khoản và index vào vector DB"""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        embedding = get_local_embedding(chunk)
        collection.add(documents=[chunk], embeddings=[embedding], ids=[str(i)])
    print(f"Indexed {len(chunks)} chunks vào vector DB.")
    print("Chunks:", chunks[:5])  

def retrieve(query: str, top_k=5):
    """Truy vấn vector DB"""
    query_emb = get_local_embedding(query)
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k
    )
    print("Kết quả truy vấn:", results)
    return results['documents'][0]
