from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_local_embedding(text: str):
    return model.encode(text).tolist()
