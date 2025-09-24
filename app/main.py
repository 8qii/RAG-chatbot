from fastapi import FastAPI
from pydantic import BaseModel
from .rag import retrieve, build_vectorstore
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

class Query(BaseModel):
    question: str

# Build vectorstore lần đầu (chỉ chạy 1 lần khi start server)
if not os.path.exists("db"):
    os.makedirs("db")
    
build_vectorstore("data/wizard_of_laos.txt")

@app.post("/ask")
async def ask(query: Query):
    context = "\n".join(retrieve(query.question))

    prompt = f"""
    Bạn là một chatbot giúp giải thích nội dung dựa trên câu hỏi của người dùng.
    Đây là các đoạn nội dung liên quan:
    {context}

    Câu hỏi: {query.question}

    Hãy trả lời dựa trên nội dung trên. 
    Nếu câu query của người dùng không liên quan đến câu chuyện, hãy trả lời linh hoạt
    Nếu không tìm thấy thông tin, hãy trả lời "Tôi không tìm thấy thông tin trong câu chuyện".
    """

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    return {"answer": response.text}

if __name__ == "__main__":
    test_query = Query(question="Dorothy ແມ່ນໃຜ?")
    import asyncio
    print(asyncio.run(ask(test_query)))