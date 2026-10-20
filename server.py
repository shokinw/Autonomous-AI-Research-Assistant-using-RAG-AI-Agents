from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag import initialize_rag, search_documents
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import requests


# Store the RAG components in memory.
rag_state = {}

def generate_answer(question, passages):
    context = "\n\n".join(
        f"Source: {item['source']}, page {item['page']}\n{item['text']}"
        for item in passages
    )

    prompt = f"""
You are a research assistant. Answer the user's question using only
the provided context.

Rules:
- Use only facts supported by the context.
- If the answer is not in the context, say:
  "I couldn't find that answer in your research paper."
- Do not invent sources or page numbers.
- Explain clearly in simple English.

Context:
{context}

Question:
{question}

Answer:
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:1b",
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not get a response from Ollama: {exc}",
        )
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG once when the API server starts."""
    print("Initializing the research assistant...")

    try:
        model, index, chunks = initialize_rag()

        rag_state["model"] = model
        rag_state["index"] = index
        rag_state["chunks"] = chunks

        print("Research assistant is ready!")

        yield

    finally:
        rag_state.clear()


app = FastAPI(
    title="Physical AI Research Assistant",
    description="A local PDF-based RAG backend for our future physical AI device",
    version="0.2.0",
    lifespan=lifespan,
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)


@app.get("/ui", include_in_schema=False)
def user_interface():
    return FileResponse(STATIC_DIR / "index.html")


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "project": "Physical AI Research Assistant",
        "status": "running",
        "message": "RAG backend is running!",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "rag_ready": bool(rag_state),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question.",
        )

    if not rag_state:
        raise HTTPException(
            status_code=503,
            detail="The RAG system is not ready yet.",
        )

    results = search_documents(
        question=question,
        model=rag_state["model"],
        index=rag_state["index"],
        chunks=rag_state["chunks"],
        top_k=3,
    )

    answer = generate_answer(question, results)

    return {
        "status": "success",
        "question": question,
        "answer": answer,
        "retrieved_passages": [
            {
                "text": result["text"],
                "source": result["source"],
                "page": result["page"],
                "similarity": round(result["score"], 3),
            }
            for result in results
        ],
    }