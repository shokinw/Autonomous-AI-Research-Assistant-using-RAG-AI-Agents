from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents():
    """Extract readable text from every PDF in the documents folder."""
    documents = []

    for pdf_path in DOCUMENTS_DIR.glob("*.pdf"):
        reader = PdfReader(str(pdf_path))

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            if text.strip():
                documents.append({
                    "text": text,
                    "source": pdf_path.name,
                    "page": page_number,
                })

    return documents


def split_into_chunks(text, chunk_size=180, overlap=40):
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def initialize_rag():
    """Load the PDF, create embeddings, and build the FAISS index."""
    pages = load_documents()

    if not pages:
        raise ValueError(
            "No readable PDF text found. "
            "Check that a text-based PDF is inside the documents folder."
        )

    chunks = []

    for page in pages:
        for chunk in split_into_chunks(page["text"]):
            chunks.append({
                "text": chunk,
                "source": page["source"],
                "page": page["page"],
            })

    if not chunks:
        raise ValueError("No text chunks were created from the PDF.")

    print(f"Loaded {len(pages)} pages.")
    print(f"Created {len(chunks)} chunks.")
    print("Loading the embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        [chunk["text"] for chunk in chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    print("RAG index is ready.")

    return model, index, chunks


def search_documents(question, model, index, chunks, top_k=3):
    """Retrieve the most relevant chunks for a question."""
    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    number_to_return = min(top_k, len(chunks))
    scores, positions = index.search(
        query_embedding,
        number_to_return,
    )

    results = []

    for score, position in zip(scores[0], positions[0]):
        if position < 0:
            continue

        results.append({
            **chunks[int(position)],
            "score": float(score),
        })

    return results


if __name__ == "__main__":
    model, index, chunks = initialize_rag()

    question = input("\nYour question:\n> ").strip()

    if not question:
        print("Please enter a question.")
    else:
        results = search_documents(question, model, index, chunks)

        print("\nMost relevant passages:\n")

        for number, result in enumerate(results, start=1):
            print(f"--- Result {number} ---")
            print(f"Source: {result['source']}")
            print(f"Page: {result['page']}")
            print(f"Similarity: {result['score']:.3f}")
            print(result["text"])
            print()