Autonomous Ai research Assistant

Ask your research paper question. Get instant ,source-backend answer.

A local RAG system that reads your PDFs and answers using a local LLM. No cloud. No API keys. Just your documents + AI.
### What it does
- **Chat with your PDF** - Ask questions in plain English
- **RAG Powered** - Retrieves real passages before answering
- **100% Local** - Runs on Ollama + Llama 3.2 on your machine
- **Shows its work** - Every answer comes with source, page no, and similarity score
- **Clean UI** - Simple web interface + REST API
- **Handles off-topic** - Says "I don't know" when the answer isn't in the PDF
<img width="1881" height="887" alt="Screenshot 2026-10-20 190657" src="https://github.com/user-attachments/assets/73032e28-e77b-409e-9965-bada2316bb86" />

### 🛠 Stack
Python • FastAPI • Sentence-Transformers • FAISS • Ollama / Llama 3.2 • PyMuPDF
![Uploading Screenshot 2026-10-20 191020.png…]()

### How it works
`PDF -> Text Chunks -> Embeddings -> FAISS Index -> Your Question -> Top Chunks -> Llama 3.2 -> Answer + Sources`
1. Ingests and chunks your PDF
2. Creates embeddings and stores in FAISS
3. Finds the most relevant chunks for your question
4. LLM generates answer using only that context

🚀 Getting Started

Prerequisites

Install the following:

Python 3.10 or a compatible version

Ollama

Git

A research paper in PDF format

1. Clone the Repository

git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY

Replace the repository URL with your actual GitHub repository URL.

2. Create a Virtual Environment

python -m venv venv

Activate it on Windows:

venv\Scripts\activate

3. Install Dependencies

pip install -r requirements.txt

4. Set Up Ollama

Install Ollama from https://ollama.com/

Download the model:

ollama pull llama3.2:1b

Ensure that Ollama is running locally before starting the application.

5. Add a Research Paper

Place your PDF in the project's document directory and follow the document-loading process implemented in the project.

Ensure that the filename and path match your configuration.

6. Start the Backend

From the project root, run:

python -m uvicorn server:app --reload

7. Open the Application

Web interface:

http://127.0.0.1:8000/ui

API documentation:

http://127.0.0.1:8000/docs

Health check:

http://127.0.0.1:8000/health

The interface and API URLs assume your FastAPI routes are configured as described.

🧪 Testing

The following scenarios have been tested manually:

Test

Expected behaviour

Observed result

Question relevant to the PDF

Generate an answer using relevant document passages

Relevant answer and retrieved passages returned

Question unrelated to the PDF

Return a fallback when the document does not contain the answer

Fallback response returned for the tested question

Source inspection

Display passages with source and page information

Source passages displayed in the UI

Web interface

Submit a question and display the response

Working locally

These are manual test results, not a formal benchmark of retrieval accuracy or hallucination rates.
