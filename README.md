# 🧠 Elegant Gemini RAG Chatbot 🤖📄

## 📜 Overview

Welcome to the Elegant Gemini RAG Chatbot! This Retrieval-Augmented Generation (RAG) chatbot uses Google Gemini for powerful, context-aware answers based on uploaded PDF documents. It stores document embeddings in ChromaDB, allowing fast retrieval of relevant passages which Gemini then uses to generate grounded responses.

Key capabilities:

- Upload PDFs and index their contents.
- Ask questions about uploaded documents.
- Receive intelligent answers grounded in the document context.

## 🚀 Features

- 📄 Upload PDFs: Seamlessly upload PDF files to be processed and indexed for quick retrieval.
- 🔍 Search & Query: Ask the chatbot questions and get answers that rely only on your uploaded documents.
- 💡 Real-Time Answering: Fast retrieval plus Gemini generation for near-instant responses.
- 📊 Powered by AI: Google Gemini for language generation and ChromaDB for embeddings and retrieval.

## ⚙️ Tech Stack

- Backend: Python + Flask
- Frontend: HTML + Tailwind CSS
- Database: ChromaDB (embeddings storage)
- AI: Google Gemini (generation & embeddings)
- PDF Parsing: PyPDF2
- Environment Management: `.env` for sensitive keys

## 📦 Installation

### 🔧 Clone the repository

```powershell
git clone https://github.com/yourusername/gemini-rag-chatbot.git
cd gemini-rag-chatbot
```

### 🐍 Set up Python environment

Create a virtual environment and install dependencies:

Windows (PowerShell):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1  # or: .\venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 🔑 Add your API key

Create a `.env` file in the root directory and add your Gemini API Key:

```
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

### 🚀 Run the Flask application

```powershell
python app.py

# Open http://127.0.0.1:5000 in your browser
```

## 🧑‍💻 Usage

1️⃣ Upload a PDF

On the left panel, select and upload your PDF file. The backend will process, chunk, and index the file into ChromaDB.

2️⃣ Ask Questions

Once indexed, type your query in the chat box. The chatbot retrieves relevant chunks and generates answers based solely on the uploaded documents.

Interaction example:

User: "What is the main theme of the uploaded document?"
Bot: "The main theme of the document is the importance of AI in modern healthcare systems."

## 🗂️ File Structure

```
gemini-rag-chatbot/
├── app.py                # Main backend logic
├── requirements.txt      # Project dependencies
├── templates/
│   └── index.html        # Frontend interface for user interaction
├── .env                  # Environment variables (Gemini API key)
├── uploaded_files/       # Directory to store uploaded PDF files
└── db/                   # ChromaDB storage for embeddings
```

## 📋 Requirements

- Python 3.7+
- Virtual environment tools (venv)
- Google Gemini API key (available from Google AI platform)

## 🔥 Getting Started

1. Clone the repository and set up your environment.
2. Add your Gemini API key to `.env`.
3. Run the app locally or deploy to your preferred environment.
4. Start interacting with the chatbot!
