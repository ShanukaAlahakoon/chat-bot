import os
from google import genai
import pypdf
import chromadb
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import numpy as np

load_dotenv()
app = Flask(__name__)
CORS(app)  # Enable CORS if needed for frontend communication

# --- 1. DEFINE PERSISTENT FOLDERS ---
UPLOAD_FOLDER = 'uploaded_files'
DB_FOLDER = 'db'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DB_FOLDER, exist_ok=True)

# --- 2. GEMINI SETUP ---
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not set. Please add your API key to the .env file.")
client = genai.Client(api_key=api_key)
print("API key configured successfully.")

# --- 3. MODEL NAMES AND ChromaDB SETUP ---
SYSTEM_INSTRUCTION = """
You are a helpful assistant. Your task is to answer the user's question based *only* on the provided context.
- Do not use any external knowledge.
- If the answer is not found in the context, you *must* state: "I'm sorry, but that information is not available in the provided document."
- Be concise and directly answer the question using only the information from the context.
"""
try:
    generation_model_name = 'gemini-2.5-flash-preview-09-2025'
    print("Generative model name set.")
    embedding_model_name = 'models/text-embedding-004'
    print("Embedding model name set.")

    chroma_client = chromadb.PersistentClient(path=DB_FOLDER)
    collection = chroma_client.get_or_create_collection(name="pdf_chatbot_collection")
    print("ChromaDB client and collection initialized.")

except Exception as e:
    print(f"Error initializing models or ChromaDB: {e}")
    generation_model_name = None
    embedding_model_name = None
    collection = None

# --- 4. HELPER FUNCTIONS ---

def extract_and_chunk_pdf(file_path):
    """Extracts text from a PDF and splits it into chunks."""
    print(f"Reading {file_path}...")
    try:
        pdf_reader = pypdf.PdfReader(file_path)
        full_text = ""
        for page in pdf_reader.pages:
            full_text += page.extract_text() + "\n\n"
        print("Chunking text...")
        chunks = full_text.split("\n\n")
        chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50]
        print(f"Created {len(chunks)} text chunks.")
        return chunks
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return []

def embed_text(text):
    """Embeds a single piece of text using the embedding model."""
    try:
        result = client.models.embed_content(
            model=embedding_model_name,
            contents=text
        )
        # result.embeddings[0].values is the actual embedding vector
        return result.embeddings[0].values
    except Exception as e:
        print(f"Error embedding text: {e}")
        return None

def get_chat_response(query, context):
    """Generates a chat response using the Gemini model, based on context."""
    if not generation_model_name:
        return "Error: Generation model is not initialized."
    final_prompt = f"""
Context:
---
{context}
---

Question:
{query}

Answer:
    """
    try:
        response_tuple = client.models.generate_content(
            model=generation_model_name,
            contents=final_prompt
            
            ),
        response = response_tuple[0] if isinstance(response_tuple, tuple) else response_tuple

    
        candidate = response.candidates[0]
        parts = candidate.content.parts
        answer_text = ""
        for part in parts:
            if hasattr(part, 'text'):
                answer_text += part.text
            return answer_text
       
    except Exception as e:
        print(f"Error generating chat response: {e}")
        return "Sorry, I encountered an error while generating the answer."

# --- 5. DIAGNOSTICS ---
def print_collection_count():
    try:
        meta = collection.get()
        docs = meta.get('documents', [])
        print(f"Current collection count: {len(docs)}")
    except Exception as e:
        print(f"Error accessing collection count: {e}")

# --- 6. FLASK ENDPOINTS ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if not collection:
        print("ERROR: No ChromaDB collection on /upload")
        return jsonify({"error": "Database not initialized"}), 500

    if 'file' not in request.files:
        print("ERROR: No file part in request")
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        print("ERROR: No selected file")
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.lower().endswith('.pdf'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        print(f"File saved to {file_path}")

        try:
            print(f"=== UPLOAD START ===")
            print(f"Filename used for doc_id: '{filename}'")

            text_chunks = extract_and_chunk_pdf(file_path)
            print(f"PDF chunk count: {len(text_chunks)}")
            if not text_chunks:
                print("No chunks extracted.")
                return jsonify({"error": "Could not extract text from PDF"}), 400

            embeddings_list, chunks_list, metadatas_list, ids_list = [], [], [], []
           
            print("Embedding chunks (this may take a moment)...")

            knowledge_base = []

            for i, chunk in enumerate(text_chunks):
                if (i+1)%10 == 0 or ((i+1) == len(text_chunks)):
                    print(f"Embedding chunk {i+1}/{len(text_chunks)}...")

                embedding = embed_text(chunk)
                if embedding:
                    knowledge_base.append({
                        'text':chunk,
                        'embedding':np.array(embedding)
                    })    
                    embeddings_list.append(embedding)
                    chunks_list.append(chunk)
                    metadatas_list.append({"doc_id": filename})
                    ids_list.append(f"{filename}_{i}")

            if embeddings_list:
                collection.add(
                    embeddings=embeddings_list,
                    documents=chunks_list,
                    metadatas=metadatas_list,
                    ids=ids_list
                )
                print(f"Successfully added {len(ids_list)} chunks to ChromaDB.")

                # # Print all doc_ids now in collection
                # meta = collection.get()
                # doc_ids_in_db = [m.get('doc_id') for m in meta.get('metadatas', [])]
                # #print(f"All doc_ids in DB: {doc_ids_in_db}")

            print("=== UPLOAD END ===")
            return jsonify({
                "message": f"File '{filename}' processed successfully.",
                "doc_id": filename
            }), 200

        except Exception as e:
            print(f"Error processing file: {e}")
            return jsonify({"error": f"Error processing file: {e}"}), 500
    else:
        print("ERROR: Invalid file type on upload")
        return jsonify({"error": "Invalid file type. Please upload a PDF."}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    if not collection or not generation_model_name:
        print("ERROR: Backend services not initialized on /ask")
        return jsonify({"error": "Backend services not initialized"}), 500

    try:
        print("=== ASK START ===")
        data = request.json
        print(f"Request JSON: {data}")
        if not data or 'query' not in data or 'doc_id' not in data:
            print("ERROR: Missing query or doc_id in request")
            return jsonify({"error": "Missing 'query' or 'doc_id' in request"}), 400

        query = data['query']
        doc_id = data['doc_id']
        normalized_doc_id = secure_filename(doc_id)
        print(f"Ask for doc_id: '{doc_id}'")
        print(f"Normalized doc_id: '{normalized_doc_id}'")

        # Print all doc_ids in DB (again, for debug)
        meta = collection.get()
        doc_ids_in_db = [m.get('doc_id') for m in meta.get('metadatas', [])]
        # print(f"All doc_ids in DB at ask time: {doc_ids_in_db}")

        if doc_id not in doc_ids_in_db:
            print(f"WARNING: doc_id '{doc_id}' does not exist in DB!")

        query_embedding = embed_text(query)
        if query_embedding is None:
            print("ERROR: Could not embed query")
            return jsonify({"error": "Could not embed query"}), 500

        #print(f"Query embedding: {query_embedding}")

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            where={"doc_id": normalized_doc_id}
        )
        # print(f"Query result keys: {list(results.keys())}")
        # print(f"Query results: {results}")

        if not results['documents'] or not results['documents'][0]:
            print("No relevant context found in DB for this doc_id.")
            print("=== ASK END ===")
            return jsonify({"answer": "I'm sorry, but that information is not available in the provided document."}), 200

        context = "\n---\n".join(results['documents'][0])
        #print(f"--- Retrieved Context ---\n{context}\n-------------------------")

        answer = get_chat_response(query, context)
        print(f"--- Generated Answer ---\n{answer}\n------------------------")
        print("=== ASK END ===")
        return jsonify({"answer": answer}), 200
    except Exception as e:
        print(f"Error during /ask: {e}")
        return jsonify({"error": f"An error occurred: {e}"}), 500

# --- 7. RUN FLASK WITHOUT AUTO-RELOAD ---
if __name__ == '__main__':
    print("Starting Flask in NO debug mode (persistence safe).")
    app.run(debug=False)
