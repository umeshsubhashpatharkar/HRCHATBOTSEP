import os
import json
import faiss
import numpy as np
from dotenv import load_dotenv
from rag_utils.chunking import simple_chunk_text

# Load configuration from .env
load_dotenv()
INDEX_STORE = os.getenv("INDEX_STORE", "vector_db")

# Ensure the storage directory exists
if not os.path.exists(INDEX_STORE):
    os.makedirs(INDEX_STORE)

INDEX_PATH = os.path.join(INDEX_STORE, "faiss.index")
DOCS_PATH = os.path.join(INDEX_STORE, "docs.json")
ALLOWED_EXTENSIONS = {'txt'}


def allowed_file(filename):
    """
    Check if the uploaded file is allowed (based on extension).

    Args:
        filename (str): The name of the uploaded file.

    Returns:
        bool: True if file extension is allowed (e.g., .txt), False otherwise.

    Other options:
        - Add support for PDF, DOCX, CSV, JSON, etc.
        - Use `python-magic` or `mimetypes` for MIME-type based validation.
        - Integrate antivirus scan or checksum deduplication here.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def build_index_and_docs(get_embedding_func, file_contents=None, docs=None):
    """
    Build or rebuild a FAISS index and chunk metadata from uploaded documents.

    Args:
        get_embedding_func (callable): Function that takes a text chunk and returns its vector embedding.
        file_contents (list of tuples): Optional. Each tuple = (filename, full_text_content).
        docs (list of dicts): Optional. Already chunked docs (used when reindexing on file deletion).

    Returns:
        tuple: (faiss.Index, list of docs [{content, meta: {filename}}])

    Behavior:
        - Chunks each document into overlapping word-based segments.
        - Embeds each chunk and adds it to FAISS index.
        - Saves `faiss.index` and `docs.json` into INDEX_STORE.

    Other options:
        - Replace FAISS with Azure AI Search, Pinecone, Qdrant, Weaviate, etc.
        - Add caching of embeddings for repeated builds.
        - Normalize embeddings if using cosine similarity search.
        - Persist additional metadata like timestamp, author, file source.
    """
    docs_to_index = []
    embeddings = []

    if file_contents:
        for filename, content in file_contents:
            chunks = simple_chunk_text(content, chunk_size=50, overlap_pct=0.1)
            for chunk in chunks:
                docs_to_index.append({"content": chunk, "meta": {"filename": filename}})
                emb = get_embedding_func(chunk)
                embeddings.append(emb)
    elif docs is not None:
        for doc in docs:
            docs_to_index.append(doc)
            emb = get_embedding_func(doc["content"])
            embeddings.append(emb)
    else:
        # If nothing to embed, clear index
        if os.path.exists(INDEX_PATH): os.remove(INDEX_PATH)
        if os.path.exists(DOCS_PATH): os.remove(DOCS_PATH)
        return None, []

    if not embeddings:
        return None, []

    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings).astype("float32"))

    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(docs_to_index, f)

    return index, docs_to_index


def load_index_and_docs():
    """
    Load FAISS index and chunk metadata from disk.

    Returns:
        tuple: (faiss.Index object, list of docs)

    Other options:
        - Add try/except for corrupt index handling.
        - Support loading per-user or per-organization vector stores.
        - In production, load from cloud storage like Azure Blob or S3.
    """
    if not os.path.exists(INDEX_PATH) or not os.path.exists(DOCS_PATH):
        return None, []

    index = faiss.read_index(INDEX_PATH)
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)

    return index, docs


def rag_chat_search(
    query, k, session, faiss_index, docs,
    get_embedding_func, get_sys_prompt, chatgpt_func
):
    """
    Run RAG (Retrieval-Augmented Generation) for a given user query.

    Args:
        query (str): The user question.
        k (int): Number of top chunks to retrieve.
        session (dict): Flask session (stores chat history).
        faiss_index (faiss.Index): The FAISS vector index.
        docs (list): List of chunks with metadata.
        get_embedding_func (callable): Embedding generator for query and text chunks.
        get_sys_prompt (callable): Returns system prompt for GPT.
        chatgpt_func (callable): Function that calls the LLM with (system, user) prompt.

    Returns:
        tuple: (response_dict, HTTP status code)
            - response_dict contains:
                - results: matched docs
                - answer: LLM response
                - history: updated session chat

    Other options:
        - Add semantic re-ranking (e.g., using a scoring model like Cohere Reranker).
        - Use cosine similarity by normalizing vectors if not using L2.
        - Perform hybrid retrieval (FAISS + keyword search).
        - Support per-user document isolation or filtering.
        - Add real-time streaming response using SSE/WebSocket.
        - Store full chat session in a DB (MongoDB, PostgreSQL).
    """
    if not docs or not faiss_index:
        return {"error": "No documents found in database. Please upload files first."}, 400
    if not query:
        return {"error": "Query text required."}, 400

    history = session.get("history", [])

    emb = get_embedding_func(query).reshape(1, -1)
    D, I = faiss_index.search(emb, min(k, len(docs)))
    top_docs = [docs[idx] for idx in I[0]]
    context = "\n".join([doc["content"] for doc in top_docs])

    # Append past 5 rounds of conversation
    conversation = ""
    for turn in history[-5:]:
        conversation += f"User: {turn['user']}\nAssistant: {turn['bot']}\n"
    conversation += f"User: {query}\nAssistant:"

    user_prompt = f"Context:\n{context}\n\n{conversation}"
    system_prompt = get_sys_prompt()
    answer = chatgpt_func(system_prompt, user_prompt)

    history.append({"user": query, "bot": answer})
    session["history"] = history

    results = []
    for doc, dist in zip(top_docs, D[0]):
        results.append({
            "score": float(dist),
            "content": doc.get("content"),
            "meta": doc.get("meta", {})
        })

    return {"results": results, "answer": answer, "history": history}, 200
if __name__ == "__main__":
    print("FAISS utils module loaded successfully.")