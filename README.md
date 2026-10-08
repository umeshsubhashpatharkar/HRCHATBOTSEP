# HR Assistant RAG Chatbot (FAISS + Flask + OpenAI)

A production-ready HR Assistant chatbot powered by Retrieval-Augmented Generation (RAG) using [FAISS](https://github.com/facebookresearch/faiss), [OpenAI/Azure OpenAI](https://platform.openai.com/docs/), and Flask.

* **Upload** HR policy docs (TXT files) from anywhere—never saved to a static data folder.
* **Chunked and embedded** on upload, indexed with FAISS for fast retrieval.
* **Interactive chat UI** with chat history, context-aware answers, and GPT-powered natural language.
* **Extensible**: Ready for per-user document isolation (add login/registration with any DB).
* **No cloud vector DB required** (pure FAISS, runs anywhere: local, Azure, etc.).

---

## 🚀 Features

* Upload, delete, and search your own HR policy/text files (TXT)
* Real-time context-aware answers using OpenAI GPT (API key required)
* No external DB for RAG—uses local FAISS index and JSON doc store (configurable location)
* Easily extend to multi-user: each user can have their own isolated vector DB and doc store
* Secure session management with Flask secret key
* Modular, maintainable Python code

---

## 🛠️ Requirements

* Python 3.8+
* [FAISS CPU](https://github.com/facebookresearch/faiss)
* Flask
* OpenAI Python SDK (for Azure or OpenAI endpoints)
* numpy
* python-dotenv
* werkzeug

**Optional:**

* scikit-learn (only if you use cosine similarity outside FAISS)
* gunicorn (for production WSGI serving)

---

## 📦 Installation

```bash
git clone https://github.com/your-org/hr-assistant-rag.git
cd hr-assistant-rag
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## ⚙️ Configuration

1. **Copy `.env.example` to `.env`** and fill in your keys:

   ```ini
   AZURE_OPENAI_KEY=your-azure-openai-key
   AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
   AZURE_OPENAI_API_VERSION=2024-12-01-preview
   AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
   AZURE_OPENAI_GPT_DEPLOYMENT=gpt-4o
   FLASK_SECRET_KEY=your_random_secret
   INDEX_STORE=vector_db
   ```
2. `INDEX_STORE` sets where your vector index and docs store will be saved.
   (Default is `vector_db/` in your project root.)

---

## 🏃‍♂️ Running the App

```bash
python main.py
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

**For production:**

```bash
gunicorn -w 4 -b 0.0.0.0:8000 main:app
```

---

## 🖼️ UI Features

* Home page: Chat interface (ask questions, get HR answers)
* "Manage" page: Upload and delete TXT files (index rebuilds automatically)
* Upload supports multi-file selection
* Chat remembers history until you click "New Chat"
* All files processed in-memory; only FAISS index and docs.json saved

---

## 👤 Per-User Document Storage (Optional)

* For multi-user support, add login/registration (SQLite, MongoDB, or other DB)
* Store each user’s FAISS/docs under `vector_db/{user_id}/`
* All retrieval, upload, delete routes reference the current user’s folder

---

## 🗂️ Project Structure

```
├── main.py
├── requirements.txt
├── .env
├── vector_db/            # FAISS index/docs store
├── rag_utils/
│    ├── __init__.py
│    ├── faiss_utils.py
│    ├── embedding.py
│    ├── gpt.py
│    ├── prompts.py
│    └── chunking.py
└── templates/
     └── index.html
```

---

## 🔐 Security Tips

* **NEVER commit `.env` or secrets to git.**
* Use a strong, random `FLASK_SECRET_KEY`.
* Set max upload size in Flask config for production.
* Always run Flask apps behind a real WSGI server (not `debug=True` in production).
* For multi-user, always hash passwords (see Flask/MongoDB/SQLite integration).

---

## 🧩 Extending

* Add login/registration (see Flask-Login or MongoDB/SQLite guide in this repo).
* Deploy to Azure Web App, AWS, GCP, or anywhere Python runs.
* Integrate more chunking, hybrid retrieval, or document types as needed.

---

## 🙏 Credits

* [Facebook FAISS](https://github.com/facebookresearch/faiss)
* [OpenAI](https://platform.openai.com/)
* [Flask](https://flask.palletsprojects.com/)
* Bootstrap (for the UI)

---

## 📬 Feedback

Open an issue or PR for improvements.
Happy RAG hacking!

---

*This project is intended for educational and prototyping purposes. For enterprise use, review security and scaling requirements carefully.*
