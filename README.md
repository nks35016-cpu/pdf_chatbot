# 📄 PDF Q&A Chatbot 🤖

An AI-powered chatbot that allows users to upload PDF documents and ask questions based on their content. The chatbot intelligently extracts, processes, and retrieves relevant information using modern NLP and vector search techniques.

---

## 🚀 Features

- 📂 Upload and analyze PDF documents
- 🔍 Ask questions in natural language
- ⚡ Fast semantic search using embeddings
- 🧠 Context-aware answers using LLMs
- 💬 Conversational interface
- 📑 Supports multiple PDFs (optional)

---

## 🛠️ Tech Stack

- Python
- LangChain / LlamaIndex
- OpenAI / Local LLM (optional)
- FAISS / ChromaDB (Vector Database)
- Streamlit / Flask (Frontend)

---

## ⚙️ How It Works

1. Upload PDF
2. Extract text from document
3. Split text into chunks
4. Convert chunks into embeddings
5. Store embeddings in vector database
6. User asks question
7. Relevant chunks retrieved
8. LLM generates answer

---

## 📦 Installation

```bash
git clone https://github.com/your-username/pdf-qa-chatbot.git
cd pdf-qa-chatbot
pip install -r requirements.txt
