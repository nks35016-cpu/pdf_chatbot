import requests
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv("api.env")
key = os.getenv("GROQ_API_KEY")

if not key:
    raise ValueError("API key not found. Check your api.env file.")

headers = {
    "Authorization": f"Bearer {key}",
    "Content-Type": "application/json"
}

# Step 1 - read PDF
PDF_FILE = "ML.PDF"

if not os.path.exists(PDF_FILE):
    raise FileNotFoundError(f"{PDF_FILE} not found. Check filename.")

reader = PdfReader(PDF_FILE)
total_pages = len(reader.pages)
print(f"Total pages in PDF: {total_pages}")

text = ""
for i, page in enumerate(reader.pages[50:350]):
    try:
        extracted = page.extract_text()
        if extracted and isinstance(extracted, str):
            text += extracted
    except Exception as e:
        print(f"Skipping page {i} due to error: {e}")
        continue

print(f"Total characters extracted: {len(text)}")

if len(text) < 100:
    raise ValueError("Not enough text extracted. Check page range.")

# Step 2 - chunk
def split_text(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

raw_chunks = split_text(text)

# Clean every chunk — this prevents the tokenizer error
chunks = []
for c in raw_chunks:
    if not c:
        continue
    if not isinstance(c, str):
        continue
    cleaned = c.strip()
    if len(cleaned) < 100:
        continue
    # Remove null bytes and weird characters
    cleaned = cleaned.replace('\x00', '')
    cleaned = cleaned.encode('utf-8', errors='ignore').decode('utf-8')
    if cleaned:
        chunks.append(cleaned)

print(f"Total clean chunks: {len(chunks)}")

if len(chunks) == 0:
    raise ValueError("No valid chunks found. Check your PDF.")

# Step 3 - embeddings
print("Creating embeddings — this takes 2-3 minutes...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode in small batches to avoid memory issues
def encode_in_batches(model, chunks, batch_size=32):
    all_embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        try:
            batch_embeddings = model.encode(batch)
            all_embeddings.extend(batch_embeddings.tolist())
            print(f"Processed {min(i+batch_size, len(chunks))}/{len(chunks)} chunks")
        except Exception as e:
            print(f"Skipping batch {i} due to error: {e}")
            # Add zero embeddings for failed batch
            for _ in batch:
                all_embeddings.append([0.0] * 384)
    return all_embeddings

embeddings = encode_in_batches(model, chunks)
print(f"Embeddings done: {len(embeddings)}")

# Step 4 - store in ChromaDB
client = chromadb.Client()

# Delete collection if exists to avoid duplicate errors
try:
    client.delete_collection("ml_book")
except:
    pass

collection = client.create_collection("ml_book")

# Add in batches to avoid memory issues
def add_in_batches(collection, chunks, embeddings, batch_size=50):
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size]
        batch_ids = [f"chunk_{j}" for j in range(i, i+len(batch_chunks))]
        try:
            collection.add(
                documents=batch_chunks,
                embeddings=batch_embeddings,
                ids=batch_ids
            )
        except Exception as e:
            print(f"Skipping batch {i} due to error: {e}")
    print(f"Stored {collection.count()} chunks in database")

add_in_batches(collection, chunks, embeddings)

# Step 5 - search by meaning
def find_relevant_chunks(question, n=5):
    try:
        question_embedding = model.encode([question]).tolist()
        results = collection.query(
            query_embeddings=question_embedding,
            n_results=min(n, collection.count())
        )
        return results["documents"][0]
    except Exception as e:
        print(f"Search error: {e}")
        return []

# Step 6 - ask AI
def ask(question):
    relevant = find_relevant_chunks(question)

    if not relevant:
        return "Sorry, I could not find relevant information."

    context = "\n\n".join(relevant)

    body = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "system",
                "content": "You are an ML tutor. Answer clearly and in simple terms based only on the context provided. If the answer is not in the context say I don't know."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=body,
            timeout=30
        )
        data = response.json()

        if "choices" not in data:
            print(f"API error: {data}")
            return "API error — check your key."

        return data["choices"][0]["message"]["content"]

    except requests.exceptions.Timeout:
        return "Request timed out. Try again."
    except Exception as e:
        return f"Error: {e}"

# Step 7 - chat loop
print("\nPDF bot ready. Type quit to exit.\n")
while True:
    try:
        question = input("You: ")
        if question.lower() == "quit":
            print("Bye!")
            break
        if not question.strip():
            print("Please type a question.")
            continue
        answer = ask(question)
        print(f"\nBot: {answer}\n")
    except KeyboardInterrupt:
        print("\nBye!")
        break