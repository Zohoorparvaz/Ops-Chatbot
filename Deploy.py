# Deploy.py


from fastapi import FastAPI, Request
from pydantic import BaseModel
import pickle, faiss, numpy as np
from openai import AzureOpenAI
import os

# === FastAPI app ===
app = FastAPI()

# === Load your embedding + chunks ===
with open("text_chunks.pkl", "rb") as f:
    text_chunks = pickle.load(f)
with open("embeddings.pkl", "rb") as f:
    embedding_matrix = np.array(pickle.load(f), dtype="float32")

# === Azure OpenAI Setup ===
client = AzureOpenAI(
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    api_version="2024-12-01-preview",
    azure_endpoint="https://aaron-mb5yqktn-eastus2.cognitiveservices.azure.com/"
)
embedding_model = "text-embedding-3-small"
chat_model = "o4-mini"

# === Model input ===
class ChatRequest(BaseModel):
    question: str

# === Chat memory ===
chat_log = []

# === Retrieve chunks ===
def retrieve_chunks_np(user_question, k=15):
    query = client.embeddings.create(
        input=user_question,
        model=embedding_model
    ).data[0].embedding
    query = np.array(query, dtype="float32")
    matrix_norm = embedding_matrix / np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    query_norm = query / np.linalg.norm(query)
    similarities = np.dot(matrix_norm, query_norm)
    top_k_idx = np.argsort(similarities)[-k:][::-1]
    top_chunks = [text_chunks[i] for i in top_k_idx]
    return "\n\n".join(top_chunks)

# === Answer generation ===
def generate_answer_from_context(context, user_question):
    prompt = f"""
You are a highly structured and detail-oriented assistant who helps employees locate and understand company procedures, sections, and forms from internal documentation. 

Your job:
1. Summarize the relevant information clearly.
2. If any **forms, section numbers, or document titles** are mentioned in the context, and they appear to have associated links, include them using **Markdown hyperlinks** like: [Tour Waiver](https://link.com).
3. Do **not** make up links. Only include a link if the name and URL appear in the context or are clearly stated.
4. Organize your answer using clear bullet points or numbers.
5. Prioritize helping the user **find what to click on** to take action.
6. Display all sections and topics that you retrieve. For example Safety, project planning, or Change Control 

Context:
{context}

Question: {user_question}

Answer:
"""

    messages = [
        {"role": "system", "content": "You are a link-aware internal assistant. Prioritize clarity and actionable hyperlinks when answering questions about procedures or forms."}
    ]
    for chat in chat_log[-3:]:
        messages.append({"role": "user", "content": chat["user"]})
        messages.append({"role": "assistant", "content": chat["response"]})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=chat_model,
        messages=messages,
        max_completion_tokens=5000
    )
    answer = response.choices[0].message.content.strip()
    chat_log.append({"user": user_question, "context": context, "response": answer})
    return answer

# === Route ===
@app.post("/chat")
async def chat_with_bot(request: ChatRequest):
    context = retrieve_chunks_np(request.question)
    answer = generate_answer_from_context(context, request.question)
    return {"answer": answer}

