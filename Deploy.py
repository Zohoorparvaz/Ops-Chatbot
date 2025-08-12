# Deploy.py
from fastapi import FastAPI
from pydantic import BaseModel
import pickle, numpy as np
from openai import AzureOpenAI
import os

app = FastAPI()

# Health check so Azure knows we're alive
@app.get("/health")
async def health():
    return {"status": "ok"}

# --- Load data ---
with open("text_chunks.pkl", "rb") as f:
    text_chunks = pickle.load(f)
with open("embeddings.pkl", "rb") as f:
    embedding_matrix = np.array(pickle.load(f), dtype="float32")

# Pre-normalize once (faster, safer)
_eps = 1e-12
row_norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True) + _eps
embedding_matrix_unit = embedding_matrix / row_norms

# --- Azure OpenAI ---
EMBEDDING_MODEL = "text-embedding-3-small"  # your deployment name
CHAT_MODEL = "o4-mini"                      # your deployment name

def get_client() -> AzureOpenAI:
    key = os.getenv("AZURE_OPENAI_API_KEY")
    if not key:
        raise RuntimeError("AZURE_OPENAI_API_KEY is not set")
    return AzureOpenAI(
        api_key=key,
        api_version="2024-12-01-preview",
        azure_endpoint=os.getenv(
            "AZURE_OPENAI_ENDPOINT",
            "https://aaron-mb5yqktn-eastus2.cognitiveservices.azure.com/",
        ),
    )

class ChatRequest(BaseModel):
    question: str

chat_log = []

def retrieve_chunks_np(user_question: str, k: int = 15) -> str:
    client = get_client()
    emb = client.embeddings.create(input=user_question, model=EMBEDDING_MODEL).data[0].embedding
    q = np.array(emb, dtype="float32")
    qn = q / (np.linalg.norm(q) + _eps)

    n = embedding_matrix_unit.shape[0]
    k = min(k, n) if n > 0 else 0
    if k == 0:
        return ""

    sims = np.dot(embedding_matrix_unit, qn)
    top_k_idx = np.argpartition(-sims, kth=k-1)[:k]
    top_k_idx = top_k_idx[np.argsort(sims[top_k_idx])[::-1]]
    top_chunks = [text_chunks[i] for i in top_k_idx]
    return "\n\n".join(top_chunks)

def generate_answer_from_context(context: str, user_question: str) -> str:
    prompt = f"""
You are a highly structured and detail-oriented assistant who helps employees locate and understand company procedures, sections, and forms from internal documentation.

Your job:
1. Summarize the relevant information clearly.
2. If any forms, section numbers, or document titles are mentioned in the context, and they appear to have associated links, include them using Markdown hyperlinks like: [Tour Waiver](https://link.com).
3. Do not make up links. Only include a link if the name and URL appear in the context or are clearly stated.
4. Organize your answer using clear bullet points or numbers.
5. Prioritize helping the user find what to click on to take action.
6. Display all sections and topics that you retrieve (e.g., Safety, Project Planning, Change Control).

Context:
{context}

Question: {user_question}

Answer:
""".strip()

    messages = [
        {"role": "system", "content": "You are a link-aware internal assistant. Prioritize clarity and actionable hyperlinks when answering questions about procedures or forms."}
    ]
    for chat in chat_log[-3:]:
        messages.append({"role": "user", "content": chat["user"]})
        messages.append({"role": "assistant", "content": chat["response"]})
    messages.append({"role": "user", "content": prompt})

    client = get_client()
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        max_completion_tokens=2000,   # <-- correct kwarg
    )
    answer = resp.choices[0].message.content.strip()
    chat_log.append({"user": user_question, "context": context, "response": answer})
    if len(chat_log) > 50:
        del chat_log[:-50]
    return answer

@app.post("/chat")
async def chat_with_bot(req: ChatRequest):
    try:
        context = retrieve_chunks_np(req.question)
        answer = generate_answer_from_context(context, req.question)
        return {"answer": answer}
    except Exception as e:
        # Temporary: surface errors as JSON so the worker doesn't crash during warmup
        return {"error": str(e)}
