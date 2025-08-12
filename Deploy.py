# Deploy.py
# Deploy.py
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
import pickle, numpy as np
from openai import AzureOpenAI
import os

app = FastAPI()

# ---------- Health ----------
@app.get("/health")
async def health():
    return {"status": "ok"}

# ---------- Lazy-load data ----------
text_chunks = None
embedding_matrix_unit = None
_eps = 1e-12

def ensure_loaded():
    global text_chunks, embedding_matrix_unit
    if text_chunks is not None and embedding_matrix_unit is not None:
        return
    with open("text_chunks.pkl", "rb") as f:
        tc = pickle.load(f)
    with open("embeddings.pkl", "rb") as f:
        em = np.array(pickle.load(f), dtype="float32")
    row_norms = np.linalg.norm(em, axis=1, keepdims=True) + _eps
    embedding_matrix_unit = em / row_norms
    text_chunks = tc

# ---------- Azure OpenAI ----------
EMBEDDING_MODEL = os.getenv("AOAI_EMBED_DEPLOY", "text-embedding-3-small")
CHAT_MODEL      = os.getenv("AOAI_CHAT_DEPLOY",  "o4-mini")  # must match your Deployment name

def get_client() -> AzureOpenAI:
    key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")  # e.g., https://<resource>.openai.azure.com/
    if not key or not endpoint:
        raise RuntimeError("AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT not set")
    return AzureOpenAI(
        api_key=key,
        api_version="2024-12-01-preview",
        azure_endpoint=endpoint,
    )

class ChatRequest(BaseModel):
    question: str

chat_log = []

def retrieve_chunks_np(user_question: str, k: int = 15) -> str:
    ensure_loaded()
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
2. If any forms, section numbers, or document titles are mentioned in the context, and they appear to have associated links, include them using Markdown hyperlinks that are present in the context.
3. Do not make up links.
4. Organize your answer using clear bullet points or numbers.
5. Prioritize helping the user find what to click on to take action.
6. Display all sections and topics that you retrieve (e.g., Safety, Project Planning, Change Control).

Context:
{context}

Question: {user_question}

Answer:
""".strip()

    messages = [{"role": "system", "content": "You are a link-aware internal assistant. Prioritize clarity and actionable hyperlinks."}]
    for chat in chat_log[-3:]:
        messages.append({"role": "user", "content": chat["user"]})
        messages.append({"role": "assistant", "content": chat["response"]})
    messages.append({"role": "user", "content": prompt})

    client = get_client()
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        #max_tokens=2000,  # correct param
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
        return {"error": str(e)}

# ---------- Bot Framework (optional) ----------
MICROSOFT_APP_ID = os.getenv("MICROSOFT_APP_ID", "")
MICROSOFT_APP_PASSWORD = os.getenv("MICROSOFT_APP_PASSWORD", "")
adapter = None

if MICROSOFT_APP_ID and MICROSOFT_APP_PASSWORD:
    try:
        from botbuilder.core import BotFrameworkAdapter, BotFrameworkAdapterSettings, TurnContext
        from botbuilder.schema import Activity, ActivityTypes

        settings = BotFrameworkAdapterSettings(app_id=MICROSOFT_APP_ID, app_password=MICROSOFT_APP_PASSWORD)
        adapter = BotFrameworkAdapter(settings)

        async def on_error(ctx: TurnContext, err: Exception):
            print(f"[BOT ERROR] {err}")
            await ctx.send_activity("Sorry—server error.")
        adapter.on_turn_error = on_error

        async def on_turn(turn_context: TurnContext):
            if turn_context.activity.type == ActivityTypes.message:
                user_q = (turn_context.activity.text or "").strip()
                ctx = retrieve_chunks_np(user_q)
                ans = generate_answer_from_context(ctx, user_q)
                await turn_context.send_activity(ans)
            else:
                await turn_context.send_activity(f"Received: {turn_context.activity.type}")

        @app.post("/api/messages")
        async def messages(request: Request):
            body = await request.json()
            activity = Activity().deserialize(body)
            auth_header = request.headers.get("Authorization", "")
            await adapter.process_activity(activity, auth_header, on_turn)
            return Response(status_code=200)

    except Exception as e:
        print(f"[STARTUP] Bot wiring failed: {e}")
        adapter = None

