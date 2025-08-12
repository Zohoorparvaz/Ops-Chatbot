# deploy.py
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
import os, pickle, numpy as np
from openai import AzureOpenAI

app = FastAPI()

# ---------- Health ----------
@app.get("/health")
async def health():
    return {"status": "ok"}

# ---------- Lazy data load (so startup never 503s) ----------
text_chunks = None
embedding_matrix_unit = None
_load_err = None
_EPS = 1e-12

def ensure_data_loaded():
    global text_chunks, embedding_matrix_unit, _load_err
    if text_chunks is not None and embedding_matrix_unit is not None:
        return
    try:
        with open("text_chunks.pkl", "rb") as f:
            tc = pickle.load(f)
        with open("embeddings.pkl", "rb") as f:
            em = np.array(pickle.load(f), dtype="float32")
        # pre-normalize
        row_norms = np.linalg.norm(em, axis=1, keepdims=True) + _EPS
        embedding_matrix_unit = em / row_norms
        text_chunks = tc
        _load_err = None
    except Exception as e:
        _load_err = e
        print(f"[STARTUP WARNING] Failed to load embeddings/chunks: {e}")

# ---------- Azure OpenAI ----------
EMBEDDING_MODEL = os.getenv("AOAI_EMBED_DEPLOY", "text-embedding-3-small")  # must be your deployment name
CHAT_MODEL = os.getenv("AOAI_CHAT_DEPLOY", "gpt-4o-mini")                   # must be your deployment name

def get_client() -> AzureOpenAI:
    key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not key or not endpoint:
        raise RuntimeError("AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT not set")
    return AzureOpenAI(
        api_key=key,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        azure_endpoint=endpoint,
    )

class ChatRequest(BaseModel):
    question: str

chat_log = []

def retrieve_chunks_np(user_question: str, k: int = 15) -> str:
    ensure_data_loaded()
    if _load_err:
        return ""
    client = get_client()
    emb = client.embeddings.create(input=user_question, model=EMBEDDING_MODEL).data[0].embedding
    q = np.array(emb, dtype="float32")
    qn = q / (np.linalg.norm(q) + _EPS)
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
    messages = [
        {"role": "system", "content": "You are a link-aware internal assistant. Prioritize clarity and actionable hyperlinks."}
    ]
    for chat in chat_log[-3:]:
        messages.append({"role": "user", "content": chat["user"]})
        messages.append({"role": "assistant", "content": chat["response"]})

    prompt = f"""Summarize clearly, include only links found in context, and list relevant sections.
Context:
{context}

Question: {user_question}
Answer:"""
    messages.append({"role": "user", "content": prompt})

    client = get_client()
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        max_completion_tokens=2000,
    )
    answer = resp.choices[0].message.content.strip()
    chat_log.append({"user": user_question, "context": context, "response": answer})
    if len(chat_log) > 50:
        del chat_log[:-50]
    return answer

@app.post("/chat")
async def chat_with_bot(req: ChatRequest):
    try:
        ctx = retrieve_chunks_np(req.question)
        if _load_err:
            return {"error": f"Data not loaded yet: {_load_err}"}
        ans = generate_answer_from_context(ctx, req.question)
        return {"answer": ans}
    except Exception as e:
        return {"error": str(e)}

# ---------- Bot Framework (optional; won’t crash startup) ----------
MICROSOFT_APP_ID = os.getenv("MICROSOFT_APP_ID", "")
MICROSOFT_APP_PASSWORD = os.getenv("MICROSOFT_APP_PASSWORD", "")
adapter = None

if MICROSOFT_APP_ID and MICROSOFT_APP_PASSWORD:
    try:
        from botbuilder.core import BotFrameworkAdapter, BotFrameworkAdapterSettings, TurnContext
        from botbuilder.schema import Activity, ActivityTypes

        settings = BotFrameworkAdapterSettings(
            app_id=MICROSOFT_APP_ID,
            app_password=MICROSOFT_APP_PASSWORD,
        )
        adapter = BotFrameworkAdapter(settings)

        async def on_error(ctx: TurnContext, err: Exception):
            print(f"[BOT ERROR] {err}")
            await ctx.send_activity("Sorry—server error.")
        adapter.on_turn_error = on_error

        async def on_turn(turn_context: TurnContext):
            if turn_context.activity.type == ActivityTypes.message:
                user_q = (turn_context.activity.text or "").strip()
                ctx = retrieve_chunks_np(user_q)
                if _load_err:
                    await turn_context.send_activity("Server warming up—try again in a minute.")
                    return
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
        adapter = None  # keep app alive
