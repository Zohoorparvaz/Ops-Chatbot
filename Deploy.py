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

# ---------- Bot Framework ----------

from fastapi.responses import HTMLResponse

@app.get("/teams")
def teams_tab():
    html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Operations Manual Chat</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    html, body { height:100%; margin:0; }
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding:16px; }
    #log { height:60vh; overflow:auto; border:1px solid #e5e7eb; padding:10px; border-radius:8px; }
    #row { margin-top:10px; display:flex; gap:8px; }
    #q { flex:1; padding:8px; }
    button { padding:8px 12px; cursor: pointer; }
    .you { font-weight:600; margin-top:8px; }
    .bot { margin:4px 0 12px 0; white-space:pre-wrap; word-wrap: break-word; overflow-wrap: anywhere; }
    .bot a { color:#2563eb; text-decoration: underline; }
  </style>
</head>
<body>
  <h3>Operations Manual Assistant</h3>
  <div id="log"></div>
  <div id="row">
    <input id="q" placeholder="Ask a question..." />
    <button id="sendBtn">Send</button>
  </div>

  <!-- Markdown renderer + sanitizer -->
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <script src="https://unpkg.com/dompurify@3.1.6/dist/purify.min.js"></script>

  <script>
    const log = document.getElementById('log');
    const input = document.getElementById('q');
    const sendBtn = document.getElementById('sendBtn');

    // Escape ONLY user-typed echoes (we don't escape assistant Markdown before rendering)
    const _escape = s => String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");

    // Turn bare URLs into Markdown-compatible links (<https://...>)
    const autolink = s => String(s).replace(/\bhttps?:\/\/[^\s<>"')]+/g, u => `<${u}>`);

    // Render assistant text (Markdown -> HTML) safely
    function renderAnswer(raw) {
      const withAuto = autolink(raw ?? "");
      const html = marked.parse(withAuto, { breaks: true });
      const clean = DOMPurify.sanitize(html, {
        ALLOWED_TAGS: ['a','p','ul','ol','li','strong','em','code','pre','br','h1','h2','h3','h4','h5','h6','blockquote'],
        ALLOWED_ATTR: ['href','target','rel']
      });
      // Force links to open safely in new tabs
      const div = document.createElement('div');
      div.innerHTML = clean;
      div.querySelectorAll('a[href]').forEach(a => {
        a.setAttribute('target', '_blank');
        a.setAttribute('rel', 'noopener noreferrer');
      });
      return div.innerHTML;
    }

    async function send() {
      const q = input.value.trim();
      if (!q) return;
      input.value = '';
      log.insertAdjacentHTML('beforeend', `<div class="you">You:</div><div>${_escape(q)}</div>`);
      log.scrollTop = log.scrollHeight;

      try {
        const r = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: q })
        });
        const j = await r.json();
        const a = j.answer || j.error || '(no answer)';

        const safeHtml = renderAnswer(a);
        log.insertAdjacentHTML(
          'beforeend',
          `<div class="you">Assistant:</div><div class="bot">${safeHtml}</div>`
        );
        log.scrollTop = log.scrollHeight;
      } catch (e) {
        log.insertAdjacentHTML('beforeend', `<div style="color:#b91c1c">Error: ${_escape(String(e))}</div>`);
      }
    }

    input.addEventListener('keydown', (e) => { if (e.key === 'Enter') send(); });
    sendBtn.addEventListener('click', send);
  </script>
</body>
</html>"""
    resp = HTMLResponse(html)
    # Allow the CDNs for marked + DOMPurify
    resp.headers["Content-Security-Policy"] = (
        "frame-ancestors https://teams.microsoft.com https://*.teams.microsoft.com "
        "https://*.office.com https://*.microsoft.com; "
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://unpkg.com; "
        "style-src  'self' 'unsafe-inline'; "
        "img-src    'self' data: blob:; "
        "connect-src 'self';"
    )
    return resp
