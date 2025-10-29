# Deploy.py
# Deploy.py
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
import pickle, numpy as np
from openai import AzureOpenAI
import os
import traceback

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
4. Prioritize helping the user find what to click on to take action.


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
        print(traceback.format_exc())
        return {"error": "Internal error. Please try again later."}

# ---------- Bot Framework ----------

from pathlib import Path
from fastapi.responses import HTMLResponse, FileResponse

LOGO_PATH = Path(__file__).parent / "bird-logo-RGB.jpg"

@app.get("/logo")
def logo():
    # Serves the logo file directly (no static folder required)
    return FileResponse(LOGO_PATH, media_type="image/jpeg")

@app.get("/teams")
def teams_tab():
    html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Operations Manual Chat</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    html, body { height:100%; margin:0; background:#fff; }
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding:16px; }

    /* Brand header */
    .brand { display:flex; align-items:center; gap:12px; margin:0 0 12px; }
    .brand img { height:36px; width:auto; display:block; }
    .brand h3 { margin:0; line-height:1.2; }
    .brand small { color:#6b7280; font-weight:500; }

    #log { height:60vh; overflow:auto; border:1px solid #e5e7eb; padding:12px; border-radius:10px; background:#fafafa; }
    #row { margin-top:10px; display:flex; gap:8px; }
    #q { flex:1; padding:10px; border:1px solid #e5e7eb; border-radius:8px; }
    button { padding:10px 14px; cursor:pointer; border:0; border-radius:10px; background:#2563eb; color:white; }

    .msg { display:flex; gap:8px; margin:10px 0; align-items:flex-start; }
    .avatar { width:28px; height:28px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:13px; color:white; overflow:hidden; }
    .avatar.you { background:#6b7280; }
    .avatar.bot { background:#2563eb; }  /* solid color circle for A */
    .bubble { max-width:720px; padding:10px 12px; border-radius:14px; }
    .bubble.you { background:#e5e7eb; }
    .bubble.bot { background:#eef2ff; }
    .bubble .meta { font-size:12px; color:#6b7280; margin-bottom:4px; }

    .bot a { color:#2563eb; text-decoration:underline; }
  </style>
</head>
<body>
  <!-- Brand header with Bird logo (served from /logo) -->
  <div class="brand">
    <img src="/logo" alt="Bird logo" onerror="this.style.display='none'">
    <div>
      <h3>Operations Manual Assistant</h3>
      # <small>Preview</small>
    </div>
  </div>

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
    const autolink = s => String(s).replace(/\\bhttps?:\\/\\/[^\\s<>"')]+/g, u => `<${u}>`);

    // Render assistant text (Markdown -> HTML) safely
    function renderAnswer(raw) {
      const withAuto = autolink(raw ?? "");
      const html = marked.parse(withAuto, { breaks: true });
      const clean = DOMPurify.sanitize(html, {
        ALLOWED_TAGS: ['a','p','ul','ol','li','strong','em','code','pre','br','h1','h2','h3','h4','h5','h6','blockquote'],
        ALLOWED_ATTR: ['href','target','rel']
      });
      const div = document.createElement('div');
      div.innerHTML = clean;
      div.querySelectorAll('a[href]').forEach(a => {
        a.setAttribute('target', '_blank');
        a.setAttribute('rel', 'noopener noreferrer');
      });
      return div.innerHTML;
    }

    function addMsg(role, html) {
      const isYou = role === 'you';
      const avatarHTML = isYou ? 'Y' : 'A';
      const msg = document.createElement('div');
      msg.className = 'msg';
      msg.innerHTML = [
        '<div class="avatar ', (isYou ? 'you' : 'bot'), '">', avatarHTML, '</div>',
        '<div class="bubble ', (isYou ? 'you' : 'bot'), '">',
          '<div class="meta">', (isYou ? 'You' : 'Assistant'), '</div>',
          '<div class="', (isYou ? '' : 'bot'), '">', html, '</div>',
        '</div>'
      ].join('');
      log.appendChild(msg);
      log.scrollTop = log.scrollHeight;
    }

    async function send() {
      const q = input.value.trim();
      if (!q) return;
      input.value = '';
      addMsg('you', _escape(q));

      try {
        const r = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: q })
        });
        const j = await r.json();
        const a = j.answer || j.error || '(no answer)';
        addMsg('bot', renderAnswer(a));
      } catch (e) {
        addMsg('bot', '<span style="color:#b91c1c">Error: ' + _escape(String(e)) + '</span>');
      }
    }

    input.addEventListener('keydown', (e) => { if (e.key === 'Enter') send(); });
    sendBtn.addEventListener('click', send);

    // Seed example messages so you can judge the look immediately
    addMsg('bot', renderAnswer(`Hi, I am your assistant — ask me anything about the Operations Manual.  

_Note: The Operations Manual is under review for some major changes and updates. Please visit me back after the release to see the updated results for your prompts as well.)_  

**Key sections available in the Operations Manual:**  

02 - SAFETY  
03 - COMPANY ORGANIZATION  
04 - PROJECT PLANNING  
05 - OWNER CONTRACT REVIEW AND TYPES  
06 - CONTRACT INSURANCE AND BONDING  
07 - PERMITS AND LICENSES  
08 - PROJECT FILING SYSTEM  
09 - CONTRACT DRAWINGS AND SPECIFICATIONS  
10 - SUBCONTRACTING  
11 - PRODUCT PURCHASING  
12 - SHOP DRAWINGS AND SUBMITTALS  
13 - SCHEDULING  
14 - COST CONTROL  
15 - CHANGE CONTROL  
16 - PROGRESS APPLICATIONS  
17 - PROFIT LOSS REPORTING  
18 - EMPLOYEE HIRING, LAYOFF & REPORTING  
19 - LABOUR RELATIONS  
20 - QUALITY CONTROL  
21 - GENERAL PROJECT ADMINISTRATION  
22 - PROJECT TURNOVER AND CLOSEOUT PROCEDURES  
23 - CONSTRUCTION ENGINEERING  
24 - CONSTRUCTION MANAGEMENT  
25 - DESIGN-BUILD CONSTRUCTION  
26 - ENVIRONMENTAL  
27 - DELAYS AND CLAIMS  
28 - EMERGENCY PROCEDURES AND PUBLICITY`));
  </script>
</body>
</html>"""
    resp = HTMLResponse(html)
    # Allow the CDNs for marked + DOMPurify; allow Teams to frame it.
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


