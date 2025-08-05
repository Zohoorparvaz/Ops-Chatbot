#!/usr/bin/env python
# coding: utf-8

# In[3]:
import streamlit as st

my_api_key = st.secrets["my_api_key"]

# Step 1: Set your user question
user_question = "Tell me about safety procedures"




import pickle, faiss, numpy as np
from openai import AzureOpenAI

# === Load text chunks and precomputed embeddings ===
with open("text_chunks.pkl", "rb") as f:
    text_chunks = pickle.load(f)

with open("embeddings.pkl", "rb") as f:
    embedding_matrix = np.array(pickle.load(f), dtype="float32")  # shape: (n_chunks, 1536)
# === Azure OpenAI setup ===
client = AzureOpenAI(
    api_key= my_api_key,  # replace this securely
    api_version="2024-12-01-preview",
    azure_endpoint="https://aaron-mb5yqktn-eastus2.cognitiveservices.azure.com/"
)
embedding_model = "text-embedding-3-small"  # your embedding deployment
chat_model = "o4-mini"                 # your chat model deployment

### Retrevial Function ###


# === Function: retrieve relevant chunks using cosine similarity ===
def retrieve_chunks_np(user_question, k=15):
    print(f"💬 Question: {user_question}")
    
    # Step 1: Embed the question
    print("🔍 Step 1: Getting embedding...")
    query = client.embeddings.create(
        input=user_question,
        model=embedding_model
    ).data[0].embedding
    query = np.array(query, dtype="float32")

    print("✅ Query shape:", query.shape)

    # Step 2: Normalize the matrix and query
    print("🔍 Step 2: Normalizing embeddings...")
    try:
        # Check for zeros or NaNs
        if np.isnan(embedding_matrix).any():
            print("❌ ERROR: NaNs found in embedding_matrix")
            return

        if np.any(np.linalg.norm(embedding_matrix, axis=1) == 0):
            print("❌ ERROR: Zero vectors found in embedding_matrix")
            return

        matrix_norm = embedding_matrix / np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        query_norm = query / np.linalg.norm(query)

        print("✅ Normalization complete.")

        # Step 3: Cosine similarity
        print("🔍 Step 3: Calculating cosine similarities...")
        similarities = np.dot(matrix_norm, query_norm)

        # Step 4: Get top chunks
        top_k_idx = np.argsort(similarities)[-k:][::-1]
        top_chunks = [text_chunks[i] for i in top_k_idx]

        print("✅ Retrieved top", k, "chunks.")
        return "\n\n".join(top_chunks)

    except Exception as e:
        print("❌ ERROR during similarity calc:", e)
        return None

#context = retrieve_chunks_np(user_question, k=5)

## LLM Function ##



# === Initialize chat log ===
# === Chat history ===
chat_log = []

# === Function: Retrieve top-k chunks using cosine similarity ===
def retrieve_chunks_np(user_question, k=15):
    print(f"💬 Question: {user_question}")
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
    print("✅ Retrieved top", k, "chunks.")
    return "\n\n".join(top_chunks)

# === Function: Generate answer with chat history ===
def generate_answer_from_context(context, user_question):
    print("💡 Sending context to Azure OpenAI...")

    prompt = f"""
You are a highly structured and detail-oriented assistant who helps employees locate and understand company procedures, sections, and forms from internal documentation. 

Your job:
1. Summarize the relevant information clearly.
2. If any **forms, section numbers, or document titles** are mentioned in the context, and they appear to have associated links, include them using **Markdown hyperlinks** like: [Tour Waiver](https://link.com).
3. Do **not** make up links. Only include a link if the name and URL appear in the context or are clearly stated.
4. Organize your answer using clear bullet points or numbers.
5. Prioritize helping the user **find what to click on** to take action.
6. Each time you retrieve a section display the relevant Forms, Other references and references. For example if you grab a section in section 15 display all of sections 15.10 Forms and 15.11 Other References 

Context:
{context}

Question: {user_question}

Answer:
"""

    # Build full message history (last 3 turns)
    messages = [
        {"role": "system", "content": "You are a link-aware internal assistant. Prioritize clarity and actionable hyperlinks when answering questions about procedures or forms."}
    ]
    for chat in chat_log[-3:]:
        messages.append({"role": "user", "content": chat["user"]})
        messages.append({"role": "assistant", "content": chat["response"]})
    
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            model=chat_model,
            messages=messages,
            max_completion_tokens=5000
        )
        answer = response.choices[0].message.content.strip()
        print("🤖 Answer:\n", answer)

        # Save this exchange to memory
        chat_log.append({
            "user": user_question,
            "context": context,
            "response": answer
        })

        return answer

    except Exception as e:
        print("❌ Exception during model call:", e)
        return f"Model error: {e}"






# In[4]:


import streamlit as st

# === UI Settings ===
st.set_page_config(page_title="Operations Manual Assistant", layout="wide")
st.title("📘 RAG Chatbot Assistant")
st.write("Ask questions about your internal documentation.")

# === Session State for chat memory ===
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []  # Each item: {"user": ..., "response": ...}

# === Question Input ===
user_question = st.text_input("🔎 Enter your question:", placeholder="e.g., Tell me about safety procedures")

if user_question:
    with st.spinner("🔍 Retrieving relevant chunks..."):
        context = retrieve_chunks_np(user_question)

    with st.spinner("🤖 Generating answer..."):
        answer = generate_answer_from_context(context, user_question)

    # === Save to memory ===
    st.session_state.chat_log.append({
        "user": user_question,
        "response": answer
    })

    # === Display Answer ===
    st.subheader("📄 Answer")
    st.write(answer)

# === Chat History Display ===
if st.session_state.chat_log:
    st.subheader("💬 Chat History")
    for entry in reversed(st.session_state.chat_log):
        st.markdown(f"**You:** {entry['user']}")
        st.markdown(f"**Assistant:** {entry['response']}")
        st.markdown("---")


# In[5]:








# In[6]:



# In[ ]:




