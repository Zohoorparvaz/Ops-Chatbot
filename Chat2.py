#!/usr/bin/env python
# coding: utf-8

# In[3]:
import streamlit as st

my_api_key = st.secrets["my_api_key"]

# Step 1: Set your user question
user_question = "Tell me about safety procedures"


# In[4]:


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
def retrieve_chunks_np(user_question, k=5):
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



def generate_answer_from_context(context, user_question):
    print("💡 Sending context to Azure OpenAI...")

    prompt = f"""
You are a highly structured and detail-oriented assistant. Your job is to extract and summarize internal operation manuals to help employees quickly understand procedures, rules, and responsibilities.

Given the following context from an operations manual, answer the user question clearly, with step-by-step detail if applicable. Use numbered or bulleted lists when appropriate. If any links are mentioned, always include the full clickable hyperlink using Markdown format: `[Link Text](URL)`.

Context:
{context}

Question: {user_question}

Answer:"""

    response = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": "You are a professional assistant that extracts and summarizes step-by-step procedures from internal manuals. Focus on clarity and structure."},
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=5000
    )

    answer = response.choices[0].message.content.strip()
    print("🤖 Answer:\n", answer)
    return answer

#generate_answer_from_context(context, user_question)


# In[5]:


import streamlit as st

# === Streamlit UI ===
st.set_page_config(page_title="Operations Manual Assistant", layout="wide")
st.title("📘 RAG Chatbot Assistant")
st.write("Ask questions about your internal documentation.")

user_question = st.text_input("🔎 Enter your question:", placeholder="e.g., Tell me about safety procedures")

if user_question:
    with st.spinner("Retrieving relevant chunks..."):
        context = retrieve_chunks_np(user_question)

    with st.spinner("Generating answer..."):
        answer = generate_answer_from_context(context, user_question)

    st.subheader("📄 Answer")
    st.write(answer)


# In[ ]:





# In[6]:


import nbformat
from nbconvert import PythonExporter

# Load your notebook
with open("3 Chat2.ipynb", "r", encoding="utf-8") as f:
    notebook_content = nbformat.read(f, as_version=4)

# Convert to Python script
python_exporter = PythonExporter()
python_script, _ = python_exporter.from_notebook_node(notebook_content)

# Save to .py file
with open("Chat2.py", "w", encoding="utf-8") as f:
    f.write(python_script)

print("✅ Notebook converted to Chat2.py")


# In[ ]:




