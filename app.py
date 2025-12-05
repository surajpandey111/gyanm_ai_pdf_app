# =====================================================================
#  AI_HUB — Research PDF Intelligence System
#  Final Streamlit App (Stable for Streamlit Cloud)
# =====================================================================

import streamlit as st
import os
import pdfplumber
import pandas as pd
import tempfile
import io
from dotenv import load_dotenv
import google.generativeai as genai
import matplotlib.pyplot as plt
import re
import numpy as np

# =====================================================================
# 1. SAFE LANGUAGE & EMBEDDINGS IMPORTS (NO CRASH IF MISSING)
# =====================================================================

has_langchain = False
has_langchain_community = False

CharacterTextSplitter = None
FAISS = None
HuggingFaceEmbeddings = None

try:
    # Newer LangChain path
    from langchain.text_splitters import CharacterTextSplitter
    has_langchain = True

    try:
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
        has_langchain_community = True
    except:
        FAISS = None
        HuggingFaceEmbeddings = None
        has_langchain_community = False

except:
    # Older fallback
    try:
        from langchain.text_splitter import CharacterTextSplitter
        has_langchain = True
    except:
        CharacterTextSplitter = None
        has_langchain = False


# =====================================================================
# 2. FALLBACK TEXT SPLITTER (ALWAYS WORKS)
# =====================================================================

class SimpleCharacterSplitter:
    def __init__(self, chunk_size=500, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def create_documents(self, texts):
        class Doc:
            def __init__(self, content): self.page_content = content

        if isinstance(texts, str):
            texts = [texts]

        docs = []
        for text in texts:
            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                docs.append(Doc(text[start:end]))
                start = end - self.chunk_overlap
        return docs


# =====================================================================
# 3. SENTENCE TRANSFORMERS FALLBACK (NO FAISS NEEDED)
# =====================================================================

use_sentence_transformers_fallback = False
sentence_model = None

try:
    from sentence_transformers import SentenceTransformer
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    use_sentence_transformers_fallback = True
except:
    use_sentence_transformers_fallback = False


# =====================================================================
# 4. AI_HUB (Gemini) CONFIG
# =====================================================================

load_dotenv()

ai_hub = None
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    ai_hub = genai.GenerativeModel("gemini-2.5-flash")
except:
    st.warning("⚠️ AI_HUB (Gemini) is not configured. AI features disabled.")


# =====================================================================
# 5. STREAMLIT UI HEADER
# =====================================================================

st.set_page_config(page_title="AI_HUB — Research PDF Intelligence", layout="wide")
st.title("🤖 AI_HUB — Research PDF Intelligence System")
st.caption("Built by **Suraj Kumar Pandey** — Research Automation & AI Extraction Suite")

uploaded_file = st.file_uploader("📄 Upload Your Research/Technical PDF", type="pdf")

# =====================================================================
# 6. PDF EXTRACTION ENGINE
# =====================================================================

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    st.success("✅ PDF Uploaded Successfully!")

    tables_list, text_chunks = [], []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # *** Extract tables ***
            pg_tables = page.extract_tables()
            if pg_tables:
                for t in pg_tables:
                    cols = list(dict.fromkeys(t[0]))
                    df_tmp = pd.DataFrame(t[1:], columns=cols)
                    df_tmp["source"] = "table"
                    tables_list.append(df_tmp)

            # *** Extract text ***
            text = page.extract_text()
            if text:
                lines = [x.strip() for x in text.split("\n") if x.strip()]
                text_chunks.extend(lines)

    # Build dataframes
    df_table = pd.concat(tables_list, ignore_index=True) if tables_list else pd.DataFrame()
    df_text = pd.DataFrame(text_chunks, columns=["Text"]) if text_chunks else pd.DataFrame()


    # =====================================================================
    # 7. DISPLAY TABLES
    # =====================================================================

    st.subheader("📊 Extracted Tables")
    if df_table.empty:
        st.info("No structured tables found in this PDF.")
    else:
        st.dataframe(df_table)

        buf = io.BytesIO()
        df_table.to_excel(buf, index=False, engine="openpyxl")
        st.download_button("⬇️ Download Tables (Excel)", buf.getvalue(), "tables.xlsx")


    # =====================================================================
    # 8. DISPLAY TEXT
    # =====================================================================

    st.subheader("📝 Extracted Text Content")
    if df_text.empty:
        st.info("No readable text found in this PDF.")
    else:
        st.dataframe(df_text, height=300)
        buf = io.BytesIO()
        df_text.to_excel(buf, index=False, engine="openpyxl")
        st.download_button("⬇️ Download Text (Excel)", buf.getvalue(), "text.xlsx")


    # =====================================================================
    # 9. SEMANTIC INDEX BUILD SYSTEM
    # =====================================================================

    st.info("⚙️ Building AI semantic index...")

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100) if CharacterTextSplitter else SimpleCharacterSplitter()
    documents = splitter.create_documents(text_chunks)

    # FAISS Mode
    if has_langchain and has_langchain_community and FAISS:
        try:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db_text = FAISS.from_documents(documents, embeddings)
        except:
            db_text = None
    # Sentence-Transformer fallback
    elif use_sentence_transformers_fallback:
        doc_texts = [d.page_content for d in documents]
        embeddings = sentence_model.encode(doc_texts, convert_to_numpy=True)
        db_text = {"docs": documents, "emb": embeddings}
    else:
        db_text = None


    # =====================================================================
    # 10. SEARCH SYSTEM (Tables + AI Semantic)
    # =====================================================================

    st.subheader("🔍 Unified Search (Keyword + Semantic)")
    query = st.text_input("Enter your question or keyword:")

    def semantic_search(db, q, k=5):
        q_emb = sentence_model.encode([q], convert_to_numpy=True)[0]
        sims = np.dot(db["emb"], q_emb) / (
            np.linalg.norm(db["emb"], axis=1) * np.linalg.norm(q_emb) + 1e-10
        )
        top = sims.argsort()[::-1][:k]
        return [db["docs"][i] for i in top]

    if query:
        # 1️⃣ Structured filter
        if not df_table.empty:
            structured = df_table[df_table.apply(lambda r: r.astype(str).str.contains(query, case=False).any(), axis=1)]
            st.write("### 🟦 Structured Matches")
            st.dataframe(structured if not structured.empty else "No matches")
        else:
            st.info("No tables available for structured search.")

        # 2️⃣ Semantic text search
        st.write("### 🟨 Semantic Text Matches")

        if db_text:
            if isinstance(db_text, dict):
                results = semantic_search(db_text, query)
            else:
                results = db_text.similarity_search(query, k=5)

            for idx, r in enumerate(results, 1):
                st.write(f"**Result {idx}:** {r.page_content}")
        else:
            st.info("Semantic search unavailable (no embeddings backend detected).")


    # =====================================================================
    # 11. AI_HUB SUMMARY / INSIGHTS / TABLE GENERATION
    # =====================================================================

    st.header("🤖 AI_HUB Intelligence Tools")

    # -------- SUMMARY ----------
    if st.button("📘 Generate Summary"):
        if ai_hub:
            txt = "\n".join(text_chunks)[:8000]
            out = ai_hub.generate_content(f"Summarize the following clearly:\n\n{txt}")
            st.success(out.text)
        else:
            st.error("AI_HUB not configured.")

    # -------- INSIGHTS ----------
    if st.button("🔎 Extract Insights"):
        if ai_hub:
            txt = "\n".join(text_chunks)[:8000]
            prompt = """
            Analyze this research text and extract:
            - Key findings
            - Methods used
            - Critical parameters
            - Weaknesses or gaps
            - Recommendations
            """
            out = ai_hub.generate_content(prompt + txt)
            st.success(out.text)
        else:
            st.error("AI_HUB not configured.")

    # -------- CHAT ----------
    st.subheader("💬 AI_HUB Research Chat")
    user_q = st.text_input("Ask anything about the PDF")

    if user_q and ai_hub:
        full = "\n".join(text_chunks)[:15000]
        prompt = f"""
        You are AI_HUB, a research analysis assistant.
        User question: {user_q}
        PDF Content:
        {full}
        Provide a clear, accurate answer.
        """
        out = ai_hub.generate_content(prompt)
        st.write(out.text)


    # =====================================================================
    st.markdown("---")
    st.caption("🛠 AI_HUB — Built by Suraj Kumar Pandey")

else:
    st.info("📄 Upload a PDF to begin.")
