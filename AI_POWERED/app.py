# app.py
import os
import io
import re
import warnings
import tempfile
from typing import List, Dict

import streamlit as st
import pandas as pd
import pdfplumber
import matplotlib.pyplot as plt

# --- Optional AI & embeddings packages (some are optional at runtime) ---
try:
    import google.generativeai as genai
except Exception:
    genai = None
    warnings.warn("google.generativeai not available. GYANM features will be disabled.")

# Robust langchain text splitter import (fallback to a simple local splitter)
try:
    from langchain.text_splitter import CharacterTextSplitter
except Exception:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter as CharacterTextSplitter
    except Exception:
        class CharacterTextSplitter:
            def __init__(self, chunk_size=1000, chunk_overlap=0):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap

            def create_documents(self, texts: List[str]):
                docs = []
                for t in texts:
                    s = str(t)
                    i = 0
                    L = self.chunk_size
                    overlap = self.chunk_overlap
                    while i < len(s):
                        end = min(i + L, len(s))
                        docs.append(type("D", (), {"page_content": s[i:end]}))
                        i = end - overlap if end < len(s) else end
                return docs

        warnings.warn(
            "langchain CharacterTextSplitter import failed; using simple fallback splitter. "
            "Install a compatible langchain version for full features."
        )

# Try to import FAISS-backed vector store or fall back to simple in-memory store
USE_FAISS = False
faiss_import_error = None
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    USE_FAISS = True
except Exception as e:
    faiss_import_error = e
    # We'll fallback to sentence-transformers in-memory store below
    warnings.warn(f"FAISS or langchain_community not available: {e}. Using in-memory fallback vector store.")

# In-memory fallback vector store using sentence-transformers
try:
    from sentence_transformers import SentenceTransformer, util
    import numpy as np
except Exception:
    SentenceTransformer = None
    util = None
    np = None
    warnings.warn("sentence-transformers not available. Install it for embeddings and fallback vector store.")

class SimpleInMemoryVectorStore:
    """
    Simple in-memory vector store using sentence-transformers for embeddings and cosine-similarity search.
    Works as a fallback when FAISS isn't available.
    """
    def __init__(self, docs: List[object], model_name: str = "all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is required for SimpleInMemoryVectorStore fallback.")
        self.model = SentenceTransformer(model_name)
        self.docs = docs
        self.embeddings = self.model.encode([d.page_content if hasattr(d, "page_content") else str(d) for d in docs], convert_to_tensor=True)

    def similarity_search(self, query: str, k: int = 5):
        q_emb = self.model.encode([query], convert_to_tensor=True)
        hits = util.semantic_search(q_emb, self.embeddings, top_k=k)[0]
        results = []
        for h in hits:
            idx = h['corpus_id']
            results.append(self.docs[int(idx)])
        return results

# --- GYANM (Gemini) setup helper ---
def configure_generative_model():
    """
    Configure google.generativeai library if available and GEMINI_API_KEY in environment.
    Returns the model object or None.
    """
    if genai is None:
        return None

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.warning("GEMINI_API_KEY not found in environment; GYANM features are disabled.")
        return None

    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.0-flash")
    except Exception as e:
        warnings.warn(f"Failed to configure generative model: {e}")
        return None

# Call once
gyanm_ai = configure_generative_model()

# ---- Utility: ask GYANM for multilingual translations ----
def translate_multilang_via_gyanm(text: str) -> Dict[str, str]:
    """
    Ask the Generative model to translate 'text' into 5 target languages.
    Returns dict with keys: 'en', 'hi', 'bho', 'ur', 'ta', 'mr'.
    If generative model is unavailable, returns the original text only in 'en'.
    """
    langs = {
        "hi": "Hindi",
        "bho": "Bhojpuri",
        "ur": "Urdu",
        "ta": "Tamil",
        "mr": "Marathi"
    }
    outputs = {"en": text}
    if gyanm_ai is None:
        # No model: return original only (frontend will show user English)
        return outputs

    # Build a single prompt asking for labeled translations
    prompt = (
        "Translate the following text into these languages: Hindi, Bhojpuri, Urdu, Tamil, Marathi. "
        "Return the result in plain text with clear headings like '### Hindi:' then content, "
        "and so on for each language.\n\n"
        f"TEXT TO TRANSLATE:\n{text[:12000]}\n\n"
        "Important: do not include extra commentary. Only labeled translations."
    )
    try:
        resp = gyanm_ai.generate_content(prompt)
        txt = resp.text if hasattr(resp, "text") else str(resp)
        # parse translations by headings
        for code, langname in langs.items():
            pattern = rf"###\s*{re.escape(langname)}\s*:\s*(.*?)(?=(\n###\s*\w+\s*:)|\Z)"
            m = re.search(pattern, txt, flags=re.S | re.I)
            if m:
                outputs[code] = m.group(1).strip()
            else:
                # try looser match: find the language name anywhere
                alt = re.search(rf"{re.escape(langname)}\s*:\s*(.*?)(?=(\n[A-Z][a-z]+:)|\Z)", txt, flags=re.S | re.I)
                outputs[code] = alt.group(1).strip() if alt else ""
    except Exception as e:
        warnings.warn(f"Translation via GYANM failed: {e}")
    return outputs

# ---- Helper: create documents and vectorstores from text content ----
def build_text_chunks(lines: List[str], chunk_size=500, chunk_overlap=100):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.create_documents(lines)
    return docs

def build_vectorstore_from_docs(docs: List[object]):
    """
    Attempts to build a FAISS-backed vector store if available,
    otherwise builds a SimpleInMemoryVectorStore.
    Returns an object with similarity_search(query,k) method.
    """
    if USE_FAISS:
        try:
            embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = FAISS.from_documents(docs, embed_model)
            return db
        except Exception as e:
            warnings.warn(f"FAISS build failed: {e}. Falling back to in-memory vector store.")
    # Fallback
    return SimpleInMemoryVectorStore(docs, model_name="all-MiniLM-L6-v2")

# --- Streamlit App UI ---
st.set_page_config(page_title="AI PDF→Excel | GYANM Multilingual", layout="wide")
st.title("🤖 AI PDF → Excel & Multilingual Insights")
st.caption("Developed by Suraj Kumar Pandey — outputs translated to Hindi, Bhojpuri, Urdu, Tamil, Marathi")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if not uploaded_file:
    st.info("Please upload a PDF to begin.")
    st.stop()

# Save to a temporary file for pdfplumber
with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
    tmp.write(uploaded_file.read())
    pdf_path = tmp.name

st.success("PDF uploaded — extracting content...")

# --- Extract tables and text lines ---
tables_list = []
text_chunks: List[str] = []

with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        # tables
        page_tables = page.extract_tables()
        if page_tables:
            for table in page_tables:
                # first row usually headers
                if len(table) < 2:
                    continue
                cols = table[0]
                # make unique column names
                def make_unique(seq):
                    seen = {}
                    out = []
                    for x in seq:
                        if x not in seen:
                            seen[x] = 1
                            out.append(x)
                        else:
                            seen[x] += 1
                            out.append(f"{x}_{seen[x]}")
                    return out
                unique_cols = make_unique(cols)
                df_tmp = pd.DataFrame(table[1:], columns=unique_cols)
                df_tmp["source"] = "table"
                tables_list.append(df_tmp)

        # text
        page_text = page.extract_text()
        if page_text:
            lines = [line.strip() for line in page_text.split("\n") if line.strip()]
            text_chunks.extend(lines)

# Assemble DataFrames
if tables_list:
    try:
        df_table = pd.concat(tables_list, ignore_index=True)
    except Exception as e:
        st.error(f"Error concatenating table DataFrames: {e}")
        df_table = pd.DataFrame()
else:
    df_table = pd.DataFrame()

df_text = pd.DataFrame({"Text": text_chunks})
if not df_text.empty:
    df_text["source"] = "text"

# Display extracted tables and text
st.subheader("Extracted Tables")
if not df_table.empty:
    st.dataframe(df_table)
    buf = io.BytesIO()
    df_table.to_excel(buf, index=False, engine="openpyxl")
    st.download_button("Download tables as Excel", data=buf.getvalue(), file_name="tables.xlsx")
else:
    st.info("No tables found in the PDF.")

st.subheader("Extracted Text Lines")
if not df_text.empty:
    st.dataframe(df_text)
    buf2 = io.BytesIO()
    df_text.to_excel(buf2, index=False, engine="openpyxl")
    st.download_button("Download text as Excel", data=buf2.getvalue(), file_name="text.xlsx")
else:
    st.info("No text lines found in the PDF.")

# --- Build semantic indexes ---
st.info("Building semantic index (for smart search). This may take a moment...")

# For text
text_docs = build_text_chunks(text_chunks) if text_chunks else []
db_text = build_vectorstore_from_docs(text_docs) if text_docs else None

# For tables (convert rows to text)
db_table = None
if not df_table.empty:
    table_rows = [" | ".join(map(str, row)) for row in df_table.values.tolist()]
    table_docs = build_text_chunks(table_rows)
    db_table = build_vectorstore_from_docs(table_docs)

st.success("Semantic index ready.")

# --- Unified Smart Search UI ---
st.subheader("Unified Smart Search (Tables + Semantic)")
query = st.text_input("Type a keyword or question and press Enter:")

if query:
    # Structured table search (simple string match)
    st.markdown("**Structured table matches:**")
    if not df_table.empty:
        structured_matches = df_table[
            df_table.apply(lambda r: r.astype(str).str.contains(query, case=False, na=False).any(), axis=1)
        ]
        if not structured_matches.empty:
            st.dataframe(structured_matches)
        else:
            st.info("No direct structured matches found in tables.")
    else:
        st.info("No tables to search.")

    # Semantic text search
    st.markdown("**Semantic free-text matches:**")
    semantic_matches = []
    try:
        if db_text:
            semantic_matches = db_text.similarity_search(query, k=5)
    except Exception as e:
        st.warning(f"Semantic search (text) failed: {e}")

    if semantic_matches:
        for i, doc in enumerate(semantic_matches, 1):
            content = doc.page_content if hasattr(doc, "page_content") else str(doc)
            st.write(f"**Result {i}:** {content}")
    else:
        st.info("No semantic free-text matches found.")

    # Semantic table matches
    st.markdown("**Semantic table matches:**")
    semantic_table_matches = []
    try:
        if db_table:
            semantic_table_matches = db_table.similarity_search(query, k=5)
    except Exception as e:
        st.warning(f"Semantic search (tables) failed: {e}")

    if semantic_table_matches:
        matched_rows = []
        for doc in semantic_table_matches:
            txt = doc.page_content if hasattr(doc, "page_content") else str(doc)
            # find matching row in df_table
            for row in df_table.values.tolist():
                joined = " | ".join(map(str, row))
                if joined == txt:
                    matched_rows.append(row)
                    break
        if matched_rows:
            df_matched = pd.DataFrame(matched_rows, columns=df_table.columns)
            st.dataframe(df_matched)
        else:
            st.info("No matching table rows found for semantic table matches.")
    else:
        st.info("No semantic table matches found.")

    # Ask GYANM to explain and produce multilingual translations
    if gyanm_ai is None:
        st.warning("GYANM (Gemini) model is not configured. Install GEMINI_API_KEY and google-generative-ai to enable AI explanations.")
    else:
        # Build context for the model: include top semantic matches and structured matches
        context_parts = []
        if semantic_matches:
            context_parts.append("\n".join([d.page_content for d in semantic_matches]))
        if not df_table.empty:
            context_parts.append(df_table.head(20).to_csv(index=False))
        context_text = "\n\n".join(context_parts)[:15000]

        prompt = (
            f"User query: {query}\n\n"
            f"Context extracted from the PDF (if any):\n{context_text}\n\n"
            "Please provide a clear, concise answer with insights and actionable items if relevant."
            "\n\nReturn the answer in English first. Then provide translations into Hindi, Bhojpuri, Urdu, Tamil, and Marathi. "
            "Label each section clearly, e.g. '### English:', '### Hindi:', etc."
        )
        try:
            gen_resp = gyanm_ai.generate_content(prompt)
            answer_text = gen_resp.text if hasattr(gen_resp, "text") else str(gen_resp)
            st.markdown("### GYANM - Full multilingual response")
            st.markdown(answer_text)

            # parse and display each language separately (if model returned labeled sections)
            translations = translate_multilang_via_gyanm(answer_text)
            st.markdown("#### English (extracted)")
            st.write(translations.get("en", ""))

            cols = st.columns(5)
            lang_codes = [("hi", "Hindi"), ("bho", "Bhojpuri"), ("ur", "Urdu"), ("ta", "Tamil"), ("mr", "Marathi")]
            for col, (code, label) in zip(cols, lang_codes):
                with col:
                    st.markdown(f"**{label}**")
                    st.write(translations.get(code, "(translation not available)"))

        except Exception as e:
            st.error(f"GYANM generation failed: {e}")

# --- Dynamic AI Table Generator (same pattern as your original) ---
st.markdown("---")
st.subheader("Dynamic Table Generator (ask GYANM to build a table from PDF)")

dynamic_query = st.text_input("Type what table you want to extract (example: 'list all deadlines and their dates'):")

if dynamic_query:
    if gyanm_ai is None:
        st.warning("GYANM not configured; cannot generate dynamic tables.")
    else:
        combined_content = "\n".join(text_chunks)
        if not df_table.empty:
            combined_content += "\n\n" + df_table.to_csv(index=False)
        dyn_prompt = (
            f"USER REQUEST: {dynamic_query}\n\n"
            f"PDF CONTENT (truncated): {combined_content[:15000]}\n\n"
            "Return a markdown table only that satisfies the user's request."
        )
        try:
            dyn_resp = gyanm_ai.generate_content(dyn_prompt)
            dyn_text = dyn_resp.text if hasattr(dyn_resp, "text") else str(dyn_resp)
            st.markdown("### GYANM - Generated Table (markdown)")
            st.markdown(dyn_text, unsafe_allow_html=True)

            # Try to parse a markdown table into pandas
            try:
                if "<table" in dyn_text.lower():
                    dfs = pd.read_html(dyn_text)
                else:
                    # convert pipe-table to csv-like and read
                    lines = [l for l in dyn_text.splitlines() if "|" in l]
                    if lines:
                        csv_like = "\n".join(lines).replace("|", ",")
                        dfs = [pd.read_csv(io.StringIO(csv_like))]
                    else:
                        dfs = []
                if dfs:
                    df_generated = dfs[0]
                    st.dataframe(df_generated)
                    bufg = io.BytesIO()
                    df_generated.to_excel(bufg, index=False, engine="openpyxl")
                    st.download_button("Download AI-generated table", data=bufg.getvalue(), file_name="ai_generated_table.xlsx")
            except Exception as e:
                st.warning(f"Could not parse AI-generated table into structured DataFrame: {e}")

        except Exception as e:
            st.error(f"Dynamic table generation failed: {e}")

# --- Simple chat interface for questions about the PDF ---
st.markdown("---")
st.subheader("General Chat (ask anything from the PDF)")

user_q = st.text_input("Ask a question about the PDF, press Enter:")

if user_q:
    if gyanm_ai is None:
        st.warning("GYANM not configured; chat is unavailable.")
    else:
        combined_content = "\n".join(text_chunks)
        if not df_table.empty:
            combined_content += "\n\n" + df_table.to_csv(index=False)

        chat_prompt = (
            f"You are an assistant. The user asked: {user_q}\n\n"
            f"Here is context from the PDF (truncated):\n{combined_content[:15000]}\n\n"
            "Answer clearly in English first, then provide translations into Hindi, Bhojpuri, Urdu, Tamil, Marathi. Label each section."
        )
        try:
            chat_resp = gyanm_ai.generate_content(chat_prompt)
            chat_text = chat_resp.text if hasattr(chat_resp, "text") else str(chat_resp)
            st.markdown("### GYANM - Chat Answer (multilingual)")
            st.markdown(chat_text)

            translations = translate_multilang_via_gyanm(chat_text)
            st.markdown("#### English (extracted)")
            st.write(translations.get("en", ""))

            cols2 = st.columns(5)
            for col, (code, label) in zip(cols2, lang_codes):
                with col:
                    st.markdown(f"**{label}**")
                    st.write(translations.get(code, "(translation not available)"))

            # Attempt to detect any markdown table included in the answer and render chart if present
            # Look for "labels:" and "values:" pattern for charts
            if "labels:" in chat_text and "values:" in chat_text:
                labels_match = re.search(r"labels:\s*(.*)", chat_text)
                values_match = re.search(r"values:\s*(.*)", chat_text)
                if labels_match and values_match:
                    try:
                        labels = [x.strip() for x in labels_match.group(1).split(",")]
                        values = [float(x.strip()) for x in values_match.group(1).split(",")]
                        fig, ax = plt.subplots()
                        ax.pie(values, labels=labels, autopct="%1.1f%%")
                        st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"Could not draw chart: {e}")

        except Exception as e:
            st.error(f"Chat generation failed: {e}")

st.markdown("---")
st.caption("⚙️ If GYANM (Gemini) features don't work, ensure GEMINI_API_KEY is set in repo secrets and google-generative-ai is installed.")
