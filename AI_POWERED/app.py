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

# --- Load dotenv if present ---
from dotenv import load_dotenv
load_dotenv()

# --- Google GenAI compatibility wrapper (google-genai preferred, fallback to google.generativeai) ---
_g_client = None
_g_client_old = None
_g_sdk_name = None

try:
    # new SDK: google-genai (import path: from google import genai)
    from google import genai as _genai_new
    try:
        _g_client = _genai_new.Client()
        _g_sdk_name = "google-genai"
    except Exception as e:
        warnings.warn(f"google-genai installed but client init failed: {e}")
        _g_client = None
except Exception:
    _genai_new = None

try:
    # older SDK fallback (deprecated): google.generativeai
    import google.generativeai as _genai_old
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                _genai_old.configure(api_key=api_key)
            except Exception:
                pass
        _g_client_old = _genai_old
        _g_sdk_name = getattr(_genai_old, "__name__", "google.generativeai")
    except Exception as e:
        warnings.warn(f"Older google.generativeai import succeeded but config failed: {e}")
        _g_client_old = None
except Exception:
    _genai_old = None
    _g_client_old = None

def gyanm_generate(prompt: str, model: str = "gemini-2.5-flash", max_output_chars: int = 4000) -> str:
    """
    Generate text using Google GenAI SDK if available.
    Tries new google-genai client first, then old google.generativeai fallback.
    Returns plain string or raises RuntimeError if no SDK available.
    """
    # new google-genai client path
    if _g_client is not None:
        try:
            resp = _g_client.models.generate_content(model=model, contents=[{"text": prompt}])
            # Common shapes:
            if hasattr(resp, "text"):
                return resp.text
            # try candidate extraction
            try:
                cand = getattr(resp, "candidates", None)
                if cand and len(cand) > 0:
                    c0 = cand[0]
                    # check for content.parts etc.
                    if hasattr(c0, "content"):
                        cont = c0.content
                        parts = cont.get("parts") if isinstance(cont, dict) else getattr(cont, "parts", None)
                        if parts:
                            first = parts[0]
                            if isinstance(first, dict) and "text" in first:
                                return first["text"]
                            elif hasattr(first, "text"):
                                return first.text
                    if hasattr(c0, "text"):
                        return c0.text
            except Exception:
                pass
            return str(resp)
        except Exception as e:
            raise RuntimeError(f"google-genai generation failed: {e}") from e

    # fallback older google.generativeai
    if _g_client_old is not None:
        try:
            try:
                model_obj = _g_client_old.GenerativeModel(model)
                resp = model_obj.generate_content(prompt)
                if hasattr(resp, "text"):
                    return resp.text
                return str(resp)
            except Exception:
                resp = _g_client_old.generate(prompt)
                if hasattr(resp, "text"):
                    return resp.text
                return str(resp)
        except Exception as e:
            raise RuntimeError(f"old google.generativeai generation failed: {e}") from e

    raise RuntimeError(
        "No Google GenAI SDK installed. Install 'google-genai' (pip install google-genai) or the older package."
    )

# --- Robust langchain CharacterTextSplitter fallback ---
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

        warnings.warn("langchain CharacterTextSplitter import failed; using simple fallback splitter.")

# --- Vectorstore: try FAISS else fallback to safe in-memory store ---
USE_FAISS = False
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    USE_FAISS = True
except Exception as e:
    warnings.warn(f"FAISS/langchain_community not available: {e}. Using in-memory fallback.")

try:
    from sentence_transformers import SentenceTransformer, util
except Exception:
    SentenceTransformer = None
    util = None
    warnings.warn("sentence-transformers not available. Semantic search fallback will be disabled.")

# --- Safe SimpleInMemoryVectorStore (does NOT raise if sentence-transformers missing) ---
class SimpleInMemoryVectorStore:
    """
    In-memory fallback vector store. If sentence-transformers is missing or fails,
    semantic search methods return empty results but app continues running.
    """
    def __init__(self, docs: List[object], model_name: str = "all-MiniLM-L6-v2"):
        self.docs = docs or []
        self.model = None
        self.embeddings = None

        if SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(model_name)
                try:
                    self.embeddings = self.model.encode(
                        [d.page_content if hasattr(d, "page_content") else str(d) for d in self.docs],
                        convert_to_tensor=True
                    )
                except Exception as e:
                    warnings.warn(f"Failed to encode docs: {e}. Semantic search disabled.")
                    self.model = None
                    self.embeddings = None
            except Exception as e:
                warnings.warn(f"Failed to init SentenceTransformer: {e}. Semantic search disabled.")
                self.model = None
                self.embeddings = None
        else:
            warnings.warn("sentence-transformers not installed. Semantic search disabled.")

    def similarity_search(self, query: str, k: int = 5):
        if self.model is None or self.embeddings is None or util is None:
            return []
        try:
            q_emb = self.model.encode([query], convert_to_tensor=True)
            hits = util.semantic_search(q_emb, self.embeddings, top_k=k)[0]
            results = []
            for h in hits:
                idx = int(h.get("corpus_id", 0)) if isinstance(h, dict) else int(getattr(h, "corpus_id", 0))
                results.append(self.docs[idx])
            return results
        except Exception as e:
            warnings.warn(f"Semantic search failed: {e}")
            return []

def build_text_chunks(lines: List[str], chunk_size=500, chunk_overlap=100):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.create_documents(lines)

def build_vectorstore_from_docs(docs: List[object]):
    if USE_FAISS:
        try:
            embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = FAISS.from_documents(docs, embed_model)
            return db
        except Exception as e:
            warnings.warn(f"FAISS build failed: {e}. Falling back to in-memory.")
    # fallback
    return SimpleInMemoryVectorStore(docs, model_name="all-MiniLM-L6-v2")

# ---- Translation helper using gyanm_generate for robust parsing ----
def translate_multilang_via_gyanm(text: str) -> Dict[str, str]:
    langs = {
        "hi": "Hindi",
        "bho": "Bhojpuri",
        "ur": "Urdu",
        "ta": "Tamil",
        "mr": "Marathi"
    }
    outputs = {"en": text}
    try:
        prompt = (
            "Translate the following English text into Hindi, Bhojpuri, Urdu, Tamil, Marathi. "
            "Return labeled sections like '### Hindi:' then translation, etc. Only return labeled translations.\n\n"
            f"TEXT:\n{text[:12000]}"
        )
        t = gyanm_generate(prompt)
    except Exception:
        return outputs

    for code, lang in langs.items():
        pattern = rf"###\s*{re.escape(lang)}\s*:\s*(.*?)(?=(\n###\s*\w+\s*:)|\Z)"
        m = re.search(pattern, t, flags=re.S | re.I)
        outputs[code] = m.group(1).strip() if m else ""
    return outputs

# --- Streamlit UI & main logic ---
st.set_page_config(page_title="AI PDF→Excel | GYANM Multilingual", layout="wide")
st.title("🤖 AI PDF → Excel & Multilingual Insights (PDF AI Parse)")
st.caption("Developed by Suraj Kumar Pandey — choose language and ask questions about the PDF")

# language selector
LANG_OPTIONS = ["English", "Hindi", "Bhojpuri", "Urdu", "Tamil", "Marathi", "All"]
chosen_lang = st.selectbox("Choose response language (or All)", LANG_OPTIONS, index=0)

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if not uploaded_file:
    st.info("Please upload a PDF to begin.")
    st.stop()

# Save to a temp file
with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
    tmp.write(uploaded_file.read())
    pdf_path = tmp.name

st.success("PDF uploaded — extracting content...")

# Extract tables and text
tables_list = []
text_chunks: List[str] = []

with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        # tables
        page_tables = page.extract_tables()
        if page_tables:
            for table in page_tables:
                if len(table) < 2:
                    continue
                cols = table[0]
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

# Display extracted content + downloads
st.subheader("Extracted Tables")
if not df_table.empty:
    st.dataframe(df_table)
    buf = io.BytesIO()
    try:
        df_table.to_excel(buf, index=False, engine="openpyxl")
        st.download_button("Download tables as Excel", data=buf.getvalue(), file_name="tables.xlsx")
    except Exception as e:
        st.warning(f"Excel export (tables) failed: {e}")
else:
    st.info("No tables found in the PDF.")

st.subheader("Extracted Text Lines")
if not df_text.empty:
    st.dataframe(df_text)
    buf2 = io.BytesIO()
    try:
        df_text.to_excel(buf2, index=False, engine="openpyxl")
        st.download_button("Download text as Excel", data=buf2.getvalue(), file_name="text.xlsx")
    except Exception as e:
        st.warning(f"Excel export (text) failed: {e}")
else:
    st.info("No text lines found in the PDF.")

# Build semantic indexes
st.info("Building semantic index (for smart search). This may take a moment...")
text_docs = build_text_chunks(text_chunks) if text_chunks else []
db_text = build_vectorstore_from_docs(text_docs) if text_docs else None

db_table = None
if not df_table.empty:
    table_rows = [" | ".join(map(str, row)) for row in df_table.values.tolist()]
    table_docs = build_text_chunks(table_rows)
    db_table = build_vectorstore_from_docs(table_docs)

st.success("Semantic index ready.")

# Search UI and prompt builder
st.subheader("Unified Smart Search (Tables + Semantic) — choose language & ask")
query = st.text_input("Type a keyword or question and press Enter:")

def prepare_prompt_for_answer(query_text: str, context_text: str, language: str):
    lang_map = {
        "English": "English",
        "Hindi": "Hindi",
        "Bhojpuri": "Bhojpuri",
        "Urdu": "Urdu",
        "Tamil": "Tamil",
        "Marathi": "Marathi"
    }
    if language == "All":
        lang_instruction = (
            "Return answer in English followed by translations into Hindi, Bhojpuri, Urdu, Tamil, Marathi. "
            "Label each section clearly like '### English:', '### Hindi:' etc."
        )
    else:
        lang_instruction = f"Return the answer ONLY in {lang_map.get(language, 'English')}."
    prompt = (
        f"PDF AI PARSE REQUEST\nUser query: {query_text}\n\n"
        f"Context extracted from the PDF (truncated):\n{context_text[:15000]}\n\n"
        f"{lang_instruction}\n\n"
        "Provide concise insights and any actionable items. If tabular data is relevant, include a markdown table."
    )
    return prompt

if query:
    # Structured table match
    st.markdown("**Structured table matches:**")
    if not df_table.empty:
        structured_matches = df_table[df_table.apply(lambda r: r.astype(str).str.contains(query, case=False, na=False).any(), axis=1)]
        if not structured_matches.empty:
            st.dataframe(structured_matches)
        else:
            st.info("No direct structured matches found in tables.")
    else:
        st.info("No tables to search.")

    # Semantic free-text matches
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

    # Ask GYANM model for the answer, respecting chosen language
    context_parts = []
    if semantic_matches:
        context_parts.append("\n".join([d.page_content for d in semantic_matches]))
    if not df_table.empty:
        context_parts.append(df_table.head(20).to_csv(index=False))
    context_text = "\n\n".join(context_parts)[:15000]

    prompt = prepare_prompt_for_answer(query, context_text, chosen_lang)
    try:
        answer_text = gyanm_generate(prompt)
        st.markdown("### GYANM - PDF AI Parse Result")
        st.markdown(answer_text, unsafe_allow_html=True)

        if chosen_lang == "All":
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

# Dynamic table generator
st.markdown("---")
st.subheader("Dynamic Table Generator (ask GYANM to build a table from PDF)")
dynamic_query = st.text_input("Type what table you want to extract (example: 'list all deadlines and their dates'):")

if dynamic_query:
    combined_content = "\n".join(text_chunks)
    if not df_table.empty:
        combined_content += "\n\n" + df_table.to_csv(index=False)
    dyn_prompt = (
        f"PDF AI PARSE - USER REQUEST: {dynamic_query}\n\n"
        f"PDF CONTENT (truncated): {combined_content[:15000]}\n\n"
        "Return ONLY a markdown table that satisfies the user's request."
    )
    try:
        dyn_text = gyanm_generate(dyn_prompt)
        st.markdown("### GYANM - Generated Table (markdown)")
        st.markdown(dyn_text, unsafe_allow_html=True)
        try:
            if "<table" in dyn_text.lower():
                dfs = pd.read_html(dyn_text)
            else:
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
                try:
                    df_generated.to_excel(bufg, index=False, engine="openpyxl")
                    st.download_button("Download AI-generated table", data=bufg.getvalue(), file_name="ai_generated_table.xlsx")
                except Exception as e:
                    st.warning(f"Excel export (AI-generated table) failed: {e}")
        except Exception as e:
            st.warning(f"Could not parse AI-generated table into DataFrame: {e}")
    except Exception as e:
        st.error(f"Dynamic table generation failed: {e}")

# Chat
st.markdown("---")
st.subheader("General Chat (ask anything from the PDF)")
user_q = st.text_input("Ask a question about the PDF, press Enter:")

if user_q:
    combined_content = "\n".join(text_chunks)
    if not df_table.empty:
        combined_content += "\n\n" + df_table.to_csv(index=False)
    chat_prompt = prepare_prompt_for_answer(user_q, combined_content, chosen_lang)
    try:
        chat_text = gyanm_generate(chat_prompt)
        st.markdown("### GYANM - Chat Answer (PDF AI Parse)")
        st.markdown(chat_text, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Chat generation failed: {e}")

st.markdown("---")
st.caption("⚙️ Ensure GEMINI_API_KEY is set as an environment variable (or in .env) and install google-genai for full GYANM features.")
