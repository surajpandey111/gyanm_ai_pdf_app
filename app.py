# app.py (ready to paste) -----------------------------------------------------
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

# --- Robust LangChain / FAISS imports (safe for Streamlit Cloud) -------------
has_langchain = False
has_langchain_community = False
CharacterTextSplitter = None
FAISS = None
HuggingFaceEmbeddings = None

try:
    # try newest path first
    from langchain.text_splitters import CharacterTextSplitter  # new path
    has_langchain = True
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
        has_langchain_community = True
    except Exception:
        FAISS = None
        HuggingFaceEmbeddings = None
        has_langchain_community = False
except Exception:
    # fallback to older path
    try:
        from langchain.text_splitter import CharacterTextSplitter  # older path
        has_langchain = True
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_community.embeddings import HuggingFaceEmbeddings
            has_langchain_community = True
        except Exception:
            FAISS = None
            HuggingFaceEmbeddings = None
            has_langchain_community = False
    except Exception:
        CharacterTextSplitter = None
        FAISS = None
        HuggingFaceEmbeddings = None
        has_langchain = False
        has_langchain_community = False

# Simple fallback splitter (mimics CharacterTextSplitter interface)
class SimpleCharacterSplitter:
    def __init__(self, chunk_size=500, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text):
        if not text:
            return []
        chunks = []
        start = 0
        L = len(text)
        cs = self.chunk_size
        co = self.chunk_overlap
        while start < L:
            end = min(start + cs, L)
            chunks.append(text[start:end])
            start = end - co if end - co > start else end
        return chunks

    def create_documents(self, texts):
        # return objects with .page_content to mimic LangChain docs
        class Doc:
            def __init__(self, content): self.page_content = content
        docs = []
        if isinstance(texts, str):
            texts = [texts]
        for t in texts:
            for c in self.split_text(t):
                docs.append(Doc(c))
        return docs

# If FAISS not available, provide an in-memory vector fallback using sentence-transformers
use_sentence_transformers_fallback = False
sentence_model = None
import numpy as np
try:
    if FAISS is None:
        # Try to import sentence-transformers encoder for in-memory embedding+cosine similarity
        from sentence_transformers import SentenceTransformer
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        use_sentence_transformers_fallback = True
except Exception:
    use_sentence_transformers_fallback = False
    sentence_model = None

# ---------------- GYANM AI config ------------------------------------------------
load_dotenv()
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # uses your .env
    gyanm_ai = genai.GenerativeModel("gemini-2.5-flash")
except Exception as e:
    # if Gemini not configured, we keep a placeholder to avoid crash
    gyanm_ai = None
    st.warning("Warning: Google Generative AI not configured or failed to initialize. Gemini features will be disabled.")

# ---------------- Streamlit page ------------------------------------------------
st.set_page_config(page_title="AI PDF to Excel & Summarizing by Suraj", layout="wide")
st.title("🤖 ARTIFICIAL INTELLIGENCE PDF TO EXCEL, SMART SEARCH & INSIGHTS")
st.caption("🛠 Developed with ❤️ by **Suraj Kumar Pandey (Founder, Gyanm AI Platform)**")

uploaded_file = st.file_uploader("📄 Upload your PDF file", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.success("✅ PDF uploaded and ready to process!")

    # 3️⃣ Extract tables + text
    tables_list = []
    text_chunks = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Tables
            page_tables = page.extract_tables()
            if page_tables:
                for table in page_tables:
                    cols = table[0]
                    # make columns unique
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

            # Text
            page_text = page.extract_text()
            if page_text:
                lines = [line.strip() for line in page_text.split("\n") if line.strip()]
                text_chunks.extend(lines)

    # DataFrames
    if tables_list:
        try:
            df_table = pd.concat(tables_list, ignore_index=True)
        except Exception as e:
            st.error(f"⚠️ Error merging tables: {e}")
            df_table = pd.DataFrame()
    else:
        df_table = pd.DataFrame()

    df_text = pd.DataFrame(text_chunks, columns=["Text"])
    df_text["source"] = "text"

    # 4️⃣ Display tables
    st.subheader("📊 Extracted Tables")
    if not df_table.empty:
        st.dataframe(df_table)

        excel_buf = io.BytesIO()
        df_table.to_excel(excel_buf, index=False, engine="openpyxl")
        st.download_button(
            "⬇️ Download Tables as Excel",
            data=excel_buf.getvalue(),
            file_name="tables.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # 5️⃣ Column-based filter
        st.subheader("🔍 Filter Structured Data by Any Column")
        filter_query = st.text_input("Type to search in tables:")

        if filter_query:
            filtered_df = df_table[
                df_table.apply(lambda row: row.astype(str).str.contains(filter_query, case=False).any(), axis=1)
            ]
            st.dataframe(filtered_df)
            if filtered_df.empty:
                st.warning("No matching rows found.")
    else:
        st.info("No tables found in the PDF.")

    # 6️⃣ Display text
    st.subheader("📝 Extracted Free Text")
    if not df_text.empty:
        st.dataframe(df_text)

        excel_buf2 = io.BytesIO()
        df_text.to_excel(excel_buf2, index=False, engine="openpyxl")
        st.download_button(
            "⬇️ Download Text as Excel",
            data=excel_buf2.getvalue(),
            file_name="text.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.info("No text lines found in the PDF.")

    # 7️⃣ Build semantic index (safe): prefer LangChain/FAISS if available; else use sentence-transformers in-memory
    st.info("🔗 Building semantic index for smart text search (if available)...")

    if has_langchain and has_langchain_community and HuggingFaceEmbeddings and FAISS:
        try:
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            split_docs = text_splitter.create_documents(text_chunks)
            embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db_text = FAISS.from_documents(split_docs, embed_model) if split_docs else None

            if not df_table.empty:
                table_text_rows = [" | ".join(str(v) for v in row) for row in df_table.values.tolist()]
                table_docs = text_splitter.create_documents(table_text_rows)
                db_table = FAISS.from_documents(table_docs, embed_model) if table_docs else None
            else:
                db_table = None
        except Exception as e:
            st.warning(f"FAISS/langchain-community failed to initialize: {e}")
            db_text = None
            db_table = None
    elif has_langchain and CharacterTextSplitter is not None and use_sentence_transformers_fallback and sentence_model:
        # Use LangChain splitter (if present) but sentence-transformers for embeddings + in-memory similarity
        try:
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100) if CharacterTextSplitter else SimpleCharacterSplitter(chunk_size=500, chunk_overlap=100)
            split_docs = text_splitter.create_documents(text_chunks)
            # compute vector matrix for split_docs using sentence_model
            doc_texts = [d.page_content for d in split_docs]
            if doc_texts:
                doc_embeddings = sentence_model.encode(doc_texts, convert_to_numpy=True)
            else:
                doc_embeddings = np.array([]).reshape(0, 384)
            # store docs and embeddings in-memory
            db_text = {"docs": split_docs, "embeddings": doc_embeddings}
            if not df_table.empty:
                table_text_rows = [" | ".join(str(v) for v in row) for row in df_table.values.tolist()]
                table_docs = text_splitter.create_documents(table_text_rows)
                table_texts = [d.page_content for d in table_docs]
                table_embeddings = sentence_model.encode(table_texts, convert_to_numpy=True) if table_texts else np.array([]).reshape(0,384)
                db_table = {"docs": table_docs, "embeddings": table_embeddings}
            else:
                db_table = None
        except Exception as e:
            st.warning(f"Sentence-transformers fallback failed: {e}")
            db_text = None
            db_table = None
    else:
        # no semantic features available; use a simple splitter so rest of app works
        text_splitter = SimpleCharacterSplitter(chunk_size=500, chunk_overlap=100)
        split_docs = text_splitter.create_documents(text_chunks)
        db_text = None
        db_table = None
        st.warning("Semantic search disabled (LangChain/FAISS or sentence-transformers not available).")

    # 8️⃣ Unified Smart Search
    st.subheader("🤖 Unified Smart Search (Tables + Semantic)")
    unified_query = st.text_input("Type your keyword or question:")

    def semantic_search_in_memory(db, query, k=5):
        # db: {"docs": [Doc], "embeddings": np.array}
        if not db or "docs" not in db or db["embeddings"].size == 0:
            return []
        q_emb = sentence_model.encode([query], convert_to_numpy=True)[0]
        # cosine similarity
        emb = db["embeddings"]
        scores = np.dot(emb, q_emb) / (np.linalg.norm(emb, axis=1) * np.linalg.norm(q_emb) + 1e-10)
        idxs = np.argsort(-scores)[:k]
        return [db["docs"][i] for i in idxs]

    if unified_query:
        structured_matches = pd.DataFrame()
        if not df_table.empty:
            structured_matches = df_table[
                df_table.apply(lambda row: row.astype(str).str.contains(unified_query, case=False).any(), axis=1)
            ]

        # semantic matches: prefer FAISS/langchain if available, else in-memory
        if db_text and isinstance(db_text, dict) and "embeddings" in db_text:
            try:
                semantic_matches = semantic_search_in_memory(db_text, unified_query, k=5)
            except Exception:
                semantic_matches = []
        elif db_text and hasattr(db_text, "similarity_search"):
            try:
                semantic_matches = db_text.similarity_search(unified_query, k=5)
            except Exception:
                semantic_matches = []
        else:
            semantic_matches = []

        # semantic table matches
        if db_table and isinstance(db_table, dict) and "embeddings" in db_table:
            try:
                semantic_table_matches = semantic_search_in_memory(db_table, unified_query, k=5)
            except Exception:
                semantic_table_matches = []
        elif db_table and hasattr(db_table, "similarity_search"):
            try:
                semantic_table_matches = db_table.similarity_search(unified_query, k=5)
            except Exception:
                semantic_table_matches = []
        else:
            semantic_table_matches = []

        st.markdown("### 🟦 Structured Table Matches")
        if not structured_matches.empty:
            st.dataframe(structured_matches)
        else:
            st.info("No structured matches found.")

        st.markdown("### 🟨 Semantic Free Text Matches")
        if semantic_matches:
            for i, doc in enumerate(semantic_matches, 1):
                st.write(f"**Result {i}:**\n{doc.page_content}\n")
        else:
            st.info("No semantic free text matches found or semantic features disabled.")

        st.markdown("### 🟧 Semantic Table Matches (AI)")
        semantic_table_rows = []
        if semantic_table_matches:
            for doc in semantic_table_matches:
                for idx, row in enumerate(df_table.values.tolist()):
                    joined_row = " | ".join(str(v) for v in row)
                    if joined_row == doc.page_content:
                        semantic_table_rows.append(df_table.iloc[idx])
                        break

            if semantic_table_rows:
                ai_table_df = pd.DataFrame(semantic_table_rows)
                st.dataframe(ai_table_df)

                excel_buf3 = io.BytesIO()
                ai_table_df.to_excel(excel_buf3, index=False, engine="openpyxl")
                st.download_button(
                    "⬇️ Download AI Matched Table Rows as Excel",
                    data=excel_buf3.getvalue(),
                    file_name="ai_table_matches.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.info("No semantic table matches found.")

        # explanation using GYANM-AI (if configured)
        table_context = "\n".join(doc.page_content for doc in semantic_table_matches) if semantic_table_matches else ""
        prompt = (
            f"Given this extracted data:\n\n{table_context}\n\n"
            f"and user query:\n{unified_query}\n\n"
            f"Provide a detailed answer with insights."
        )
        try:
            if gyanm_ai:
                gyanm_response = gyanm_ai.generate_content(prompt)
                st.markdown(f"### 🧠 GYANM - AI Explanation\n{gyanm_response.text}")
            else:
                st.info("GYANM-AI not configured; can't produce AI explanation.")
        except Exception as e:
            st.error(f"❌ GYANM - AI error: {e}")

    # 9️⃣ Dynamic AI Table Generator (rest of your app behavior)
    st.subheader("📊 Dynamic Table Generator with GYANM - AI")
    dynamic_query = st.text_input("Type what table you want to extract:")

    if dynamic_query:
        try:
            combined_content = "\n".join(text_chunks)
            if not df_table.empty:
                combined_content += "\n\n" + df_table.to_csv(index=False)
            dynamic_prompt = f"""
            From the PDF content below, form a structured table in markdown as per the user request:
            USER REQUEST: {dynamic_query}
            PDF CONTENT: {combined_content[:15000]}
            """
            if gyanm_ai:
                gyanm_ai_response = gyanm_ai.generate_content(dynamic_prompt)
                gyanm_table_md = gyanm_ai_response.text
                st.markdown("### 📋 Generated Table by GYANM - AI")
                st.markdown(gyanm_table_md, unsafe_allow_html=True)

                # SAFELY parse table using read_html
                dfs = pd.read_html(f"<table>{gyanm_table_md}</table>") if "<table>" not in gyanm_table_md else pd.read_html(gyanm_table_md)
                df_gyanm = dfs[0]
                excel_buf4 = io.BytesIO()
                df_gyanm.to_excel(excel_buf4, index=False, engine="openpyxl")
                st.download_button("⬇️ Download GYANM-AI Generated Table as Excel",
                                data=excel_buf4.getvalue(),
                                file_name="gyanm_ai_generated_table.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.info("GYANM-AI not configured; dynamic table generation requires Gemini.")
        except Exception as e:
            st.error(f"❌ GYANM - AI dynamic table error: {e}")

    # 🔟 Summary
    if st.button("🧠 Generate AI Summary of Entire Text"):
        try:
            combined_text = "\n".join(text_chunks)
            summary_prompt = f"Summarize clearly, listing important points:\n\n{combined_text[:8000]}"
            if gyanm_ai:
                response = gyanm_ai.generate_content(summary_prompt)
                st.subheader("🧠 GYANM - AI Summary")
                st.success(response.text)
            else:
                st.info("GYANM-AI not configured; summary requires Gemini.")
        except Exception as e:
            st.error(f"❌ GYANM - AI summary error: {e}")

    # insights
    if st.button("🤖 Generate AI Insights"):
        try:
            combined_text = "\n".join(text_chunks)
            insights_prompt = f"""
            Analyze the following text. Provide:
            1. Key people or organizations
            2. Any actionable items or deadlines
            3. Most critical details
            4. Recommended next steps

            TEXT:
            {combined_text[:8000]}
            """
            if gyanm_ai:
                insights_response = gyanm_ai.generate_content(insights_prompt)
                st.subheader("🔎 GYANM - AI Insights")
                st.success(insights_response.text)
            else:
                st.info("GYANM-AI not configured; insights require Gemini.")
        except Exception as e:
            st.error(f"❌ GYANM - AI insights error: {e}")

    # 1️⃣1️⃣ General Chat with table/chart ability
    st.subheader("💬 GYANM - AI General Chat with Table/Chart Builder")
    user_question = st.text_input("Ask anything from the PDF, even chart/table requests:")

    if user_question:
        try:
            combined_content = "\n".join(text_chunks)
            if not df_table.empty:
                combined_content += "\n\n" + df_table.to_csv(index=False)

            chat_prompt = f"""
            You are GYANM-AI, an intelligent assistant.

            The user question is: "{user_question}"

            Please answer clearly, and if appropriate, also produce:
            - a markdown table if it fits
            - or a chart data description (example: "labels: X, values: Y")
            based on the following PDF contents:

            {combined_content[:15000]}

            Return any tables in markdown, charts as text data description, and a textual explanation.
            """

            if gyanm_ai:
                chat_response = gyanm_ai.generate_content(chat_prompt)
                answer = chat_response.text
                st.subheader("💬 GYANM - AI Answer")
                st.markdown(answer, unsafe_allow_html=True)

                # parse possible table
                table_lines = [line for line in answer.splitlines() if "|" in line]
                if table_lines:
                    csv_like = "\n".join(table_lines).replace("|", ",")
                    import io as _io
                    df_ai = pd.read_csv(_io.StringIO(csv_like), skipinitialspace=True)

                    st.dataframe(df_ai)
                    excel_buf5 = io.BytesIO()
                    df_ai.to_excel(excel_buf5, index=False, engine="openpyxl")
                    st.download_button(
                        "⬇️ Download GYANM-AI General Chat Table as Excel",
                        data=excel_buf5.getvalue(),
                        file_name="gyanm_ai_general_chat_table.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                # parse chart instructions
                if "labels:" in answer and "values:" in answer:
                    labels_match = re.search(r"labels:\s*(.*)", answer)
                    values_match = re.search(r"values:\s*(.*)", answer)
                    if labels_match and values_match:
                        labels = [x.strip() for x in labels_match.group(1).split(",")]
                        values = [float(x.strip()) for x in values_match.group(1).split(",")]
                        fig, ax = plt.subplots()
                        ax.pie(values, labels=labels, autopct="%1.1f%%")
                        st.pyplot(fig)
            else:
                st.info("GYANM-AI not configured; chat requires Gemini.")
        except Exception as e:
            st.error(f"❌ GYANM - AI chat error: {e}")

    st.markdown("---")
    st.markdown("🛠 ** Developed by Suraj Kumar Pandey (Founder, Gyanm AI Platform)**")

else:
    st.info("Please upload a PDF file to get started.")
# -----------------------------------------------------------------------------
