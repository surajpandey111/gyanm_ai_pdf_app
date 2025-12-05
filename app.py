import streamlit as st
import os
import pdfplumber
import pandas as pd
import tempfile
import io
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # FIXED: New import for compatibility
import matplotlib.pyplot as plt
import re

# 1️⃣ AI HUB config
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # uses your .env
ai_hub = genai.GenerativeModel("gemini-2.5-flash")  # FIXED: Correct model name

# 2️⃣ Streamlit page
st.set_page_config(page_title="AI PDF to Excel & Summarizing by Suraj", layout="wide")
st.title("🤖 ARTIFICIAL INTELLIGENCE PDF TO EXCEL, SMART SEARCH & INSIGHTS")
st.caption("🛠 Developed with ❤️ by **Suraj Kumar Pandey (Founder, AI Hub Platform)**")

uploaded_file = st.file_uploader("📄 Upload your PDF file", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name
    st.success("✅ PDF uploaded and ready to process!")
    
    # 3️⃣ Extract tables + text
    tables_list = []
    text_chunks = []
    try:
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
    except Exception as e:
        st.error(f"⚠️ Error reading PDF: {e}")
        os.unlink(pdf_path)  # Clean up temp file
        st.stop()
    
    os.unlink(pdf_path)  # Clean up temp file
    
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
    
    # 7️⃣ Build semantic index
    st.info("🔗 Building semantic index for smart text search...")
    try:
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_docs = text_splitter.create_documents(text_chunks)
      
        embed_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},  # FIXED: Force CPU to avoid meta tensor errors
            encode_kwargs={'device': 'cpu'}  # FIXED: Same for encoding
        )
      
        if split_docs:  # safe check
            db_text = FAISS.from_documents(split_docs, embed_model)
        else:
            db_text = None
            st.warning("⚠️ No text available to build semantic index.")
        
        if not df_table.empty:
            table_text_rows = [
                " | ".join(str(v) for v in row)
                for row in df_table.values.tolist()
            ]
            table_docs = text_splitter.create_documents(table_text_rows)
            db_table = FAISS.from_documents(table_docs, embed_model)
        else:
            db_table = None
    except Exception as e:
        st.error(f"⚠️ Semantic index build failed: {e}. Using basic search fallback.")
        db_text = None
        db_table = None
    
    # 8️⃣ Unified Smart Search
    st.subheader("🤖 Unified Smart Search (Tables + Semantic)")
    unified_query = st.text_input("Type your keyword or question:")
    if unified_query:
        structured_matches = pd.DataFrame()
        if not df_table.empty:
            structured_matches = df_table[
                df_table.apply(lambda row: row.astype(str).str.contains(unified_query, case=False).any(), axis=1)
            ]
        if db_text:
             semantic_matches = db_text.similarity_search(unified_query, k=5)
        else:
            semantic_matches = []
        semantic_table_matches = []
        if db_table:
           semantic_table_matches = db_table.similarity_search(unified_query, k=5)
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
            st.info("No semantic free text matches found.")
        
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
                ai_table_df = pd.DataFrame(semantic_table_rows)  # FIXED: Ensure DataFrame
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
        
        # explanation
        table_context = "\n".join(doc.page_content for doc in semantic_table_matches) if semantic_table_matches else ""
        prompt = (
            f"Given this extracted data:\n\n{table_context}\n\n"
            f"and user query:\n{unified_query}\n\n"
            f"Provide a detailed answer with insights."
        )
        try:
            ai_hub_response = ai_hub.generate_content(prompt)
            st.markdown(f"### 🧠 AI HUB Explanation\n{ai_hub_response.text}")
        except Exception as e:
            st.error(f"❌ AI HUB error: {e}")
    
    # 9️⃣ Dynamic AI Table Generator
    st.subheader("📊 Dynamic Table Generator with AI HUB")
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
            ai_hub_response = ai_hub.generate_content(dynamic_prompt)
            ai_hub_table_md = ai_hub_response.text
            st.markdown("### 📋 Generated Table by AI HUB")
            st.markdown(ai_hub_table_md, unsafe_allow_html=True)
            # SAFELY parse table using read_html
            if "<table>" not in ai_hub_table_md:
                ai_hub_table_md = f"<table>{ai_hub_table_md}</table>"  # Wrap if needed
            dfs = pd.read_html(ai_hub_table_md)
            df_ai_hub = dfs[0]
            excel_buf4 = io.BytesIO()
            df_ai_hub.to_excel(excel_buf4, index=False, engine="openpyxl")
            st.download_button("⬇️ Download AI HUB Generated Table as Excel",
                               data=excel_buf4.getvalue(),
                               file_name="ai_hub_generated_table.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as e:
            st.error(f"❌ AI HUB dynamic table error: {e}")
    
    # 🔟 Summary
    if st.button("🧠 Generate AI Summary of Entire Text"):
        try:
            combined_text = "\n".join(text_chunks)
            summary_prompt = f"Summarize clearly, listing important points:\n\n{combined_text[:8000]}"
            response = ai_hub.generate_content(summary_prompt)
            st.subheader("🧠 AI HUB Summary")
            st.success(response.text)
        except Exception as e:
            st.error(f"❌ AI HUB summary error: {e}")
    
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
            insights_response = ai_hub.generate_content(insights_prompt)
            st.subheader("🔎 AI HUB Insights")
            st.success(insights_response.text)
        except Exception as e:
            st.error(f"❌ AI HUB insights error: {e}")
    
    # 1️⃣1️⃣ General Chat with table/chart ability
    st.subheader("💬 AI HUB General Chat with Table/Chart Builder")
    user_question = st.text_input("Ask anything from the PDF, even chart/table requests:")
    if user_question:
        try:
            combined_content = "\n".join(text_chunks)
            if not df_table.empty:
                combined_content += "\n\n" + df_table.to_csv(index=False)
            chat_prompt = f"""
            You are AI HUB, an intelligent assistant.
            The user question is: "{user_question}"
            Please answer clearly, and if appropriate, also produce:
            - a markdown table if it fits
            - or a chart data description (example: "labels: X, values: Y")
            based on the following PDF contents:
            {combined_content[:15000]}
            Return any tables in markdown, charts as text data description, and a textual explanation.
            """
            chat_response = ai_hub.generate_content(chat_prompt)
            answer = chat_response.text
            st.subheader("💬 AI HUB Answer")
            st.markdown(answer, unsafe_allow_html=True)
            # parse possible table
            table_lines = [line for line in answer.splitlines() if "|" in line]
            if table_lines:
                csv_like = "\n".join(table_lines).replace("|", ",")
                df_ai = pd.read_csv(io.StringIO(csv_like), skipinitialspace=True)
                st.dataframe(df_ai)
                excel_buf5 = io.BytesIO()
                df_ai.to_excel(excel_buf5, index=False, engine="openpyxl")
                st.download_button(
                    "⬇️ Download AI HUB General Chat Table as Excel",
                    data=excel_buf5.getvalue(),
                    file_name="ai_hub_general_chat_table.xlsx",
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
        except Exception as e:
            st.error(f"❌ AI HUB chat error: {e}")
    
    st.markdown("---")
    st.markdown("🛠 ** Developed by Suraj Kumar Pandey (Founder, AI Hub Platform)**")
else:
    st.info("Please upload a PDF file to get started.")

