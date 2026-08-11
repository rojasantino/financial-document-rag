import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Financial Document Intelligence", page_icon="💰")
st.title("💰 Financial Document Intelligence")
st.caption("Upload financial documents and ask questions grounded in their content.")

with st.sidebar:
    st.header("1. Upload a document")
    uploaded_file = st.file_uploader("Upload a financial PDF", type=["pdf"])

    if uploaded_file and st.button("Index document"):
        with st.spinner("Extracting, chunking, and embedding..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            resp = requests.post(f"{API_URL}/upload", files=files)

        if resp.status_code == 200:
            data = resp.json()
            st.success(f"Indexed {data['chunks_indexed']} chunks from {data['filename']}.")
        else:
            st.error(resp.json().get("detail", "Upload failed."))

st.header("2. Ask a question")
question = st.text_input("e.g. What was revenue in 2025?")

if st.button("Ask") and question:
    with st.spinner("Retrieving relevant context and generating an answer..."):
        resp = requests.post(f"{API_URL}/ask", json={"question": question})

    if resp.status_code == 200:
        result = resp.json()
        st.subheader("Answer")
        st.write(result["answer"])

        if result.get("sources"):
            st.subheader("Sources")
            for s in result["sources"]:
                st.write(f"📄 **{s['source']}** — Page {s['page']} (relevance: {s['relevance']})")
    else:
        st.error("Something went wrong. Is the backend running?")
