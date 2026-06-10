import streamlit as st
import httpx

API = "http://localhost:8000"

st.set_page_config(page_title="ResearchMind", page_icon="🔬", layout="wide")
st.title("🔬 ResearchMind")
st.caption("Intelligent research assistant — upload docs, ask questions")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Ingest Documents")

    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded and st.button("Ingest PDF"):
        with st.spinner("Ingesting..."):
            r = httpx.post(
                f"{API}/ingest/pdf",
                files={"file": (uploaded.name, uploaded, "application/pdf")}
            )
            st.success(r.json().get("message", "Done"))

    url_input = st.text_input("Or paste a URL")
    if url_input and st.button("Ingest URL"):
        with st.spinner("Fetching and ingesting..."):
            r = httpx.post(f"{API}/ingest/url", json={"url": url_input})
            st.success(r.json().get("message", "Done"))

with col2:
    st.subheader("Ask a Question")

    question = st.text_area("Your question:", height=100)
    use_agent = st.checkbox("Use AI Agent (recommended)", value=True)

    if question and st.button("Search", type="primary"):
        with st.spinner("Thinking..."):
            r = httpx.post(
                f"{API}/query",
                json={"question": question, "use_agent": use_agent},
                timeout=60
            )
            data = r.json()

        st.subheader("Answer")
        st.write(data.get("answer", "No answer"))

        if data.get("sources"):
            st.subheader("Sources")
            for src in data["sources"]:
                st.code(src)

        passes = data.get("retrieval_passes", 1)
        st.caption(f"Retrieval passes: {passes}")