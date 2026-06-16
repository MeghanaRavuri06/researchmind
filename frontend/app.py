import streamlit as st
import httpx

API = "http://api:8000"

st.set_page_config(page_title="ResearchMind", page_icon="🔬", layout="wide")
st.title("🔬 ResearchMind")
st.caption("Intelligent research assistant — upload docs, ask questions")

col1, col2 = st.columns([1, 1])

TIMEOUT = httpx.Timeout(None, connect=30.0)

with col1:
    st.subheader("Ingest Documents")

    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded and st.button("Ingest PDF"):
        with st.spinner("Ingesting... please wait"):
            try:
                with httpx.Client(timeout=TIMEOUT) as client:
                    r = client.post(
                        f"{API}/ingest/pdf",
                        files={"file": (uploaded.name, uploaded, "application/pdf")}
                    )
                st.success(r.json().get("message", "Done"))
            except Exception as e:
                st.error(f"Error: {str(e)}")

    url_input = st.text_input("Or paste a URL")
    if url_input and st.button("Ingest URL"):
        with st.spinner("Fetching and ingesting..."):
            try:
                with httpx.Client(timeout=TIMEOUT) as client:
                    r = client.post(f"{API}/ingest/url", json={"url": url_input})
                st.success(r.json().get("message", "Done"))
            except Exception as e:
                st.error(f"Error: {str(e)}")

with col2:
    st.subheader("Ask a Question")

    question = st.text_area("Your question:", height=100)
    use_agent = st.checkbox("Use AI Agent (recommended)", value=True)

    if question and st.button("Search", type="primary"):
        with st.spinner("Thinking... please wait"):
            try:
                with httpx.Client(timeout=TIMEOUT) as client:
                    r = client.post(
                        f"{API}/query",
                        json={"question": question, "use_agent": use_agent}
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
            except Exception as e:
                st.error(f"Error: {str(e)}")