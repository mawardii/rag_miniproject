# main.py
import os
import time

import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.manager import get_openai_callback
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

from embeddings import load_or_build_vectorstore  # Reuse embedding logic

load_dotenv()


def build_qa(
    model_name: str,
    temperature: float,
    top_k: int,
    return_sources: bool,
    chunk_size: int,
    chunk_overlap: int,
    rebuild_counter: int,
):
    """Build the RAG chain with DeepSeek."""
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        max_tokens=1000,
    )
    db = load_or_build_vectorstore(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, force_rebuild=(rebuild_counter > 0)
    )
    retriever = db.as_retriever(search_kwargs={"k": top_k})
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=return_sources,
    )
    return qa


def main():
    st.set_page_config(page_title="Simple RAG QA", page_icon="💬", layout="centered")
    st.title("💬 Simple RAG QA (DeepSeek)")
    st.caption("FAISS + DeepSeek + LangChain. Optimized for responsiveness.")

    # Sidebar settings
    st.sidebar.header("⚙️ Settings")
    fast_mode = st.sidebar.checkbox(
        "Fast mode", value=True, help="Lower k and temperature for speed."
    )
    model = st.sidebar.selectbox(
        "Model",
        options=["deepseek-chat", "deepseek-coder"],
        index=0,
        help="Select a DeepSeek model.",
    )
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    top_k = st.sidebar.slider("Top-k (retrieval)", 1, 8, 3)
    return_sources = st.sidebar.checkbox("Return sources", value=True)
    chunk_size = st.sidebar.number_input(
        "Chunk size", min_value=200, max_value=1500, value=500, step=50
    )
    chunk_overlap = st.sidebar.number_input(
        "Chunk overlap", min_value=0, max_value=400, value=50, step=10
    )

    if fast_mode:
        temperature = min(temperature, 0.15)
        top_k = min(top_k, 2)

    if "rebuild_counter" not in st.session_state:
        st.session_state.rebuild_counter = 0
    if st.sidebar.button("Rebuild index"):
        st.session_state.rebuild_counter += 1

    if not os.path.exists("docs"):
        os.makedirs("docs", exist_ok=True)
        with open(os.path.join("docs", "example.txt"), "w", encoding="utf-8") as f:
            f.write(
                "This is an example note. Add your .txt or .pdf files inside the 'docs' folder."
            )

    docs_empty = len(os.listdir("docs")) == 0
    if docs_empty:
        st.warning("No documents found in 'docs/'. Add .txt or .pdf files and rerun.")
        st.stop()

    with st.spinner("Indexing (first run may take a moment)..."):
        qa = build_qa(
            model_name=model,
            temperature=temperature,
            top_k=top_k,
            return_sources=return_sources,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            rebuild_counter=st.session_state.rebuild_counter,
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant" and m.get("sources"):
                with st.expander("Sources"):
                    for i, s in enumerate(m["sources"], 1):
                        st.markdown(f"**{i}.** `{s}`")

    q = st.chat_input("Ask a question based on your notes")
    if q:
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("Thinking…")
            t0 = time.time()
            try:
                with get_openai_callback() as cb:
                    resp = qa.invoke({"query": q})
                    answer = resp.get("result", "")
                    sources_docs = resp.get("source_documents", []) if return_sources else []
                    sources = [d.metadata.get("source", "unknown") for d in sources_docs]
                    elapsed = time.time() - t0

                    placeholder.markdown(answer if answer else "(No answer)")
                    if return_sources and sources:
                        with st.expander("Sources"):
                            for i, s in enumerate(sources, 1):
                                st.markdown(f"**{i}.** `{s}`")
                    st.caption(
                        f"Response time: {elapsed:.2f}s | Tokens: {cb.total_tokens}"
                        f" | Est. Cost: ${cb.total_cost:.6f}"
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources if return_sources else [],
                        }
                    )
            except Exception as e:
                placeholder.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
