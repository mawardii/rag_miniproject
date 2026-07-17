"""Streamlit chat UI — a thin client over rag.pipeline.RAGPipeline.

No pipeline logic lives here: no chunking knobs, no retrieval parameters.
Those belong to rag.config so that serving, CLI, and evaluation all run the
same pipeline. This file only renders chat state and calls pipeline.answer().
"""

import streamlit as st

from rag.config import get_settings
from rag.pipeline import RAGPipeline
from rag.retrieval.vectorstore import build_index


@st.cache_resource(show_spinner="Loading pipeline (first run builds the index)...")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


def render_sources(sources) -> None:
    with st.expander("Sources"):
        for i, ref in enumerate(sources, 1):
            location = f"{ref.source}, p.{ref.page + 1}" if ref.page is not None else ref.source
            st.markdown(f"**[{i}]** `{location}`")
            st.caption(ref.snippet)


def main() -> None:
    st.set_page_config(page_title="Road-Damage Research RAG", page_icon="🛣️", layout="centered")
    st.title("🛣️ Road-Damage Research RAG")

    settings = get_settings()
    with st.sidebar:
        st.header("Pipeline config")
        st.caption(
            f"model: `{settings.chat_model}`\n\n"
            f"embeddings: `{settings.embedding_model}`\n\n"
            f"chunk: {settings.chunk_size}/{settings.chunk_overlap} · top-k: {settings.top_k}"
        )
        st.caption("Change values in .env / rag.config — not here — so eval and serving match.")
        if st.button("Rebuild index"):
            with st.spinner("Rebuilding index..."):
                build_index(settings)
            get_pipeline.clear()
            st.success("Index rebuilt.")

    pipeline = get_pipeline()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"])

    question = st.chat_input("Ask about the road-damage research corpus")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    result = pipeline.answer(question)
            except Exception:
                st.error("Something went wrong answering that — check the server logs.")
                raise
            st.markdown(result.answer)
            render_sources(result.sources)
            st.caption(
                f"{result.latency_s:.2f}s · {result.input_tokens} in / "
                f"{result.output_tokens} out tokens"
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": result.answer, "sources": result.sources}
            )


main()
