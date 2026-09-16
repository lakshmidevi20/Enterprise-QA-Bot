import streamlit as st

from src.config import settings
from src.rag import answer_question

st.set_page_config(
    page_title="Enterprise RAG Q&A",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Enterprise RAG Q&A")
st.caption(
    f"Pinecone index: {settings.pinecone_index_name} · "
    f"Embedding: {settings.embedding_model} · "
    f"LLM: {settings.chat_model}"
)

with st.sidebar:
    st.header("How to use")
    st.write("1. Put PDF, DOCX, TXT, or Markdown files in `data/`.")
    st.write("2. Run `python -m src.ingest`.")
    st.write("3. Ask questions here.")
    st.info(
        "The assistant is instructed to answer only from retrieved enterprise documents "
        "and to refuse when evidence is insufficient."
    )

question = st.text_input(
    "Ask a question about your enterprise documents",
    placeholder="Example: How many days of paid parental leave are available?",
)

if st.button("Ask", type="primary", disabled=not question.strip()):
    with st.spinner("Retrieving and answering..."):
        try:
            result = answer_question(question.strip())
        except Exception as exc:
            st.error(str(exc))
        else:
            st.subheader("Answer")
            st.write(result.answer)

            st.caption(
                f"Max retrieval relevance: {result.max_relevance_score:.3f} · "
                f"Retrieval sufficient: {result.retrieval_sufficient}"
            )

            st.subheader("Retrieved sources")
            if not result.sources:
                st.write("No sources retrieved.")
            for i, src in enumerate(result.sources, start=1):
                page_or_chunk = (
                    f"page {src.page}" if src.page is not None
                    else f"chunk {src.chunk_index}"
                )
                with st.expander(
                    f"{i}. {src.source} · {page_or_chunk} · "
                    f"score {src.relevance_score:.3f}"
                ):
                    st.write(src.excerpt)
