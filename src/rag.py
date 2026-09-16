from dataclasses import dataclass, asdict
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import settings, validate_settings
from src.vectorstore import get_vector_store

SYSTEM_PROMPT = """You are an enterprise knowledge-base assistant.

Answer ONLY from the supplied context.

Rules:
1. If the context does not contain enough information, say exactly:
   "I don't have enough information in the knowledge base to answer that."
2. Do not invent policies, dates, limits, product behavior, or compliance rules.
3. When several documents are needed, combine them carefully and identify any conflict.
4. If the question is ambiguous, state the most likely interpretation and ask a concise clarifying question.
5. Cite sources inline as [Source: filename, page/chunk] when making factual claims.
6. Keep answers concise but complete.
"""

@dataclass
class Source:
    source: str
    page: int | None
    chunk_index: int | None
    relevance_score: float
    excerpt: str

@dataclass
class RagResult:
    question: str
    answer: str
    sources: list[Source]
    max_relevance_score: float
    retrieval_sufficient: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def _format_context(scored_docs: list[tuple[Document, float]]) -> str:
    blocks = []
    for doc, score in scored_docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        chunk = doc.metadata.get("chunk_index")
        locator = f"page {page}" if page else f"chunk {chunk}"
        blocks.append(
            f"[Source: {source}, {locator}; relevance={score:.3f}]\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(blocks)

def answer_question(question: str) -> RagResult:
    validate_settings()
    vector_store = get_vector_store()

    scored_docs = vector_store.similarity_search_with_relevance_scores(
        question,
        k=settings.top_k,
    )

    max_score = max((score for _, score in scored_docs), default=0.0)
    sufficient = bool(scored_docs) and max_score >= settings.min_relevance_score

    sources = [
        Source(
            source=doc.metadata.get("source", "unknown"),
            page=doc.metadata.get("page"),
            chunk_index=doc.metadata.get("chunk_index"),
            relevance_score=float(score),
            excerpt=doc.page_content[:320].replace("\n", " "),
        )
        for doc, score in scored_docs
    ]

    if not sufficient:
        return RagResult(
            question=question,
            answer="I don't have enough information in the knowledge base to answer that.",
            sources=sources,
            max_relevance_score=float(max_score),
            retrieval_sufficient=False,
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "Question:\n{question}\n\n"
                "Retrieved context:\n{context}\n\n"
                "Answer using only the retrieved context.",
            ),
        ]
    )

    llm = ChatOpenAI(
        model=settings.chat_model,
        temperature=0,
    )

    chain = prompt | llm
    response = chain.invoke(
        {
            "question": question,
            "context": _format_context(scored_docs),
        }
    )

    return RagResult(
        question=question,
        answer=response.content,
        sources=sources,
        max_relevance_score=float(max_score),
        retrieval_sufficient=True,
    )
