import argparse
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings, validate_settings
from src.loaders import load_documents
from src.vectorstore import get_vector_store

def stable_chunk_id(source: str, chunk_index: int, text: str) -> str:
    digest = hashlib.sha256(
        f"{source}|{chunk_index}|{text}".encode("utf-8")
    ).hexdigest()
    return digest

def ingest(data_dir: str = "data") -> int:
    validate_settings()

    docs = load_documents(data_dir)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    ids = []
    per_source_counter: dict[str, int] = {}

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        chunk_index = per_source_counter.get(source, 0)
        per_source_counter[source] = chunk_index + 1

        chunk.metadata["chunk_index"] = chunk_index
        ids.append(stable_chunk_id(source, chunk_index, chunk.page_content))

    vector_store = get_vector_store()
    vector_store.add_documents(chunks, ids=ids)

    print(f"Loaded {len(docs)} document units.")
    print(f"Created/upserted {len(chunks)} chunks into Pinecone.")
    print(f"Index: {settings.pinecone_index_name}")
    return len(chunks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest enterprise documents into Pinecone.")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    ingest(args.data_dir)
