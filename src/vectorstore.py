import time
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from src.config import settings

def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        dimensions=settings.embedding_dimension,
    )

def ensure_index() -> object:
    pc = Pinecone(api_key=settings.pinecone_api_key)

    if not pc.has_index(settings.pinecone_index_name):
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )

        # Newly-created serverless indexes can take a moment to become ready.
        for _ in range(60):
            description = pc.describe_index(settings.pinecone_index_name)
            if description.status.get("ready"):
                break
            time.sleep(1)

    return pc.Index(settings.pinecone_index_name)

def get_vector_store() -> PineconeVectorStore:
    index = ensure_index()
    return PineconeVectorStore(
        index=index,
        embedding=get_embeddings(),
    )
