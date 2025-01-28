import asyncio
import os

from dotenv import load_dotev

from src.embedder import LaBSEEmbedder
from src.knowledgebase import QdrantKnowledgeBase

if __name__ == "__main__":
    load_dotenv()
    qdrant_host = os.getenv("QDRANT_HOST")
    qdrant_port = int(os.getenv("QDRANT_PORT"))
    embedder_name = os.getenv("EMBEDDER_NAME")
    collection_name = os.getenv("COLLECTION_NAME")
    sujet_path = os.getenv("SUJET_DOC_PATH")
    collision_path = os.getenv("COLLISION_DOC_PATH")

    embedder = LaBSEEmbedder(model_name=embedder_name)
    kb = QdrantKnowledgeBase(embedder, host=qdrant_host, port=qdrant_port)

    asyncio.run(
        kb.fill_base(
            docs_path=sujet_path, collection_name=collection_name, rule_type="Rules"
        )
    )
    asyncio.run(
        kb.fill_base(
            docs_path=collision_path, collection_name=collection_name, rule_type="Rules"
        )
    )
