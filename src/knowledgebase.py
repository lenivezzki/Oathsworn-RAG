import asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels
from typing import List, Optional
import numpy as np
import os
from src.embedder import LaBSEEmbedder

class QdrantKnowledgeBase:

    def __init__(self, embedder: LaBSEEmbedder, host:str, port:int):
        self.client = AsyncQdrantClient(host=host, port=port)
        self.embedder = embedder

    async def _add_texts(self, texts:List[str], embeddings:List[np.array], collection_name:str, rule_type:str, vector_size:int=768):
        """Добавление текстов в Qdrant"""

        if not await self.client.collection_exists(collection_name):
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
        )

        points = [
            qmodels.PointStruct(
                id=i, vector=embedding.tolist(), payload={"text": text, "rule_type": rule_type}
            )
            for i, (embedding, text) in enumerate(zip(embeddings, texts))
        ]
        await self.client.upsert(collection_name=collection_name, points=points)

    async def fill_base(self, docs_path:str, collection_name:str, rule_type:str, separator:str='||', vector_size:int=768):
    # pages_dir = 'pages'
        rules = []
        for page in os.listdir(docs_path):
            if '.csv' in page:
                with open(os.path.join(docs_path, page), 'r') as csvfile:
                    csvtext = csvfile.readlines()
            page_text = ' '.join(csvtext).split(separator)
            rules.extend(page_text)
        rule_embeddings = await self.embedder.embed(rules)
        await self._add_texts(rules, rule_embeddings, collection_name, rule_type, vector_size)
    

    async def search(self, query_embedding: np.array, collection_name:str, top_k: int = 5, rule_type: Optional[str] = None):
        """Поиск ближайших текстов в Qdrant с возможностью фильтрации по типу правил"""
        filters = None
        if rule_type:
            filters = qmodels.Filter(
                must=[qmodels.FieldCondition(
                    key="rule_type",
                    match=qmodels.MatchValue(value=rule_type)
                )]
            )
        search_result = await self.client.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            query_filter=filters
        )
        return [(hit.payload["text"], hit.score) for hit in search_result]