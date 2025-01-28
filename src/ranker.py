import asyncio
from typing import List

from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

from src.embedder import LaBSEEmbedder


class ChunksReranker:
    def __init__(self, embedder: LaBSEEmbedder, alpha: int):
        self.embedder = embedder
        self.alpha = alpha
        self.bm25 = None
        self.chunks = None
        self.chunk_embs = None

    async def fit(self, chunks: List[str]):
        self.chunks = chunks
        tokenized_chunks = [chunk.split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
        self.chunk_embs = await self.embedder.embed(chunks)

    async def rank(self, query: str, top_k: int = 3):
        if self.bm25 is None or self.chunk_embs is None:
            raise ValueError(
                "Ранкер должен быть подготовлен с помощью метода `fit` перед использованием."
            )
        bm25_scores = self.bm25.get_scores(query.split())
        query_emb = await self.embedder.embed([query])
        cosine_scores = cosine_similarity(query_emb, self.chunk_embs).flatten()
        final_scores = self.alpha * cosine_scores + (1 - self.alpha) * bm25_scores
        ranked_chunks = [
            (text, score)
            for score, text in sorted(zip(final_scores, self.chunks), reverse=True)
        ]
        return ranked_chunks[:top_k]
