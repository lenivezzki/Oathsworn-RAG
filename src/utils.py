import asyncio

from src.assistant import RulesAssistant
from src.embedder import LaBSEEmbedder
from src.knowledgebase import QdrantKnowledgeBase
from src.ranker import ChunksReranker


async def response(
    query: str,
    system_prompt,
    embedder: LaBSEEmbedder,
    knowledgbase: QdrantKnowledgeBase,
    collection_name: str,
    ranker: ChunksReranker,
    assistant: RulesAssistant,
    max_new_tokens: int = 350,
):
    query_embedding = await embedder.embed([query])
    query_embedding = query_embedding[0]
    results = await knowledgbase.search(query_embedding, collection_name, top_k=5)
    results = [(text, score) for text, score in results if score > 0.4]
    chunks = [result[0] for result in results]
    if chunks == []:
        chunks = ["Пустой контекст"]
    await ranker.fit(chunks)
    ranked_result = await ranker.rank(query)
    top_ranked = [result[0] for result in ranked_result]

    context = " ".join(top_ranked)
    answer = await assistant.generate_response(
        query, context, system_prompt, max_new_tokens
    )
    return answer
