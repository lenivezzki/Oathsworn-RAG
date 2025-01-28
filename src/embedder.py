import asyncio
from typing import List

import torch
from transformers import AutoModel, AutoTokenizer


class LaBSEEmbedder:
    def __init__(self, model_name: str, device: str = "mps"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    async def embed(self, texts: List[str]):
        """Получает эмбеддинги для списка текстов"""
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt", max_length=512
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings.cpu().numpy()
