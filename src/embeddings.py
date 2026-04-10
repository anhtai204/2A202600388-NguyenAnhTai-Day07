from __future__ import annotations

import hashlib
import math
import requests

LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_PROVIDER_ENV = "EMBEDDING_PROVIDER"


class MockEmbedder:
    """Deterministic embedding backend used by tests and default classroom runs."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self._backend_name = "mock embeddings fallback"

    def __call__(self, text: str) -> list[float]:
        digest = hashlib.md5(text.encode()).hexdigest()
        seed = int(digest, 16)
        vector = []
        for _ in range(self.dim):
            seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
            vector.append((seed / 0xFFFFFFFF) * 2 - 1)
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class LocalEmbedder:
    """Sentence Transformers-backed local embedder."""

    def __init__(self, model_name: str = LOCAL_EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._backend_name = model_name
        self.model = SentenceTransformer(model_name)

    def __call__(self, text: str) -> list[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return [float(value) for value in embedding]


class OpenAIEmbedder:
    """OpenAI embeddings API-backed embedder."""

    def __init__(self, model_name: str = OPENAI_EMBEDDING_MODEL) -> None:
        from openai import OpenAI

        self.model_name = model_name
        self._backend_name = model_name
        self.client = OpenAI()

    def __call__(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model_name, input=text)
        return [float(value) for value in response.data[0].embedding]

# src/embeddings.py

class GeminiEmbedder:
    def __init__(self, model_name: str = "models/gemini-embedding-001"):
        import google.generativeai as genai
        import os
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.genai = genai
        self.model_name = model_name

    def __call__(self, text: str | list[str]) -> list[float] | list[list[float]]:
        # Nếu truyền vào một danh sách, Gemini sẽ xử lý batch
        result = self.genai.embed_content(
            model=self.model_name,
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    
class OllamaEmbedder:
    def __init__(self, model_name: str = "qwen2.5:7b"):
        self.model_name = model_name
        self.url = "http://localhost:11434/api/embeddings"

    def __call__(self, text: str | list[str]) -> list[float] | list[list[float]]:
        is_single = isinstance(text, str)
        input_data = [text] if is_single else text
        embeddings = []

        for t in input_data:
            # Sửa 'prompt' thành 'input' vì đây là chuẩn mới của Ollama API
            payload = {"model": self.model_name, "input": t}
            try:
                response = requests.post(self.url, json=payload, timeout=15)
                if response.status_code == 200:
                    embedding = response.json().get("embedding")
                    embeddings.append(embedding)
                else:
                    # Nếu lỗi API, trả về vector 0 (độ dài 3584 cho Qwen 7B)
                    embeddings.append([0.0] * 3584)
            except Exception as e:
                print(f"❌ Lỗi Ollama: {e}")
                embeddings.append([0.0] * 3584)

        return embeddings[0] if is_single else embeddings

_mock_embed = MockEmbedder()
