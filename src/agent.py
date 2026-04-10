from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        # TODO: store references to store and llm_fn
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # TODO: retrieve chunks, build prompt, call llm_fn
        # 1. Retrieve
        results = self.store.search(question, top_k=top_k)
        context = "\n---\n".join([r["content"] for r in results])
        
        # 2. Build Prompt
        prompt = f"""Bạn là một trợ lý thông minh. Sử dụng các thông tin ngữ cảnh dưới đây để trả lời câu hỏi. 
                Nếu thông tin không có trong ngữ cảnh, hãy nói bạn không biết, đừng tự bịa ra câu trả lời.

                NGỮ CẢNH:
                {context}

                CÂU HỎI: {question}

        TRẢ LỜI:"""
        
        # 3. Call LLM
        return self.llm_fn(prompt)

