from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document
from .chunking import _dot, compute_similarity
import time



class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(self, collection_name: str = "documents", embedding_fn: Callable[[str], list[float]] | None = None) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._store: list[dict[str, Any]] = []
        self._use_chroma = False

        try:
            import chromadb
            self._client = chromadb.Client()
            self._collection = self._client.get_or_create_collection(name=collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False

    def _make_record(self, doc: Document) -> dict[str, Any]:
        # TODO: build a normalized stored record for one document
        embedding = self._embedding_fn(doc.content)
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": doc.metadata,
            "embedding": embedding
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        # TODO: run in-memory similarity search over provided records
        query_vec = self._embedding_fn(query)
        from .chunking import compute_similarity
        
        scored = []
        for r in records:
            score = compute_similarity(query_vec, r["embedding"])
            scored.append({**r, "score": score})
        
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    # def add_documents(self, docs: list[Document]) -> None:
    #     """
    #     Embed each document's content and store it.

    #     For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
    #     For in-memory: append dicts to self._store
    #     """
    #     # TODO: embed each doc and add to store
    #     if self._use_chroma:
    #         self._collection.add(
    #             ids=[d.id for d in docs],
    #             documents=[d.content for d in docs],
    #             embeddings=[self._embedding_fn(d.content) for d in docs],
    #             metadatas=[d.metadata for d in docs]
    #         )
    #     else:
    #         for doc in docs:
    #             self._store.append(self._make_record(doc))


    # def add_documents(self, docs: list[Document]) -> None:
    #     """
    #     Sửa đổi để hỗ trợ Batch Embedding, tránh lỗi Quota 429.
    #     """
    #     batch_size = 16  # Số lượng docs gửi đi trong một lần (Gemini hỗ trợ tới 100)
        
    #     # 1. Chuẩn bị dữ liệu
    #     all_ids = [d.id for d in docs]
    #     all_contents = [d.content for d in docs]
    #     all_metadatas = [d.metadata for d in docs]
    #     all_embeddings = []

    #     print(f"📦 Đang bắt đầu nhúng {len(docs)} đoạn văn theo từng nhóm {batch_size}...")

    #     # 2. Xử lý theo từng nhóm (Batching)
    #     for i in range(0, len(all_contents), batch_size):
    #         batch_texts = all_contents[i : i + batch_size]
            
    #         # Gửi cả nhóm cho Gemini xử lý 1 lần duy nhất
    #         try:
    #             batch_embs = self._embedding_fn(batch_texts)
    #             all_embeddings.extend(batch_embs)
                
    #             # 💡 Nghỉ 1 chút (khoảng 1s) giữa các batch để tránh bị Rate Limit
    #             if i + batch_size < len(all_contents):
    #                 time.sleep(1.0) 
    #         except Exception as e:
    #             print(f"❌ Lỗi tại nhóm thứ {i//batch_size + 1}: {e}")
    #             raise e

    #     # 3. Lưu trữ vào DB (Chroma hoặc In-memory)
    #     if self._use_chroma:
    #         self._collection.add(
    #             ids=all_ids,
    #             documents=all_contents,
    #             embeddings=all_embeddings,
    #             metadatas=all_metadatas
    #         )
    #     else:
    #         for i in range(len(docs)):
    #             record = {
    #                 "id": all_ids[i],
    #                 "content": all_contents[i],
    #                 "metadata": all_metadatas[i],
    #                 "embedding": all_embeddings[i]
    #             }
    #             self._store.append(record)
        
    #     print(f"✅ Đã lưu trữ thành công toàn bộ {len(docs)} chunks.")

    # src/store.py

    def add_documents(self, docs: list[Document]) -> None:
        all_ids = [d.id for d in docs]
        all_contents = [d.content for d in docs]
        all_metadatas = [d.metadata for d in docs]
        all_embeddings = []

        # Chia nhỏ để gửi theo Batch (giúp Ollama không bị treo khi chạy web)
        batch_size = 4  # Để số nhỏ cho an toàn trên Streamlit
        for i in range(0, len(all_contents), batch_size):
            batch_texts = all_contents[i : i + batch_size]
            batch_embs = self._embedding_fn(batch_texts)
            
            # Đảm bảo kết quả trả về là list
            if isinstance(batch_embs, list):
                all_embeddings.extend(batch_embs)

        # --- ĐOẠN SỬA LỖI INDEX ERROR ---
        # Chỉ lặp qua số lượng tối thiểu có được để tránh "Out of range"
        safe_range = min(len(docs), len(all_embeddings))
        
        for i in range(safe_range):
            record = {
                "id": all_ids[i],
                "content": all_contents[i],
                "metadata": all_metadatas[i],
                "embedding": all_embeddings[i]
            }
            self._store.append(record)
        
        print(f"✅ Đã nạp thành cô`ng {safe_range}/{len(docs)} đoạn văn.")

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        # TODO: embed query, compute similarities, return top_k
        query_vec = self._embedding_fn(query)
        
        if self._use_chroma:
            results = self._collection.query(query_embeddings=[self._embedding_fn(query)], n_results=top_k)
            return [{"content": d, "metadata": m, "id": i} 
                    for d, m, i in zip(results['documents'][0], results['metadatas'][0], results['ids'][0])]
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        # TODO
        return self._collection.count() if self._use_chroma else len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        # TODO: filter by metadata, then search among filtered chunks
        if not metadata_filter:
            return self.search(query, top_k)
        
        if self._use_chroma:
            results = self._collection.query(query_embeddings=[self._embedding_fn(query)], where=metadata_filter, n_results=top_k)
            return [{"content": d, "metadata": m} for d, m in zip(results['documents'][0], results['metadatas'][0])]
        
        filtered = [r for r in self._store if all(r["metadata"].get(k) == v for k, v in metadata_filter.items())]
        return self._search_records(query, filtered, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        # TODO: remove all stored chunks where metadata['doc_id'] == doc_id
        if self._use_chroma:
            # Kiểm tra xem có tồn tại trước khi xóa (Chroma không trả về boolean trực tiếp)
            existing = self._collection.get(ids=[doc_id])
            if not existing['ids']:
                return False
            self._collection.delete(ids=[doc_id])
            return True
        
        initial_count = len(self._store)
        # SỬA LỖI TẠI ĐÂY: Lọc theo r["id"] thay vì metadata
        self._store = [r for r in self._store if r["id"] != doc_id]
        
        return len(self._store) < initial_count
