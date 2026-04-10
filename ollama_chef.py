import sys
import requests
import json
import time
from src.chunking import RecipeChunker
from src.store import EmbeddingStore
from src.models import Document
from src.embeddings import OllamaEmbedder

# --- CẤU HÌNH ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"
DATA_PATH = "data/mon_an_truyen_thong_viet_nam.md"

def stream_print(token: str):
    """Hiệu ứng chữ chạy tới đâu hiện tới đó."""
    sys.stdout.write(token)
    sys.stdout.flush()

def init_rag_system():
    """Khởi tạo hệ thống: Đọc file -> Chunking -> Local Embedding."""
    print("⏳ Bước 1: Đang đọc dữ liệu nấu ăn...")
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    
    print("✂️ Bước 2: Đang chia nhỏ tài liệu (Chunking)...")
    chunker = RecipeChunker(chunk_size=600)
    chunks = chunker.chunk(content)
    
    print(f"🧠 Bước 3: Đang dùng {MODEL_NAME} để tạo Vector (Local Ollama)...")
    
    # Khởi tạo với model Qwen2.5:7b
    store = EmbeddingStore(embedding_fn=OllamaEmbedder(model_name=MODEL_NAME))
    
    docs = [Document(id=f"p_{i}", content=c) for i, c in enumerate(chunks)]
    
    # Nạp vào bộ nhớ
    start_time = time.time()
    store.add_documents(docs)
    end_time = time.time()
    
    print(f"✅ Đã nạp {len(chunks)} công thức. (Tốn {end_time - start_time:.2f}s)")
    return store

def chat_with_qwen(query: str, store: EmbeddingStore):
    """Tìm kiếm ngữ cảnh local và gọi Ollama."""
    # 1. Retrieval (Tìm kiếm ngữ nghĩa trong Store)
    results = store.search(query, top_k=3)
    context = "\n---\n".join([r["content"] for r in results])
    
    # 2. Tạo Prompt chuyên cho Qwen
    full_prompt = f"""Bạn là đầu bếp Việt Nam chuyên nghiệp. Dựa vào ngữ cảnh dưới đây, hãy trả lời câu hỏi của người dùng.
Nếu không có thông tin, hãy nói lịch sự là "Tôi không biết".

NGỮ CẢNH:
{context}

CÂU HỎI: {query}
TRẢ LỜI:"""

    print("\n🤖 Chef Qwen: ", end="")
    
    # 3. Request tới Ollama Local
    payload = {
        "model": MODEL_NAME,
        "prompt": full_prompt,
        "stream": True
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, stream=True)
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                token = chunk.get("response", "")
                stream_print(token)
                if chunk.get("done"):
                    break
        print("\n")
    except Exception as e:
        print(f"\n❌ Lỗi kết nối Ollama: {str(e)}. Hãy chắc chắn Ollama đang chạy!")

def main():
    try:
        store = init_rag_system()
    except Exception as e:
        print(f"❌ Lỗi khởi tạo: {e}")
        return

    print("\n" + "="*45)
    print("🍳 CHÀO MỪNG ĐẾN VỚI CLI COOKING CHATBOT (LOCAL)")
    print("   (Gõ 'exit' để dừng nấu ăn)")
    print("="*45)

    while True:
        try:
            user_input = input("👤 You: ").strip()
            if user_input.lower() in ['exit', 'quit', 'thoát']:
                print("👋 Tạm biệt Chef!")
                break
            if not user_input:
                continue
            
            chat_with_qwen(user_input, store)
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()