# cli_chatbot.py

import sys
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Import các công cụ bạn đã viết
from src.chunking import RecipeChunker
from src.store import EmbeddingStore
from src.models import Document
from src.embeddings import GeminiEmbedder 

# --- CẤU HÌNH ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
DATA_PATH = "data/mon_an_truyen_thong_viet_nam.md"

# Khởi tạo "Bộ não" Gemini để chat (Flash là bản nhanh và miễn phí tốt nhất)
genai.configure(api_key=API_KEY)
llm_model = genai.GenerativeModel('gemini-1.5-flash')

def stream_print(token: str):
    """In từng chữ ra terminal tạo hiệu ứng streaming."""
    sys.stdout.write(token)
    sys.stdout.flush()

def init_rag_system():
    """Quy trình: Đọc file -> Chia nhỏ -> Tạo Vector -> Lưu trữ."""
    print("⏳ Đang thiết lập hệ thống bếp ảo (Cloud RAG)...")
    
    if not API_KEY:
        print("❌ Lỗi: Bạn quên chưa dán API KEY vào file .env rồi!")
        sys.exit(1)

    # Đọc file hướng dẫn nấu ăn của Tài
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    
    chunker = RecipeChunker(chunk_size=200)
    chunks = chunker.chunk(content)
    
    store = EmbeddingStore(embedding_fn=GeminiEmbedder())
    
    docs = [Document(id=f"recipe_part_{i}", content=c) for i, c in enumerate(chunks)]
    store.add_documents(docs)
    
    print(f"✅ Hệ thống sẵn sàng! Đã nạp {len(chunks)} đoạn dữ liệu nấu ăn.")
    return store

def chat(query: str, store: EmbeddingStore):
    """Tìm dữ liệu liên quan và nhờ Gemini trả lời."""
    # 1. Tìm 3 đoạn văn liên quan nhất đến câu hỏi
    results = store.search(query, top_k=40)
    context = "\n---\n".join([r["content"] for r in results])
    
    # 2. Xây dựng Prompt "huấn luyện" Gemini làm đầu bếp
    prompt = f"""Bạn là một đầu bếp Việt Nam tài ba. Hãy dựa vào ngữ cảnh sau để trả lời.
Nếu thông tin không có trong tài liệu, hãy trả lời lịch sự rằng bạn không biết.

NGỮ CẢNH:
{context}

CÂU HỎI: {query}
TRẢ LỜI:"""

    print("\n👨‍🍳 Gemini Chef: ", end="")
    try:
        # 3. Chat streaming
        response = llm_model.generate_content(prompt, stream=True)
        for chunk in response:
            if chunk.text:
                stream_print(chunk.text)
        print("\n")
    except Exception as e:
        print(f"\n❌ Lỗi khi chat: {str(e)}")

def main():
    # Khởi tạo store một lần duy nhất khi mở app
    store = init_rag_system()
    
    print("\n" + "="*45)
    print("      🍳 TRỢ LÝ BẾP CLOUD - GEMINI 1.5 🍳      ")
    print("          (Gõ 'exit' để dừng nấu ăn)          ")
    print("="*45)

    while True:
        user_input = input("👤 Bạn muốn nấu gì: ").strip()
        if user_input.lower() in ['exit', 'quit', 'thoát']:
            print("👋 Chào Chef Tài, hẹn gặp lại trong gian bếp!")
            break
        if user_input:
            chat(user_input, store)

if __name__ == "__main__":
    main()