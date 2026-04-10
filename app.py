import streamlit as st
import requests
import json
import time
from src.chunking import RecipeChunker
from src.store import EmbeddingStore
from src.models import Document
from src.embeddings import OllamaEmbedder

# --- CẤU HÌNH ---
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"
EMBED_MODEL = "nomic-embed-text" 
DATA_PATH = "data/mon_an_truyen_thong_viet_nam.md"
# DATA_PATH = "data/huong_dan_nau_an.md"

# --- SYSTEM PROMPT (Luật lệ cho AI) ---
SYSTEM_PROMPT = """Bạn là trợ lý chuyên gia về Ẩm thực Truyền thống Việt Nam. 
Nhiệm vụ của bạn là chỉ trả lời các câu hỏi dựa trên NGỮ CẢNH được cung cấp.

QUY TẮC BẮT BUỘC:
1. Nếu câu hỏi KHÔNG liên quan đến ẩm thực, nấu ăn hoặc thông tin trong tài liệu, hãy trả lời: 
   "Xin lỗi, tôi chỉ được huấn luyện để giải đáp các thông tin về ẩm thực trong tài liệu này."
2. Nếu thông tin không có trong NGỮ CẢNH, hãy nói rõ là bạn không tìm thấy thông tin trong tài liệu.
3. Không tự ý sử dụng kiến thức bên ngoài tài liệu để bịa đặt công thức.
4. Trả lời ngắn gọn, súc tích và đúng trọng tâm."""
# SYSTEM_PROMPT=""

# 1. Cấu hình giao diện Streamlit
st.set_page_config(page_title="Gemini Chef Tài", page_icon="🍳", layout="centered")

# 2. Cache hệ thống RAG
@st.cache_resource
def load_rag_system():
    with st.spinner("⏳ Đang chuẩn bị nguyên liệu (Khởi tạo dữ liệu)..."):
        try:
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                content = f.read()
            
            chunker = RecipeChunker(chunk_size=600)
            chunks = chunker.chunk(content)
            
            store = EmbeddingStore(embedding_fn=OllamaEmbedder(model_name=EMBED_MODEL))
            docs = [Document(id=f"p_{i}", content=c) for i, c in enumerate(chunks)]
            store.add_documents(docs)
            return store
        except Exception as e:
            st.error(f"❌ Lỗi nạp dữ liệu: {e}")
            return None

# Khởi tạo Store
if "store" not in st.session_state:
    st.session_state.store = load_rag_system()

# 3. Giao diện Sidebar
with st.sidebar:
    st.title("👨‍🍳 Thông tin Bếp")
    st.info(f"**Model:** {MODEL_NAME}\n\n**Data:** {DATA_PATH.split('/')[-1]}")
    if st.button("🔄 Làm mới dữ liệu"):
        st.cache_resource.clear()
        st.rerun()

# 4. Giao diện Chat chính
st.title("🍳 Trợ Lý Nấu Ăn Của Tài")
st.caption("Chỉ giải đáp các món ăn có trong tài liệu truyền thống Việt Nam")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Xử lý input
if prompt := st.chat_input("Hỏi về món ăn trong tài liệu..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # A. Retrieval: Tìm kiếm ngữ cảnh từ database local
        results = st.session_state.store.search(prompt, top_k=3)
        context = "\n---\n".join([r["content"] for r in results])
        
        # B. Cấu hình Payload với System Prompt và Temperature thấp
        payload = {
            "model": MODEL_NAME,
            
            "prompt": f"NGỮ CẢNH: {context}\n\nCÂU HỎI: {prompt}",
            "system": SYSTEM_PROMPT, # Truyền luật lệ vào đây
            "stream": True,
            "options": {
                "temperature": 0.1,  # Giảm sáng tạo, tăng độ chính xác
                "num_ctx": 4096      # Đảm bảo đủ bộ nhớ context
            }
        }

        try:
            response = requests.post(OLLAMA_GENERATE_URL, json=payload, stream=True)
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    full_response += token
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"❌ Lỗi kết nối Ollama: {str(e)}")