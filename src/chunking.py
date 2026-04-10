from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text: return []
        # Tách câu: split on ". ", "! ", "? " or ".\n"
        sentences = re.split(r'(?<=[.!?])\s+|(?<=\n)', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i : i + self.max_sentences_per_chunk]
            chunks.append(" ".join(group))
        return chunks


class RecursiveChunker:
    """
    Recursively split text using remaining_separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(current_text) <= self.chunk_size or not remaining_separators:
            return [current_text]

        sep = remaining_separators[0]
        next_seps = remaining_separators[1:]
        
        parts = current_text.split(sep)
        final_chunks = []
        current_buffer = ""

        for part in parts:
            # Nếu thêm part vào buffer vượt quá chunk_size
            if current_buffer and len(current_buffer) + len(sep) + len(part) > self.chunk_size:
                final_chunks.append(current_buffer)
                current_buffer = ""
            
            # Nếu bản thân part đã quá lớn, đệ quy tiếp với separator tiếp theo
            if len(part) > self.chunk_size:
                final_chunks.extend(self._split(part, next_seps))
            else:
                if current_buffer:
                    current_buffer += sep + part
                else:
                    current_buffer = part

        if current_buffer:
            final_chunks.append(current_buffer)
        return [c for c in final_chunks if c]


import re
from typing import List

class RecipeChunker:
    # Các mục con thường gặp trong dữ liệu của bạn
    SECTION_HEADERS = [
        "Mô tả",
        "Nguyên liệu",
        "Cách làm",
        "Trình bày",
        "Thưởng thức",
        "Pha nước mắm",
        "Xốt cà"
    ]

    def __init__(self, chunk_size: int = 500):
        self.chunk_size = chunk_size

    # ===== PUBLIC =====

    def chunk(self, text: str) -> List[str]:
        # Tách theo từng món ăn trước (dựa vào ##)
        recipes = self._split_into_recipes(text)

        final_chunks = []
        for recipe in recipes:
            if len(recipe) <= self.chunk_size:
                final_chunks.append(recipe)
            else:
                # Nếu một món quá dài (như Phở hay Bún Bò), tách nhỏ theo các mục ###
                final_chunks.extend(self._split_large_recipe(recipe))

        return [c.strip() for c in final_chunks if c.strip()]

    # ===== BƯỚC 1: TÁCH THEO TÊN MÓN ĂN (H2) =====

    def _split_into_recipes(self, text: str) -> List[str]:
        """
        Dựa vào cấu trúc Markdown '## Tên món' để tách.
        Sử dụng lookahead (?=...) để giữ lại tiêu đề trong chunk.
        """
        pattern = r"(?=\n##\s)"
        recipes = re.split(pattern, text)
        
        # Lọc bỏ phần header rác (nếu có) trước khi vào món đầu tiên
        return [r.strip() for r in recipes if "##" in r]

    # ===== BƯỚC 2: TÁCH CÁC MÓN QUÁ DÀI THEO MỤC CON (H3) =====

    def _split_large_recipe(self, recipe: str) -> List[str]:
        """
        Tách một món ăn thành các phần dựa trên các thẻ H3 (###).
        """
        # Tách dựa trên tiêu đề cấp 3 (###)
        pattern = r"(?=\n###\s)"
        sections = re.split(pattern, recipe)

        chunks = []
        buffer = ""

        # Gom các mục nhỏ lại cho đến khi gần chạm ngưỡng chunk_size
        for part in sections:
            if not part.strip(): continue
            
            if len(buffer) + len(part) > self.chunk_size and buffer:
                chunks.append(buffer)
                buffer = part
            else:
                buffer = (buffer + "\n\n" + part).strip() if buffer else part

        if buffer:
            chunks.append(buffer)

        # Nếu sau khi tách theo mục mà vẫn có phần quá to (ví dụ: cách làm quá dài)
        # thì mới dùng đến phương án cuối cùng là tách theo dòng/đoạn.
        final_pieces = []
        for c in chunks:
            if len(c) > self.chunk_size:
                final_pieces.extend(self._recursive_fallback(c))
            else:
                final_pieces.append(c)

        return final_pieces

    # ===== BƯỚC 3: PHƯƠNG ÁN DỰ PHÒNG KHI VẪN QUÁ DÀI =====

    def _recursive_fallback(self, text: str) -> List[str]:
        # Tách theo đoạn (2 dòng trống), sau đó theo dòng, cuối cùng là dấu chấm
        separators = ["\n\n", "\n", ". "]
        
        for sep in separators:
            parts = text.split(sep)
            if len(parts) == 1:
                continue

            chunks = []
            buffer = ""

            for part in parts:
                if len(buffer) + len(part) > self.chunk_size:
                    if buffer: chunks.append(buffer)
                    buffer = part
                else:
                    buffer = (buffer + sep + part) if buffer else part

            if buffer:
                chunks.append(buffer)

            return chunks

        return [text]

def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    # TODO: implement cosine similarity formula
    mag_a = math.sqrt(sum(x**2 for x in vec_a))
    mag_b = math.sqrt(sum(x**2 for x in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return _dot(vec_a, vec_b) / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:        
        # Cập nhật key 'by_sentences' để khớp với yêu cầu của Test suite
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=2),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
            "recipe_chunk": RecipeChunker(chunk_size=chunk_size)
        }
        
        results = {}
        for name, strategy in strategies.items():
            chunks = strategy.chunk(text)
            results[name] = {
                "count": len(chunks),
                "avg_length": sum(len(c) for c in chunks) / len(chunks) if chunks else 0,
                "chunks": chunks
            }
        return results

    # def compare(self, text: str, chunk_size: int = 200) -> dict:
    #     strategies = {
    #         "fixed_size": FixedSizeChunker(chunk_size=chunk_size),
    #         "by_sentences": SentenceChunker(max_sentences_per_chunk=2),
    #         "recursive": RecursiveChunker(chunk_size=chunk_size)
    #     }

    #     results = {}
    #     for name, strategy in strategies.items():
    #         chunks = strategy.chunk(text)
    #         results[name] = {
    #             "count": len(chunks),
    #             "avg_length": sum(len(c) for c in chunks) / len(chunks) if chunks else 0
    #         }
    #     return results
    

# input_file = "../data/mon_an_truyen_thong_viet_nam.md"

# with open(input_file, "r", encoding="utf-8") as f:
#     text = f.read()
# comparator = ChunkingStrategyComparator()

# result = comparator.compare(text, 200)

# for name, data in result.items():
#     print(f"{name}:")
#     print(f"  count = {data['count']}")
#     print(f"  avg_length = {data['avg_length']:.2f}")