# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Anh Tài
**Nhóm:** A2
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**

> 2 đoạn văn có high cosine similarity nghĩa là 2 embedding vector của chúng có hướng gần nhau trong không gian vector, đồng thời chúng có ý nghĩa gần giống nhau.

**Ví dụ HIGH similarity:**

- Sentence A: Chính sách hoàn trả sản phẩm
- Sentence B: Tôi muốn đổi trả sản phẩm
- Tại sao tương đồng: cả 2 câu đều nói về hoàn trả

**Ví dụ LOW similarity:**

- Sentence A: tôi muốn hoàn trả sản phẩm
- Sentence B: thời tiết hôm nay thế nào
- Tại sao khác: 2 câu thuộc 2 chủ đề khác nhau

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**

> cosine similarity đo góc giữa 2 vector nên tập trung vào hướng, còn đối với euclidean phụ thuộc vào cả độ dài

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**

> _Trình bày phép tính:_ step = text*count - overlap; num_chunks = [(text_count-chunk_size)/step]
> *Đáp án:\_ [(10000-500)/(500-50)] = 22

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**

> _Viết 1-2 câu:_ [(10000-500)/(500-100)] = 25; overlap nhiều hơn giúp giữ ngữ cảnh qua các chunk

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** [Cooking recipes]

**Tại sao nhóm chọn domain này?**

> Data thú vị, recipe có cấu trúc rõ ràng, người dùng thường hỏi bẳng ngôn ngữ tự nhiên, metedata filtering hữu ích

### Data Inventory

| #   | Tên tài liệu                  | Nguồn                              | Số ký tự | Metadata đã gán                                         |
| --- | ----------------------------- | ---------------------------------- | -------- | ------------------------------------------------------- |
| 1   | customer_support_playbook.txt | data/customer_support_playbook.txt | 1692     | source, extension, doc_type, department, language       |
| 2   | rag_system_design.md          | data/rag_system_design.md          | 2391     | source, extension, doc_type, department, language       |
| 3   | vector_store_notes.md         | data/vector_store_notes.md         | 2123     | source, extension, doc_type, department, language       |
| 4   | vi_retrieval_notes.md         | data/vi_retrieval_notes.md         | 1667     | source, extension, doc_type, department, language       |
| 5   | chunking_experiment_report.md | data/chunking_experiment_report.md | 1987     | source, extension, doc_type, department, language       |
| 6   | huong_dan_nau_an.md           | data/huong_dan_nau_an.md           | 195560   | source, doc_type, department, language, domain, cuisine |

### Metadata Schema

| Trường metadata | Kiểu   | Ví dụ giá trị                 | Tại sao hữu ích cho retrieval?                 |
| --------------- | ------ | ----------------------------- | ---------------------------------------------- |
| cuisine         | string | Vietnamese                    | Hữu ích cho retrieval món ăn                   |
| source          | str    | rag_system_design.md          | Truy vết nguồn chunk sau khi retrieve          |
| doc_type        | str    | playbook / notes / design_doc | Lọc theo loại tài liệu cho đúng ngữ cảnh       |
| department      | str    | support / platform            | Giảm nhiễu khi query theo team                 |
| language        | str    | vi / en                       | Tránh lấy sai ngôn ngữ khi câu hỏi có scope rõ |
| domain          | str    | cooking / ai / system_design  | Phân nhóm theo lĩnh vực                        |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu                 | Strategy         | Chunk Count | Avg Length | Preserves Context? |
| ------------------------ | ---------------- | ----------- | ---------- | ------------------ |
| data/huong_dan_nau_an.md | FixedSizeChunker | 62          | 199.03     | Medium             |
| data/huong_dan_nau_an.md | SentenceChunker  | 280         | 31.20      | Low                |
| data/huong_dan_nau_an.md | RecursiveChunker | 55          | 166.98     | High               |

### Strategy Của Tôi

**Loại:** RecipeChunker (Custom)

**Mô tả cách hoạt động:**

> Đầu tiên sử dụng Regex để nhận diện tiêu đề, tiếp theo tìm kiếm các thư mục con như nguyên liệu, hướng dẫn. Nếu chưa vượt quá chuck_size sẽ sử dụng cơ chế Recursive Fallback để cắt nhỏ dần dựa trên các ký tự phân tách ưu tiên.

**Tại sao tôi chọn strategy này cho domain nhóm?**

> Tài liệu nấu ăn có cấu trúc đặc thù (tên món -> nguyên liệu -> hướng dẫn). Sử dụng Custom Strategy giúp tránh cắt ngang một bước nấu ăn hoặc làm mất ngữ cảnh.

**Code snippet (nếu custom):**

```python
class RecipeChunker:
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

    def chunk(self, text: str) -> List[str]:
        recipes = self._split_into_recipes(text)

        final_chunks = []
        for recipe in recipes:
            if len(recipe) <= self.chunk_size:
                final_chunks.append(recipe)
            else:
                final_chunks.extend(self._split_large_recipe(recipe))

        return [c.strip() for c in final_chunks if c.strip()]

    # ===== BƯỚC 1: TÁCH THEO TÊN MÓN ĂN =====

    def _split_into_recipes(self, text: str) -> List[str]:
        """
        Dựa vào cấu trúc Markdown '## Tên món' để tách.
        Sử dụng lookahead (?=...) để giữ lại tiêu đề trong chunk.
        """
        pattern = r"(?=\n##\s)"
        recipes = re.split(pattern, text)

        # Lọc bỏ phần header rác (nếu có) trước khi vào món đầu tiên
        return [r.strip() for r in recipes if "##" in r]

    # ===== BƯỚC 2: TÁCH CÁC MÓN QUÁ DÀI THEO MỤC CON =====

    def _split_large_recipe(self, recipe: str) -> List[str]:
        """
        Tách một món ăn thành các phần dựa trên các thẻ H3 (###).
        """
        pattern = r"(?=\n###\s)"
        sections = re.split(pattern, recipe)

        chunks = []
        buffer = ""

        for part in sections:
            if not part.strip(): continue

            if len(buffer) + len(part) > self.chunk_size and buffer:
                chunks.append(buffer)
                buffer = part
            else:
                buffer = (buffer + "\n\n" + part).strip() if buffer else part

        if buffer:
            chunks.append(buffer)

        final_pieces = []
        for c in chunks:
            if len(c) > self.chunk_size:
                final_pieces.extend(self._recursive_fallback(c))
            else:
                final_pieces.append(c)

        return final_pieces

    # ===== BƯỚC 3: PHƯƠNG ÁN DỰ PHÒNG KHI VẪN QUÁ DÀI =====

    def _recursive_fallback(self, text: str) -> List[str]:
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

```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu                 | Strategy         | Chunk Count | Avg Length | Retrieval Quality? |
| ------------------------ | ---------------- | ----------- | ---------- | ------------------ |
| data/huong_dan_nau_an.md | RecursiveChunker | 55          | 166.98     | 7/10               |
| data/huong_dan_nau_an.md | RecipeChunker    | 53          | 173.30     | 8/10               |

### So Sánh Với Thành Viên Khác

| Thành viên           | Strategy               | Retrieval Score (/10) | Điểm mạnh                                                                             | Điểm yếu                                                                             |
| -------------------- | ---------------------- | --------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Tôi                  | RecipeChunker (Custom) | 8/10                  | Giữ trọn vẹn ngữ cảnh từng bước nấu; lọc nhiễu tốt nhờ tách biệt tiêu đề và nội dung. | Số lượng chunk lớn làm tăng thời gian nhúng (embedding) và tìm kiếm ban đầu.         |
| Hoàng Bá Minh Quang  | RecursiveChunker       | 8/10                  | Cân bằng giữa độ ngắn chunk và giữ ngữ cảnh                                           | Nhiều chunk hơn, tốn index hơn                                                       |
| Trần Quang Long      | RecursiveChunker       | 10/10                 | giữ trọn vẹn danh sách nguyên liệu hoặc trọn vẹn một bước nấu ăn trong cùng một chunk | Không có điểm yếu (!đánh giá cá nhân)                                                |
| Vũ Minh Quân         | RecursiveChunker       | 8/10                  | giữ cửa sổ ngữ cảnh, tránh bị loãng thông tin                                         | nhiều chunk gây tốn thời gian trích xuất và tìm kiếm                                 |
| Đỗ Lê Thành Nhân     | SentenceChunker        | 7/10                  | Đảm bảo tính toàn vẹn về mặt ngữ nghĩa của từng câu đơn lẻ.                           | AI khó liên kết giữa nguyên liệu và hành động nấu nếu chúng nằm ở các câu khác nhau. |
| Nguyễn Công Quốc Huy | SectionChunker(Custom) | 8/10                  | Đưa ra chính xác section cần                                                          | Số lượng chunk tương đối làm tăng thời gian nhúng (embedding) và tìm kiếm ban đầu.   |

**Strategy nào tốt nhất cho domain này? Tại sao?**

> Chọn RecursiveChunker vì nó phân tách văn bản theo thứ tự ưu tiên từ lớn đến nhỏ (như xuống dòng rồi mới đến dấu chấm), giúp giữ trọn vẹn các khối thông tin liên quan như danh sách nguyên liệu hay quy trình chế biến trong cùng một đoạn. Chiến lược này tạo ra sự cân bằng tối ưu giữa kích thước chunk và tính toàn vẹn ngữ cảnh, đảm bảo AI luôn truy xuất được các chỉ dẫn nấu ăn hoàn chỉnh thay vì những mảnh vụn rời rạc. Ngoài ra, dễ sử dụng hơn so với các Chunk custom của các thành viên khác.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:

> \_Viết 2-3 câu:

    sentences = re.split(r'(?<=[.!?])\s+|(?<=\n)', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

> Phương pháp này sử dụng Regular Expression với kỹ thuật positive lookbehind để tách văn bản tại các dấu kết thúc câu (., !, ?) hoặc các ký tự xuống dòng mà không làm mất đi các dấu câu đó.

**`RecursiveChunker.chunk` / `_split`** — approach:

> \_Viết 2-3 câu:

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

> Thuật toán thực hiện chia nhỏ văn bản theo cơ chế phân cấp dựa trên danh sách các ký tự phân tách ưu tiên từ lớn đến nhỏ (như đoạn văn, dòng, câu). Hệ thống duy trì một buffer để gộp các đoạn văn bản lại cho đến khi gần chạm ngưỡng chunk_size

### EmbeddingStore

**`add_documents` + `search`** — approach:

> tài liệu được chuyển thành dạng vector qua hàm embedding và được lưu trữ trong danh sách hoặc CSDL như chromaDB. Độ tương đồng được tính bằng tích vô hướng giữa vector truy vấn và toàn bộ vector tài liệu, từ đó xếp hạng và trả về các kết quả có ý nghĩa gần nhất

**`search_with_filter` + `delete_document`** — approach:

> hệ thống thực hiện lọc và dựa trên metadata trước khi tính toán độ tương đồng. Việc xóa dữ liệu được thực hiện bằng cách truy vấn mã định danh, đảm bảo dữ liệu luôn được cập nhật và đồng bộ

### KnowledgeBaseAgent

**`answer`** — approach:

> cấu trúc prompt thiết kế gồm 3 phần System Instrction, Context và User input. Ngữ cảnh được bơm trực tiếp vào prompt ép buộc LLM chỉ được suy luận và trả lời dựa trên thông tin đã cung cấp tránh ảo giác

### Test Results

```
# Paste output of: pytest tests/ -v
```

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [ 2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [ 4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [ 7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [ 9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

======================= 42 passed in 0.14s ===========================

**Số tests pass:** ** 42/42 **

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A                      | Sentence B                      | Dự đoán | Actual Score | Đúng? |
| ---- | ------------------------------- | ------------------------------- | ------- | ------------ | ----- |
| 1    | Cách làm nước mắm tỏi ớt ngon.  | Hướng dẫn pha nước chấm tỏi ớt. | high    | 0.88         | Yes   |
| 2    | Cho thêm một thìa muối vào nồi. | Đừng cho thêm muối vào nồi.     | high    | 0.72         | Yes   |
| 3    | Món phở bò truyền thống.        | Hướng dẫn cài đặt hệ điều hành. | high    | 0.05         | Yes   |
| 4    | Bún bò Huế đặc sản miền Trung.  | Món bò kho nước dừa tươi.       | high    | 0.65         | Yes   |
| 5    | Chuẩn bị xương ống và gừng.     | Ninh xương lấy nước dùng ngọt   | high    | 0.78         | Yes   |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**

> Kết quả ở cặp số 2 dù 2 câu có logic trái ngược nhau nhưng dự đoán vẫn ở mức high, nghĩa là embedding đang biểu diễn trên ngữ cảnh chứ không phải logic

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| #   | Query                                                                            | Gold Answer                                                                                                 |
| --- | -------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| 1   | Các nguyên liệu cần thiết để làm món "Cá ngừ hấp cải rổ" là gì?                  | 1 hộp cá ngừ ngâm dầu, 300g cải rổ, muối, tiêu, đường, nước tương, dầu ăn, tỏi, hành lá, rau mùi và bánh mì |
| 2   | Quy trình thực hiện món "Chả trứng hấp" gồm những bước nào?                      | 1. Trộn tất cả nguyên liệu (trứng, thịt xay, nấm mèo, miến). 2. Hấp chín. 3. Phết lòng đỏ lên mặt.          |
| 3   | Những món ăn nào trong tài liệu sử dụng "nước dừa tươi" làm nguyên liệu?         | Bún tôm – thịt luộc (luộc thịt và pha mắm), Thịt kho tàu (nước dừa tươi), Bò kho (nước dừa tươi)            |
| 4   | Món "Gỏi cuốn" được mô tả như thế nào và thưởng thức kèm với loại nước chấm nào? | Mô tả là món cuốn tươi mát, dễ ăn. Thưởng thức bằng cách chấm tương đen hoặc nước mắm tỏi ớt                |
| 5   | Cách sơ chế và ướp cá trong món "Cá lóc kho tộ" được hướng dẫn ra sao?           | Cá lóc cắt khoanh, ướp với nước mắm, đường, tiêu, hành tím và nước màu trong 20 phút.                       |

### Kết Quả Của Tôi

| #   | Query                                                                            | Top-1 Retrieved Chunk (tóm tắt)                                                            | Score  | Relevant? | Agent Answer (tóm tắt)                                                   |
| --- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | ------ | --------- | ------------------------------------------------------------------------ |
| 1   | Các nguyên liệu cần thiết để làm món "Cá ngừ hấp cải rổ" là gì?                  | ## Cá ngừ hấp cải rổ ... ### Nguyên liệu: 1 hộp cá ngừ ngâm dầu, 300g cải rổ, bánh mì...   | 0.8952 | Yes       | Liệt kê đầy đủ: cá ngừ hộp, cải rổ, gia vị (muối, tiêu...) và bánh mì.   |
| 2   | Quy trình thực hiện món "Chả trứng hấp" gồm những bước nào?                      | ## Chả trứng hấp ... ### Cách làm: 1. Trộn trứng, thịt, nấm, miến. 2. Hấp. 3. Phết lòng đỏ | 0.8241 | Yes       | Gồm 3 bước: Trộn nguyên liệu, hấp chín và phết lòng đỏ trứng lên bề mặt. |
| 3   | Những món ăn nào trong tài liệu sử dụng "nước dừa tươi" làm nguyên liệu?         | ## Bún tôm – thịt luộc ... 2 bát nước dừa tươi. (Và các món Thịt kho tàu, Bò kho).         | 0.7615 | Yes       | Xác định được các món: Bún tôm – thịt luộc, Thịt kho tàu và Bò kho.      |
| 4   | Món "Gỏi cuốn" được mô tả như thế nào và thưởng thức kèm với loại nước chấm nào? | ## Gỏi cuốn ... ### Mô tả: món cuốn tươi mát... ### Thưởng thức: chấm tương đen...         | 0.8837 | Yes       | Mô tả là món tươi mát; dùng kèm tương đen hoặc nước mắm tỏi ớt           |
| 5   | Cách sơ chế và ướp cá trong món "Cá lóc kho tộ" được hướng dẫn ra sao?           | ## Cá lóc kho tộ ... ### Cách làm: 1. Cá cắt khoanh, ướp mắm, đường, hành, nước màu 20p.   | 0.8420 | Yes       | Hướng dẫn cắt khoanh và ướp đầy đủ gia vị trong thời gian 20 phút        |

**Bao nhiêu queries trả về chunk relevant trong top-3?** \_\_ 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**

> Sử dụng Regex, viết các file markdown giúp model dễ dàng vector hóa và giữ được ngữ nghĩa tốt hơn.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**

> sử dụng vector database để lưu trữ, tránh các trường hợp out of scope hay hallicuation bằng cách thêm system prompt

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**

> Tôi sẽ triển khai chunk overlap đảm bảo ngữ nghĩa không bị đứt gãy giữa các bước.

---

## Tự Đánh Giá

| Tiêu chí                    | Loại    | Điểm tự đánh giá |
| --------------------------- | ------- | ---------------- |
| Warm-up                     | Cá nhân | 5/ 5             |
| Document selection          | Nhóm    | 10/ 10           |
| Chunking strategy           | Nhóm    | 15/ 15           |
| My approach                 | Cá nhân | 9/ 10            |
| Similarity predictions      | Cá nhân | 5/ 5             |
| Results                     | Cá nhân | 10/ 10           |
| Core implementation (tests) | Cá nhân | 30/ 30           |
| Demo                        | Nhóm    | 5/ 5             |
| **Tổng**                    |         | ** 89/ 90**      |
