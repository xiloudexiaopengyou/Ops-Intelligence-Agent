"""
自研语义分块器 — 基于相邻句子语义相似度的自适应文本切分。

算法:
1. 按句号/换行拆分句子
2. BGE-small 逐句编码 (CPU)
3. 相邻句计算余弦相似度
4. 相似度 < threshold → 切分点
5. 合并块直到接近 max_chunk_size 上限
"""

import re
import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticChunker:
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        similarity_threshold: float = 0.7,
        max_chunk_size: int = 512,
        device: str = "cpu",
    ):
        self._embedding_model_name = embedding_model
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.device = device
        self._model = None

    @property
    def model(self):
        """延迟加载 embedding 模型（节约启动时间）"""
        if self._model is None:
            self._model = SentenceTransformer(
                self.embedding_model_name, device=self.device
            )
        return self._model

    @property
    def embedding_model_name(self):
        return self._embedding_model_name

    def _split_sentences(self, text: str) -> list[str]:
        """按句号、换行、分号拆分句子，过滤空白"""
        raw = re.split(r'[。\n；;！!？?]+', text)
        return [s.strip() for s in raw if s.strip()]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算两个向量的余弦相似度"""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def chunk_text(self, text: str) -> list[str]:
        """对单文本进行语义分块"""
        text = text.strip()
        if not text:
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        if len(sentences) == 1:
            return [sentences[0]]

        # 逐句编码
        embeddings = self.model.encode(sentences, show_progress_bar=False)

        # 计算相邻句相似度，确定切分点
        chunks = []
        current_chunk = sentences[0]

        for i in range(1, len(sentences)):
            sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])

            if sim < self.similarity_threshold or len(current_chunk) + len(sentences[i]) > self.max_chunk_size:
                # 切分
                chunks.append(current_chunk)
                current_chunk = sentences[i]
            else:
                # 合并
                current_chunk += sentences[i]

        # 最后一块
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def chunk_documents(self, documents: list[dict]) -> list[dict]:
        """批量处理文档，返回带元数据的块列表

        Args:
            documents: [{"text": str, "metadata": dict}, ...]

        Returns:
            [{"text": str, "metadata": dict}, ...]
        """
        results = []
        for doc in documents:
            doc_text = doc.get("text", "")
            doc_meta = doc.get("metadata", {})
            text_chunks = self.chunk_text(doc_text)
            for chunk in text_chunks:
                results.append({"text": chunk, "metadata": doc_meta})
        return results
