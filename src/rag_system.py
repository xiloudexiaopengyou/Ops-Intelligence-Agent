"""
RAG 检索增强系统 — TF-IDF 向量检索（零网络依赖，无需下载模型）

用法:
    from src.rag_system import RAGSystem
    rag = RAGSystem(doc_dir="./docs")
    result = rag.query("VPN怎么配置？")
    print(result["answer"], result["sources"])
"""

import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class RAGSystem:
    def __init__(
        self,
        doc_dir: str = "./docs",
        max_chunk_size: int = 512,
        top_k: int = 3,
    ):
        self.doc_dir = Path(doc_dir)
        self.top_k = top_k
        self.max_chunk_size = max_chunk_size

        # 读取文档 → 分块 → 构建 TF-IDF 索引
        self.chunks = []          # list of {"text": str, "source": str}
        self.vectorizer = None
        self.chunk_vectors = None
        self._build_index()

    def _read_files(self):
        """递归读取 docs/ 下所有 .md/.txt 文件"""
        texts = []
        for fpath in self.doc_dir.rglob("*"):
            if fpath.is_file() and fpath.suffix.lower() in (".md", ".txt"):
                try:
                    content = fpath.read_text(encoding="utf-8", errors="ignore")
                    if content.strip():
                        texts.append({
                            "text": content,
                            "source": str(fpath.relative_to(self.doc_dir)),
                        })
                except Exception:
                    pass
        return texts

    def _split_text(self, text: str) -> list[str]:
        """简单分块：按段落 + 长度限制"""
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current) + len(para) < self.max_chunk_size:
                current += para + "\n"
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = para + "\n"
        if current.strip():
            chunks.append(current.strip())
        return chunks or [text[:self.max_chunk_size]]

    def _build_index(self):
        """构建 TF-IDF 向量索引"""
        print(f"📚 构建 TF-IDF 索引: {self.doc_dir}")

        # 读取
        docs = self._read_files()
        print(f"  读取 {len(docs)} 个文件")

        if not docs:
            print("  ⚠️ 未找到文档，使用空索引")
            self.chunks = [{"text": "暂无文档", "source": "N/A"}]
            self.vectorizer = TfidfVectorizer(max_features=1000)
            self.chunk_vectors = self.vectorizer.fit_transform(
                [c["text"] for c in self.chunks]
            )
            return

        # 分块
        for doc in docs:
            for chunk_text in self._split_text(doc["text"]):
                self.chunks.append({
                    "text": chunk_text,
                    "source": doc["source"],
                })
        print(f"  分块: {len(self.chunks)} 个")

        # TF-IDF 向量化
        self.vectorizer = TfidfVectorizer(max_features=2000)
        self.chunk_vectors = self.vectorizer.fit_transform(
            [c["text"] for c in self.chunks]
        )
        print(f"✅ 索引构建完成: {len(self.chunks)} 个块, {self.chunk_vectors.shape[1]} 维")

    def query(self, question: str, top_k: int | None = None) -> dict:
        """检索并返回最相关的文档块

        Returns:
            {
                "answer": str,         # 拼接的最相关块
                "sources": [...],
                "source_count": int,
            }
        """
        if top_k is None:
            top_k = self.top_k

        if self.chunk_vectors is None or self.chunk_vectors.shape[0] == 0:
            return {
                "answer": "知识库中暂无文档，请将 IT 运维文档放入 ./docs/ 目录。",
                "sources": [],
                "source_count": 0,
            }

        # 向量化查询
        q_vec = self.vectorizer.transform([question])

        # 余弦相似度
        scores = cosine_similarity(q_vec, self.chunk_vectors)[0]

        # Top-K
        top_indices = np.argsort(scores)[::-1][:top_k]

        # 组装结果
        sources = []
        answer_parts = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < 0.01:
                continue
            chunk = self.chunks[idx]
            sources.append({
                "text": chunk["text"][:300],
                "score": score,
                "metadata": {"file_name": chunk["source"]},
            })
            answer_parts.append(chunk["text"][:600])

        if not answer_parts:
            return {
                "answer": "未找到与您问题相关的文档内容，请尝试更具体的关键词。",
                "sources": [],
                "source_count": 0,
            }

        return {
            "answer": "\n\n---\n\n".join(answer_parts),
            "sources": sources,
            "source_count": len(sources),
        }
