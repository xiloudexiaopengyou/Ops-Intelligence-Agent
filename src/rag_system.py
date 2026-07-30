"""
RAG 检索增强系统 — ChromaDB + 自研语义分块器 + LlamaIndex 编排

用法:
    from src.rag_system import RAGSystem
    rag = RAGSystem(doc_dir="./docs")
    result = rag.query("VPN怎么配置？")
    print(result["answer"], result["sources"])
"""

from pathlib import Path

import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.chunker import SemanticChunker


class RAGSystem:
    def __init__(
        self,
        doc_dir: str = "./docs",
        persist_dir: str = "./chroma_db",
        collection_name: str = "docs",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        similarity_threshold: float = 0.7,
        max_chunk_size: int = 512,
        top_k: int = 3,
    ):
        self.doc_dir = Path(doc_dir)
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.top_k = top_k

        # Embedding 模型 — 强制 CPU
        self.embed_model = HuggingFaceEmbedding(
            model_name=embedding_model,
            device="cpu",
        )
        Settings.embed_model = self.embed_model

        # 自研分块器
        self.chunker = SemanticChunker(
            embedding_model=embedding_model,
            similarity_threshold=similarity_threshold,
            max_chunk_size=max_chunk_size,
            device="cpu",
        )

        # 向量库
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=str(self.persist_dir))

        # 加载或构建索引
        self.index = self._load_or_build()

    def _load_or_build(self):
        """加载已有索引，不存在则构建"""
        try:
            collection = self.chroma_client.get_collection(self.collection_name)
            if collection.count() > 0:
                print(f"📚 加载已有索引: {collection.count()} 个块")
                vector_store = ChromaVectorStore(chroma_collection=collection)
                return VectorStoreIndex.from_vector_store(
                    vector_store, embed_model=self.embed_model
                )
        except Exception:
            pass

        return self.build_index(str(self.doc_dir))

    def build_index(self, doc_dir: str) -> int:
        """构建/重建索引，返回 chunk 数量"""
        doc_dir = Path(doc_dir)
        print(f"📚 构建索引: {doc_dir}")

        # 读取文档
        documents = SimpleDirectoryReader(str(doc_dir)).load_data()
        print(f"  读取 {len(documents)} 个文档")

        # 语义分块
        docs_for_chunker = [
            {"text": doc.text, "metadata": doc.metadata} for doc in documents
        ]
        chunks = self.chunker.chunk_documents(docs_for_chunker)
        print(f"  分块: {len(chunks)} 个语义块")

        # 写入 ChromaDB
        try:
            self.chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass
        collection = self.chroma_client.create_collection(self.collection_name)

        for i, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk["text"]],
                metadatas=[chunk.get("metadata", {})],
                ids=[f"chunk_{i}"],
            )

        # 创建 LlamaIndex 索引
        vector_store = ChromaVectorStore(chroma_collection=collection)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=self.embed_model
        )

        print(f"✅ 索引构建完成: {len(chunks)} 个块")
        return len(chunks)

    def query(self, question: str, top_k: int | None = None) -> dict:
        """检索并生成答案

        Returns:
            {
                "answer": str,
                "sources": [{"text": str, "score": float, "metadata": dict}],
                "source_count": int,
            }
        """
        if top_k is None:
            top_k = self.top_k

        query_engine = self.index.as_query_engine(
            similarity_top_k=top_k,
            response_mode="compact",
        )

        response = query_engine.query(question)

        sources = []
        for node in response.source_nodes:
            sources.append({
                "text": node.node.text[:200],
                "score": round(node.score or 0.0, 4),
                "metadata": node.node.metadata,
            })

        return {
            "answer": str(response),
            "sources": sources,
            "source_count": len(sources),
        }
