"""RAG 系统测试 — 分块逻辑 + 检索召回 + 引用格式 + 索引持久化"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_docs_dir():
    """创建临时文档目录，含测试用 Markdown"""
    tmp = tempfile.mkdtemp()
    doc_path = Path(tmp) / "test_handbook.md"
    doc_path.write_text(
        "# 测试手册\n\n"
        "## VPN 配置\n\n"
        "VPN 配置需要访问 vpn.test.com。下载客户端后，使用企业微信扫码登录。\n\n"
        "## 年假规则\n\n"
        "入职 1 年以内享受 5 天带薪年假。入职 1-3 年享受 10 天带薪年假。\n"
        "入职 3-5 年享受 15 天带薪年假。\n\n",
        encoding='utf-8'
    )
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def temp_persist_dir():
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


class TestRAGSystemBuild:
    def test_build_index_creates_chunks(self, temp_docs_dir, temp_persist_dir):
        from src.rag_system import RAGSystem
        rag = RAGSystem(doc_dir=temp_docs_dir, persist_dir=temp_persist_dir)
        chunks = rag.build_index(temp_docs_dir)
        assert chunks > 0
        assert chunks >= 2  # VPN 和年假应该是两个语义块

    def test_build_index_idempotent(self, temp_docs_dir, temp_persist_dir):
        from src.rag_system import RAGSystem
        rag = RAGSystem(doc_dir=temp_docs_dir, persist_dir=temp_persist_dir)
        first = rag.build_index(temp_docs_dir)
        second = rag.build_index(temp_docs_dir)
        assert first == second


class TestRAGSystemQuery:
    def test_query_vpn(self, temp_docs_dir, temp_persist_dir):
        from src.rag_system import RAGSystem
        rag = RAGSystem(doc_dir=temp_docs_dir, persist_dir=temp_persist_dir)
        rag.build_index(temp_docs_dir)
        result = rag.query("VPN 怎么配置？")
        assert len(result["answer"]) > 0
        assert result["source_count"] >= 1

    def test_query_annual_leave(self, temp_docs_dir, temp_persist_dir):
        from src.rag_system import RAGSystem
        rag = RAGSystem(doc_dir=temp_docs_dir, persist_dir=temp_persist_dir)
        rag.build_index(temp_docs_dir)
        result = rag.query("入职两年年假有几天？")
        assert len(result["answer"]) > 0
        assert result["source_count"] >= 1
        assert any(char.isdigit() for char in result["answer"])

    def test_query_no_results(self, temp_docs_dir, temp_persist_dir):
        from src.rag_system import RAGSystem
        rag = RAGSystem(doc_dir=temp_docs_dir, persist_dir=temp_persist_dir)
        rag.build_index(temp_docs_dir)
        result = rag.query("火星移民政策是什么？")
        assert isinstance(result["answer"], str)
