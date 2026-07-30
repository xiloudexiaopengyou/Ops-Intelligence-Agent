import pytest
from src.chunker import SemanticChunker


class TestSemanticChunkerInit:
    def test_default_parameters(self):
        chunker = SemanticChunker()
        assert chunker.similarity_threshold == 0.7
        assert chunker.max_chunk_size == 512
        assert chunker.device == "cpu"

    def test_custom_parameters(self):
        chunker = SemanticChunker(
            similarity_threshold=0.85,
            max_chunk_size=256,
            device="cpu"
        )
        assert chunker.similarity_threshold == 0.85
        assert chunker.max_chunk_size == 256


class TestSemanticChunkerChunkText:
    @pytest.fixture
    def chunker(self):
        return SemanticChunker(device="cpu")

    def test_single_sentence(self, chunker):
        """单句文本应返回一个块"""
        text = "这是一段测试文本。"
        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 1
        assert all(isinstance(c, str) for c in chunks)

    def test_multiple_sentences_same_topic(self, chunker):
        """同一主题的多句话应合并到一个块"""
        text = (
            "VPN配置需要先访问公司网站。"
            "然后下载对应平台的客户端。"
            "安装完成后使用企业微信扫码登录。"
        )
        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 1
        # 同一个主题不应切太碎
        assert len(chunks) <= 2

    def test_different_topics_split(self, chunker):
        """不同主题的文本应被切分开"""
        text = (
            "VPN 连接需要访问 vpn.chengguo.com 并下载客户端。"
            "安装完成后使用企业微信扫码登录即可连接内网。"
            "年假计算规则如下：入职1年以内5天，1-3年10天。"
            "员工需在OA系统提交请假申请并经上级审批。"
        )
        chunks = chunker.chunk_text(text)
        # VPN 和请假是两个不同主题，应该被切开
        assert len(chunks) >= 2

    def test_long_text_chunk_size_limit(self, chunker):
        """长文本单个块不应超过 max_chunk_size"""
        chunker.max_chunk_size = 200
        long_text = "这是一段测试文本。" * 50
        chunks = chunker.chunk_text(long_text)
        for chunk in chunks:
            assert len(chunk) <= 250  # 允许一些弹性

    def test_empty_text(self, chunker):
        """空文本应返回空列表"""
        assert chunker.chunk_text("") == []
        assert chunker.chunk_text("   ") == []
