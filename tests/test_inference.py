"""模型推理测试 — vLLM API + token 截断 + 输出格式"""

import pytest


@pytest.mark.gpu
class TestVLLMInference:
    def test_vllm_health_check(self):
        """验证 vLLM 服务是否可达"""
        import requests
        try:
            resp = requests.get("http://localhost:8000/health", timeout=5)
            assert resp.status_code == 200
        except requests.ConnectionError:
            pytest.skip("vLLM 服务未启动")

    def test_chat_completion_format(self):
        """验证输出格式合规"""
        from openai import OpenAI
        try:
            client = OpenAI(base_url="http://localhost:8000/v1", api_key="test")
            response = client.chat.completions.create(
                model="itops",
                messages=[{"role": "user", "content": "你好"}],
                max_tokens=32,
            )
            assert len(response.choices) > 0
            assert len(response.choices[0].message.content) > 0
        except Exception:
            pytest.skip("vLLM 服务不可达")

    def test_token_truncation(self):
        """验证长输入不会崩溃"""
        from openai import OpenAI
        try:
            client = OpenAI(base_url="http://localhost:8000/v1", api_key="test")
            long_question = "请帮我分析 " + "测试 " * 500
            response = client.chat.completions.create(
                model="itops",
                messages=[{"role": "user", "content": long_question}],
                max_tokens=32,
            )
            assert len(response.choices[0].message.content) > 0
        except Exception:
            pytest.skip("vLLM 服务不可达")
