import json
import tempfile
import os
import pytest

# 这些函数将定义在 src/__init__.py 中
# 为了避免循环导入，测试中直接复制签名，运行时再导入


def write_temp_jsonl(path, items):
    with open(path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


class TestLoadJsonl:
    def test_load_valid_file(self):
        from src import load_jsonl
        items = [
            {"instruction": "问题1", "output": "答案1"},
            {"instruction": "问题2", "output": "答案2"},
        ]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl',
                                          delete=False, encoding='utf-8') as f:
            write_temp_jsonl(f.name, items)
            f.flush()
            result = load_jsonl(f.name)
        os.unlink(f.name)
        assert len(result) == 2
        assert result[0]["instruction"] == "问题1"

    def test_load_empty_file(self):
        from src import load_jsonl
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl',
                                          delete=False, encoding='utf-8') as f:
            f.flush()
            result = load_jsonl(f.name)
        os.unlink(f.name)
        assert result == []

    def test_load_nonexistent_file(self):
        from src import load_jsonl
        with pytest.raises(FileNotFoundError):
            load_jsonl("/nonexistent/path.jsonl")


class TestValidateSftFormat:
    def test_valid_data(self):
        from src import validate_sft_format
        data = [
            {"instruction": "你好", "output": "你好！有什么可以帮你？"},
            {"instruction": "VPN怎么连？", "output": "访问 vpn.chengguo.com 下载客户端"},
        ]
        errors = validate_sft_format(data, min_output_len=1)
        assert len(errors) == 0

    def test_missing_fields(self):
        from src import validate_sft_format
        data = [
            {"instruction": "问题"},
            {"output": "答案"},
        ]
        errors = validate_sft_format(data, min_output_len=1)
        assert len(errors) == 2

    def test_empty_content(self):
        from src import validate_sft_format
        data = [
            {"instruction": "", "output": "答案"},
            {"instruction": "问题", "output": ""},
        ]
        errors = validate_sft_format(data, min_output_len=1)
        assert len(errors) == 2

    def test_short_output(self):
        from src import validate_sft_format
        data = [
            {"instruction": "问题", "output": "短"},  # 1 字符
        ]
        errors = validate_sft_format(data)
        assert len(errors) == 1
