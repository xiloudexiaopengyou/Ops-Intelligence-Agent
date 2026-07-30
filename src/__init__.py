"""智能 IT 运维助手 — 核心包"""

import json
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    """加载 JSONL 文件"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def validate_sft_format(data: list[dict], min_output_len: int = 30) -> list[str]:
    """验证 SFT 数据格式，返回错误列表"""
    errors = []
    for i, item in enumerate(data):
        inst = item.get('instruction', '').strip()
        out = item.get('output', '').strip()
        if not inst:
            errors.append(f"第 {i} 条: instruction 为空")
        if not out:
            errors.append(f"第 {i} 条: output 为空")
        elif len(out) < min_output_len:
            errors.append(f"第 {i} 条: output 长度 {len(out)} < {min_output_len}")
    return errors
