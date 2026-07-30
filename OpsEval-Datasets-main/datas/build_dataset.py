"""
OpsEval → SFT 训练数据集构建脚本
Step 1: 开放问答直接提取 (去重 + 长度过滤)
Step 2-3: 选择题 → SFT 格式转换 (调用 LLM API)
Step 4: 数据集划分

使用方法:
  python build_dataset.py                    # 使用 DeepSeek API (需设置 DEEPSEEK_API_KEY)
  python build_dataset.py --no-llm           # 纯规则转换，不调用 LLM
  python build_dataset.py --model deepseek   # 默认
  python build_dataset.py --model openai     # 使用 OpenAI API
"""
import json
import os
import re
import sys
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "datas"

# ============================================================
# STEP 1: 开放问答直接提取
# ============================================================
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def dedup_by_edit_distance(items, threshold=5):
    """基于编辑距离去重"""
    def levenshtein(s1, s2):
        if len(s1) < len(s2):
            return levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                curr.append(min(
                    prev[j+1] + 1,
                    curr[j] + 1,
                    prev[j] + (c1 != c2)
                ))
            prev = curr
        return prev[-1]

    kept = []
    for item in items:
        q = item['question'].strip()
        is_dup = False
        for k in kept:
            dist = levenshtein(q, k['question'].strip())
            if dist < threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(item)
    return kept

def filter_by_length(items, min_len=20, max_len=1000, keep_regex=True):
    """长度过滤，保留正则表达式答案"""
    filtered = []
    for item in items:
        answer = item.get('answer', '').strip()
        a_len = len(answer)
        if a_len < min_len:
            # 检查是否是正则表达式（运维常用，应保留）
            if keep_regex and is_regex_pattern(answer):
                filtered.append(item)
            continue
        if a_len > max_len:
            continue
        filtered.append(item)
    return filtered

def is_regex_pattern(text):
    """检查文本是否包含正则表达式特征"""
    regex_indicators = [
        r'\\d', r'\\w', r'\\s', r'\\S', r'\\D', r'\\W',
        r'\(?<', r'\[', r'\]', r'\^', r'\$',
        r'\\+', r'\*', r'\.\*', r'\.\+',
    ]
    count = sum(1 for pattern in regex_indicators if re.search(pattern, text))
    return count >= 2

def step1_extract_open_qa():
    """提取开放问答并清洗"""
    print("=" * 60)
    print("STEP 1: 开放问答直接提取")
    print("=" * 60)

    all_items = []

    # zh/test/Log Analysis.json
    la = load_json(DATA_DIR / "zh" / "test" / "Log Analysis.json")
    print(f"  Log Analysis (zh/test): {len(la)} 条")
    all_items.extend(la)

    # dev 开放题
    for fname in ['Log Analysis.json', 'Oracle Database.json']:
        fp = DATA_DIR / "zh" / "dev" / fname
        if fp.exists():
            items = load_json(fp)
            # 只取没有 choices 的（开放题）
            open_items = [item for item in items if 'choices' not in item or not item.get('choices')]
            print(f"  {fname} (zh/dev): {len(items)} 条, 开放题 {len(open_items)} 条")
            all_items.extend(open_items)

    total_before = len(all_items)
    print(f"\n  去重前: {total_before} 条")

    # 去重
    all_items = dedup_by_edit_distance(all_items, threshold=5)
    print(f"  去重后: {len(all_items)} 条 (移除 {total_before - len(all_items)} 条)")

    # 长度过滤
    all_items = filter_by_length(all_items, min_len=20, max_len=1000)
    print(f"  长度过滤后: {len(all_items)} 条")

    return all_items


# ============================================================
# STEP 2-3: 选择题数据提取
# ============================================================

def extract_mcq_all():
    """提取所有选择题数据，区分有/无 solution"""
    print("\n" + "=" * 60)
    print("STEP 2-3: 选择题数据提取")
    print("=" * 60)

    zh_with_solution = []
    zh_without_solution = []
    en_with_solution = []
    en_without_solution = []

    # 中文 test + dev
    for split in ['test', 'dev']:
        folder = DATA_DIR / "zh" / split
        for fpath in sorted(folder.glob("*.json")):
            items = load_json(fpath)
            mcq_items = [item for item in items if 'choices' in item and item.get('choices')]
            for item in mcq_items:
                has_sol = bool(item.get('solution', '').strip())
                if has_sol:
                    zh_with_solution.append({**item, 'source': f"zh/{split}/{fpath.name}"})
                else:
                    zh_without_solution.append({**item, 'source': f"zh/{split}/{fpath.name}"})

    print(f"  中文有solution: {len(zh_with_solution)} 条")
    print(f"  中文无solution: {len(zh_without_solution)} 条")

    # 英文 test + dev
    for split in ['test', 'dev']:
        folder = DATA_DIR / "en" / split
        for fpath in sorted(folder.glob("*.json")):
            items = load_json(fpath)
            mcq_items = [item for item in items if 'choices' in item and item.get('choices')]
            for item in mcq_items:
                has_sol = bool(item.get('solution', '').strip())
                if has_sol:
                    en_with_solution.append({**item, 'source': f"en/{split}/{fpath.name}"})
                else:
                    en_without_solution.append({**item, 'source': f"en/{split}/{fpath.name}"})

    print(f"  英文有solution: {len(en_with_solution)} 条")
    print(f"  英文无solution: {len(en_without_solution)} 条")
    total = len(zh_with_solution) + len(zh_without_solution) + len(en_with_solution) + len(en_without_solution)
    print(f"  选择题总计: {total} 条")

    return zh_with_solution, zh_without_solution, en_with_solution, en_without_solution


# ============================================================
# LLM 转换函数
# ============================================================

def build_rewrite_prompt(questions_data, lang="zh"):
    """构建选择题改写 prompt"""
    items_text = ""
    for i, item in enumerate(questions_data):
        q = item['question'].strip()
        choices = item.get('choices', [])
        choices_text = "\n".join([f"{chr(65+j)}. {c}" for j, c in enumerate(choices)])
        sol = item.get('solution', '').strip()
        items_text += f"--- 题目 {i+1} ---\n问题: {q}\n选项:\n{choices_text}\n解析: {sol}\n\n"

    if lang == "zh":
        prompt = f"""请将以下选择题改写成 IT 工程师向运维助手提问的 SFT 训练数据格式。每条输出一行 JSON，格式为 {{"instruction": "...", "output": "..."}}。

要求：
1. instruction: 把题目和选项组合成自然的用户提问口吻（不是考试口吻），例如 "请帮我分析一下..."、"请问关于...应该选择哪个选项？"
2. output: 基于解析字段给出详细解答，逐项分析每个选项为什么对或错
3. 只输出 JSONL 格式，每行一个 JSON 对象，不要有其他文字

{items_text}"""
    else:
        # English translation + conversion
        prompt = f"""Please translate the following multiple-choice questions into Chinese, then convert them into SFT training data format (IT engineer asking an ops assistant). Output one JSON object per line in format: {{"instruction": "...", "output": "..."}}.

Requirements:
1. instruction: Combine question and choices into a natural Chinese user query (not exam style)
2. output: Based on the solution field, provide detailed Chinese explanation analyzing each option
3. Only output JSONL format, one JSON per line, no other text

{items_text}"""

    return prompt

def build_generate_solution_prompt(questions_data, lang="zh"):
    """为无 solution 的题生成解析的 prompt"""
    items_text = ""
    for i, item in enumerate(questions_data):
        q = item['question'].strip()
        choices = item.get('choices', [])
        choices_text = "\n".join([f"{chr(65+j)}. {c}" for j, c in enumerate(choices)])
        answer = item.get('answer', '').strip()
        items_text += f"--- 题目 {i+1} ---\n问题: {q}\n选项:\n{choices_text}\n正确答案: {answer}\n\n"

    if lang == "zh":
        prompt = f"""请将以下选择题改写成 SFT 训练数据，同时为每道题生成详细解析。输出 JSONL 格式。

要求：
1. instruction: 将题目+选项改写成 IT 工程师向运维助手的自然提问
2. output: 先给出正确答案，然后逐项分析每个选项为什么对或错，解释要专业详细（至少150字）
3. 只输出 JSONL 格式，每行 {{"instruction": "...", "output": "..."}}

{items_text}"""
    else:
        prompt = f"""Translate the following multiple-choice questions into Chinese, generate detailed solutions for each, and convert to SFT training format. Output JSONL format.

Requirements:
1. instruction: Chinese natural query from an IT engineer to ops assistant (question + choices combined)
2. output: First state the correct answer, then analyze each option in detail (at least 150 Chinese characters)
3. Only output JSONL, one {{"instruction": "...", "output": "..."}} per line

{items_text}"""

    return prompt


# ============================================================
# API 调用
# ============================================================

def call_deepseek_api(prompt, api_key, model="deepseek-chat"):
    """调用 DeepSeek API"""
    import urllib.request
    import urllib.error

    url = "https://api.deepseek.com/v1/chat/completions"
    data = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 4096,
    }).encode('utf-8')

    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    })

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            return result['choices'][0]['message']['content']
    except urllib.error.HTTPError as e:
        print(f"  API Error: {e.code} - {e.read().decode()}")
        return None


def parse_jsonl_response(response_text):
    """解析 LLM 返回的 JSONL"""
    results = []
    if not response_text:
        return results
    for line in response_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        # 尝试提取 JSON 对象
        try:
            obj = json.loads(line)
            if 'instruction' in obj and 'output' in obj:
                results.append(obj)
        except json.JSONDecodeError:
            # 尝试从行中提取 JSON
            match = re.search(r'\{[^}]+\}', line)
            if match:
                try:
                    obj = json.loads(match.group())
                    if 'instruction' in obj and 'output' in obj:
                        results.append(obj)
                except:
                    pass
    return results


# ============================================================
# 规则转换（无 LLM 回退方案）
# ============================================================

def build_choice_map(choices):
    """构建选项字母到内容的映射"""
    return {chr(65+j): c for j, c in enumerate(choices)}

def build_choices_text(choices):
    """构造选项文本"""
    return "  ".join([f"{chr(65+j)}. {c}" for j, c in enumerate(choices)])

def generate_basic_explanation(item, lang="zh"):
    """为无 solution 的题生成基本解释（纯规则，非 LLM）"""
    answer = item.get('answer', '').strip()
    choices = item.get('choices', [])
    question = item.get('question', '').strip()
    choice_map = build_choice_map(choices)

    # 解析正确答案
    correct_labels = [a.strip() for a in answer.split(',') if a.strip() and a.strip() in choice_map]
    correct_texts = [f"{label}. {choice_map[label]}" for label in correct_labels if label in choice_map]

    # 错误选项
    wrong_labels = [k for k in choice_map if k not in correct_labels]

    if lang == "zh":
        output = f"正确答案是 {answer}。"
        if correct_texts:
            output += f"正确选项为{'；'.join(correct_texts)}。"

        # 简要排除错误选项
        if wrong_labels and len(wrong_labels) <= 5:
            wrong_texts = [f"{label}. {choice_map[label]}" for label in wrong_labels]
            output += "其他选项不符合题意。"

        # 补充题目背景
        q_short = question[:80]
        if len(question) > 10:
            output += f" 该题涉及{q_short}相关知识，建议结合实际场景理解各选项的含义与区别。"
    else:
        output = f"The correct answer is {answer}. "
        if correct_texts:
            output += f"Correct option(s): {'; '.join(correct_texts)}. "
        if wrong_labels and len(wrong_labels) <= 5:
            output += "Other options are incorrect. "

    return output

def rule_based_convert_zh(item):
    """规则转换中文选择题（不用 LLM）"""
    q = item['question'].strip()
    choices = item.get('choices', [])
    sol = item.get('solution', '').strip()
    answer = item.get('answer', '').strip()

    choices_text = build_choices_text(choices)
    instruction = f"请帮我分析以下问题：{q} 选项：{choices_text} 请逐一分析每个选项并给出正确答案。"

    if sol:
        output = f"正确答案是 {answer}。{sol}"
    else:
        output = generate_basic_explanation(item, lang="zh")

    return {"instruction": instruction, "output": output}

def rule_based_convert_en(item):
    """规则转换英文选择题 — 格式转为中文 SFT，内容保持英文原意"""
    q = item['question'].strip()
    choices = item.get('choices', [])
    sol = item.get('solution', '').strip()
    answer = item.get('answer', '').strip()

    choices_text = build_choices_text(choices)

    # instruction: 中文引导 + 英文题目
    if len(q) > 200:
        instruction = f"请帮我解答以下英文技术问题：\n{q}\n选项：{choices_text}\n请逐一分析每个选项并给出正确答案。"
    else:
        instruction = f"请帮我解答以下英文技术问题：{q} 选项：{choices_text} 请逐一分析每个选项并给出正确答案。"

    if sol:
        output = f"正确答案是 {answer}。{sol}"
    else:
        output = generate_basic_explanation(item, lang="en")

    return {"instruction": instruction, "output": output}


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="OpsEval SFT 数据集构建工具")
    parser.add_argument("--no-llm", action="store_true", help="不使用 LLM API，纯规则转换")
    parser.add_argument("--model", default="deepseek", choices=["deepseek", "openai"], help="LLM 模型")
    parser.add_argument("--api-key", default=None, help="API Key (或设置环境变量 DEEPSEEK_API_KEY)")
    parser.add_argument("--batch-size", type=int, default=30, help="每批处理的题目数")
    parser.add_argument("--limit", type=int, default=0, help="限制处理条数（0=全部）")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    use_llm = not args.no_llm and bool(api_key)

    if use_llm:
        print(f"✅ 使用 LLM API ({args.model}), batch_size={args.batch_size}")
    else:
        print("⚠️  纯规则转换模式（不调用 LLM），建议提供 API Key 以获得更好质量")

    OUTPUT_DIR.mkdir(exist_ok=True)

    # ========== Step 1 ==========
    open_qa = step1_extract_open_qa()
    print(f"\n  ✅ Step 1 完成: {len(open_qa)} 条开放问答")

    # 存中间结果
    with open(OUTPUT_DIR / "step1_open_qa.jsonl", 'w', encoding='utf-8') as f:
        for item in open_qa:
            f.write(json.dumps({
                "instruction": item['question'].strip(),
                "output": item['answer'].strip(),
                "source": item.get('id', 'unknown'),
                "type": "open_qa"
            }, ensure_ascii=False) + '\n')
    print(f"  保存到: datas/step1_open_qa.jsonl")

    # ========== Step 2-3: 提取选择题 ==========
    zh_with, zh_without, en_with, en_without = extract_mcq_all()
    print(f"\n  ✅ 选择题提取完成")

    # ========== 转换处理 ==========
    # 开放问答转为 SFT 格式
    open_qa_sft = [{
        "instruction": item['question'].strip(),
        "output": item['answer'].strip(),
        "_source": item.get('id', 'unknown'),
        "_type": "open_qa"
    } for item in open_qa]
    all_sft = list(open_qa_sft)  # 先加入开放问答
    open_qa_count = len(open_qa_sft)

    # 中文有 solution
    print(f"\n--- 转换中文选择题 (有solution, {len(zh_with)} 条) ---")
    if args.limit > 0:
        zh_with = zh_with[:args.limit]

    if use_llm:
        # 分批调用 LLM
        converted = []
        for i in range(0, len(zh_with), args.batch_size):
            batch = zh_with[i:i + args.batch_size]
            prompt = build_rewrite_prompt(batch, lang="zh")
            print(f"  批次 {i//args.batch_size + 1}/{(len(zh_with)-1)//args.batch_size + 1}: 处理 {len(batch)} 条...", end=" ")
            resp = call_deepseek_api(prompt, api_key)
            if resp:
                parsed = parse_jsonl_response(resp)
                converted.extend(parsed)
                print(f"✅ 获得 {len(parsed)} 条")
            else:
                print(f"❌ 失败，回退到规则转换")
                converted.extend([rule_based_convert_zh(item) for item in batch])
    else:
        converted = [rule_based_convert_zh(item) for item in zh_with]
        print(f"  规则转换: {len(converted)} 条")

    all_sft.extend(converted)

    # 英文有 solution
    print(f"\n--- 转换英文选择题 (有solution, {len(en_with)} 条) ---")
    if args.limit > 0:
        en_with = en_with[:args.limit]

    if use_llm:
        en_converted = []
        for i in range(0, len(en_with), args.batch_size):
            batch = en_with[i:i + args.batch_size]
            prompt = build_rewrite_prompt(batch, lang="en")
            print(f"  批次 {i//args.batch_size + 1}/{(len(en_with)-1)//args.batch_size + 1}: 处理 {len(batch)} 条...", end=" ")
            resp = call_deepseek_api(prompt, api_key)
            if resp:
                parsed = parse_jsonl_response(resp)
                en_converted.extend(parsed)
                print(f"✅ 获得 {len(parsed)} 条")
            else:
                print(f"❌ 失败，回退到规则转换")
                en_converted.extend([rule_based_convert_en(item) for item in batch])
    else:
        en_converted = [rule_based_convert_en(item) for item in en_with]
        print(f"  规则转换: {len(en_converted)} 条")

    all_sft.extend(en_converted)

    # 中文无 solution (生成解析)
    print(f"\n--- 处理中文选择题 (无solution, {len(zh_without)} 条) ---")
    if args.limit > 0:
        zh_without = zh_without[:args.limit]

    if use_llm:
        zh_nosol_converted = []
        for i in range(0, len(zh_without), args.batch_size):
            batch = zh_without[i:i + args.batch_size]
            prompt = build_generate_solution_prompt(batch, lang="zh")
            print(f"  批次 {i//args.batch_size + 1}/{(len(zh_without)-1)//args.batch_size + 1}: 处理 {len(batch)} 条...", end=" ")
            resp = call_deepseek_api(prompt, api_key)
            if resp:
                parsed = parse_jsonl_response(resp)
                zh_nosol_converted.extend(parsed)
                print(f"✅ 获得 {len(parsed)} 条")
            else:
                print(f"❌ 失败，回退到规则转换")
                zh_nosol_converted.extend([rule_based_convert_zh(item) for item in batch])
    else:
        zh_nosol_converted = [rule_based_convert_zh(item) for item in zh_without]
        print(f"  规则转换: {len(zh_nosol_converted)} 条")

    all_sft.extend(zh_nosol_converted)

    # 英文无 solution
    print(f"\n--- 处理英文选择题 (无solution, {len(en_without)} 条) ---")
    if args.limit > 0:
        en_without = en_without[:args.limit]

    if use_llm:
        en_nosol_converted = []
        for i in range(0, len(en_without), args.batch_size):
            batch = en_without[i:i + args.batch_size]
            prompt = build_generate_solution_prompt(batch, lang="en")
            print(f"  批次 {i//args.batch_size + 1}/{(len(en_without)-1)//args.batch_size + 1}: 处理 {len(batch)} 条...", end=" ")
            resp = call_deepseek_api(prompt, api_key)
            if resp:
                parsed = parse_jsonl_response(resp)
                en_nosol_converted.extend(parsed)
                print(f"✅ 获得 {len(parsed)} 条")
            else:
                print(f"❌ 失败，回退到规则转换")
                en_nosol_converted.extend([rule_based_convert_en(item) for item in batch])
    else:
        en_nosol_converted = [rule_based_convert_en(item) for item in en_without]
        print(f"  规则转换: {len(en_nosol_converted)} 条")

    all_sft.extend(en_nosol_converted)

    # ========== Step 4: 数据集划分 ==========
    print("\n" + "=" * 60)
    print("STEP 4: 数据集划分")
    print("=" * 60)

    # 清洗：去空 instruction/output + 三类过滤
    cleaned = []
    skipped_empty = 0
    skipped_short = 0
    skipped_template = 0      # 规则模板垃圾数据
    skipped_english = 0       # 英文output混入中文
    skipped_overlong = 0      # 超长样本（>2048字符，4060显存放不下）
    for item in all_sft:
        inst = item.get('instruction', '').strip()
        out = item.get('output', '').strip()
        if not inst or not out:
            skipped_empty += 1
            continue
        # 过滤1: output 太短（<30字，无实质内容）
        if len(out) < 30:
            skipped_short += 1
            continue
        # 过滤2: 规则模板垃圾数据（"其他选项不符合题意" / "Other options are incorrect"）
        if "其他选项不符合题意" in out or "Other options are incorrect" in out:
            skipped_template += 1
            continue
        # 过滤3: 英文output混入中文训练集（output前50字无中文字符）
        if not any('一' <= c <= '鿿' for c in out[:50]):
            skipped_english += 1
            continue
        # 过滤4: instruction+output总长超2048字符
        if len(inst) + len(out) > 2048:
            skipped_overlong += 1
            continue
        cleaned.append({"instruction": inst, "output": out})

    print(f"  清洗统计:")
    print(f"    移除空内容:     {skipped_empty} 条")
    print(f"    移除过短<30字:  {skipped_short} 条")
    print(f"    移除规则模板:   {skipped_template} 条")
    print(f"    移除英文output: {skipped_english} 条")
    print(f"    移除超长>2048:  {skipped_overlong} 条")

    print(f"  清洗后总数: {len(cleaned)} 条")

    # 划分：90% train, 10% test
    import random
    random.seed(42)
    random.shuffle(cleaned)

    split_idx = int(len(cleaned) * 0.9)
    train_data = cleaned[:split_idx]
    test_data = cleaned[split_idx:]

    # 写 train.jsonl
    with open(OUTPUT_DIR / "train.jsonl", 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 写 test.jsonl
    with open(OUTPUT_DIR / "test.jsonl", 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"  ✅ train.jsonl: {len(train_data)} 条")
    print(f"  ✅ test.jsonl:  {len(test_data)} 条")

    # ========== 统计报告 ==========
    print("\n" + "=" * 60)
    print("📊 构建报告")
    print("=" * 60)
    print(f"  开放问答:           {open_qa_count} 条")
    print(f"  中文有solution:      {len(zh_with)} 条")
    print(f"  中文无solution:      {len(zh_without)} 条")
    print(f"  英文有solution:      {len(en_with)} 条")
    print(f"  英文无solution:      {len(en_without)} 条")
    print(f"  总数据集:           {len(cleaned)} 条")
    print(f"  训练集:             {len(train_data)} 条")
    print(f"  验证集:             {len(test_data)} 条")

    avg_inst = sum(len(item['instruction']) for item in cleaned) / len(cleaned) if cleaned else 0
    avg_out = sum(len(item['output']) for item in cleaned) / len(cleaned) if cleaned else 0
    print(f"  平均 instruction:   {avg_inst:.0f} 字")
    print(f"  平均 output:        {avg_out:.0f} 字")
    print(f"\n  清洗详情:")
    print(f"    移除规则模板:     {skipped_template} 条")
    print(f"    移除英文output:   {skipped_english} 条")
    print(f"    移除过短<30字:    {skipped_short} 条")
    print(f"    移除超长>2048:    {skipped_overlong} 条")
    print(f"    清洗后保留率:     {len(cleaned)/max(len(all_sft),1)*100:.1f}%")
    print(f"\n  📁 输出文件:")
    print(f"     {OUTPUT_DIR / 'train.jsonl'}")
    print(f"     {OUTPUT_DIR / 'test.jsonl'}")
    print(f"     {OUTPUT_DIR / 'step1_open_qa.jsonl'} (中间产物)")


if __name__ == "__main__":
    main()
