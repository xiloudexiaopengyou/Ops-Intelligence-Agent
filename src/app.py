"""
Gradio 主界面 — 5 Tab + GPU 状态栏

用法:
    python src/app.py [--port 7860]
"""

import os
import argparse

import gradio as gr

# 延迟导入 — 让未加载的模块不影响 Gradio 启动
try:
    from src.rag_system import RAGSystem
    _rag_available = True
except Exception as e:
    print(f"⚠️ RAG 模块不可用: {e}")
    _rag_available = False

try:
    from src.agent_engine import AgentEngine
    from openai import OpenAI
    _agent_available = True
except Exception as e:
    print(f"⚠️ Agent 模块不可用: {e}")
    _agent_available = False


# ============================================================
# GPU 状态监控
# ============================================================

def _get_gpu_status():
    """获取 GPU 状态 HTML"""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        used_gb = info.used / 1024**3
        total_gb = info.total / 1024**3
        pct = used_gb / total_gb * 100

        if pct < 85:
            color, status = "#34D399", "● GPU Normal"
        elif pct < 95:
            color, status = "#F59E0B", "● GPU Warning"
        else:
            color, status = "#F87171", "● GPU Critical"

        return f"""
        <div style="display:flex;align-items:center;gap:20px;padding:6px 16px;font-family:monospace;font-size:11px;color:#8B909F;background:#1A1D2A;border-radius:6px">
            <span style="color:{color}">{status}</span>
            <span>VRAM <b style="color:#E4E6ED">{used_gb:.1f}</b>/<b style="color:#E4E6ED">{total_gb:.1f}</b> GB</span>
            <span style="color:#2D3245">|</span>
            <span>模型 <b style="color:#4DA6FF">Qwen2.5-7B-LoRA</b></span>
            <span style="color:#2D3245">|</span>
            <span>vLLM :8000</span>
        </div>"""
    except Exception:
        return '<div style="color:#8B909F;font-size:11px">⚠️ GPU 监控不可用</div>'


# ============================================================
# Tab 回调函数
# ============================================================

def _rag_query(message, history):
    if not _rag_available:
        return "⚠️ RAG 模块未加载，请检查依赖和文档目录。"
    if not hasattr(_rag_query, '_instance'):
        _rag_query._instance = RAGSystem(doc_dir="./docs")
    result = _rag_query._instance.query(message)
    answer = result["answer"]
    if result["source_count"] > 0:
        src = result["sources"][0]
        fname = src.get('metadata', {}).get('file_name', '文档')
        score = src.get('score', 0)
        answer += f"\n\n📚 来源: {fname} · 关联度 {score:.0%}"
    return answer


def _lora_query(message, history):
    try:
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
        resp = client.chat.completions.create(
            model="itops",
            messages=[{"role": "user", "content": message}],
            temperature=0.7, max_tokens=512,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ vLLM 服务不可用: {e}\n请先启动: make serve"


def _compare_query(message):
    """对比基座和 LoRA"""
    try:
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
        base = client.chat.completions.create(
            model="unsloth/Qwen2.5-7B-Instruct",
            messages=[{"role": "user", "content": message}],
            temperature=0.1, max_tokens=512,
        )
        lora = client.chat.completions.create(
            model="itops",
            messages=[{"role": "user", "content": message}],
            temperature=0.1, max_tokens=512,
        )
        return (
            f"### 🔴 基座模型\n{base.choices[0].message.content}\n\n"
            f"---\n\n"
            f"### 🟢 LoRA 微调\n{lora.choices[0].message.content}"
        )
    except Exception as e:
        return f"⚠️ vLLM 不可用: {e}"


async def _agent_query(message, history):
    if not _agent_available:
        return "⚠️ Agent 模块不可用"
    try:
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
        engine = AgentEngine(llm_client=client)
        result = await engine.run(message)

        output_parts = []
        for step in result.steps:
            output_parts.append(f"**步骤 {step.step_num}** 🤔 {step.thought[:100]}...")
            if step.action:
                output_parts.append(f"> 🔧 `{step.action}`")
            if step.observation:
                output_parts.append(f"> 👁️ {step.observation[:150]}")
            output_parts.append("")

        output_parts.append(f"\n### ✅ 最终答案\n{result.final_answer}")
        return "\n".join(output_parts)
    except Exception as e:
        return f"⚠️ Agent 错误: {e}"


# ============================================================
# UI 构建
# ============================================================

def build_ui(port: int = 7860):
    theme = gr.themes.Base(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="gray",
    ).set(
        body_background_fill="#0F1119",
        body_background_fill_dark="#0F1119",
        block_background_fill="#1A1D2A",
        block_background_fill_dark="#1A1D2A",
        block_border_color="#2D3245",
        input_background_fill="#242838",
        button_primary_background_fill="#4DA6FF",
        button_primary_text_color="#FFFFFF",
        color_accent_soft="#4DA6FF",
        font=["IBM Plex Sans", "system-ui"],
        font_mono=["JetBrains Mono", "monospace"],
    )

    with gr.Blocks(
        title="智能IT运维助手",
        theme=theme,
        css="""
        .status-bar { position: sticky; top: 0; z-index: 100; }
        .source-tag { border-left: 3px solid #A78BFA; padding: 4px 12px; background: rgba(167,139,250,0.06); font-size: 12px; }
        """,
    ) as demo:
        # 顶部状态栏
        gr.HTML(_get_gpu_status, every=5)

        # 标题
        gr.Markdown(
            "# 🚀 智能IT运维助手\n"
            "基于 Qwen2.5-7B · QLoRA · RAG · LangGraph Agent · RTX 4060",
        )

        with gr.Tab("📚 RAG 知识问答"):
            gr.ChatInterface(fn=_rag_query, title="RAG 检索增强问答")

        with gr.Tab("🧠 LoRA 微调问答"):
            gr.ChatInterface(fn=_lora_query, title="LoRA 微调 IT 运维问答")

        with gr.Tab("📊 模型对比"):
            gr.Markdown("输入问题，对比基座模型和微调模型的效果差异")
            with gr.Row():
                compare_input = gr.Textbox(
                    label="问题", placeholder="Oracle 连接池耗尽怎么排查？", lines=2
                )
                compare_btn = gr.Button("对比 ⚡", variant="primary")
            compare_output = gr.Markdown()
            compare_btn.click(
                _compare_query, inputs=compare_input, outputs=compare_output
            )

        with gr.Tab("🤖 Agent 推理"):
            gr.ChatInterface(fn=_agent_query, title="Agent 多步推理")

        with gr.Tab("📈 评估报告"):
            gr.Markdown("## 模型评估对比")
            eval_img = "evaluation.png"
            if os.path.exists(eval_img):
                gr.Image(eval_img, label="基座 vs 微调")
            else:
                gr.Markdown("⚠️ 评估图未生成，请先运行 `python src/evaluate_model.py`")

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    ui = build_ui(args.port)
    ui.launch(server_name="0.0.0.0", server_port=args.port, share=False)
