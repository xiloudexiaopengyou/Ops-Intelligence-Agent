"""
Gradio 主界面 — 5 Tab + GPU 状态栏

用法:
    python src/app.py [--port 7860]
"""

import os
import sys
import argparse
import traceback

# ── 确保项目根目录在 Python 路径中 ──
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import gradio as gr

# ── 绕过 torchcodec 的 FFmpeg 依赖（sentence_transformers → torchcodec 仅用于音视频，RAG 用不到）──
import types
import importlib

def _make_fake_module(name: str):
    """创建带 __spec__ 的假模块，避免 transformers 的 importlib 检查崩溃"""
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    mod.__path__ = []
    mod.__package__ = name
    return mod

_tc = _make_fake_module("torchcodec")
_tc_core = _make_fake_module("torchcodec._core")
_tc_decoders = _make_fake_module("torchcodec.decoders")
_tc_decoders.AudioDecoder = None
_tc_decoders.VideoDecoder = None
sys.modules["torchcodec"] = _tc
sys.modules["torchcodec._core"] = _tc_core
sys.modules["torchcodec.decoders"] = _tc_decoders

# ── OpenAI 客户端（LoRA/Compare/Agent 共用，独立导入）──
try:
    from openai import OpenAI
    _openai_available = True
except Exception as e:
    print(f"⚠️ OpenAI 库不可用: {e}")
    _openai_available = False

# ── RAG 模块 ──
try:
    from src.rag_system import RAGSystem
    _rag_available = True
except Exception as e:
    print(f"⚠️ RAG 模块不可用: {e}")
    traceback.print_exc()
    _rag_available = False

# ── Agent 模块 ──
try:
    from src.agent_engine import AgentEngine
    _agent_available = True
except Exception as e:
    print(f"⚠️ Agent 模块不可用: {e}")
    traceback.print_exc()
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
            color, status = "#7BC67E", "● GPU Normal"
        elif pct < 95:
            color, status = "#E5B85C", "● GPU Warning"
        else:
            color, status = "#E06C6C", "● GPU Critical"

        return f"""
        <div style="display:flex;align-items:center;gap:20px;padding:6px 16px;font-family:monospace;font-size:11px;color:#9A9DA8;background:#222428;border-radius:6px">
            <span style="color:{color}">{status}</span>
            <span>VRAM <b style="color:#C8CCD4">{used_gb:.1f}</b>/<b style="color:#C8CCD4">{total_gb:.1f}</b> GB</span>
            <span style="color:#33353A">|</span>
            <span>模型 <b style="color:#6B9FD4">Qwen2.5-7B-LoRA</b></span>
            <span style="color:#33353A">|</span>
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
    try:
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
    except Exception as e:
        return f"⚠️ RAG 查询失败: {e}\n\n请确保 ./docs/ 目录存在且包含文档文件。"


def _lora_query(message, history):
    if not _openai_available:
        return "⚠️ OpenAI 库未安装，请运行: pip install openai"
    try:
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
        resp = client.chat.completions.create(
            model="itops",
            messages=[{"role": "user", "content": message}],
            temperature=0.7, max_tokens=256,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ vLLM 服务不可用: {e}\n请先启动: bash scripts/start_vllm_lora.sh"


def _compare_query(message):
    """对比基座和 LoRA"""
    if not _openai_available:
        return "⚠️ OpenAI 库未安装，请运行: pip install openai"
    try:
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
        base = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
            messages=[{"role": "user", "content": message}],
            temperature=0.1, max_tokens=256,
        )
        lora = client.chat.completions.create(
            model="itops",
            messages=[{"role": "user", "content": message}],
            temperature=0.1, max_tokens=256,
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
        return "⚠️ Agent 模块不可用，请检查 src/agent_engine.py 和 src/tools.py 是否正常。"
    if not _openai_available:
        return "⚠️ OpenAI 库未安装，请运行: pip install openai"
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
    # ── 配色系统：暖黑底 + 低饱和蓝 + 柔和层次 ──
    theme = gr.themes.Base(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="gray",
    ).set(
        body_background_fill="#1A1B1E",
        body_background_fill_dark="#1A1B1E",
        block_background_fill="#222428",
        block_background_fill_dark="#222428",
        block_border_color="#33353A",
        input_background_fill="#2A2C30",
        button_primary_background_fill="#6B9FD4",
        button_primary_text_color="#FFFFFF",
        color_accent_soft="#6B9FD4",
    )

    with gr.Blocks(title="智能IT运维助手") as demo:
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
            gr.HTML('<p class="compare-desc">输入问题，对比基座模型和微调模型的效果差异</p>')
            with gr.Row():
                compare_input = gr.Textbox(
                    label="问题", placeholder="Oracle 连接池耗尽怎么排查？", lines=2
                )
                compare_btn = gr.Button("对比 ⚡", variant="primary")
            compare_output = gr.Markdown(elem_classes=["compare-output"])
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

    return demo, theme


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    ui, theme = build_ui(args.port)
    ui.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=False,
        theme=theme,
        css="""
        .status-bar { position: sticky; top: 0; z-index: 100; }
        .source-tag { border-left: 3px solid #6B9FD4; padding: 4px 12px; background: rgba(107,159,212,0.08); font-size: 12px; }
        textarea, input, [data-testid="textbox"] textarea, [data-testid="textbox"] input,
        .chatbot-container textarea, .chatbot-container input,
        .gradio-container textarea, .gradio-container input,
        .prose :where(textarea):not(:where([class~="not-prose"],[class~="not-prose"] *)),
        input[type="text"], input[type="email"], input[type="password"] {
            color: #C8CCD4 !important;
        }
        /* ── 主标题 ── */
        .main-header h1, h1:first-of-type {
            color: #9A9DA8 !important;
        }
        /* ── Tab 标签 未选中 ── */
        .tabs > .tab-nav > button:not(.selected),
        [role="tablist"] button:not([aria-selected="true"]),
        .tab-nav button:not(.selected) {
            color: #7A7D85 !important;
        }
        /* ── ChatInterface 标题 ── */
        .chatbot-container .label-text,
        [data-testid="chatbot"] + div label,
        .chat-interface .heading,
        .chatbot .header {
            color: #9A9DA8 !important;
        }
        /* ── 聊天回答内容 纯黑 ── */
        .bubble p, .bubble li, .bubble h1, .bubble h2, .bubble h3,
        .bubble h4, .bubble h5, .bubble h6, .bubble strong, .bubble em,
        .bubble code, .bubble pre, .bubble blockquote, .bubble div,
        .chatbot .message p, .chatbot .message li,
        .chatbot .message h1, .chatbot .message h2, .chatbot .message h3,
        .chatbot .message h4, .chatbot .message h5, .chatbot .message h6,
        .chatbot .message strong, .chatbot .message em,
        .chatbot .message code, .chatbot .message pre, .chatbot .message div,
        [data-testid="chatbot"] p, [data-testid="chatbot"] li,
        [data-testid="chatbot"] h1, [data-testid="chatbot"] h2, [data-testid="chatbot"] h3,
        [data-testid="chatbot"] strong, [data-testid="chatbot"] code, [data-testid="chatbot"] div {
            color: #000000 !important;
        }
        /* ── 副标题 淡灰 ── */
        h1 + p, .prose h1 + p, .md h1 + p, .markdown h1 + p {
            color: #9A9DA8 !important;
        }
        /* ── 聊天操作按钮 未选中：白色 ── */
        .chatbot .message button, .bubble button,
        [data-testid="chatbot"] .message button,
        .message-actions button, .icon-button,
        .chatbot button[aria-label], .bubble button[aria-label],
        .chatbot .message footer button, .bubble footer button {
            color: #FFFFFF !important;
            fill: #FFFFFF !important;
            stroke: #FFFFFF !important;
            background: transparent !important;
            border-color: #555860 !important;
        }
        /* SVG 图标跟随按钮颜色 */
        .chatbot .message button svg, .bubble button svg,
        .message-actions button svg, .icon-button svg {
            color: #FFFFFF !important;
            fill: #FFFFFF !important;
            stroke: #FFFFFF !important;
        }
        /* ── 操作按钮 选中/active/hover：白底黑字 ── */
        .chatbot .message button:hover, .bubble button:hover,
        .chatbot .message button:active, .bubble button:active,
        .chatbot .message button:focus-visible, .bubble button:focus-visible,
        [data-testid="chatbot"] .message button:hover,
        [data-testid="chatbot"] .message button:active,
        .message-actions button:hover, .message-actions button:active,
        .icon-button:hover, .icon-button:active,
        .chatbot button[aria-label]:hover, .chatbot button[aria-label]:active,
        .chatbot .message footer button:hover,
        .chatbot .message footer button:active {
            background: #FFFFFF !important;
            color: #000000 !important;
            fill: #000000 !important;
            stroke: #000000 !important;
            border-color: #FFFFFF !important;
        }
        /* SVG 图标跟随 hover/active */
        .chatbot .message button:hover svg, .bubble button:hover svg,
        .chatbot .message button:active svg, .bubble button:active svg,
        .message-actions button:hover svg, .icon-button:hover svg,
        .message-actions button:active svg, .icon-button:active svg {
            color: #000000 !important;
            fill: #000000 !important;
            stroke: #000000 !important;
        }
        /* ── 模型对比页 描述文字 + 输出 淡灰 ── */
        .compare-desc {
            color: #9A9DA8 !important;
        }
        .compare-output, .compare-output p, .compare-output h1,
        .compare-output h2, .compare-output h3, .compare-output li,
        .compare-output strong, .compare-output em {
            color: #B0B4BD !important;
        }
        /* ── 链接 ── */
        .bubble a, .chatbot .message a, [data-testid="chatbot"] a {
            color: #6B9FD4 !important;
        }
        """,
    )
