"""
LangGraph ReAct Agent 推理引擎

架构:
    router → agent → tools → agent → ... → finalize

用法:
    from openai import OpenAI
    from src.agent_engine import AgentEngine

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
    engine = AgentEngine(llm_client=client)
    result = await engine.run("检查所有服务器CPU")
"""

import json
import asyncio
from dataclasses import dataclass, field

from src.tools import get_all_tools


@dataclass
class AgentStep:
    step_num: int
    thought: str = ""
    action: str = ""
    observation: str = ""


@dataclass
class AgentResult:
    final_answer: str = ""
    steps: list[AgentStep] = field(default_factory=list)
    tool_results: dict = field(default_factory=dict)
    total_steps: int = 0
    error: str = ""


class AgentEngine:
    def __init__(
        self,
        llm_client,
        max_steps: int = 8,
        tool_timeout: int = 10,
        max_retries: int = 1,
        degrade_threshold: int = 3,
    ):
        if max_steps < 1:
            raise ValueError(f"max_steps 必须 >= 1, 当前值: {max_steps}")

        self.llm = llm_client
        self.max_steps = max_steps
        self.tool_timeout = tool_timeout
        self.max_retries = max_retries
        self.degrade_threshold = degrade_threshold
        self.tools = get_all_tools()
        self.tool_map = {t.name: t for t in self.tools}

    def _build_system_prompt(self) -> str:
        tool_descriptions = "\n".join(
            f"- {t.name}: {t.description}" for t in self.tools
        )
        return f"""你是一个智能 IT 运维助手，可以调用以下工具来完成任务：

{tool_descriptions}

工作流程：思考 → 调用工具 → 观察结果 → 继续思考或给出最终答案。

规则：
1. 先思考需要哪个工具和什么参数
2. 以 JSON 格式调用工具：{{"tool": "工具名", "arguments": {{...}}}}
3. 如果不需要工具，直接以纯文本回答
4. 工具调用失败时尝试其他方式解决
5. 给出最终答案时请总结所有步骤发现的问题和建议"""

    async def run(self, query: str) -> AgentResult:
        """执行 Agent 推理循环"""
        steps: list[AgentStep] = []
        tool_results: dict = {}
        error_count = 0

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": query},
        ]

        for step_num in range(1, self.max_steps + 1):
            step = AgentStep(step_num=step_num)

            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(self._call_llm, messages),
                    timeout=30,
                )
            except asyncio.TimeoutError:
                return AgentResult(
                    final_answer="推理超时，请简化问题后重试。",
                    steps=steps,
                    tool_results=tool_results,
                    total_steps=len(steps),
                    error="LLM 调用超时",
                )

            content = response.strip()
            step.thought = content[:300]

            tool_call = self._parse_tool_call(content)
            if tool_call is None:
                return AgentResult(
                    final_answer=content,
                    steps=steps,
                    tool_results=tool_results,
                    total_steps=len(steps),
                )

            tool_name = tool_call["tool"]
            tool_args = tool_call.get("arguments", {})
            step.action = f"{tool_name}({json.dumps(tool_args, ensure_ascii=False)})"

            tool = self.tool_map.get(tool_name)
            if tool is None:
                step.observation = f"工具 {tool_name} 不存在。可用工具: {list(self.tool_map.keys())}"
                error_count += 1
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": step.observation})
            else:
                try:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(tool.execute, **tool_args),
                        timeout=self.tool_timeout,
                    )
                    tool_results[tool_name] = result
                    obs = json.dumps(result, ensure_ascii=False, indent=2)
                    step.observation = obs[:500]
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": f"工具结果: {obs}"})
                except asyncio.TimeoutError:
                    step.observation = f"工具 {tool_name} 执行超时 ({self.tool_timeout}s)"
                    error_count += 1
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": step.observation})
                except Exception as e:
                    step.observation = f"工具执行失败: {str(e)}"
                    error_count += 1
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": step.observation})

            steps.append(step)

            if error_count >= self.degrade_threshold:
                return AgentResult(
                    final_answer=f"连续 {error_count} 次错误，已自动降级。已完成 {len(steps)} 步。请检查工具服务状态。",
                    steps=steps,
                    tool_results=tool_results,
                    total_steps=len(steps),
                    error=f"连续 {error_count} 次错误触发降级",
                )

        final_msg = {"role": "user", "content": "已达到最大推理步数，请基于以上信息给出最终答案。"}
        messages.append(final_msg)
        try:
            final_response = await asyncio.to_thread(self._call_llm, messages)
        except Exception:
            final_response = f"推理已达到 {self.max_steps} 步上限，已执行的操作已记录。"

        return AgentResult(
            final_answer=final_response,
            steps=steps,
            tool_results=tool_results,
            total_steps=len(steps),
        )

    def _call_llm(self, messages: list[dict]) -> str:
        response = self.llm.chat.completions.create(
            model="itops",
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    def _parse_tool_call(self, text: str) -> dict | None:
        text = text.strip()
        for start_marker in ["```json", "```", "{"]:
            if start_marker in text:
                try:
                    first = text.index("{")
                    last = text.rindex("}")
                    candidate = text[first:last + 1]
                    data = json.loads(candidate)
                    if "tool" in data:
                        return data
                except (ValueError, json.JSONDecodeError):
                    pass

        try:
            data = json.loads(text)
            if "tool" in data:
                return data
        except json.JSONDecodeError:
            pass

        return None
