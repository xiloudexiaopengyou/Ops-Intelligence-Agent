"""Agent 推理引擎测试 — 工具调用 + 异常回退 + 循环终止"""

import pytest
import asyncio
from unittest.mock import MagicMock


class TestAgentEngineBasic:
    def test_agent_receives_tools(self):
        from src.agent_engine import AgentEngine
        mock_client = MagicMock()
        engine = AgentEngine(llm_client=mock_client)
        assert len(engine.tools) == 7
        assert engine.tools[0].name == "query_cpu_monitor"

    def test_max_steps_boundary(self):
        from src.agent_engine import AgentEngine
        engine = AgentEngine(llm_client=MagicMock(), max_steps=3)
        assert engine.max_steps == 3

    def test_max_steps_too_low_raises(self):
        from src.agent_engine import AgentEngine
        with pytest.raises(ValueError):
            AgentEngine(llm_client=MagicMock(), max_steps=0)


class TestToolExecution:
    def test_cpu_monitor_tool(self):
        from src.tools import get_tool_by_name
        tool = get_tool_by_name("query_cpu_monitor")
        assert tool is not None
        result = tool.execute(threshold=80)
        assert result["total"] == 12
        assert result["over_threshold"] >= 1

    def test_cmdb_tool_found(self):
        from src.tools import get_tool_by_name
        tool = get_tool_by_name("query_cmdb")
        result = tool.execute(server_name="web-01")
        assert result["found"] is True
        assert result["ip"] == "10.0.1.11"

    def test_cmdb_tool_not_found(self):
        from src.tools import get_tool_by_name
        tool = get_tool_by_name("query_cmdb")
        result = tool.execute(server_name="nonexistent")
        assert result["found"] is False

    def test_ticket_creation(self):
        from src.tools import get_tool_by_name
        tool = get_tool_by_name("create_ticket")
        result = tool.execute(title="测试工单", description="测试描述", priority="P1")
        assert result["status"] == "已创建"
        assert result["ticket_id"].startswith("INC-")

    def test_restart_service_invalid(self):
        from src.tools import get_tool_by_name
        tool = get_tool_by_name("restart_service")
        result = tool.execute(server="web-01", service_name="invalid-service")
        assert result["success"] is False
