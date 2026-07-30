"""Agent 推理引擎测试 — 工具调用 + 异常回退 + 循环终止 + 图表生成"""

import pytest
import asyncio
import os
from unittest.mock import MagicMock


class TestAgentEngineBasic:
    def test_agent_receives_tools(self):
        from src.agent_engine import AgentEngine
        mock_client = MagicMock()
        engine = AgentEngine(llm_client=mock_client)
        assert len(engine.tools) == 8
        assert engine.tools[0].name == "query_cpu_monitor"
        assert engine.tools[7].name == "generate_chart"

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


class TestChartGeneration:
    """图表生成工具测试"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """每个测试前后清理 charts 目录"""
        import shutil
        if os.path.exists("charts"):
            shutil.rmtree("charts")
        yield
        if os.path.exists("charts"):
            shutil.rmtree("charts")

    def test_chart_tool_exists(self):
        from src.tools import get_tool_by_name
        tool = get_tool_by_name("generate_chart")
        assert tool is not None
        assert tool.name == "generate_chart"

    def test_bar_chart(self):
        from src.tools import get_tool_by_name
        tool = get_tool_by_name("generate_chart")
        data = {
            "labels": ["web-01", "web-02", "web-03", "db-01", "db-02"],
            "values": [92, 45, 85, 88, 33],
        }
        result = tool.execute(
            chart_type="bar", title="服务器CPU使用率",
            data=data, xlabel="服务器", ylabel="CPU (%)", color_theme="blue",
        )
        assert result["success"] is True
        assert result["chart_type"] == "bar"
        assert os.path.exists(result["path"])

    def test_hbar_chart(self):
        from src.tools import get_tool_by_name
        tool = get_tool_by_name("generate_chart")
        data = {
            "labels": ["db-01", "cache-01", "web-01", "api-01", "web-02"],
            "values": [94, 91, 78, 66, 55],
        }
        result = tool.execute(
            chart_type="hbar", title="服务器内存使用率排行",
            data=data, xlabel="内存 (%)", ylabel="服务器", color_theme="orange",
        )
        assert result["success"] is True
        assert os.path.exists(result["path"])

    def test_pie_chart(self):
        from src.tools import get_tool_by_name
        tool = get_tool_by_name("generate_chart")
        data = {
            "labels": ["生产环境", "测试环境", "开发环境"],
            "values": [9, 1, 2],
        }
        result = tool.execute(
            chart_type="pie", title="服务器环境分布",
            data=data, color_theme="mixed",
        )
        assert result["success"] is True
        assert result["chart_type"] == "pie"
        assert os.path.exists(result["path"])

    def test_line_chart(self):
        from src.tools import get_tool_by_name
        tool = get_tool_by_name("generate_chart")
        data = {
            "labels": ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00"],
            "values": [25, 18, 55, 88, 92, 60],
        }
        result = tool.execute(
            chart_type="line", title="web-01 CPU 24小时趋势",
            data=data, xlabel="时间", ylabel="CPU (%)", color_theme="red",
        )
        assert result["success"] is True
        assert os.path.exists(result["path"])

    def test_grouped_bar_chart(self):
        from src.tools import get_tool_by_name
        tool = get_tool_by_name("generate_chart")
        data = {
            "labels": ["web-01", "web-02", "web-03", "db-01", "db-02"],
            "series": {
                "CPU (%)": [92, 45, 85, 88, 33],
                "内存 (%)": [78, 55, 71, 94, 82],
            },
        }
        result = tool.execute(
            chart_type="grouped_bar", title="服务器资源对比",
            data=data, xlabel="服务器", ylabel="使用率 (%)", color_theme="mixed",
        )
        assert result["success"] is True
        assert os.path.exists(result["path"])

    def test_chart_output_directory_created_on_execute(self):
        from src.tools import get_tool_by_name
        tool = get_tool_by_name("generate_chart")
        # 仅获取工具不创建目录
        assert not os.path.isdir("charts")
        # 执行后自动创建
        tool.execute(chart_type="bar", title="测试", data={"labels": ["a"], "values": [1]})
        assert os.path.isdir("charts")

    def test_invalid_chart_type(self):
        from src.tools import get_tool_by_name
        tool = get_tool_by_name("generate_chart")
        result = tool.execute(
            chart_type="scatter", title="测试",
            data={"labels": ["a"], "values": [1]},
        )
        assert result["success"] is False
        assert "不支持的图表类型" in result["error"]
