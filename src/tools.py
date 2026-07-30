"""
Agent 工具集 — Mock 实现，生产环境替换实现即可

所有工具继承 BaseTool，实现 execute 方法。
get_all_tools() 返回已注册工具列表，供 Agent 使用。
"""

import time
import uuid
import os
from pathlib import Path
from abc import ABC, abstractmethod


# ============================================================
# Mock 数据
# ============================================================

_MOCK_SERVERS = [
    {"hostname": "web-01",   "ip": "10.0.1.11", "owner": "devops-team", "dept": "平台研发", "env": "生产"},
    {"hostname": "web-02",   "ip": "10.0.1.12", "owner": "devops-team", "dept": "平台研发", "env": "生产"},
    {"hostname": "web-03",   "ip": "10.0.1.13", "owner": "devops-team", "dept": "平台研发", "env": "生产"},
    {"hostname": "api-01",   "ip": "10.0.2.11", "owner": "backend-team","dept": "后端研发", "env": "生产"},
    {"hostname": "api-02",   "ip": "10.0.2.12", "owner": "backend-team","dept": "后端研发", "env": "生产"},
    {"hostname": "db-01",    "ip": "10.0.3.11", "owner": "dba-team",    "dept": "DBA",     "env": "生产"},
    {"hostname": "db-02",    "ip": "10.0.3.12", "owner": "dba-team",    "dept": "DBA",     "env": "生产"},
    {"hostname": "cache-01", "ip": "10.0.4.11", "owner": "infra-team",  "dept": "基础架构","env": "生产"},
    {"hostname": "monitor-01","ip": "10.0.5.11", "owner": "infra-team",  "dept": "基础架构","env": "生产"},
    {"hostname": "test-01",  "ip": "10.0.99.11","owner": "qa-team",     "dept": "测试",    "env": "测试"},
    {"hostname": "dev-01",   "ip": "10.0.98.11","owner": "dev-team",    "dept": "研发",    "env": "开发"},
    {"hostname": "dev-02",   "ip": "10.0.98.12","owner": "dev-team",    "dept": "研发",    "env": "开发"},
]

_MOCK_CPU = {
    "web-01": 92, "web-02": 45, "web-03": 85, "api-01": 62,
    "api-02": 71, "db-01": 88, "db-02": 33, "cache-01": 28,
    "monitor-01": 15, "test-01": 8, "dev-01": 55, "dev-02": 42,
}

_MOCK_MEMORY = {
    "web-01": 78, "web-02": 55, "web-03": 71, "api-01": 66,
    "api-02": 59, "db-01": 94, "db-02": 82, "cache-01": 91,
    "monitor-01": 34, "test-01": 22, "dev-01": 61, "dev-02": 48,
}

_MOCK_LOGS = [
    {"timestamp": "2026-07-30 14:23:01", "level": "ERROR", "message": "Connection pool exhausted: max 100 connections reached"},
    {"timestamp": "2026-07-30 14:22:45", "level": "WARN",  "message": "Disk usage on /data exceeds 85% threshold"},
    {"timestamp": "2026-07-30 14:20:12", "level": "ERROR", "message": "OOM killer terminated process java (PID 28421)"},
    {"timestamp": "2026-07-30 14:18:33", "level": "INFO",  "message": "Scheduled backup completed successfully"},
    {"timestamp": "2026-07-30 14:15:00", "level": "WARN",  "message": "Slow query detected: SELECT * FROM orders WHERE... took 12.3s"},
]


# ============================================================
# 抽象基类
# ============================================================

class BaseTool(ABC):
    name: str = ""
    description: str = ""
    parameters: dict = {}

    @abstractmethod
    def execute(self, **kwargs) -> dict:
        ...

    def to_openai_tool(self) -> dict:
        """转为 OpenAI function calling 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ============================================================
# 具体工具
# ============================================================

class QueryCpuMonitor(BaseTool):
    name = "query_cpu_monitor"
    description = "查询所有服务器或指定服务器的 CPU 使用率。threshold 为告警阈值(%)，返回超过阈值的服务器列表。"
    parameters = {
        "type": "object",
        "properties": {
            "threshold": {
                "type": "integer",
                "description": "CPU 告警阈值，如 80 表示超过 80% 的服务器",
                "default": 80,
            },
            "hostname": {
                "type": "string",
                "description": "可选，指定查询的服务器主机名",
            },
        },
        "required": ["threshold"],
    }

    def execute(self, threshold: int = 80, hostname: str | None = None, **kwargs) -> dict:
        if hostname:
            cpu = _MOCK_CPU.get(hostname)
            if cpu is None:
                return {"error": f"服务器 {hostname} 不存在", "servers": []}
            return {
                "servers": [{"hostname": hostname, "cpu_pct": cpu, "status": "超标" if cpu > threshold else "正常"}],
                "total": 1,
                "over_threshold": 1 if cpu > threshold else 0,
            }

        servers = []
        for host, cpu in _MOCK_CPU.items():
            if cpu > threshold:
                servers.append({"hostname": host, "cpu_pct": cpu, "status": "超标"})
        return {
            "servers": servers,
            "total": len(_MOCK_CPU),
            "over_threshold": len(servers),
        }


class QueryMemoryMonitor(BaseTool):
    name = "query_memory_monitor"
    description = "查询所有服务器或指定服务器的内存使用率。threshold 为告警阈值(%)。"
    parameters = {
        "type": "object",
        "properties": {
            "threshold": {"type": "integer", "description": "内存告警阈值", "default": 85},
            "hostname": {"type": "string", "description": "可选，指定服务器"},
        },
        "required": ["threshold"],
    }

    def execute(self, threshold: int = 85, hostname: str | None = None, **kwargs) -> dict:
        if hostname:
            mem = _MOCK_MEMORY.get(hostname)
            if mem is None:
                return {"error": f"服务器 {hostname} 不存在"}
            return {"servers": [{"hostname": hostname, "mem_pct": mem, "status": "超标" if mem > threshold else "正常"}]}

        servers = []
        for host, mem in _MOCK_MEMORY.items():
            if mem > threshold:
                servers.append({"hostname": host, "mem_pct": mem, "status": "超标"})
        return {"servers": servers, "total": len(_MOCK_MEMORY), "over_threshold": len(servers)}


class SendAlertEmail(BaseTool):
    name = "send_alert_email"
    description = "发送告警邮件到指定收件人。servers 为受影响的服务器列表。"
    parameters = {
        "type": "object",
        "properties": {
            "servers": {"type": "array", "items": {"type": "string"}, "description": "受影响服务器主机名列表"},
            "recipient": {"type": "string", "description": "收件人邮箱", "default": "it-team@chengguo.com"},
            "subject": {"type": "string", "description": "邮件主题"},
        },
        "required": ["servers"],
    }

    def execute(self, servers: list[str], recipient: str = "it-team@chengguo.com",
                subject: str = "服务器告警通知", **kwargs) -> dict:
        msg_id = f"msg_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        print(f"📧 [Mock] 发送告警邮件 -> {recipient}")
        print(f"   主题: {subject}")
        print(f"   服务器: {', '.join(servers)}")
        return {"sent": True, "message_id": msg_id, "recipient": recipient, "server_count": len(servers)}


class QueryCmdb(BaseTool):
    name = "query_cmdb"
    description = "查询 CMDB 资产信息，获取服务器归属、IP、环境等详细信息。"
    parameters = {
        "type": "object",
        "properties": {
            "server_name": {"type": "string", "description": "服务器主机名，如 web-01"},
        },
        "required": ["server_name"],
    }

    def execute(self, server_name: str, **kwargs) -> dict:
        for srv in _MOCK_SERVERS:
            if srv["hostname"] == server_name:
                return {"found": True, **srv}
        return {"found": False, "error": f"未找到服务器 {server_name}"}


class CreateTicket(BaseTool):
    name = "create_ticket"
    description = "在工单系统创建 IT 工单。priority 可选 P1(紧急)、P2(高)、P3(中)、P4(低)。"
    parameters = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "工单标题"},
            "description": {"type": "string", "description": "工单详细描述"},
            "priority": {"type": "string", "description": "优先级: P1/P2/P3/P4", "default": "P2"},
        },
        "required": ["title", "description"],
    }

    def execute(self, title: str, description: str, priority: str = "P2", **kwargs) -> dict:
        ticket_id = f"INC-2026-{uuid.uuid4().hex[:4].upper()}"
        print(f"🎫 [Mock] 创建工单: {ticket_id} [{priority}] {title}")
        return {
            "ticket_id": ticket_id,
            "status": "已创建",
            "priority": priority,
            "url": f"https://ticket.chengguo.com/incident/{ticket_id}",
        }


class RestartService(BaseTool):
    name = "restart_service"
    description = "重启指定服务器上的服务。支持的服务: nginx, docker, postgresql, redis, java-app。"
    parameters = {
        "type": "object",
        "properties": {
            "server": {"type": "string", "description": "服务器主机名"},
            "service_name": {"type": "string", "description": "服务名。支持: nginx, docker, postgresql, redis, java-app"},
        },
        "required": ["server", "service_name"],
    }

    def execute(self, server: str, service_name: str, **kwargs) -> dict:
        if server not in _MOCK_CPU:
            return {"success": False, "output": f"服务器 {server} 不存在"}
        valid = {"nginx", "docker", "postgresql", "redis", "java-app"}
        if service_name not in valid:
            return {"success": False, "output": f"不支持的服务: {service_name}。支持: {', '.join(sorted(valid))}"}
        print(f"🔄 [Mock] 重启 {server} 上的 {service_name}...")
        time.sleep(0.5)
        return {"success": True, "output": f"{service_name} on {server} restarted successfully", "server": server, "service": service_name}


class QueryLogs(BaseTool):
    name = "query_logs"
    description = "查询服务器日志。支持按关键字和时间范围过滤。"
    parameters = {
        "type": "object",
        "properties": {
            "server": {"type": "string", "description": "服务器主机名"},
            "keyword": {"type": "string", "description": "搜索关键字，如 ERROR"},
            "time_range": {"type": "string", "description": "时间范围，如 1h/24h/7d", "default": "1h"},
        },
        "required": ["server", "keyword"],
    }

    def execute(self, server: str, keyword: str, time_range: str = "1h", **kwargs) -> dict:
        if server not in _MOCK_CPU:
            return {"error": f"服务器 {server} 不存在"}
        matched = [
            log for log in _MOCK_LOGS
            if keyword.upper() in log["message"].upper()
        ]
        return {"server": server, "keyword": keyword, "time_range": time_range, "count": len(matched), "logs": matched}


class GenerateChart(BaseTool):
    name = "generate_chart"
    description = (
        "根据数据生成图表并保存为 PNG 文件。"
        "支持柱状图(bar)、横向柱状图(hbar)、饼图(pie)、折线图(line)、多系列分组柱状图(grouped_bar)。"
        "data 中 labels 为标签列表，values 为数值列表（饼图/单系列），"
        "或多系列时使用 series 字典: {\"系列1\": [值列表], \"系列2\": [值列表]}。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "chart_type": {
                "type": "string",
                "enum": ["bar", "hbar", "pie", "line", "grouped_bar"],
                "description": "图表类型: bar=柱状图, hbar=横向柱状图, pie=饼图, line=折线图, grouped_bar=多系列分组柱状图",
            },
            "title": {"type": "string", "description": "图表标题"},
            "data": {
                "type": "object",
                "description": (
                    "图表数据。单系列: {\"labels\": [...], \"values\": [...]}。"
                    "多系列(grouped_bar): {\"labels\": [...], \"series\": {\"系列1\": [...], \"系列2\": [...]}}。"
                    "饼图会在每个扇区标注百分比。"
                ),
            },
            "xlabel": {"type": "string", "description": "X 轴标签（饼图忽略）", "default": ""},
            "ylabel": {"type": "string", "description": "Y 轴标签（饼图忽略）", "default": ""},
            "color_theme": {
                "type": "string",
                "enum": ["blue", "green", "red", "orange", "purple", "mixed"],
                "description": "配色主题，默认 blue",
                "default": "blue",
            },
        },
        "required": ["chart_type", "title", "data"],
    }

    # 配色方案
    COLOR_MAP = {
        "blue":   ["#4DA6FF", "#1A6DD4", "#0D47A1", "#82C4FF", "#B3DCFF"],
        "green":  ["#34D399", "#1B8A5A", "#065F3E", "#6EE7B7", "#A7F3D0"],
        "red":    ["#F87171", "#D32F2F", "#B71C1C", "#FCA5A5", "#FECACA"],
        "orange": ["#F59E0B", "#E65100", "#BF360C", "#FBBF24", "#FDE68A"],
        "purple": ["#A78BFA", "#6D28D9", "#4C1D95", "#C4B5FD", "#DDD6FE"],
        "mixed":  ["#4DA6FF", "#34D399", "#F59E0B", "#F87171", "#A78BFA", "#82C4FF"],
    }

    def execute(self, chart_type: str, title: str, data: dict,
                xlabel: str = "", ylabel: str = "",
                color_theme: str = "blue", **kwargs) -> dict:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        import numpy as np

        # ── 中文字体设置 ──
        self._setup_chinese_font(fm, matplotlib)

        # ── 创建输出目录 ──
        charts_dir = Path("charts")
        charts_dir.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{chart_type}_{ts}_{uuid.uuid4().hex[:6]}.png"
        save_path = str(charts_dir / filename)

        # ── 选择配色 ──
        colors = self.COLOR_MAP.get(color_theme, self.COLOR_MAP["blue"])

        # ── 按类型生成 ──
        if chart_type == "pie":
            self._draw_pie(title, data, colors, save_path)
        elif chart_type == "bar":
            self._draw_bar(title, data, xlabel, ylabel, colors, save_path)
        elif chart_type == "hbar":
            self._draw_hbar(title, data, xlabel, ylabel, colors, save_path)
        elif chart_type == "line":
            self._draw_line(title, data, xlabel, ylabel, colors, save_path)
        elif chart_type == "grouped_bar":
            self._draw_grouped_bar(title, data, xlabel, ylabel, colors, save_path)
        else:
            return {"success": False, "error": f"不支持的图表类型: {chart_type}"}

        return {
            "success": True,
            "chart_type": chart_type,
            "title": title,
            "path": save_path,
            "filename": filename,
        }

    # ── 各图表绘制方法 ──

    def _setup_chinese_font(self, fm, matplotlib):
        """尝试配置中文字体"""
        try:
            # Windows 常用中文字体
            for font_name in ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC", "WenQuanYi Micro Hei"]:
                found = [f for f in fm.findSystemFonts() if font_name.lower().replace(" ", "") in f.lower().replace(" ", "")]
                if found:
                    matplotlib.rcParams["font.family"] = font_name
                    return
            # 回退: 扫描系统所有字体找一个 CJK 字体
            for f in fm.findSystemFonts():
                if any(kw in f.lower() for kw in ["yahei", "simhei", "simsun", "cjk", "chinese", "noto"]):
                    prop = fm.FontProperties(fname=f)
                    matplotlib.rcParams["font.family"] = prop.get_name()
                    return
        except Exception:
            pass
        # 最终回退：sans-serif
        matplotlib.rcParams["font.family"] = "sans-serif"

    def _draw_bar(self, title, data, xlabel, ylabel, colors, save_path):
        import matplotlib.pyplot as plt
        import numpy as np

        labels = data.get("labels", [])
        values = data.get("values", [])
        if not labels or not values:
            raise ValueError("bar 图表需要 labels 和 values")

        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), 5.5))
        bar_colors = [colors[i % len(colors)] for i in range(len(labels))]
        bars = ax.bar(labels, values, color=bar_colors, edgecolor="#1A1D2A", linewidth=0.5)

        # 柱顶标注
        for bar_obj, val in zip(bars, values):
            ax.text(bar_obj.get_x() + bar_obj.get_width() / 2, bar_obj.get_height() + max(values) * 0.01,
                    str(val), ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=11)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        plt.xticks(rotation=30, ha="right", fontsize=9)
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def _draw_hbar(self, title, data, xlabel, ylabel, colors, save_path):
        import matplotlib.pyplot as plt
        import numpy as np

        labels = data.get("labels", [])
        values = data.get("values", [])
        if not labels or not values:
            raise ValueError("hbar 图表需要 labels 和 values")

        fig, ax = plt.subplots(figsize=(7, max(4, len(labels) * 0.45)))
        bar_colors = [colors[i % len(colors)] for i in range(len(labels))]
        bars = ax.barh(labels, values, color=bar_colors, edgecolor="#1A1D2A", linewidth=0.5)

        for bar_obj, val in zip(bars, values):
            ax.text(bar_obj.get_width() + max(values) * 0.01, bar_obj.get_y() + bar_obj.get_height() / 2,
                    str(val), va="center", fontsize=9, fontweight="bold")

        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=11)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, axis="x", alpha=0.25)
        ax.set_axisbelow(True)
        ax.invert_yaxis()
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def _draw_pie(self, title, data, colors, save_path):
        import matplotlib.pyplot as plt

        labels = data.get("labels", [])
        values = data.get("values", [])
        if not labels or not values:
            raise ValueError("pie 图表需要 labels 和 values")

        fig, ax = plt.subplots(figsize=(7, 7))
        explode = [0.03] * len(labels)

        wedges, texts, autotexts = ax.pie(
            values, labels=None, autopct="%1.1f%%", startangle=140,
            colors=colors[:len(labels)], explode=explode,
            pctdistance=0.6, wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        for t in autotexts:
            t.set_fontsize(10)
            t.set_fontweight("bold")

        # 图例放在右侧
        ax.legend(wedges, [f"{l} ({v})" for l, v in zip(labels, values)],
                  title="图例", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)

        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def _draw_line(self, title, data, xlabel, ylabel, colors, save_path):
        import matplotlib.pyplot as plt
        import numpy as np

        labels = data.get("labels", [])
        values = data.get("values", [])
        if not labels or not values:
            raise ValueError("line 图表需要 labels 和 values")

        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.5), 5.5))
        ax.plot(labels, values, color=colors[0], marker="o", linewidth=2,
                markersize=6, markerfacecolor="white", markeredgewidth=2, markeredgecolor=colors[0])

        # 数据点标注
        for i, (x, y) in enumerate(zip(labels, values)):
            ax.annotate(str(y), (x, y), textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=8, fontweight="bold", color=colors[0])

        ax.fill_between(range(len(labels)), values, alpha=0.08, color=colors[0])
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=11)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)
        plt.xticks(rotation=30, ha="right", fontsize=9)
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def _draw_grouped_bar(self, title, data, xlabel, ylabel, colors, save_path):
        import matplotlib.pyplot as plt
        import numpy as np

        labels = data.get("labels", [])
        series = data.get("series", {})
        if not labels or not series:
            raise ValueError("grouped_bar 图表需要 labels 和 series")

        series_names = list(series.keys())
        n_groups = len(labels)
        n_series = len(series_names)
        x = np.arange(n_groups)
        width = 0.8 / n_series

        fig, ax = plt.subplots(figsize=(max(10, n_groups * 1.0), 6))

        for i, sname in enumerate(series_names):
            vals = series[sname]
            offset = (i - (n_series - 1) / 2) * width
            bar_color = colors[i % len(colors)]
            bars = ax.bar(x + offset, vals, width, label=sname, color=bar_color,
                          edgecolor="#1A1D2A", linewidth=0.3)
            # 柱顶标注（仅在系列数 ≤3 时标注，避免拥挤）
            if n_series <= 3:
                for bar_obj, val in zip(bars, vals):
                    ax.text(bar_obj.get_x() + bar_obj.get_width() / 2,
                            bar_obj.get_height() + max(max(v) for v in series.values()) * 0.01,
                            str(val), ha="center", va="bottom", fontsize=7, fontweight="bold")

        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=11)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)


# ============================================================
# 注册表
# ============================================================

_ALL_TOOLS: list[BaseTool] = []

def _register():
    global _ALL_TOOLS
    _ALL_TOOLS = [
        QueryCpuMonitor(),
        QueryMemoryMonitor(),
        SendAlertEmail(),
        QueryCmdb(),
        CreateTicket(),
        RestartService(),
        QueryLogs(),
        GenerateChart(),
    ]

_register()


def get_all_tools() -> list[BaseTool]:
    return _ALL_TOOLS


def get_tool_by_name(name: str) -> BaseTool | None:
    for tool in _ALL_TOOLS:
        if tool.name == name:
            return tool
    return None


def get_tools_for_openai() -> list[dict]:
    return [t.to_openai_tool() for t in _ALL_TOOLS]
