"""
Agent 工具集 — Mock 实现，生产环境替换实现即可

所有工具继承 BaseTool，实现 execute 方法。
get_all_tools() 返回已注册工具列表，供 Agent 使用。
"""

import time
import uuid
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
