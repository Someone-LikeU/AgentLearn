# encoding: utf-8
import json
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    # 搜索结果里可能包含 GBK 控制台无法编码的字符，手动运行时统一按 UTF-8 输出。
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from external_mcp_manager import ExternalMCPManager


EXA_QUERY = "F1方程式2026赛季摩纳哥站结果"
TAVILY_QUERY = "F1方程式2026赛季摩纳哥站结果"


def _build_search_arguments(parameters: dict, query: str) -> dict:
    properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
    args = {"query": query}
    # 不同搜索 MCP 的分页参数命名不完全一致，这里只补常见且安全的结果数量参数。
    for name in ("numResults", "num_results", "max_results", "limit"):
        if name in properties:
            args[name] = 5
            break
    return args


def _pick_search_tool(manager: ExternalMCPManager, prefix: str) -> dict:
    tools = manager.list_tools_by_server(prefix)
    preferred_keywords = ("web_search", "search")
    for keyword in preferred_keywords:
        for tool in tools:
            if keyword in str(tool.get("name", "")).lower():
                return tool
    raise RuntimeError(f"未找到 {prefix} 搜索工具，可用工具：{[tool.get('name') for tool in tools]}")


def _run_provider_search(prefix: str, query: str):
    manager = ExternalMCPManager(
        PROJECT_ROOT / "tools" / "external_mcp_servers.json",
        refresh_tools=True,
    )
    manager.start()
    try:
        tool = _pick_search_tool(manager, prefix)
    except RuntimeError as error:
        print(str(error))
        errors = manager.get_start_errors()
        if errors:
            print("MCP 加载错误：")
            print(json.dumps(errors, ensure_ascii=False, indent=2))
        return None
    if not query:
        print(f"{prefix} 搜索内容为空，请先在脚本里填写 query。")
        print(f"将使用的工具：{tool['name']}")
        print("参数 schema：")
        print(json.dumps(tool.get("parameters", {}), ensure_ascii=False, indent=2))
        return None
    result = manager.call_tool(tool["name"], _build_search_arguments(tool.get("parameters", {}), query))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def test_exa_search(query: str = EXA_QUERY):
    result = _run_provider_search("exa", query)
    assert result is None or result.get("ok") is True


def test_tavily_search(query: str = TAVILY_QUERY):
    result = _run_provider_search("tavily", query)
    assert result is None or result.get("ok") is True


if __name__ == "__main__":
    test_exa_search()
    test_tavily_search()
