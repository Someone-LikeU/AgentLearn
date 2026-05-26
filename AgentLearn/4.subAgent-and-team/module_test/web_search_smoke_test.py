# encoding: utf-8
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.tool_manager import DDGS, ToolManager, ToolManagerConfig, AgentToolHandlers
from tools.tool_names import ToolNameConstant


class SmokeSummaryClient:
    """用于 smoke test 的假模型客户端，避免测试搜索连通性时额外依赖真实模型。"""

    class _Chat:
        class _Completions:
            @staticmethod
            def create(**kwargs):
                class _Message:
                    content = "SMOKE_SUMMARY: web search returned results"

                class _Choice:
                    message = _Message()

                class _Resp:
                    choices = [_Choice()]

                return _Resp()

        completions = _Completions()

    chat = _Chat()


def run_web_search_tool_smoke_test(query: str = "OpenAI", max_results: int = 1) -> dict:
    """
    通过 ToolManager 的真实工具分发路径测试 WEB_SEARCH 是否可用。
    :param query: 搜索关键词
    :param max_results: 搜索结果数量
    :return: 测试结果字典
    """
    tool_manager = ToolManager(
        config=ToolManagerConfig(
            project_root=str(PROJECT_ROOT),
            client=SmokeSummaryClient(),
            model="smoke-test-model",
            temperature=0,
            is_main_agent=False,
        ),
        handlers=AgentToolHandlers(),
        mcp_client=None,
    )

    start_time = time.perf_counter()
    # 这里必须走 available_functions，覆盖 Agent 实际调用工具的路径。
    result = tool_manager.available_functions[ToolNameConstant.WEB_SEARCH](
        query=query,
        max_results=max_results,
    )
    elapsed_seconds = round(time.perf_counter() - start_time, 3)
    result_text = str(result)
    ok = (
        result_text.startswith("SMOKE_SUMMARY:")
        and "WEB_SEARCH 执行失败" not in result_text
        and "未搜索到" not in result_text
    )

    smoke_result = {
        "ok": ok,
        "elapsed_seconds": elapsed_seconds,
        "query": query,
        "max_results": max_results,
        "result": result_text,
    }
    print(smoke_result)
    return smoke_result


def run_duckduckgo_raw_search_test(query: str = "OpenAI", max_results: int = 1) -> dict:
    """
    直接调用 ddgs，查看搜索端点返回的原始结果。
    :param query: 搜索关键词
    :param max_results: 搜索结果数量
    :return: 原始搜索测试结果
    """
    max_results = max(1, min(int(max_results), 10))
    if DDGS is None:
        result = {
            "ok": False,
            "phase": "duckduckgo_raw_search",
            "error": "missing dependency ddgs",
        }
        print(result)
        return result

    start_time = time.perf_counter()
    try:
        # 这里直接打到 DuckDuckGo，能看到标题、摘要和链接，方便判断搜索质量。
        with DDGS() as ddgs:
            raw_items = list(ddgs.text(query, max_results=max_results))
        elapsed_seconds = round(time.perf_counter() - start_time, 3)
        normalized_items = [
            {
                "title": item.get("title", ""),
                "body": item.get("body", ""),
                "href": item.get("href", ""),
            }
            for item in raw_items
        ]
        result = {
            "ok": bool(normalized_items),
            "phase": "duckduckgo_raw_search",
            "elapsed_seconds": elapsed_seconds,
            "query": query,
            "max_results": max_results,
            "raw_results": normalized_items,
        }
    except Exception as error:
        elapsed_seconds = round(time.perf_counter() - start_time, 3)
        result = {
            "ok": False,
            "phase": "duckduckgo_raw_search",
            "elapsed_seconds": elapsed_seconds,
            "query": query,
            "max_results": max_results,
            "error": str(error),
        }
    print(result)
    return result


def run_bing_raw_search_test(query: str = "OpenAI", max_results: int = 1) -> dict:
    """
    通过 ToolManager 的 Bing fallback 搜索方法查看原始结果。
    :param query: 搜索关键词
    :param max_results: 搜索结果数量
    :return: Bing 原始搜索测试结果
    """
    tool_manager = ToolManager(
        config=ToolManagerConfig(
            project_root=str(PROJECT_ROOT),
            client=SmokeSummaryClient(),
            model="smoke-test-model",
            temperature=0,
            is_main_agent=False,
        ),
        handlers=AgentToolHandlers(),
        mcp_client=None,
    )
    start_time = time.perf_counter()
    try:
        raw_items = tool_manager._search_with_bing(query, max_results)
        result = {
            "ok": bool(raw_items),
            "phase": "bing_raw_search",
            "elapsed_seconds": round(time.perf_counter() - start_time, 3),
            "query": query,
            "max_results": max_results,
            "raw_results": raw_items,
        }
    except Exception as error:
        result = {
            "ok": False,
            "phase": "bing_raw_search",
            "elapsed_seconds": round(time.perf_counter() - start_time, 3),
            "query": query,
            "max_results": max_results,
            "error": str(error),
        }
    print(result)
    return result


def run_web_search_full_smoke_test(query: str = "OpenAI", max_results: int = 1) -> dict:
    """
    分两步测试 WEB_SEARCH：先看 DuckDuckGo 原始结果，再测试 ToolManager 分发路径。
    :param query: 搜索关键词
    :param max_results: 搜索结果数量
    :return: 汇总测试结果
    """
    raw_result = run_duckduckgo_raw_search_test(query, max_results)
    bing_result = run_bing_raw_search_test(query, max_results)
    tool_result = run_web_search_tool_smoke_test(query, max_results)
    result = {
        "ok": (bool(raw_result.get("ok")) or bool(bing_result.get("ok"))) and bool(tool_result.get("ok")),
        "duckduckgo_raw_search": raw_result,
        "bing_raw_search": bing_result,
        "tool_path": tool_result,
    }
    print(result)
    return result


if __name__ == "__main__":
    search_query = sys.argv[1] if len(sys.argv) > 1 else "OpenAI"
    search_max_results = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    mode = sys.argv[3].lower() if len(sys.argv) > 3 else "full"
    if mode == "bing":
        run_bing_raw_search_test(search_query, search_max_results)
    elif mode in ("duckduckgo", "ddg"):
        run_duckduckgo_raw_search_test(search_query, search_max_results)
    elif mode == "tool":
        run_web_search_tool_smoke_test(search_query, search_max_results)
    else:
        run_web_search_full_smoke_test(search_query, search_max_results)
