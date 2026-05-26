# encoding: utf-8
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 测试可能从 module_test 或仓库根目录启动，这里确保阶段目录可被导入。
project_root_str = str(PROJECT_ROOT)
if project_root_str in sys.path:
    sys.path.remove(project_root_str)
sys.path.insert(0, project_root_str)

import tempfile
import unittest
from unittest.mock import Mock, patch

from tools.tool_manager import ToolManager, ToolManagerConfig, AgentToolHandlers
from tools.tool_scheduler import ToolScheduler, ToolCallTask
from tools.tool_names import ToolNameConstant


class DummyClient:
    class _Chat:
        class _Completions:
            @staticmethod
            def create(**kwargs):
                class _Message:
                    content = "总结结果"

                class _Choice:
                    message = _Message()

                class _Resp:
                    choices = [_Choice()]

                return _Resp()

        completions = _Completions()

    chat = _Chat()


class DummyMCPClient:
    def list_tools(self):
        return [
            {"name": "QUERY_WEATHER", "description": "query weather", "parameters": {"type": "object", "properties": {}}},
            {"name": "QUERY_FLIGHT_TICKETS", "description": "query flights", "parameters": {"type": "object", "properties": {}}},
        ]

    def call_tool(self, tool_name, kwargs):
        return {"tool": tool_name, "args": kwargs}


class ToolManagerLocalToolsTest(unittest.TestCase):
    def setUp(self):
        self.project_root = Path(__file__).resolve().parents[1]
        self.tm = ToolManager(
            config=ToolManagerConfig(
                project_root=str(self.project_root),
                client=DummyClient(),
                model="dummy-model",
                temperature=0,
                is_main_agent=False,
            ),
            handlers=AgentToolHandlers(),
            mcp_client=None,
        )

    def test_read_write_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a.txt"
            self.tm.write_file(str(p), "line1\nline2\n")
            text = self.tm.read_file(str(p))
            self.assertIn("line1", text)
            self.assertIn("line2", text)

    def test_edit(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "b.txt"
            p.write_text("hello world", encoding="utf-8")
            result = self.tm.edit(str(p), "world", "python")
            self.assertIn("Successfully edited", result)
            self.assertEqual(p.read_text(encoding="utf-8"), "hello python")

    def test_glob_and_list_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = Path(tmp) / "x.txt"
            p2 = Path(tmp) / "y.log"
            p1.write_text("1", encoding="utf-8")
            p2.write_text("2", encoding="utf-8")
            out = self.tm.glob(f"{tmp}/*")
            self.assertIn(str(p1), out)
            listing = self.tm.list_dir(tmp)
            self.assertIn("[file] x.txt", listing)
            self.assertIn("[file] y.log", listing)

    def test_grep(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "g.txt"
            p.write_text("alpha\nbeta\n", encoding="utf-8")
            out = self.tm.grep("alpha", tmp)
            self.assertIn("alpha", out)

    def test_execute_bash_safe(self):
        out = self.tm.execute_bash("echo hello")
        self.assertIn("hello", out)

    def test_execute_bash_dangerous(self):
        out = self.tm.execute_bash("rm -rf /")
        self.assertIn("dangerous", out)

    def test_web_search(self):
        search_results = [{"title": "test result", "body": "b", "href": "https://example.com", "source": "duckduckgo"}]
        with patch.object(
            self.tm.web_tool,
            "_web_search_backend_order",
            return_value=[("duckduckgo", Mock(return_value=search_results))],
        ), patch.object(
            self.tm.web_tool,
            "web_extract",
            return_value=[{"ok": True, "href": "https://example.com", "text": "detail"}],
        ):
            out = self.tm.web_search("test", max_results=1)
        self.assertEqual(out, "总结结果")
        self.assertEqual(self.tm.last_web_search_results["extracted_pages"][0]["text"], "detail")

    def test_web_search_via_available_functions(self):
        search_results = [{"title": "test result", "body": "b", "href": "https://example.com", "source": "duckduckgo"}]
        # 覆盖 Agent 的真实工具调用路径：function_impl(**function_args)。
        with patch.object(
            self.tm.web_tool,
            "_web_search_backend_order",
            return_value=[("duckduckgo", Mock(return_value=search_results))],
        ), patch.object(
            self.tm.web_tool,
            "web_extract",
            return_value=[{"ok": True, "href": "https://example.com", "text": "detail"}],
        ):
            out = self.tm.available_functions[ToolNameConstant.WEB_SEARCH](query="test", max_results=1)
        self.assertEqual(out, "总结结果")

    def test_web_search_fallback_to_bing(self):
        duck_search = Mock(side_effect=TimeoutError("timed out"))
        bing_search = Mock(
            return_value=[{"title": "test Bing 结果", "body": "摘要", "href": "https://example.com", "source": "bing"}]
        )
        with patch.object(
            self.tm.web_tool,
            "_web_search_backend_order",
            return_value=[("duckduckgo", duck_search), ("bing", bing_search)],
        ), patch.object(
            self.tm.web_tool,
            "web_extract",
            return_value=[{"ok": True, "href": "https://example.com", "text": "detail"}],
        ):
            out = self.tm.available_functions[ToolNameConstant.WEB_SEARCH](query="test", max_results=1)
        bing_search.assert_called_once_with("test", 1)
        self.assertEqual(out, "总结结果")

    def test_web_search_default_backend_prefers_bing(self):
        backend_names = [name for name, _ in self.tm._web_search_backend_order()]
        self.assertEqual(backend_names[:2], ["bing", "duckduckgo"])

    def test_web_extract_quotes_chinese_url(self):
        raw_url = "https://example.com/zh-hans/2026年世界一级方程式锦标赛?kw=结果"
        quoted_url = self.tm.web_tool._quote_url_for_request(raw_url)
        self.assertIn("2026%E5%B9%B4", quoted_url)
        self.assertIn("kw=%E7%BB%93%E6%9E%9C", quoted_url)

    def test_web_search_filters_irrelevant_bing_results(self):
        duck_search = Mock(
            return_value=[{"title": "2026赛季F1大奖赛积分榜", "body": "F1 积分", "href": "https://example.com", "source": "duckduckgo"}]
        )
        bing_search = Mock(
            return_value=[{"title": "2026年_百度百科", "body": "十五五规划", "href": "https://baike.baidu.com/item/2026年", "source": "bing"}]
        )
        with patch.object(
            self.tm.web_tool,
            "_web_search_backend_order",
            return_value=[("bing", bing_search), ("duckduckgo", duck_search)],
        ), patch.object(
            self.tm.web_tool,
            "web_extract",
            return_value=[{"ok": True, "href": "https://example.com", "text": "detail"}],
        ):
            out = self.tm.available_functions[ToolNameConstant.WEB_SEARCH](
                query="2026赛季F1大奖赛积分榜",
                max_results=1,
            )
        self.assertEqual(out, "总结结果")
        self.assertEqual(self.tm.last_web_search_results["backend"], "duckduckgo")

    def test_mcp_capability_parallel_read(self):
        tm = ToolManager(
            config=ToolManagerConfig(
                project_root=str(self.project_root),
                client=DummyClient(),
                model="dummy-model",
                temperature=0,
                is_main_agent=False,
            ),
            handlers=AgentToolHandlers(),
            mcp_client=DummyMCPClient(),
        )
        self.assertTrue(tm.is_parallel_read_tool("QUERY_WEATHER"))
        self.assertTrue(tm.is_parallel_read_tool("QUERY_FLIGHT_TICKETS"))

    def test_scheduler_v2_scope_isolation(self):
        profile_map = {
            "A": {"is_read_only": True, "is_concurrency_safe": True, "side_effect_scope": "network"},
            "B": {"is_read_only": True, "is_concurrency_safe": True, "side_effect_scope": "network"},
            "C": {"is_read_only": False, "is_concurrency_safe": False, "side_effect_scope": "filesystem"},
            "D": {"is_read_only": True, "is_concurrency_safe": True, "side_effect_scope": "runtime"},
        }
        scheduler = ToolScheduler(get_profile=lambda name: profile_map[name])
        tasks = [
            ToolCallTask(tool_call_id="1", function_name="A", raw_arguments="{}", function_args={}),
            ToolCallTask(tool_call_id="2", function_name="B", raw_arguments="{}", function_args={}),
            ToolCallTask(tool_call_id="3", function_name="C", raw_arguments="{}", function_args={}),
            ToolCallTask(tool_call_id="4", function_name="D", raw_arguments="{}", function_args={}),
        ]
        batches = scheduler.plan_batches(tasks)
        self.assertEqual(len(batches), 3)
        self.assertTrue(batches[0].parallel)
        self.assertEqual([t.function_name for t in batches[0].tasks], ["A", "B"])
        self.assertFalse(batches[1].parallel)
        self.assertEqual([t.function_name for t in batches[1].tasks], ["C"])
        self.assertTrue(batches[2].parallel)
        self.assertEqual([t.function_name for t in batches[2].tasks], ["D"])


def debug_web_search(query: str, max_results: int = 5) -> dict:
    """
    手动调试 WEB_SEARCH 工具，打印各搜索后端返回的完整结果。
    :param query: 搜索内容
    :param max_results: 每个后端最多返回的结果数
    :return: 调试结果字典
    """
    max_results = max(1, min(int(max_results), 10))
    tool_manager = ToolManager(
        config=ToolManagerConfig(
            project_root=str(PROJECT_ROOT),
            client=DummyClient(),
            model="debug-model",
            temperature=0,
            is_main_agent=False,
        ),
        handlers=AgentToolHandlers(),
        mcp_client=None,
    )

    backend_results = []
    first_success_results = []
    for backend_name, search_func in tool_manager.web_tool._web_search_backend_order():
        start_time = time.perf_counter()
        try:
            # 直接调用搜索后端，绕过模型总结，方便看到标题、摘要、链接等原始搜索结果。
            results = search_func(query, max_results)
            if results and not first_success_results:
                first_success_results = results
            backend_results.append(
                {
                    "backend": backend_name,
                    "ok": bool(results),
                    "elapsed_seconds": round(time.perf_counter() - start_time, 3),
                    "result_count": len(results),
                    "results": results,
                }
            )
        except Exception as error:
            backend_results.append(
                {
                    "backend": backend_name,
                    "ok": False,
                    "elapsed_seconds": round(time.perf_counter() - start_time, 3),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )

    if first_success_results:
        extract_start_time = time.perf_counter()
        try:
            extracted_pages = tool_manager.web_tool.web_extract(
                first_success_results,
                max_pages=min(max_results, 5),
            )
            extract_result = {
                "ok": True,
                "elapsed_seconds": round(time.perf_counter() - extract_start_time, 3),
                "pages": extracted_pages,
            }
        except Exception as error:
            extract_result = {
                "ok": False,
                "elapsed_seconds": round(time.perf_counter() - extract_start_time, 3),
                "error_type": type(error).__name__,
                "error": str(error),
            }
    else:
        extract_result = {"ok": False, "reason": "no successful search results"}

    tool_start_time = time.perf_counter()
    tool_result = tool_manager.available_functions[ToolNameConstant.WEB_SEARCH](
        query=query,
        max_results=max_results,
    )
    debug_result = {
        "query": query,
        "max_results": max_results,
        "backend_results": backend_results,
        "extract_result": extract_result,
        "tool_path": {
            "elapsed_seconds": round(time.perf_counter() - tool_start_time, 3),
            "result": tool_result,
            "raw_search_results": tool_manager.last_web_search_results,
        },
    }
    print(json.dumps(debug_result, ensure_ascii=False, indent=2))
    return debug_result


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "debug_web_search":
        search_query = sys.argv[2] if len(sys.argv) > 2 else "OpenAI"
        search_max_results = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        debug_web_search(search_query, search_max_results)
    else:
        unittest.main()
