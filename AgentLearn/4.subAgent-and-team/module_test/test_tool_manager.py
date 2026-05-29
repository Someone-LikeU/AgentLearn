# encoding: utf-8
import json
import sys
import time
from pathlib import Path
from threading import Lock

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

    def test_bash_auto_approve_status_is_owned_by_tool_manager(self):
        self.assertEqual(self.tm.bash_approve_status_text(), "自动确认（无需手动确认）")

        self.tm.set_bash_auto_approve(False)

        self.assertEqual(self.tm.bash_approve_status_text(), "手动确认（每次需确认）")
        with patch("builtins.input", return_value="n"):
            out = self.tm.execute_bash("echo skipped")
        self.assertIn("skipped by user", out)

        self.tm.set_bash_auto_approve(True)
        self.assertEqual(self.tm.bash_approve_status_text(), "自动确认（无需手动确认）")

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

    def test_web_search_default_backend_uses_duckduckgo(self):
        backend_names = [name for name, _ in self.tm._web_search_backend_order()]
        self.assertEqual(backend_names, ["duckduckgo"])

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

    def test_web_search_fallback_uses_raw_results_when_filter_empty(self):
        search_results = [
            {
                "title": "2024年夏季奥林匹克运动会乒乓球男子单打比赛",
                "body": "樊振东 在决赛中获得冠军",
                "href": "https://example.com/table-tennis",
                "source": "duckduckgo",
            }
        ]
        with patch.object(
            self.tm.web_tool,
            "_web_search_backend_order",
            return_value=[("duckduckgo", Mock(return_value=search_results))],
        ), patch.object(
            self.tm.web_tool,
            "_filter_relevant_results",
            return_value=[],
        ), patch.object(
            self.tm.web_tool,
            "web_extract",
            return_value=[{"ok": True, "href": "https://example.com/table-tennis", "text": "detail"}],
        ):
            out = self.tm.available_functions[ToolNameConstant.WEB_SEARCH](
                query="2024年世界乒乓球锦标赛男单冠军",
                max_results=1,
            )
        self.assertEqual(out, "总结结果")
        self.assertTrue(self.tm.last_web_search_results["fallback_used"])
        self.assertEqual(self.tm.last_web_search_results["extracted_pages"][0]["text"], "detail")

    def test_web_summary_input_avoids_repeated_body_for_extracted_pages(self):
        results = [
            {
                "title": "已抓取页面",
                "body": "这段摘要不应该重复发送",
                "href": "https://example.com/ok",
                "source": "duckduckgo",
            },
            {
                "title": "抓取失败页面",
                "body": "失败页面保留搜索摘要",
                "href": "https://example.com/fail",
                "source": "duckduckgo",
            },
        ]
        pages = [
            {"ok": True, "href": "https://example.com/ok", "title": "已抓取页面", "text": "正文详情"},
            {
                "ok": False,
                "href": "https://example.com/fail",
                "title": "抓取失败页面",
                "error_type": "HTTPError",
                "error": "HTTP Error 403: Forbidden",
            },
        ]
        search_text = self.tm.web_tool._format_search_results(results, pages)
        detail_text = self.tm.web_tool._format_extracted_pages(pages)

        self.assertNotIn("这段摘要不应该重复发送", search_text)
        self.assertIn("失败页面保留搜索摘要", search_text)
        self.assertIn("读取失败: HTTPError", detail_text)
        self.assertNotIn("HTTP Error 403: Forbidden", detail_text)

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

    def test_compare_parallel_and_serial_tool_call_results(self):
        report = compare_parallel_and_serial_tool_calls()
        self.assertTrue(report["normalized_result_equal"])
        self.assertEqual(report["parallel_results_normalized"], report["serial_results_normalized"])
        self.assertTrue(report["planned_batches"][0]["parallel"])


def compare_parallel_and_serial_tool_calls(output_dir: Path | None = None) -> dict:
    """
    对比调度器并发执行和手动串行执行同一组工具调用的结果。
    只使用 ToolManager 中已标记为只读且并发安全的本地工具，便于断点观察真实调度路径。
    """
    output_dir = output_dir or PROJECT_ROOT / "module_test" / "tool_call_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_test_path = output_dir / "test_write_tool.txt"
    tool_manager = ToolManager(
        config=ToolManagerConfig(
            project_root=str(PROJECT_ROOT),
            client=DummyClient(),
            model="debug-model",
            temperature=0,
            is_main_agent=False,
        ),
        handlers=AgentToolHandlers(),
        mcp_client=DummyMCPClient(),
    )
    tasks = [
        ToolCallTask(
            tool_call_id="call_read_file",
            function_name=ToolNameConstant.READ_FILE,
            raw_arguments=json.dumps(
                {"path": str(PROJECT_ROOT / "module_test" / "test_tool_manager.py"), "offset": 0, "limit": 5},
                ensure_ascii=False,
            ),
            function_args={
                "path": str(PROJECT_ROOT / "module_test" / "test_tool_manager.py"),
                "offset": 0,
                "limit": 5,
            },
        ),
        ToolCallTask(
            tool_call_id="call_list_dir",
            function_name=ToolNameConstant.LIST_DIR,
            raw_arguments=json.dumps({"path": str(PROJECT_ROOT / "module_test")}, ensure_ascii=False),
            function_args={"path": str(PROJECT_ROOT / "module_test")},
        ),
        ToolCallTask(
            tool_call_id="call_glob",
            function_name=ToolNameConstant.GLOB,
            raw_arguments=json.dumps({"pattern": str(PROJECT_ROOT / "module_test" / "*.py")}, ensure_ascii=False),
            function_args={"pattern": str(PROJECT_ROOT / "module_test" / "*.py")},
        ),
        ToolCallTask(
            tool_call_id="call_read_agent_teams",
            function_name=ToolNameConstant.READ_FILE,
            raw_arguments=json.dumps(
                {"path": str(PROJECT_ROOT / "agent_Teams.py"), "offset": 0, "limit": 5},
                ensure_ascii=False,
            ),
            function_args={"path": str(PROJECT_ROOT / "agent_Teams.py"), "offset": 0, "limit": 5},
        ),
        ToolCallTask(
            tool_call_id="call_get_time",
            function_name=ToolNameConstant.GET_TIME,
            raw_arguments="{}",
            function_args={},
        ),
        ToolCallTask(
            tool_call_id="call_write_file",
            function_name=ToolNameConstant.WRITE_FILE,
            raw_arguments=json.dumps(
                {"path": str(write_test_path), "content": "test_123"},
                ensure_ascii=False,
            ),
            function_args={"path": str(write_test_path), "content": "test_123"},
        ),
        ToolCallTask(
            tool_call_id="call_query_flight",
            function_name="QUERY_FLIGHT_TICKETS",
            raw_arguments=json.dumps(
                {"from_city": "上海", "to_city": "北京", "direct": False},
                ensure_ascii=False,
            ),
            function_args={"from_city": "上海", "to_city": "北京", "direct": False},
        ),
        ToolCallTask(
            tool_call_id="call_query_weather",
            function_name="QUERY_WEATHER",
            raw_arguments=json.dumps({"city": "北京", "days": 3}, ensure_ascii=False),
            function_args={"city": "北京", "days": 3},
        ),
        ToolCallTask(
            tool_call_id="call_read_written_file",
            function_name=ToolNameConstant.READ_FILE,
            raw_arguments=json.dumps({"path": str(write_test_path)}, ensure_ascii=False),
            function_args={"path": str(write_test_path)},
        ),
    ]
    scheduler = ToolScheduler(get_profile=tool_manager.get_tool_runtime_profile, max_workers=3)
    delay_by_call_id = {
        "call_read_file": 0.20,
        "call_list_dir": 0.05,
        "call_glob": 0.10,
        "call_read_agent_teams": 0.15,
        "call_get_time": 0.03,
        "call_write_file": 0.02,
        "call_query_flight": 0.12,
        "call_query_weather": 0.08,
        "call_read_written_file": 0.04,
    }
    parallel_started: list[str] = []
    parallel_finished: list[str] = []
    serial_started: list[str] = []
    serial_finished: list[str] = []
    execution_order_lock = Lock()

    def invoke_debug_tool(task: ToolCallTask, *, mode: str) -> dict:
        # 用不同耗时模拟乱序完成，验证 execute_batches 最终仍按原始 tool_call 顺序回填结果。
        with execution_order_lock:
            if mode == "parallel":
                parallel_started.append(task.tool_call_id)
            else:
                serial_started.append(task.tool_call_id)
        time.sleep(delay_by_call_id[task.tool_call_id])
        function_impl = tool_manager.available_functions[task.function_name]
        response = function_impl(**task.function_args)
        with execution_order_lock:
            if mode == "parallel":
                parallel_finished.append(task.tool_call_id)
            else:
                serial_finished.append(task.tool_call_id)
        return {
            "tool_call_id": task.tool_call_id,
            "function_name": task.function_name,
            "response": response,
        }

    planned_batches = scheduler.plan_batches(tasks)
    scheduled_execution_order = [
        {
            "batch_index": batch_index,
            "parallel": batch.parallel,
            "task_order": [
                {
                    "tool_call_id": task.tool_call_id,
                    "function_name": task.function_name,
                }
                for task in batch.tasks
            ],
        }
        for batch_index, batch in enumerate(planned_batches, 1)
    ]
    print("调度执行顺序:")
    for batch in scheduled_execution_order:
        mode = "并发" if batch["parallel"] and len(batch["task_order"]) > 1 else "串行"
        names = " -> ".join(item["function_name"] for item in batch["task_order"])
        print(f"  batch {batch['batch_index']} [{mode}]: {names}")

    parallel_start = time.perf_counter()
    parallel_pairs = scheduler.execute_batches(
        planned_batches,
        lambda task: invoke_debug_tool(task, mode="parallel"),
    )
    parallel_elapsed = time.perf_counter() - parallel_start

    serial_start = time.perf_counter()
    serial_pairs = [(task, invoke_debug_tool(task, mode="serial")) for task in tasks]
    serial_elapsed = time.perf_counter() - serial_start

    def normalize(pairs: list[tuple[ToolCallTask, dict]]) -> list[dict]:
        return [
            {
                "task_id": task.tool_call_id,
                "function_name": task.function_name,
                "result": result,
            }
            for task, result in pairs
        ]

    def normalize_dynamic_results(results: list[dict]) -> list[dict]:
        normalized = []
        for item in results:
            copy_item = json.loads(json.dumps(item, ensure_ascii=False))
            if copy_item["function_name"] == ToolNameConstant.GET_TIME:
                copy_item["result"]["response"] = "<dynamic-current-time>"
            normalized.append(copy_item)
        return normalized

    parallel_results = normalize(parallel_pairs)
    serial_results = normalize(serial_pairs)
    parallel_results_normalized = normalize_dynamic_results(parallel_results)
    serial_results_normalized = normalize_dynamic_results(serial_results)
    report = {
        "tool_profiles": {
            task.function_name: tool_manager.get_tool_runtime_profile(task.function_name)
            for task in tasks
        },
        "scheduled_execution_order": scheduled_execution_order,
        "parallel_started_order": parallel_started,
        "parallel_finished_order": parallel_finished,
        "serial_started_order": serial_started,
        "serial_finished_order": serial_finished,
        "planned_batches": [
            {
                "parallel": batch.parallel,
                "tasks": [task.function_name for task in batch.tasks],
            }
            for batch in planned_batches
        ],
        "parallel_elapsed_seconds": round(parallel_elapsed, 4),
        "serial_elapsed_seconds": round(serial_elapsed, 4),
        "parallel_results": parallel_results,
        "serial_results": serial_results,
        "result_equal": parallel_results == serial_results,
        "parallel_results_normalized": parallel_results_normalized,
        "serial_results_normalized": serial_results_normalized,
        "normalized_result_equal": parallel_results_normalized == serial_results_normalized,
    }

    # 调试输出统一落到 module_test/tool_call_test，避免散落到项目根目录。
    output_path = output_dir / "parallel_vs_serial_tool_calls.json"
    report["output_path"] = str(output_path)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


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
            print(f"使用backend_name: {backend_name}执行搜索。。。")
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
            print(f"发生异常， {error}")
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
            print("开始调用web_extract")
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
    print("debug结果：")
    print(json.dumps(debug_result, ensure_ascii=False, indent=2))
    return debug_result


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "debug_web_search":
        search_query = sys.argv[2] if len(sys.argv) > 2 else "OpenAI"
        search_max_results = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        debug_web_search(search_query, search_max_results)
    elif len(sys.argv) > 1 and sys.argv[1] == "debug_tool_call_parallel":
        debug_report = compare_parallel_and_serial_tool_calls()
        print(json.dumps(debug_report, ensure_ascii=False, indent=2))
    else:
        unittest.main()
