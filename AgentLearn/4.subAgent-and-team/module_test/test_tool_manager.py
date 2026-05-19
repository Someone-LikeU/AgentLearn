# encoding: utf-8
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 测试可能从 module_test 或仓库根目录启动，这里确保阶段目录可被导入。
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "duckduckgo_search" not in sys.modules:
    mock_module = types.ModuleType("duckduckgo_search")

    class _StubDDGS:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def text(self, query, max_results=10):
            return []

    mock_module.DDGS = _StubDDGS
    sys.modules["duckduckgo_search"] = mock_module

import tempfile
import unittest
from unittest.mock import patch

from tools.tool_manager import ToolManager, ToolManagerConfig, AgentToolHandlers
from tools.tool_scheduler import ToolScheduler, ToolCallTask


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

    @patch("tools.tool_manager.DDGS")
    def test_web_search(self, ddgs_cls):
        ddgs = ddgs_cls.return_value.__enter__.return_value
        ddgs.text.return_value = [{"title": "t", "body": "b", "href": "u"}]
        out = self.tm.web_search("test", max_results=1)
        self.assertEqual(out, "总结结果")

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


if __name__ == "__main__":
    unittest.main()
