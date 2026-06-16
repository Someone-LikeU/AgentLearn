# encoding: utf-8
import asyncio
import logging
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from external_mcp_manager import ExternalMCPManager, ExternalMCPServerConfig, _ToolRoute


class ExternalMCPManagerTest(unittest.TestCase):
    def test_streamable_http_transport_logger_is_suppressed(self):
        logger = logging.getLogger("mcp.client.streamable_http")

        self.assertFalse(logger.propagate)
        self.assertGreaterEqual(logger.level, logging.CRITICAL)

    def test_external_tool_timeout_returns_structured_error(self):
        manager = ExternalMCPManager.__new__(ExternalMCPManager)
        server = ExternalMCPServerConfig(name="exa", timeout_seconds=10)
        manager._started = True
        manager._tools = []
        manager._tool_counts_by_server = {}
        manager._start_errors = {}
        manager._tool_routes = {"web_fetch_exa": _ToolRoute(server=server, raw_name="web_fetch_exa")}
        manager.start = lambda: None
        manager._tool_timeout_seconds = lambda _server, _raw_name: 0.01

        async def slow_call(*_args, **_kwargs):
            await asyncio.sleep(1)
            return {"ok": True}

        manager._call_tool_on_server = slow_call

        result = manager.call_tool("web_fetch_exa", {"urls": ["https://example.com"]})

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_type"], "tool_timeout")
        self.assertEqual(result["timeout_seconds"], 0.01)
        self.assertIn("timed out", result["message"])


if __name__ == "__main__":
    unittest.main()
