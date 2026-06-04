# encoding: utf-8
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_Teams import Agent
from session_manager import SessionManager


class AgentStatusTest(unittest.TestCase):
    def _agent(self, output: io.StringIO):
        agent = Agent.__new__(Agent)
        agent.model = "test-model"
        agent._max_context_tokens = 258000
        agent.messages = [{"role": "user", "content": "hello"}]
        agent._all_tools = []
        agent.console = Console(file=output, force_terminal=False, color_system=None, width=120)
        return agent

    def test_status_uses_latest_total_tokens_as_context_usage(self):
        class FakeSessionManager:
            current_session_id = "session_1"

            def calculate_session_usage(self, session_id, include_deleted=False):
                return {
                    "prompt_tokens": 7607,
                    "completion_tokens": 58,
                    "total_tokens": 7665,
                    "response_count": 3,
                    "has_real_usage": True,
                }

        output = io.StringIO()
        agent = self._agent(output)
        agent.session_manager = FakeSessionManager()

        agent._handle_cmd_status(())

        text = output.getvalue()
        self.assertIn("上下文 Token：7665", text)
        self.assertIn("上下文使用率：2.97%", text)
        self.assertNotIn("当前会话 API Token 累计", text)
        self.assertNotIn("prompt:", text)

    def test_status_shows_zero_without_real_usage(self):
        output = io.StringIO()
        agent = self._agent(output)
        agent.session_manager = None
        agent._used_token = 123

        agent._handle_cmd_status(())

        text = output.getvalue()
        self.assertIn("上下文 Token：0", text)
        self.assertIn("上下文使用率：0.00%", text)

    def test_independent_usage_does_not_replace_current_context_usage(self):
        output = io.StringIO()
        agent = self._agent(output)
        agent.session_manager = None
        agent._used_token = 123
        response = SimpleNamespace(usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10})

        agent._update_and_record_response_usage(response, "task_completion_check")

        self.assertEqual(agent._used_token, 123)

    def test_status_shows_zero_after_only_task_is_deleted(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            output = io.StringIO()
            agent = self._agent(output)
            manager = SessionManager(project_root=tmp)
            session_id = manager.start_session("test-model")
            turn_id = manager.create_turn_id()
            manager.record_response_usage(
                turn_id=turn_id,
                usage={"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
                response_kind="assistant_response",
            )
            manager.mark_turn_deleted(turn_id)
            agent.session_manager = manager

            agent._handle_cmd_status(())

            text = output.getvalue()
            self.assertIn("上下文 Token：0", text)
            self.assertIn("上下文使用率：0.00%", text)


if __name__ == "__main__":
    unittest.main()
