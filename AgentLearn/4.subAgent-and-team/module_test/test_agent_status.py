# encoding: utf-8
import io
import sys
import unittest
from pathlib import Path

from rich.console import Console


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_Teams import Agent


class AgentStatusTest(unittest.TestCase):
    def _agent(self, output: io.StringIO):
        agent = Agent.__new__(Agent)
        agent.model = "test-model"
        agent._max_context_tokens = 258000
        agent.messages = [{"role": "user", "content": "hello"}]
        agent._all_tools = []
        agent.console = Console(file=output, force_terminal=False, color_system=None, width=120)
        return agent

    def test_status_uses_real_session_usage_without_detail_line(self):
        class FakeSessionManager:
            current_session_id = "session_1"

            def calculate_session_usage(self, session_id):
                return {
                    "prompt_tokens": 19728,
                    "completion_tokens": 194,
                    "total_tokens": 19922,
                    "response_count": 3,
                    "has_real_usage": True,
                }

        output = io.StringIO()
        agent = self._agent(output)
        agent.session_manager = FakeSessionManager()

        agent._handle_cmd_status(())

        text = output.getvalue()
        self.assertIn("Token 用量：19922", text)
        self.assertIn("Token 使用率：7.72%", text)
        self.assertNotIn("当前会话 API Token 累计", text)
        self.assertNotIn("prompt:", text)

    def test_status_shows_zero_without_real_usage(self):
        output = io.StringIO()
        agent = self._agent(output)
        agent.session_manager = None
        agent._used_token = 123

        agent._handle_cmd_status(())

        text = output.getvalue()
        self.assertIn("Token 用量：0", text)
        self.assertIn("Token 使用率：0.00%", text)


if __name__ == "__main__":
    unittest.main()
