# encoding: utf-8
import sys
import unittest
import io
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from agent_Teams import Agent


class AgentCompletionGuardTest(unittest.TestCase):
	def _agent(self):
		agent = Agent.__new__(Agent)
		agent.messages = []
		agent.session_manager = None
		agent._current_turn_id = None
		agent._current_task_full_context = None
		agent.max_iterations = 3
		agent._all_tools = []
		agent._all_tools_without_make_plan = []
		agent.plan_mode = False
		return agent

	def test_trivial_task_does_not_call_completion_checker(self):
		agent = self._agent()
		agent._request_next_model_message = lambda _: SimpleNamespace(content="你好，有什么可以帮你？", tool_calls=None)
		agent._check_task_complete = lambda *_: self.fail("hello 不应该触发完成判断")

		result = agent._run_agent_step([], task_goal="hello")

		self.assertIn("你好", result)

	def test_completion_checker_uses_recent_five_messages(self):
		class FakeCompletions:
			def __init__(self):
				self.messages = None

			def create(self, **kwargs):
				self.messages = kwargs["messages"]
				return SimpleNamespace(
					choices=[SimpleNamespace(message=SimpleNamespace(content="CONTINUE"))],
					usage=None,
				)

		agent = self._agent()
		completions = FakeCompletions()
		agent.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
		agent.model = "test-model"
		agent.messages = [
			{"role": "user", "content": f"msg_{index}"}
			for index in range(7)
		]

		status = agent._check_task_complete("写入文件", "还没写")

		self.assertEqual(status, "CONTINUE")
		check_prompt = completions.messages[-1]["content"]
		self.assertNotIn("msg_0", check_prompt)
		self.assertNotIn("msg_1", check_prompt)
		for index in range(2, 7):
			self.assertIn(f"msg_{index}", check_prompt)

	def test_plan_steps_use_step_as_task_goal(self):
		agent = self._agent()
		captured_goals = []
		agent._append_message = lambda *_, **__: None

		def fake_run_agent_step(_tools, task_goal=None):
			captured_goals.append(task_goal)
			return f"done {task_goal}"

		agent._run_agent_step = fake_run_agent_step

		with redirect_stdout(io.StringIO()):
			result = agent._run_plan_steps(["步骤一", "步骤二"], [])

		self.assertEqual(captured_goals, ["步骤一", "步骤二"])
		self.assertIn("done 步骤一", result)
		self.assertIn("done 步骤二", result)

	def test_completion_status_parser_accepts_labeled_response(self):
		agent = self._agent()

		self.assertEqual(agent._parse_completion_status("status: continue"), "CONTINUE")
		self.assertEqual(agent._parse_completion_status('{"status":"NEED_USER"}'), "NEED_USER")
		self.assertEqual(agent._parse_completion_status("unknown"), "DONE")


if __name__ == "__main__":
	unittest.main()
