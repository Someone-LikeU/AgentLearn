# encoding: utf-8
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from agent_Teams import Agent, ToolFailureGuardState
from tools.tool_scheduler import ToolCallTask


class AgentStreamResponseTest(unittest.TestCase):
	def _agent(self):
		agent = Agent.__new__(Agent)
		agent._used_token = 0
		agent._total_token = 0
		agent.session_manager = None
		agent._current_turn_id = None
		return agent

	def _chunk(self, content=None, tool_calls=None, usage=None):
		delta = SimpleNamespace()
		if content is not None:
			delta.content = content
		if tool_calls is not None:
			delta.tool_calls = tool_calls
		chunk = SimpleNamespace(choices=[SimpleNamespace(delta=delta)])
		if usage is not None:
			chunk.usage = usage
		return chunk

	def _usage_chunk(self, usage):
		return SimpleNamespace(choices=[], usage=usage)

	def _tool_delta(self, index, tool_id=None, name=None, arguments=None):
		function = SimpleNamespace()
		if name is not None:
			function.name = name
		if arguments is not None:
			function.arguments = arguments
		tool_call = SimpleNamespace(index=index, function=function)
		if tool_id is not None:
			tool_call.id = tool_id
		return tool_call

	def test_stream_content_is_printed_and_joined(self):
		agent = self._agent()
		stream = [
			self._chunk(content="你", usage=SimpleNamespace(total_tokens=7)),
			self._chunk(content="好"),
		]
		output = io.StringIO()

		with redirect_stdout(output):
			message = agent._deal_stream_response(stream)

		self.assertEqual(output.getvalue(), "你好")
		self.assertEqual(message.content, "你好")
		self.assertIsNone(message.tool_calls)
		self.assertEqual(message.usage["total_tokens"], 7)
		self.assertEqual(agent._used_token, 7)

	def test_stream_usage_chunk_without_choices_is_accepted(self):
		agent = self._agent()
		stream = [
			self._chunk(content="ok"),
			self._usage_chunk(SimpleNamespace(prompt_tokens=3, completion_tokens=4, total_tokens=7)),
		]
		output = io.StringIO()

		with redirect_stdout(output):
			message = agent._deal_stream_response(stream)

		self.assertEqual(output.getvalue(), "ok")
		self.assertEqual(message.content, "ok")
		self.assertEqual(message.usage["prompt_tokens"], 3)
		self.assertEqual(message.usage["completion_tokens"], 4)
		self.assertEqual(message.usage["total_tokens"], 7)
		self.assertEqual(agent._used_token, 7)

	def test_run_agent_step_retries_empty_response_after_tool_result(self):
		agent = self._agent()
		agent.max_iterations = 5
		agent._all_tools = []
		agent._all_tools_without_make_plan = []
		tool_call_message = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id="call_1")])
		empty_message = SimpleNamespace(content=None, tool_calls=None)
		final_message = SimpleNamespace(content="done", tool_calls=None)
		responses = iter([tool_call_message, empty_message, final_message])
		appended_messages = []
		continue_prompts = []

		def handle_tool_calls(_tool_calls, guard_state):
			guard_state.executed_tool_count += 1
			return None

		agent._request_next_model_message = lambda _tools: next(responses)
		agent._append_assistant_response = lambda message: appended_messages.append(message)
		agent._handle_tool_calls = handle_tool_calls
		agent._append_empty_tool_followup_prompt = lambda task_goal: continue_prompts.append(task_goal)
		agent._should_check_task_complete = lambda _task_goal, _message, _guard_state: False

		with redirect_stdout(io.StringIO()):
			result = agent._run_agent_step(agent._all_tools, task_goal="今天是星期几？")

		self.assertEqual(result, "done")
		self.assertEqual(continue_prompts, ["今天是星期几？"])
		self.assertEqual(appended_messages, [tool_call_message, empty_message, final_message])

	def test_repeated_success_tool_call_blocks_exact_next_call(self):
		agent = self._agent()
		appended_messages = []
		agent._append_message = lambda message, **kwargs: appended_messages.append((message, kwargs)) or {}
		guard_state = ToolFailureGuardState(active_tools=[], disabled_tools=set())
		task_1 = ToolCallTask(
			tool_call_id="call_1",
			function_name="EXECUTE_BASH",
			raw_arguments='{"command":"dir x"}',
			function_args={"command": "dir x"},
		)
		task_2 = ToolCallTask(
			tool_call_id="call_2",
			function_name="EXECUTE_BASH",
			raw_arguments='{"command":"dir x"}',
			function_args={"command": "dir x"},
		)

		agent._append_tool_result_and_check_guard(task_1, "same result", guard_state)
		agent._append_tool_result_and_check_guard(task_2, "same result", guard_state)

		self.assertIn(agent._tool_call_repeat_key(task_2), guard_state.blocked_duplicate_call_keys)
		self.assertEqual(guard_state.consecutive_success_duplicate_count, 2)
		self.assertTrue(any(
			kwargs.get("session_metadata", {}).get("tool_repeat_guard")
			for _, kwargs in appended_messages
		))

		different_task = ToolCallTask(
			tool_call_id="call_3",
			function_name="EDIT",
			raw_arguments='{"path":"a.txt"}',
			function_args={"path": "a.txt"},
		)
		agent._append_tool_result_and_check_guard(different_task, "edited", guard_state)

		self.assertEqual(guard_state.blocked_duplicate_call_keys, set())

	def test_blocked_repeated_tool_call_is_not_executed(self):
		agent = self._agent()
		appended_messages = []
		executed_calls = []
		agent._append_message = lambda message, **kwargs: appended_messages.append((message, kwargs)) or {}
		agent._available_functions = {
			"EXECUTE_BASH": lambda **kwargs: executed_calls.append(kwargs) or "should not run"
		}
		agent._tool_manager = SimpleNamespace(
			get_tool_runtime_profile=lambda _tool_name: {
				"is_read_only": False,
				"is_concurrency_safe": False,
				"side_effect_scope": "system",
			}
		)
		guard_state = ToolFailureGuardState(active_tools=[], disabled_tools=set())
		task = ToolCallTask(
			tool_call_id="call_1",
			function_name="EXECUTE_BASH",
			raw_arguments='{"command":"dir x"}',
			function_args={"command": "dir x"},
		)
		guard_state.blocked_duplicate_call_keys.add(agent._tool_call_repeat_key(task))
		tool_call = SimpleNamespace(
			id="call_1",
			function=SimpleNamespace(name="EXECUTE_BASH", arguments='{"command":"dir x"}'),
		)

		stop_reason = agent._handle_tool_calls([tool_call], guard_state)

		self.assertEqual(executed_calls, [])
		self.assertIn("重复请求", stop_reason)
		self.assertIn("repeated_identical_call", appended_messages[-1][0]["content"])

	def test_tool_call_chunks_are_merged_in_order(self):
		agent = self._agent()
		stream = [
			self._chunk(tool_calls=[self._tool_delta(0, name="WR", arguments='{"path"')]),
			self._chunk(tool_calls=[self._tool_delta(0, tool_id="call_1", name="ITE_FILE", arguments=':"a.txt"}')]),
		]

		message = agent._deal_stream_response(stream)

		self.assertEqual(message.content, "")
		self.assertEqual(len(message.tool_calls), 1)
		self.assertEqual(message.tool_calls[0].id, "call_1")
		self.assertEqual(message.tool_calls[0].function.name, "WRITE_FILE")
		self.assertEqual(message.tool_calls[0].function.arguments, '{"path":"a.txt"}')

	def test_multiple_tool_calls_keep_index_order(self):
		agent = self._agent()
		stream = [
			self._chunk(
				tool_calls=[
					self._tool_delta(1, tool_id="call_2", name="GET_TIME", arguments="{}"),
					self._tool_delta(0, tool_id="call_1", name="LIST_DIR", arguments='{"path":"."}'),
				]
			)
		]

		message = agent._deal_stream_response(stream)

		self.assertEqual([tool_call.id for tool_call in message.tool_calls], ["call_1", "call_2"])

	def test_spinner_stops_when_first_chunk_arrives(self):
		agent = self._agent()
		spinner = SimpleNamespace(stop_count=0)

		def stop():
			spinner.stop_count += 1

		spinner.stop = stop

		with redirect_stdout(io.StringIO()):
			agent._deal_stream_response([self._chunk(content="ok")], spinner=spinner)

		self.assertEqual(spinner.stop_count, 1)

	def test_stream_error_records_partial_content_and_reraises(self):
		class FakeSessionManager:
			def __init__(self):
				self.errors = []

			def record_model_stream_error(self, turn_id, partial_content, error):
				self.errors.append(
					{
						"turn_id": turn_id,
						"partial_content": partial_content,
						"error": error,
					}
				)

		def broken_stream():
			yield self._chunk(content="partial")
			raise RuntimeError("network closed")

		agent = self._agent()
		agent.session_manager = FakeSessionManager()
		agent._current_turn_id = "turn_1"

		with self.assertRaises(RuntimeError), redirect_stdout(io.StringIO()):
			agent._deal_stream_response(broken_stream())

		self.assertEqual(agent.session_manager.errors[0]["turn_id"], "turn_1")
		self.assertEqual(agent.session_manager.errors[0]["partial_content"], "partial")
		self.assertIn("network closed", agent.session_manager.errors[0]["error"])


if __name__ == "__main__":
	unittest.main()
