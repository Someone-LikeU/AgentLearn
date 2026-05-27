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

from agent_Teams import Agent


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
		self.assertEqual(agent._used_token, 7)

	def test_tool_call_chunks_are_merged_in_order(self):
		agent = self._agent()
		stream = [
			self._chunk(tool_calls=[self._tool_delta(0, name="WR", arguments='{"path"')]),
			self._chunk(tool_calls=[self._tool_delta(0, tool_id="call_1", name="ITE_FILE", arguments=':"a.txt"}')]),
		]

		message = agent._deal_stream_response(stream)

		self.assertIsNone(message.content)
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
