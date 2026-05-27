# encoding: utf-8
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

import agent_Teams
from agent_Teams import Agent
from task_history_viewer import TaskHistoryResult


class AgentHistoryCommandTest(unittest.TestCase):
	def test_history_reopens_after_delete(self):
		class FakeSessionManager:
			current_session_id = "session_1"

			def __init__(self):
				self.tasks = [
					{"index": 1, "turn_id": "turn_1"},
					{"index": 2, "turn_id": "turn_2"},
				]
				self.list_sizes = []

			def list_tasks(self):
				self.list_sizes.append(len(self.tasks))
				return list(self.tasks)

		class FakeConsole:
			def __init__(self):
				self.messages = []

			def print(self, message):
				self.messages.append(message)

		agent = Agent.__new__(Agent)
		agent.session_manager = FakeSessionManager()
		agent.console = FakeConsole()
		deleted_turn_ids = []
		viewer_task_counts = []

		def fake_delete(task):
			deleted_turn_ids.append(task["turn_id"])
			agent.session_manager.tasks = [
				existing_task
				for existing_task in agent.session_manager.tasks
				if existing_task["turn_id"] != task["turn_id"]
			]
			return True

		def fake_open_task_history_viewer(tasks):
			viewer_task_counts.append(len(tasks))
			if len(viewer_task_counts) == 1:
				return TaskHistoryResult(action="delete", task=tasks[-1])
			return TaskHistoryResult(action="quit")

		original_open = agent_Teams.open_task_history_viewer
		agent._delete_history_task = fake_delete
		agent_Teams.open_task_history_viewer = fake_open_task_history_viewer
		try:
			self.assertEqual(agent._handle_cmd_history(()), (True, False))
		finally:
			agent_Teams.open_task_history_viewer = original_open

		self.assertEqual(deleted_turn_ids, ["turn_2"])
		self.assertEqual(viewer_task_counts, [2, 1])
		self.assertEqual(agent.session_manager.list_sizes[:2], [2, 1])


if __name__ == "__main__":
	unittest.main()
