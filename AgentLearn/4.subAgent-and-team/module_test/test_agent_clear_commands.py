# encoding: utf-8
import io
import sys
import tempfile
import unittest
from pathlib import Path

from rich.console import Console


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_Teams import Agent
from session_manager import SessionManager


class FakeMemoryManager:
    def __init__(self):
        self.delete_calls = []

    def delete_memories(self, **kwargs):
        self.delete_calls.append(kwargs)
        task_ids = kwargs.get("task_ids") or []
        return {
            "deleted_task_count": len(task_ids),
            "deleted_task_ids": list(task_ids),
            "deleted_full_context_count": len(task_ids),
        }


class AgentClearCommandsTest(unittest.TestCase):
    def _build_history_delete_agent(self, tmp):
        manager = SessionManager(project_root=tmp)
        session_id = manager.start_session("test-model")
        manager.append_message({"role": "system", "content": "system prompt"})
        for index in range(1, 4):
            turn_id = manager.create_turn_id()
            task_id = f"task_20260603_12000{index}_abcde{index}"
            manager.append_message(
                {"role": "user", "content": f"任务 {index}"},
                turn_id=turn_id,
                metadata={"is_task_entry": True},
            )
            manager.record_memory_saved(turn_id, task_id, str(Path(tmp) / f"{task_id}.json"))
            manager.record_response_usage(
                turn_id=turn_id,
                usage={"prompt_tokens": index * 10, "completion_tokens": index, "total_tokens": index * 11},
                response_kind="assistant_response",
            )

        agent = Agent.__new__(Agent)
        agent.session_manager = manager
        agent.memory_manager = FakeMemoryManager()
        agent.console = Console(file=io.StringIO(), force_terminal=False, color_system=None)
        agent._wait_for_pending_memory_updates = lambda: None
        agent._reload_messages_after_turn_delete = lambda _session_id: None
        return agent, manager, session_id

    def test_clear_current_session_keeps_session_id_and_records_clear_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = SessionManager(project_root=tmp)
            session_id = manager.start_session("test-model")
            agent = Agent.__new__(Agent)
            agent.session_manager = manager
            agent.memory_manager = None
            agent.console = Console(file=io.StringIO(), force_terminal=False, color_system=None)
            agent.messages = [{"role": "system", "content": "old"}]
            agent._all_tools = []
            agent._confirm_irreversible_delete = lambda _title, _details: True
            agent._confirm_default_yes = lambda _prompt: True
            agent._wait_for_pending_memory_updates = lambda: None
            agent._delete_memories_for_session_turns = lambda session_id, turn_ids: {
                "task_ids": ["task_20260519_120000_abcdef"],
                "deleted_task_ids": ["task_20260519_120000_abcdef"],
                "deleted_task_count": 1,
                "deleted_full_context_count": 1,
            }
            reset_calls = []

            def reset_messages(record_system_message):
                reset_calls.append(record_system_message)
                agent.messages = [{"role": "system", "content": "new"}]

            agent._reset_messages_for_session = reset_messages
            agent._restore_session_usage_summary = lambda _session_id: {}

            agent._handle_cmd_clear_current_session(())

            events = manager.load_session(session_id)
            clear_events = [event for event in events if event.get("event") == "session_cleared"]
            self.assertEqual(manager.current_session_id, session_id)
            self.assertEqual(reset_calls, [True])
            self.assertEqual(len(clear_events), 1)
            self.assertTrue(clear_events[0]["delete_memory"])
            self.assertEqual(clear_events[0]["memory_task_ids"], ["task_20260519_120000_abcdef"])

    def test_history_delete_middle_task_with_following_tasks_and_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent, manager, session_id = self._build_history_delete_agent(tmp)
            tasks = manager.list_tasks(session_id)
            confirm_answers = iter([True, True])
            agent._confirm_default_yes = lambda _prompt: next(confirm_answers)

            deleted = agent._delete_history_task(tasks[1])
            events = manager.load_session(session_id)
            deleted_turn_ids = [event.get("turn_id") for event in events if event.get("event") == "turn_deleted"]

            self.assertTrue(deleted)
            self.assertEqual(deleted_turn_ids, [tasks[1]["turn_id"], tasks[2]["turn_id"]])
            self.assertEqual(
                agent.memory_manager.delete_calls[0]["turn_ids"],
                [tasks[1]["turn_id"], tasks[2]["turn_id"]],
            )
            self.assertEqual(
                agent.memory_manager.delete_calls[0]["task_ids"],
                tasks[1]["memory_task_ids"] + tasks[2]["memory_task_ids"],
            )

    def test_history_delete_middle_task_without_following_or_memory_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent, manager, session_id = self._build_history_delete_agent(tmp)
            tasks = manager.list_tasks(session_id)
            confirm_answers = iter([False, False])
            agent._confirm_default_yes = lambda _prompt: next(confirm_answers)

            deleted = agent._delete_history_task(tasks[1])
            events = manager.load_session(session_id)
            deleted_turn_ids = [event.get("turn_id") for event in events if event.get("event") == "turn_deleted"]

            self.assertTrue(deleted)
            self.assertEqual(deleted_turn_ids, [tasks[1]["turn_id"]])
            self.assertEqual(agent.memory_manager.delete_calls, [])

    def test_history_delete_last_task_skips_following_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent, manager, session_id = self._build_history_delete_agent(tmp)
            tasks = manager.list_tasks(session_id)
            confirm_answers = iter([True])
            agent._confirm_default_yes = lambda _prompt: next(confirm_answers)

            deleted = agent._delete_history_task(tasks[-1])
            events = manager.load_session(session_id)
            deleted_turn_ids = [event.get("turn_id") for event in events if event.get("event") == "turn_deleted"]

            self.assertTrue(deleted)
            self.assertEqual(deleted_turn_ids, [tasks[-1]["turn_id"]])
            self.assertEqual(agent.memory_manager.delete_calls[0]["turn_ids"], [tasks[-1]["turn_id"]])
            self.assertEqual(agent.memory_manager.delete_calls[0]["task_ids"], tasks[-1]["memory_task_ids"])


if __name__ == "__main__":
    unittest.main()
