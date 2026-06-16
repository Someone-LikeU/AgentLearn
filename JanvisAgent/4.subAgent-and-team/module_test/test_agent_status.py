# encoding: utf-8
import io
import sys
import tempfile
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

    def test_skill_list_commands_print_skill_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = io.StringIO()
            agent = self._agent(output)
            skills_dir = Path(tmp) / "skills"
            (skills_dir / "alpha").mkdir(parents=True)
            (skills_dir / "beta").mkdir()
            (skills_dir / "alpha" / "SKILL.md").write_text(
                "---\nname: alpha\ndescription: Alpha skill description\n---\n# Alpha\n",
                encoding="utf-8",
            )
            (skills_dir / "beta" / "SKILL.md").write_text(
                "---\nname: beta\ndescription: Beta skill description\n---\n# Beta\n",
                encoding="utf-8",
            )
            agent.skills_dir = str(skills_dir)
            agent._skills_cache = {}

            handled, should_exit = agent._handle_user_command("skills", ())
            self.assertTrue(handled)
            self.assertFalse(should_exit)
            text = output.getvalue()
            self.assertIn("alpha", text)
            self.assertIn("Alpha skill description", text)
            self.assertIn("beta", text)
            self.assertIn("Beta skill description", text)

            output.seek(0)
            output.truncate(0)
            handled, should_exit = agent._handle_user_command("skill_list", ())
            self.assertTrue(handled)
            self.assertFalse(should_exit)
            self.assertIn("Alpha skill description", output.getvalue())

    def test_vision_model_commands_read_and_append_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = io.StringIO()
            agent = self._agent(output)
            root = Path(tmp)
            config_path = root / "agent" / "config" / "local_vision_model.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(
                '{"default_model":"gemma4:e4b","models":[{"name":"gemma4:e4b"}]}',
                encoding="utf-8",
            )
            agent._agent_file_path = lambda relative_path: str(root / relative_path)

            handled, should_exit = agent._handle_user_command("vision model list", ())

            self.assertTrue(handled)
            self.assertFalse(should_exit)
            self.assertIn("gemma4:e4b", output.getvalue())

            output.seek(0)
            output.truncate(0)
            agent.console.input = lambda _prompt: "qwen3.5:9b"
            handled, should_exit = agent._handle_user_command("add vision model", ())

            self.assertTrue(handled)
            self.assertFalse(should_exit)
            self.assertIn("qwen3.5:9b", config_path.read_text(encoding="utf-8"))

    def test_status_shows_zero_after_only_task_is_deleted(self):
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

    def test_compact_updates_status_context_tokens(self):
        class FakeCompletions:
            def create(self, **_kwargs):
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="压缩摘要"))],
                    usage={"prompt_tokens": 1000, "completion_tokens": 10, "total_tokens": 1010},
                )

        with tempfile.TemporaryDirectory() as tmp:
            output = io.StringIO()
            agent = self._agent(output)
            agent.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
            agent._start_spinner = lambda: SimpleNamespace(stop=lambda: None)
            agent._is_main_agent = False
            agent.memory_manager = None
            agent._KEEP_RECENT = 2
            agent._MIDDLE_COMPACT_RATIO = 0.5
            agent._current_task_start_index = None
            manager = SessionManager(project_root=tmp)
            session_id = manager.start_session("test-model")
            manager.append_message({"role": "system", "content": "system prompt"})
            for index in range(4):
                turn_id = manager.create_turn_id()
                manager.append_message(
                    {"role": "user", "content": f"任务 {index} " + "内容" * 30},
                    turn_id=turn_id,
                    metadata={"is_task_entry": True},
                )
                manager.append_message({"role": "assistant", "content": f"结果 {index}"}, turn_id=turn_id)
            manager.record_response_usage(
                turn_id=turn_id,
                usage={"prompt_tokens": 436000, "completion_tokens": 93, "total_tokens": 436093},
                response_kind="assistant_response",
            )
            agent.session_manager = manager
            agent.messages = manager.rebuild_messages(session_id)

            compacted = agent._compact_messages(agent._all_tools, force=True, reason="manual")
            agent._handle_cmd_status(())

            usage = manager.calculate_session_usage(session_id)
            text = output.getvalue()
            self.assertTrue(compacted)
            self.assertEqual(usage["usage_source"], "conversation_compacted")
            self.assertNotEqual(usage["total_tokens"], 436093)
            self.assertIn(f"上下文 Token：{usage['total_tokens']}", text)
            self.assertNotIn("上下文 Token：436093", text)


if __name__ == "__main__":
    unittest.main()
