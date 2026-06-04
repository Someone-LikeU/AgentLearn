# encoding: utf-8
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory_manager import MemoryManager


class MemoryManagerTest(unittest.TestCase):
    def _manager(self, tmp: str) -> MemoryManager:
        project_root = Path(tmp)
        prompts_dir = project_root / "agent" / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        (prompts_dir / "memory_prompt_view.md").write_text(
            "GLOBAL\n<global_summary>\nRECENT\n<recent_tasks>\nINDEX <task_index_path>\nFULL <full_context_dir>",
            encoding="utf-8",
        )
        (prompts_dir / "memory_extraction_system.md").write_text("extract", encoding="utf-8")
        (prompts_dir / "memory_index_compaction_system.md").write_text("compact", encoding="utf-8")
        return MemoryManager(project_root=project_root, client=None, model="test-model")

    def test_deleted_task_id_is_filtered_from_recent_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self._manager(tmp)
            manager.task_index_file.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "task_id": "task_20260522_100000_aaaaaa",
                                "timestamp": "2026-05-22 10:00:00",
                                "title": "保留任务",
                                "tags": [],
                                "summary": "keep",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "task_id": "task_20260522_110000_bbbbbb",
                                "timestamp": "2026-05-22 11:00:00",
                                "title": "删除任务",
                                "tags": [],
                                "summary": "delete",
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            manager.record_deleted_task(
                session_id="session_1",
                turn_id="turn_1",
                task_ids=["task_20260522_110000_bbbbbb"],
            )

            memory_view = manager.load_prompt_memory_view()
            self.assertIn("保留任务", memory_view)
            self.assertNotIn("删除任务", memory_view)

    def test_deleted_turn_id_filters_later_saved_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self._manager(tmp)
            manager.record_deleted_task(session_id="session_1", turn_id="turn_1", task_ids=[])
            manager.task_index_file.write_text(
                json.dumps(
                    {
                        "task_id": "task_20260522_120000_cccccc",
                        "timestamp": "2026-05-22 12:00:00",
                        "title": "异步保存后也应过滤",
                        "tags": [],
                        "summary": "deleted by turn",
                        "session_id": "session_1",
                        "turn_id": "turn_1",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            memory_view = manager.load_prompt_memory_view()
            self.assertNotIn("异步保存后也应过滤", memory_view)
            self.assertIn("No previous tasks recorded.", memory_view)

    def test_delete_memories_removes_session_records_and_full_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self._manager(tmp)
            deleted_task_id = "task_20260522_130000_dddddd"
            kept_task_id = "task_20260522_140000_eeeeee"
            deleted_context = manager.full_context_dir / f"{deleted_task_id}.json"
            kept_context = manager.full_context_dir / f"{kept_task_id}.json"
            deleted_context.write_text("{}", encoding="utf-8")
            kept_context.write_text("{}", encoding="utf-8")
            manager.task_index_file.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "task_id": deleted_task_id,
                                "timestamp": "2026-05-22 13:00:00",
                                "title": "删除的会话任务",
                                "summary": "delete",
                                "session_id": "session_delete",
                                "full_context_path": str(deleted_context),
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "task_id": kept_task_id,
                                "timestamp": "2026-05-22 14:00:00",
                                "title": "保留的会话任务",
                                "summary": "keep",
                                "session_id": "session_keep",
                                "full_context_path": str(kept_context),
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            manager.task_summaries_file.write_text(
                "\n".join(
                    [
                        json.dumps({"task_id": deleted_task_id, "title": "删除摘要", "result_summary": "delete"}, ensure_ascii=False),
                        json.dumps({"task_id": kept_task_id, "title": "保留摘要", "result_summary": "keep"}, ensure_ascii=False),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = manager.delete_memories(session_ids={"session_delete"}, full_context_paths={str(deleted_context)})

            self.assertEqual(result["deleted_task_count"], 1)
            self.assertFalse(deleted_context.exists())
            self.assertTrue(kept_context.exists())
            self.assertNotIn("删除的会话任务", manager.task_index_file.read_text(encoding="utf-8"))
            self.assertIn("保留的会话任务", manager.task_index_file.read_text(encoding="utf-8"))
            self.assertNotIn("删除摘要", manager.task_summaries_file.read_text(encoding="utf-8"))
            self.assertFalse(hasattr(manager, "global_summary_file"))

    def test_delete_memories_by_turn_removes_skipped_task_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self._manager(tmp)
            task_id = "task_20260522_150000_ffffff"
            context_path = manager.full_context_dir / f"{task_id}.json"
            context_path.write_text("{}", encoding="utf-8")
            manager.task_index_file.write_text(
                json.dumps(
                    {
                        "task_id": task_id,
                        "timestamp": "2026-05-22 15:00:00",
                        "title": "简单任务",
                        "summary": "skipped",
                        "memory_status": "skipped",
                        "session_id": "session_1",
                        "turn_id": "turn_1",
                        "full_context_path": str(context_path),
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            result = manager.delete_memories(session_ids={"session_1"}, turn_ids={"turn_1"})

            self.assertEqual(result["deleted_task_ids"], [task_id])
            self.assertEqual(result["deleted_task_ids_by_turn"], {"turn_1": [task_id]})
            self.assertFalse(context_path.exists())
            self.assertEqual(manager.task_index_file.read_text(encoding="utf-8"), "")

    def test_delete_rebuilds_memory_index_from_remaining_topics_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self._manager(tmp)
            deleted_task_id = "task_20260522_160000_aaaaaa"
            kept_task_id = "task_20260522_170000_bbbbbb"
            manager.task_index_file.write_text(
                "\n".join(
                    [
                        json.dumps({"task_id": deleted_task_id, "session_id": "session_1"}, ensure_ascii=False),
                        json.dumps({"task_id": kept_task_id, "session_id": "session_2"}, ensure_ascii=False),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            topic_file = manager.topics_dir / "projects.md"
            topic_file.write_text(
                "# projects\n"
                f"- 2026-05-22 | fact | 删除内容 (confidence: high; source: {deleted_task_id})\n"
                f"- 2026-05-22 | fact | 保留内容 (confidence: high; source: {kept_task_id})\n",
                encoding="utf-8",
            )
            manager.memory_items_file.write_text(
                "\n".join(
                    [
                        json.dumps({"task_id": deleted_task_id, "topic": "projects", "content": "删除内容"}, ensure_ascii=False),
                        json.dumps({"task_id": kept_task_id, "topic": "projects", "content": "保留内容"}, ensure_ascii=False),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            manager.memory_index_file.write_text("# MEMORY\n\n删除内容\n保留内容", encoding="utf-8")

            result = manager.delete_memories(task_ids={deleted_task_id})

            self.assertEqual(result["deleted_memory_item_count"], 1)
            rebuilt_memory = manager.memory_index_file.read_text(encoding="utf-8")
            self.assertNotIn("删除内容", rebuilt_memory)
            self.assertIn("保留内容", rebuilt_memory)

    def test_memory_model_call_uses_short_timeout_without_retries(self):
        class FakeCompletions:
            def __init__(self, owner):
                self.owner = owner

            def create(self, **kwargs):
                self.owner.create_kwargs = kwargs
                return object()

        class FakeChat:
            def __init__(self, owner):
                self.completions = FakeCompletions(owner)

        class FakeClient:
            def __init__(self):
                self.options = None
                self.create_kwargs = None
                self.chat = FakeChat(self)

            def with_options(self, **kwargs):
                self.options = kwargs
                return self

        client = FakeClient()
        manager = MemoryManager.__new__(MemoryManager)
        manager.client = client
        manager.memory_request_timeout = 7.5
        manager.memory_max_retries = 0

        manager._create_memory_completion(model="test-model", messages=[])

        self.assertEqual(client.options, {"timeout": 7.5, "max_retries": 0})
        self.assertEqual(client.create_kwargs["timeout"], 7.5)

    def test_trivial_task_skips_model_summary(self):
        class FailingClient:
            def with_options(self, **_kwargs):
                raise AssertionError("trivial task should not call model")

        with tempfile.TemporaryDirectory() as tmp:
            manager = self._manager(tmp)
            manager.client = FailingClient()

            record = manager._process_memory_update("Hello", "Hi", [], {"session_id": "s1", "turn_id": "t1"})

            self.assertEqual(record["memory_status"], "skipped")
            self.assertEqual(record["skip_reason"], "trivial_task")
            self.assertEqual(manager.task_summaries_file.read_text(encoding="utf-8"), "")

    def test_valuable_task_extracts_topic_item_without_global_summary_update(self):
        class FakeCompletions:
            def create(self, **_kwargs):
                content = json.dumps(
                    {
                        "should_save": True,
                        "task_summary": "用户确认了长期偏好",
                        "tags": ["preference"],
                        "memory_items": [
                            {
                                "topic": "user_profile",
                                "type": "preference",
                                "content": "用户希望 Agent 记住通用型任务偏好。",
                                "confidence": "high",
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
                return type("Resp", (), {"choices": [type("Choice", (), {"message": type("Msg", (), {"content": content})()})()]})()

        class FakeClient:
            def __init__(self):
                self.chat = type("Chat", (), {"completions": FakeCompletions()})()

            def with_options(self, **_kwargs):
                return self

        with tempfile.TemporaryDirectory() as tmp:
            manager = self._manager(tmp)
            manager.client = FakeClient()

            record = manager._process_memory_update(
                "请记住我的偏好",
                "已记录",
                [{"role": "assistant", "tool_calls": [{"id": "1"}]}],
                {"session_id": "s1", "turn_id": "t1"},
            )

            self.assertEqual(record["memory_status"], "captured")
            self.assertIn("用户希望 Agent", (manager.topics_dir / "user_profile.md").read_text(encoding="utf-8"))
            self.assertIn("用户确认了长期偏好", manager.task_summaries_file.read_text(encoding="utf-8"))
            self.assertFalse(hasattr(manager, "global_summary_file"))

    def test_memory_compact_skips_when_no_dirty_items(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self._manager(tmp)

            result = manager.compact_memory_index(force=True, reason="manual")

            self.assertEqual(result["status"], "skipped")
            self.assertIn("memory_compact_skipped", manager.maintenance_log_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
