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


if __name__ == "__main__":
    unittest.main()
