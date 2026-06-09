# encoding: utf-8
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SESSION_TEST_TEMP_DIR = PROJECT_ROOT / "module_test" / "session_test_temp"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from session_manager import SessionManager
from task_history_viewer import TaskHistoryViewer


class SessionManagerTest(unittest.TestCase):
    def test_session_event_flow_and_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = SessionManager(project_root=tmp)
            session_id = manager.start_session("test-model", metadata={"role": "Main Agent"})
            turn_id = manager.create_turn_id()

            manager.append_message({"role": "system", "content": "system prompt"})
            manager.append_message(
                {"role": "user", "content": "用户真实任务"},
                turn_id=turn_id,
                metadata={"is_task_entry": True},
            )
            manager.append_message(
                {"role": "user", "content": "计划模式内部子步骤"},
                turn_id=turn_id,
                metadata={"is_task_entry": False},
            )
            manager.append_message(
                {
                    "role": "assistant",
                    "content": "需要调用工具",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "READ_FILE", "arguments": "{}"},
                        }
                    ],
                },
                turn_id=turn_id,
            )
            manager.append_message(
                {"role": "tool", "tool_call_id": "call_1", "content": "{\"ok\": true}"},
                turn_id=turn_id,
            )
            manager.append_message(
                {"role": "assistant", "content": "最终任务结果"},
                turn_id=turn_id,
            )
            manager.record_memory_saved(turn_id, "task_20260519_120000_abcdef", str(Path(tmp) / "task.json"))
            manager.record_model_stream_error(turn_id, "partial", "network closed")

            events = manager.load_session(session_id)
            self.assertEqual(events[0]["event"], "session_start")
            self.assertNotIn("model_stream_delta", {event.get("event") for event in events})

            tasks = manager.list_tasks(session_id)
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0]["content"], "用户真实任务")
            self.assertEqual(tasks[0]["assistant_message_count"], 2)
            self.assertEqual(tasks[0]["tool_call_count"], 1)
            self.assertEqual(tasks[0]["tool_result_count"], 1)
            self.assertEqual(tasks[0]["memory_task_ids"], ["task_20260519_120000_abcdef"])
            self.assertEqual(tasks[0]["final_output"], "最终任务结果")

            messages = manager.rebuild_messages(session_id)
            self.assertEqual([message["role"] for message in messages], ["system", "user", "user", "assistant", "tool", "assistant"])
            self.assertEqual(messages[3]["tool_calls"][0]["id"], "call_1")

    def test_turn_deleted_filters_rebuild_and_task_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = SessionManager(project_root=tmp)
            session_id = manager.start_session("test-model")
            turn_id = manager.create_turn_id()

            manager.append_message({"role": "system", "content": "system prompt"})
            manager.append_message(
                {"role": "user", "content": "要删除的任务"},
                turn_id=turn_id,
                metadata={"is_task_entry": True},
            )
            manager.append_message({"role": "assistant", "content": "任务结果"}, turn_id=turn_id)
            manager.mark_turn_deleted(turn_id)

            self.assertEqual(manager.list_tasks(session_id), [])
            self.assertEqual(manager.rebuild_messages(session_id), [{"role": "system", "content": "system prompt"}])

    def test_session_cleared_filters_prior_context_tasks_and_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = SessionManager(project_root=tmp)
            session_id = manager.start_session("test-model")
            old_turn_id = manager.create_turn_id()
            new_turn_id = manager.create_turn_id()

            manager.append_message({"role": "system", "content": "old system"})
            manager.append_message(
                {"role": "user", "content": "旧任务"},
                turn_id=old_turn_id,
                metadata={"is_task_entry": True},
            )
            manager.append_message({"role": "assistant", "content": "旧结果"}, turn_id=old_turn_id)
            manager.record_response_usage(
                turn_id=old_turn_id,
                usage={"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
                response_kind="assistant_response",
            )
            manager.record_session_cleared(delete_memory=True, memory_task_ids=["task_20260519_120000_abcdef"])
            manager.append_message({"role": "system", "content": "new system"})
            manager.append_message(
                {"role": "user", "content": "新任务"},
                turn_id=new_turn_id,
                metadata={"is_task_entry": True},
            )
            manager.record_response_usage(
                turn_id=new_turn_id,
                usage={"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22},
                response_kind="assistant_response",
            )

            messages = manager.rebuild_messages(session_id)
            tasks = manager.list_tasks(session_id)
            usage = manager.calculate_session_usage(session_id)

            self.assertEqual(messages, [{"role": "system", "content": "new system"}, {"role": "user", "content": "新任务"}])
            self.assertEqual([task["content"] for task in tasks], ["新任务"])
            self.assertEqual(usage["prompt_tokens"], 20)
            self.assertEqual(usage["total_tokens"], 22)

    def test_response_usage_uses_latest_assistant_response_for_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = SessionManager(project_root=tmp)
            session_id = manager.start_session("test-model")
            turn_id = manager.create_turn_id()

            event = manager.record_response_usage(
                turn_id=turn_id,
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                response_kind="assistant_response",
                model="test-model",
                message_id="msg_1",
            )
            manager.record_response_usage(
                turn_id=turn_id,
                usage={"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22},
                response_kind="assistant_response",
                model="test-model",
                message_id="msg_2",
            )
            manager.record_response_usage(
                turn_id=turn_id,
                usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
                response_kind="task_completion_check",
                model="test-model",
            )

            summary = manager.calculate_session_usage(session_id)

            self.assertEqual(event["event"], "response_usage")
            self.assertEqual(event["usage"]["total_tokens"], 15)
            self.assertEqual(summary["prompt_tokens"], 20)
            self.assertEqual(summary["completion_tokens"], 2)
            self.assertEqual(summary["total_tokens"], 22)
            self.assertEqual(summary["response_count"], 2)
            self.assertTrue(summary["has_real_usage"])

    def test_response_usage_excludes_deleted_turns_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = SessionManager(project_root=tmp)
            session_id = manager.start_session("test-model")
            turn_id = manager.create_turn_id()

            manager.record_response_usage(
                turn_id=turn_id,
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                response_kind="assistant_response",
            )
            manager.mark_turn_deleted(turn_id)

            default_summary = manager.calculate_session_usage(session_id)
            included_summary = manager.calculate_session_usage(session_id, include_deleted=True)

            self.assertEqual(default_summary["total_tokens"], 0)
            self.assertFalse(default_summary["has_real_usage"])
            self.assertEqual(included_summary["total_tokens"], 15)
            self.assertEqual(included_summary["response_count"], 1)

    def test_turn_delete_invalidates_earlier_context_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = SessionManager(project_root=tmp)
            session_id = manager.start_session("test-model")
            first_turn_id = manager.create_turn_id()
            second_turn_id = manager.create_turn_id()
            third_turn_id = manager.create_turn_id()

            manager.record_response_usage(
                turn_id=first_turn_id,
                usage={"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
                response_kind="assistant_response",
            )
            manager.record_response_usage(
                turn_id=second_turn_id,
                usage={"prompt_tokens": 20, "completion_tokens": 1, "total_tokens": 21},
                response_kind="assistant_response",
            )
            manager.record_response_usage(
                turn_id=third_turn_id,
                usage={"prompt_tokens": 30, "completion_tokens": 1, "total_tokens": 31},
                response_kind="assistant_response",
            )
            manager.mark_turn_deleted(second_turn_id)

            stale_summary = manager.calculate_session_usage(session_id)
            included_summary = manager.calculate_session_usage(session_id, include_deleted=True)

            self.assertEqual(stale_summary["total_tokens"], 0)
            self.assertFalse(stale_summary["has_real_usage"])
            self.assertEqual(included_summary["total_tokens"], 31)

            manager.record_response_usage(
                turn_id=third_turn_id,
                usage={"prompt_tokens": 25, "completion_tokens": 1, "total_tokens": 26},
                response_kind="assistant_response",
            )
            refreshed_summary = manager.calculate_session_usage(session_id)

            self.assertEqual(refreshed_summary["prompt_tokens"], 25)
            self.assertEqual(refreshed_summary["total_tokens"], 26)
            self.assertTrue(refreshed_summary["has_real_usage"])

    def test_delete_last_task_restores_previous_task_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = SessionManager(project_root=tmp)
            session_id = manager.start_session("test-model")
            turn_ids = [manager.create_turn_id() for _ in range(3)]
            for index, turn_id in enumerate(turn_ids, start=1):
                manager.append_message(
                    {"role": "user", "content": f"任务 {index}"},
                    turn_id=turn_id,
                    metadata={"is_task_entry": True},
                )
                manager.record_response_usage(
                    turn_id=turn_id,
                    usage={"prompt_tokens": index * 10, "completion_tokens": index, "total_tokens": index * 11},
                    response_kind="assistant_response",
                )

            manager.mark_turn_deleted(turn_ids[-1])
            summary = manager.calculate_session_usage(session_id)

            self.assertEqual(summary["prompt_tokens"], 20)
            self.assertEqual(summary["total_tokens"], 22)
            self.assertTrue(summary["has_real_usage"])

    def test_delete_middle_task_suffix_restores_prefix_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = SessionManager(project_root=tmp)
            session_id = manager.start_session("test-model")
            turn_ids = [manager.create_turn_id() for _ in range(3)]
            for index, turn_id in enumerate(turn_ids, start=1):
                manager.append_message(
                    {"role": "user", "content": f"任务 {index}"},
                    turn_id=turn_id,
                    metadata={"is_task_entry": True},
                )
                manager.record_response_usage(
                    turn_id=turn_id,
                    usage={"prompt_tokens": index * 10, "completion_tokens": index, "total_tokens": index * 11},
                    response_kind="assistant_response",
                )

            manager.mark_turn_deleted(turn_ids[1])
            manager.mark_turn_deleted(turn_ids[2])
            summary = manager.calculate_session_usage(session_id)

            self.assertEqual(summary["prompt_tokens"], 10)
            self.assertEqual(summary["total_tokens"], 11)
            self.assertTrue(summary["has_real_usage"])

    def test_delete_middle_task_kept_following_marks_usage_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = SessionManager(project_root=tmp)
            session_id = manager.start_session("test-model")
            turn_ids = [manager.create_turn_id() for _ in range(3)]
            for index, turn_id in enumerate(turn_ids, start=1):
                manager.append_message(
                    {"role": "user", "content": f"任务 {index}"},
                    turn_id=turn_id,
                    metadata={"is_task_entry": True},
                )
                manager.record_response_usage(
                    turn_id=turn_id,
                    usage={"prompt_tokens": index * 10, "completion_tokens": index, "total_tokens": index * 11},
                    response_kind="assistant_response",
                )

            manager.mark_turn_deleted(turn_ids[1])
            summary = manager.calculate_session_usage(session_id)

            self.assertEqual(summary["total_tokens"], 0)
            self.assertFalse(summary["has_real_usage"])
            self.assertTrue(summary["is_stale"])

    def test_list_sessions_uses_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = SessionManager(project_root=tmp)
            first = manager.start_session("model-a")
            second = manager.start_session("model-b")

            sessions = manager.list_sessions(limit=2)
            self.assertEqual(len(sessions), 2)
            self.assertEqual(sessions[0]["session_id"], second)
            self.assertEqual(sessions[1]["session_id"], first)

    def test_list_sessions_skips_deleted_session_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = SessionManager(project_root=tmp)
            deleted_session_id = manager.start_session("model-a")
            deleted_session_path = manager.current_session_path
            kept_session_id = manager.start_session("model-b")

            deleted_session_path.unlink()
            sessions = manager.list_sessions(limit=10)
            session_ids = {item["session_id"] for item in sessions}

            self.assertIn(kept_session_id, session_ids)
            self.assertNotIn(deleted_session_id, session_ids)

    def test_session_title_update_and_user_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = SessionManager(project_root=tmp)
            session_id = manager.start_session("test-model")

            self.assertEqual(manager.get_current_session_info()["title"], "未命名会话")
            auto_event = manager.update_title("实现会话管理", source="auto")
            self.assertEqual(auto_event["event"], "session_title_updated")
            self.assertEqual(manager.get_current_session_info()["title"], "实现会话管理")
            self.assertEqual(manager.get_current_session_info()["title_source"], "auto")

            user_event = manager.update_title("我的会话", source="user")
            self.assertEqual(user_event["title"], "我的会话")
            skipped = manager.update_title("自动覆盖", source="auto")
            self.assertEqual(skipped["event"], "session_title_update_skipped")
            self.assertEqual(manager.get_current_session_info()["title"], "我的会话")

            events = manager.load_session(session_id)
            self.assertIn("session_title_updated", {event.get("event") for event in events})
            sessions = manager.list_sessions(limit=1)
            self.assertEqual(sessions[0]["title"], "我的会话")
            self.assertEqual(sessions[0]["title_source"], "user")

    def test_index_updates_are_not_written_per_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = SessionManager(project_root=tmp)
            manager.start_session("test-model")
            turn_id = manager.create_turn_id()

            manager.append_message({"role": "user", "content": "任务"}, turn_id=turn_id)
            manager.append_message({"role": "assistant", "content": "结果"}, turn_id=turn_id)
            manager.append_message({"role": "tool", "tool_call_id": "call_1", "content": "工具结果"}, turn_id=turn_id)
            manager.record_memory_saved(turn_id, "task_20260521_120000_abcdef", str(Path(tmp) / "task.json"))

            index_file = Path(tmp) / "sessions" / "session_index.jsonl"
            index_events = [json.loads(line) for line in index_file.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([event.get("event") for event in index_events], ["session_index"])

            manager.touch_session()
            manager.end_session()
            index_events = [json.loads(line) for line in index_file.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(
                [event.get("event") for event in index_events],
                ["session_index", "session_index_update", "session_index_update"],
            )
            self.assertEqual(index_events[-1]["status"], "ended")

    def test_switch_session_restores_current_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = SessionManager(project_root=tmp)
            first = manager.start_session("model-a", title="第一个会话")
            manager.update_title("用户标题", source="user")
            manager.append_message({"role": "system", "content": "system prompt"})
            manager.append_message({"role": "user", "content": "历史任务"}, turn_id=manager.create_turn_id())
            manager.end_session()

            second = manager.start_session("model-b", title="第二个会话")
            self.assertEqual(manager.get_current_session_info()["session_id"], second)

            info = manager.switch_session(first)
            self.assertEqual(info["session_id"], first)
            self.assertEqual(info["title"], "用户标题")
            self.assertEqual(info["title_source"], "user")
            self.assertEqual(info["model"], "model-a")
            self.assertEqual(info["status"], "active")

            events = manager.load_session(first)
            self.assertEqual(events[-1]["event"], "session_resumed")
            sessions = manager.list_sessions(limit=1)
            self.assertEqual(sessions[0]["session_id"], first)
            self.assertEqual(sessions[0]["status"], "active")
            self.assertEqual(
                [message["role"] for message in manager.rebuild_messages(first)],
                ["system", "user"],
            )

    def test_memory_checked_cursor_can_be_recorded_for_loaded_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = SessionManager(project_root=tmp)
            first = manager.start_session("model-a")
            turn_id = manager.create_turn_id()
            manager.append_message(
                {"role": "user", "content": "历史任务"},
                turn_id=turn_id,
                metadata={"is_task_entry": True},
            )
            manager.end_session()

            manager.start_session("model-b")
            info = manager.switch_session(first)
            cursor = manager.record_memory_checked(
                turn_id=None,
                last_memory_checked_event_id=info["resume_event_id"],
                status="cursor_only",
                metadata={"reason": "session_resumed"},
            )

            self.assertEqual(cursor["event"], "memory_checked")
            self.assertEqual(manager.latest_memory_checked_event_id(first), info["resume_event_id"])

    def test_record_session_interrupted_before_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = SessionManager(project_root=tmp)
            session_id = manager.start_session("test-model")
            turn_id = manager.create_turn_id()

            interrupted = manager.record_session_interrupted("keyboard_interrupt", turn_id=turn_id)
            manager.end_session()

            self.assertEqual(interrupted["event"], "session_interrupted")
            self.assertEqual(interrupted["reason"], "keyboard_interrupt")
            self.assertEqual(interrupted["turn_id"], turn_id)

            events = manager.load_session(session_id)
            self.assertEqual([event["event"] for event in events[-2:]], ["session_interrupted", "session_end"])

    def test_list_sessions_can_skip_empty_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = SessionManager(project_root=tmp)
            empty_session_id = manager.start_session("model-empty")
            manager.end_session()

            task_session_id = manager.start_session("model-task", title="任务会话", title_source="user")
            turn_id = manager.create_turn_id()
            manager.append_message({"role": "system", "content": "system prompt"})
            manager.append_message(
                {"role": "user", "content": "真实任务"},
                turn_id=turn_id,
                metadata={"is_task_entry": True},
            )
            manager.append_message({"role": "assistant", "content": "任务结果"}, turn_id=turn_id)
            manager.touch_session(reason="task_completed")

            all_sessions = manager.list_sessions(limit=10)
            non_empty_sessions = manager.list_sessions(limit=10, include_empty=False)

            self.assertIn(empty_session_id, {item["session_id"] for item in all_sessions})
            self.assertEqual([item["session_id"] for item in non_empty_sessions], [task_session_id])
            self.assertEqual(non_empty_sessions[0]["title_source"], "user")
            self.assertTrue(non_empty_sessions[0]["has_user_task"])

    def test_agent_session_management_commands_step_by_step(self):
        target_session_file = "sessions/2026/06/08/session_20260608_224848_23587a.jsonl"
        target_session_path = Path(target_session_file)
        if not target_session_path.is_absolute():
            target_session_path = PROJECT_ROOT / target_session_path
        target_session_id = target_session_path.stem
        output_dir = SESSION_TEST_TEMP_DIR / "session_management_commands"
        copied_session_path = output_dir / f"{target_session_id}.jsonl"
        output_path = output_dir / f"{target_session_id}_context.json"

        self.assertTrue(target_session_path.exists(), f"target session file not found: {target_session_path}")
        output_dir.mkdir(parents=True, exist_ok=True)
        # 集成式 session 命令测试只读取真实历史会话，所有写入都落到 session_test_temp。
        copied_session_path.write_text(target_session_path.read_text(encoding="utf-8"), encoding="utf-8")
        manager = SessionManager(project_root=PROJECT_ROOT, sessions_dir=output_dir)

        print("\n$ session new")
        new_session_id = manager.start_session(
            "test-model",
            metadata={"role": "Main Agent", "name": "session-command-test", "is_main_agent": True},
        )
        print(json.dumps(manager.get_current_session_info(), ensure_ascii=False, indent=2))

        print("\n$ session title session_test")
        title_event = manager.update_title("session_test", source="user")
        print(json.dumps(title_event, ensure_ascii=False, indent=2))
        self.assertEqual(manager.get_current_session_info()["title"], "session_test")

        print(f"\n$ session load {target_session_id}")
        loaded_info = manager.switch_session(target_session_id)
        print(json.dumps(loaded_info, ensure_ascii=False, indent=2))
        self.assertEqual(loaded_info["session_id"], target_session_id)

        print("\n$ session current")
        current_info = manager.get_current_session_info()
        print(json.dumps(current_info, ensure_ascii=False, indent=2))
        self.assertEqual(current_info["session_id"], target_session_id)

        events = manager.load_session(target_session_id)
        messages = manager.rebuild_messages(target_session_id)
        context = {
            "session": current_info,
            "source_session_file": str(target_session_path),
            "copied_session_file": str(copied_session_path),
            "created_test_session_id": new_session_id,
            "event_count": len(events),
            "message_count": len(messages),
            "events": events,
            "messages": messages,
        }

        # 将事件日志和还原后的 messages 一起落盘，便于人工核对完整上下文。
        output_path.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n$ rebuild messages")
        print(json.dumps({"message_count": len(messages), "output_path": str(output_path)}, ensure_ascii=False, indent=2))

        self.assertGreater(len(events), 0)
        self.assertGreater(len(messages), 0)
        self.assertTrue(output_path.exists())

    def test_rebuild_messages_from_session_copy(self):
        target_session_file = "sessions/2026/06/08/session_20260608_224848_23587a.jsonl"
        target_session_path = Path(target_session_file)
        if not target_session_path.is_absolute():
            target_session_path = PROJECT_ROOT / target_session_path
        target_session_id = target_session_path.stem
        copy_dir = PROJECT_ROOT / "module_test" / "session_copy"
        rebuild_dir = PROJECT_ROOT / "module_test" / "session_rebuild"
        copied_session_path = copy_dir / f"{target_session_id}.jsonl"
        output_path = rebuild_dir / f"{target_session_id}_rebuild_context.json"

        self.assertTrue(target_session_path.exists(), f"target session file not found: {target_session_path}")

        copy_dir.mkdir(parents=True, exist_ok=True)
        rebuild_dir.mkdir(parents=True, exist_ok=True)
        # 只读取副本进行重建验证，避免污染真实 session 文件。
        copied_session_path.write_text(target_session_path.read_text(encoding="utf-8"), encoding="utf-8")

        manager = SessionManager(project_root=PROJECT_ROOT, sessions_dir=copy_dir)
        events = manager.load_session(target_session_id)
        messages = manager.rebuild_messages(target_session_id)
        context = {
            "source_session_file": str(target_session_path),
            "copied_session_file": str(copied_session_path),
            "event_count": len(events),
            "message_count": len(messages),
            "events": events,
            "messages": messages,
        }
        output_path.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")

        print("\n$ copy session")
        print(json.dumps({"source": str(target_session_path), "copy": str(copied_session_path)}, ensure_ascii=False, indent=2))
        print("\n$ rebuild messages from copy")
        print(json.dumps({"message_count": len(messages), "output_path": str(output_path)}, ensure_ascii=False, indent=2))

        self.assertGreater(len(events), 0)
        self.assertGreater(len(messages), 0)
        self.assertTrue(copied_session_path.exists())
        self.assertTrue(output_path.exists())

    def test_history_delete_first_task_with_session_copy(self):
        target_session_file = "sessions/2026/05/22/session_20260522_100508_c4fb08.jsonl"
        target_session_path = PROJECT_ROOT / target_session_file
        target_session_id = target_session_path.stem
        output_dir = PROJECT_ROOT / "module_test" / "session_test"
        copied_session_path = output_dir / f"{target_session_id}.jsonl"
        before_delete_path = output_dir / f"{target_session_id}_before_delete.jsonl"
        after_delete_path = output_dir / f"{target_session_id}_after_delete.jsonl"
        simulate_output_path = output_dir / "delete_simulate.txt"

        self.assertTrue(target_session_path.exists(), f"target session file not found: {target_session_path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        # 只操作 session_test 下的副本，避免测试污染真实历史会话。
        copied_session_path.write_text(target_session_path.read_text(encoding="utf-8"), encoding="utf-8")

        manager = SessionManager(project_root=PROJECT_ROOT, sessions_dir=output_dir)
        loaded_info = manager.switch_session(target_session_id)
        self.assertEqual(loaded_info["session_id"], target_session_id)

        messages_before_delete = manager.rebuild_messages(target_session_id)
        tasks = manager.list_tasks(target_session_id)
        self.assertGreater(len(tasks), 0)

        viewer = TaskHistoryViewer(tasks)
        while viewer.selected_task() and viewer.selected_task().get("index") != 1:
            viewer.move(-1)
        selected_task = viewer.selected_task()
        self.assertIsNotNone(selected_task)
        self.assertEqual(selected_task["index"], 1)

        before_delete_path.write_text(copied_session_path.read_text(encoding="utf-8"), encoding="utf-8")
        manager.mark_turn_deleted(
            turn_id=selected_task["turn_id"],
            task_ids=selected_task.get("memory_task_ids") or [],
        )

        messages_after_delete = manager.rebuild_messages(target_session_id)
        after_delete_path.write_text(copied_session_path.read_text(encoding="utf-8"), encoding="utf-8")

        report = {
            "loaded_session": loaded_info,
            "copied_session_path": str(copied_session_path),
            "deleted_task": selected_task,
            "messages_before_delete": messages_before_delete,
            "messages_after_delete": messages_after_delete,
        }
        simulate_output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        self.assertTrue(simulate_output_path.exists())
        self.assertTrue(before_delete_path.exists())
        self.assertTrue(after_delete_path.exists())
        self.assertGreater(len(messages_before_delete), len(messages_after_delete))
        self.assertNotIn(selected_task["turn_id"], json.dumps(messages_after_delete, ensure_ascii=False))

        after_events = manager.load_session(target_session_id)
        self.assertEqual(after_events[-1]["event"], "turn_deleted")
        self.assertEqual(after_events[-1]["turn_id"], selected_task["turn_id"])


if __name__ == "__main__":
    unittest.main()
