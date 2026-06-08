# encoding: utf-8
import json
import re
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, wait
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from threading import Lock, RLock
from typing import Any
from prompt_loader import load_prompt


class MemoryManager:
    """管理分层长期记忆，并把记忆整理流程隐藏在 Agent 运行时内部。"""

    DEFAULT_TOPICS = {
        "user_profile",
        "projects",
        "decisions",
        "workflows",
        "references",
        "open_items",
        "learnings",
    }

    def __init__(
        self,
        project_root: str | Path,
        client: Any,
        model: str,
        temperature: float = 0.1,
        recent_limit: int = 10,
        prompt_char_limit: int = 6000,
        memory_request_timeout: float = 20.0,
        memory_max_retries: int = 0,
        topic_compact_threshold: int = 10,
        memory_index_line_limit: int = 200,
        memory_worker_count: int = 5,
    ):
        self.project_root = Path(project_root)
        self.prompts_dir = self.project_root / "agent" / "prompts"
        self.client = client
        self.model = model
        self.temperature = temperature
        self.recent_limit = recent_limit
        self.prompt_char_limit = prompt_char_limit
        # 记忆整理是后台非关键路径；模型服务慢或网络半断开时必须快速降级，
        # 否则 OpenAI SDK 默认分钟级 timeout/retry 会让退出阶段长时间等待。
        self.memory_request_timeout = memory_request_timeout
        self.memory_max_retries = memory_max_retries
        self.topic_compact_threshold = topic_compact_threshold
        self.memory_index_line_limit = memory_index_line_limit
        self.memory_worker_count = max(1, int(memory_worker_count or 1))

        self.memory_dir = self.project_root / "agent" / "memory"
        self.full_context_dir = self.memory_dir / "full_context"
        self.topics_dir = self.memory_dir / "topics"
        self.memory_index_file = self.memory_dir / "MEMORY.md"
        self.task_index_file = self.memory_dir / "task_index.jsonl"
        self.task_summaries_file = self.memory_dir / "task_summaries.jsonl"
        self.memory_items_file = self.memory_dir / "memory_items.jsonl"
        self.memory_state_file = self.memory_dir / "memory_state.json"
        self.maintenance_log_file = self.memory_dir / "memory_maintenance.log"
        self.deleted_tasks_file = self.memory_dir / "deleted_tasks.jsonl"
        self.error_log_file = self.memory_dir / "memory_errors.log"

        # 模型摘要可并发执行；文件写入和 MEMORY.md 压缩由锁保证一致性。
        self._storage_lock = RLock()
        self._futures_lock = RLock()
        self._compaction_lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=self.memory_worker_count, thread_name_prefix="memory-manager")
        self._futures = []
        self._shutdown = False
        self._ensure_memory_files()

    def _ensure_memory_files(self):
        # 启动时按需创建目录和空文件，避免新检出项目需要手动初始化记忆结构。
        self.full_context_dir.mkdir(parents=True, exist_ok=True)
        self.topics_dir.mkdir(parents=True, exist_ok=True)
        for file_path in (
            self.memory_index_file,
            self.task_index_file,
            self.task_summaries_file,
            self.memory_items_file,
            self.maintenance_log_file,
            self.deleted_tasks_file,
            self.error_log_file,
        ):
            file_path.touch(exist_ok=True)
        if not self.memory_state_file.exists():
            self._write_memory_state(self._default_memory_state())
        for topic in self.DEFAULT_TOPICS:
            topic_path = self._topic_file(topic)
            if not topic_path.exists():
                topic_path.write_text(f"# {topic}\n", encoding="utf-8")

    def enqueue(
        self,
        task: str,
        result: str,
        context: list[dict[str, Any]],
        session_id: str | None = None,
        turn_id: str | None = None,
        last_memory_checked_event_id: str | None = None,
    ):
        if self._shutdown:
            return None
        # Agent.chat() 结束后会清空当前任务上下文，因此这里先复制快照再交给后台线程。
        context_snapshot = deepcopy(context or [])
        metadata = {
            "session_id": session_id,
            "turn_id": turn_id,
            "last_memory_checked_event_id": last_memory_checked_event_id,
        }
        future = self._executor.submit(self._process_memory_update, task, result, context_snapshot, metadata)
        with self._futures_lock:
            self._futures.append(future)
        self._cleanup_finished_futures()
        return future

    def has_pending(self) -> bool:
        self._cleanup_finished_futures()
        with self._futures_lock:
            return any(not future.done() for future in self._futures)

    def pending_count(self) -> int:
        self._cleanup_finished_futures()
        with self._futures_lock:
            return sum(1 for future in self._futures if not future.done())

    def wait_for_pending(self, timeout: float | None = None) -> bool:
        # Agent 退出前调用这里，等待用户无感的后台记忆整理任务完成。
        self._cleanup_finished_futures()
        with self._futures_lock:
            futures = list(self._futures)
        if not futures:
            return True
        done, not_done = wait(futures, timeout=timeout)
        for future in done:
            self._log_future_exception(future)
        with self._futures_lock:
            self._futures = [future for future in self._futures if not future.done()]
            return not any(not future.done() for future in self._futures)

    def shutdown(self):
        if self._shutdown:
            return
        self.wait_for_pending()
        self._executor.shutdown(wait=True)
        self._shutdown = True

    def load_prompt_memory_view(self) -> str:
        return self._load_prompt_memory_view()

    def _load_prompt_memory_view(self) -> str:
        # system prompt 只注入 MEMORY.md 短索引和最近任务索引，详细 topic/full_context 按需读取。
        self._ensure_memory_files()
        with self._storage_lock:
            memory_index = self.memory_index_file.read_text(encoding="utf-8").strip()
        recent_tasks = self._load_recent_tasks()

        recent_task_lines = []
        if recent_tasks:
            for item in recent_tasks:
                tags = ", ".join(item.get("tags") or [])
                recent_task_lines.append(
                    f"- [{item.get('task_id')}] {item.get('timestamp')} | "
                    f"{item.get('title')} | tags: {tags or 'none'} | {item.get('summary')}"
                )
        else:
            recent_task_lines.append("- No previous tasks recorded.")

        memory_view = load_prompt(
            "memory_prompt_view.md",
            prompts_dir=self.prompts_dir,
            global_summary=memory_index or "No stable long-term memory recorded.",
            recent_tasks="\n".join(recent_task_lines),
            task_index_path=self.task_index_file,
            task_summaries_path=self.task_summaries_file,
            full_context_dir=self.full_context_dir,
        )
        return self._limit_text(memory_view, self.prompt_char_limit)

    def clear(self):
        self.wait_for_pending()
        # 项目规则禁止批量删除文件，所以这里不会删除 full_context 下的历史上下文文件。
        # 清空记忆只重置会进入 prompt 的索引、摘要和错误日志。
        self.memory_index_file.write_text("", encoding="utf-8")
        self.task_index_file.write_text("", encoding="utf-8")
        self.task_summaries_file.write_text("", encoding="utf-8")
        self.memory_items_file.write_text("", encoding="utf-8")
        self.maintenance_log_file.write_text("", encoding="utf-8")
        self.deleted_tasks_file.write_text("", encoding="utf-8")
        self.error_log_file.write_text("", encoding="utf-8")
        self._write_memory_state(self._default_memory_state())
        for topic in self.DEFAULT_TOPICS:
            self._topic_file(topic).write_text(f"# {topic}\n", encoding="utf-8")

    def delete_memories(
            self,
            task_ids: list[str] | set[str] | None = None,
            session_ids: list[str] | set[str] | None = None,
            turn_ids: list[str] | set[str] | None = None,
            full_context_paths: list[str] | set[str] | None = None,
            rebuild_index: bool = True,
    ) -> dict[str, Any]:
        """
        硬删除指定 session 或 task 关联的长期记忆。
        :param task_ids: 要删除的 task_id
        :param session_ids: 要删除的 session_id
        :param turn_ids: 要删除的用户任务 turn_id；传入 session_ids 时只匹配这些 session 内的 turn
        :param full_context_paths: session 事件中记录的完整上下文路径
        :param rebuild_index: 删除 topic item 后是否立即用剩余 topics 重建 MEMORY.md
        :return: 删除统计
        """
        self.wait_for_pending()
        self._ensure_memory_files()
        target_session_ids = {str(session_id) for session_id in (session_ids or []) if session_id}
        target_turn_ids = {str(turn_id) for turn_id in (turn_ids or []) if turn_id}
        target_task_ids = self._normalize_task_ids(task_ids or [])
        context_paths = {path for path in (full_context_paths or []) if path}

        index_records = self._read_jsonl_records(self.task_index_file)
        for record in index_records:
            task_id = record.get("task_id")
            if self._memory_record_matches(record, target_task_ids, target_session_ids, target_turn_ids):
                if task_id:
                    target_task_ids.add(str(task_id))
                if record.get("full_context_path"):
                    context_paths.add(str(record.get("full_context_path")))

        matched_index_records = [
            record
            for record in index_records
            if self._memory_record_matches(record, target_task_ids, target_session_ids, target_turn_ids)
        ]
        deleted_task_ids_by_turn: dict[str, list[str]] = {}
        for record in matched_index_records:
            turn_id = record.get("turn_id")
            task_id = record.get("task_id")
            if not turn_id or not task_id:
                continue
            deleted_task_ids_by_turn.setdefault(str(turn_id), [])
            if str(task_id) not in deleted_task_ids_by_turn[str(turn_id)]:
                deleted_task_ids_by_turn[str(turn_id)].append(str(task_id))

        remaining_index = [
            record
            for record in index_records
            if not self._memory_record_matches(record, target_task_ids, target_session_ids, target_turn_ids)
        ]
        summary_records = self._read_jsonl_records(self.task_summaries_file)
        remaining_summaries = [
            record
            for record in summary_records
            if str(record.get("task_id") or "") not in target_task_ids
        ]
        deleted_records = self._read_jsonl_records(self.deleted_tasks_file)
        remaining_deleted_records = [
            record
            for record in deleted_records
            if not self._memory_record_matches(record, target_task_ids, target_session_ids, target_turn_ids)
        ]

        self._write_jsonl_records(self.task_index_file, remaining_index)
        self._write_jsonl_records(self.task_summaries_file, remaining_summaries)
        self._write_jsonl_records(self.deleted_tasks_file, remaining_deleted_records)
        remaining_items = [
            record
            for record in self._read_jsonl_records(self.memory_items_file)
            if str(record.get("task_id") or "") not in target_task_ids
        ]
        self._write_jsonl_records(self.memory_items_file, remaining_items)
        removed_topic_items = self._remove_topic_items_for_tasks(target_task_ids)
        index_rebuild_result = None
        if removed_topic_items and rebuild_index:
            index_rebuild_result = self._rebuild_memory_index_from_topics(reason="memory_deleted")

        deleted_files = 0
        missing_files = 0
        context_candidates = set(context_paths)
        for task_id in target_task_ids:
            context_candidates.add(str(self.full_context_dir / f"{task_id}.json"))
        for path_value in sorted(context_candidates):
            result = self._delete_full_context_file(path_value)
            if result == "deleted":
                deleted_files += 1
            elif result == "missing":
                missing_files += 1

        return {
            "deleted_task_count": len(target_task_ids),
            "deleted_task_ids": sorted(target_task_ids),
            "deleted_task_ids_by_turn": deleted_task_ids_by_turn,
            "deleted_memory_item_count": removed_topic_items,
            "deleted_full_context_count": deleted_files,
            "missing_full_context_count": missing_files,
            "rewritten_index_count": len(remaining_index),
            "rewritten_summary_count": len(remaining_summaries),
            "memory_index": index_rebuild_result,
        }

    def delete_all_memories(self) -> dict[str, Any]:
        """
        硬删除所有长期记忆和完整上下文文件。
        :return: 删除统计
        """
        self.wait_for_pending()
        self._ensure_memory_files()
        task_ids = set()
        for file_path in (self.task_index_file, self.task_summaries_file, self.deleted_tasks_file):
            for record in self._read_jsonl_records(file_path):
                task_id = record.get("task_id")
                if task_id:
                    task_ids.add(str(task_id))

        deleted_files = 0
        # 遍历后逐个删除明确路径文件，不使用批量删除。
        for path in sorted(self.full_context_dir.glob("task_*.json")):
            if self._delete_full_context_file(str(path)) == "deleted":
                deleted_files += 1

        self.memory_index_file.write_text("", encoding="utf-8")
        self.task_index_file.write_text("", encoding="utf-8")
        self.task_summaries_file.write_text("", encoding="utf-8")
        self.memory_items_file.write_text("", encoding="utf-8")
        self.maintenance_log_file.write_text("", encoding="utf-8")
        self.deleted_tasks_file.write_text("", encoding="utf-8")
        self.error_log_file.write_text("", encoding="utf-8")
        self._write_memory_state(self._default_memory_state())
        for topic in self.DEFAULT_TOPICS:
            self._topic_file(topic).write_text(f"# {topic}\n", encoding="utf-8")
        return {
            "deleted_task_count": len(task_ids),
            "deleted_full_context_count": deleted_files,
        }

    def record_deleted_task(
        self,
        session_id: str | None,
        turn_id: str,
        task_ids: list[str] | None = None,
        reason: str = "user_deleted_from_session",
    ) -> list[dict[str, Any]]:
        """
        记录被用户从会话中删除的长期记忆任务。
        :param session_id: 会话 id
        :param turn_id: 用户任务 turn_id
        :param task_ids: 已关联的长期记忆 task_id
        :param reason: 删除原因
        :return: 写入的删除记录
        """
        self._ensure_memory_files()
        timestamp = datetime.now().isoformat(timespec="seconds")
        normalized_task_ids = [task_id for task_id in (task_ids or []) if task_id]
        if not normalized_task_ids:
            normalized_task_ids = [None]

        records = []
        for task_id in normalized_task_ids:
            record = {
                "task_id": task_id,
                "session_id": session_id,
                "turn_id": turn_id,
                "deleted_at": timestamp,
                "reason": reason,
            }
            self._append_jsonl(self.deleted_tasks_file, record)
            records.append(record)
        return records

    def get_task_full_context(self, task_id: str) -> dict[str, Any]:
        # 只允许读取 MemoryManager 自己生成的 task_id，避免模型传入任意路径。
        normalized_task_id = str(task_id or "").strip()
        if normalized_task_id.endswith(".json"):
            normalized_task_id = normalized_task_id[:-5]
        if not re.fullmatch(r"task_\d{8}_\d{6}_[0-9a-f]{6}", normalized_task_id):
            return {
                "error": "invalid_task_id",
                "message": "task_id must look like task_YYYYMMDD_HHMMSS_xxxxxx",
            }

        full_context_path = self.full_context_dir / f"{normalized_task_id}.json"
        if not full_context_path.exists():
            return {
                "error": "not_found",
                "task_id": normalized_task_id,
                "message": "No full context file exists for this task_id.",
            }

        try:
            return json.loads(full_context_path.read_text(encoding="utf-8"))
        except Exception:
            self._log_error("load full context failed", traceback.format_exc())
            return {
                "error": "read_failed",
                "task_id": normalized_task_id,
                "message": "Failed to read or parse the full context file.",
            }

    def _process_memory_update(
        self,
        task: str,
        result: str,
        context: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ):
        try:
            # 先保存完整上下文和轻量索引；是否调用模型由本地价值判断决定。
            metadata = metadata or {}
            task_id = self._create_task_id()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            full_context_path = self._save_task_full_context(task_id, timestamp, task, result, context, metadata)
            local_assessment = self._assess_memory_value(task, result, context)
            summary = self._fallback_task_summary(task_id, timestamp, task, result)
            summary["memory_status"] = "skipped"
            summary["skip_reason"] = local_assessment["reason"]
            memory_items: list[dict[str, Any]] = []

            if local_assessment["should_save"]:
                extraction = self._extract_task_memory_items(task_id, timestamp, task, result, context)
                if extraction.get("should_save") and extraction.get("memory_items"):
                    summary = self._normalize_extracted_summary(task_id, timestamp, extraction, summary)
                    memory_items = self._normalize_memory_items(task_id, timestamp, extraction.get("memory_items") or [])
                    self._append_memory_items(memory_items)
                    self._append_topic_items(memory_items)
                    self._increase_dirty_topic_count(len(memory_items))
                    summary["memory_status"] = "captured"
                    summary["skip_reason"] = None
                else:
                    summary["memory_status"] = "skipped"
                    summary["skip_reason"] = extraction.get("reason") or "model_found_no_reusable_memory"

            index_record = self._build_index_record(task_id, timestamp, summary, full_context_path, metadata)
            self._append_task_index(index_record)
            if summary.get("memory_status") == "captured":
                self._append_task_summary(summary)
                self._compact_memory_index_if_needed(reason="topic_dirty_threshold")
            return index_record
        except Exception:
            self._log_error("memory update failed", traceback.format_exc())
            return None

    def _create_task_id(self) -> str:
        return f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    def _save_task_full_context(
        self,
        task_id: str,
        timestamp: str,
        task: str,
        result: str,
        context: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        path = self.full_context_dir / f"{task_id}.json"
        payload = {
            "task_id": task_id,
            "timestamp": timestamp,
            "task": task,
            "result": result,
            "messages": context,
            "metadata": metadata or {},
        }
        with self._storage_lock:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    def _assess_memory_value(self, task: str, result: str, context: list[dict[str, Any]]) -> dict[str, Any]:
        """
        本地判断当前任务是否值得进入长期记忆抽取，避免简单任务消耗模型 token。
        """
        task_text = " ".join(str(task or "").split())
        result_text = " ".join(str(result or "").split())
        combined = f"{task_text}\n{result_text}"
        lower_combined = combined.lower()
        has_tool_activity = any(
            message.get("role") == "tool" or message.get("tool_calls")
            for message in context
            if isinstance(message, dict)
        )
        durable_keywords = (
            "记住", "以后", "偏好", "长期", "下次", "todo", "待办", "决策", "决定",
            "方案", "设计", "重构", "架构", "bug", "错误", "测试", "修改", "新增",
            "删除", "配置", "项目", "总结", "规则", "流程", "reference", "preference",
            "remember", "decision", "followup", "workflow", "project",
        )
        has_keyword = any(keyword in lower_combined for keyword in durable_keywords)
        if has_tool_activity:
            return {"should_save": True, "reason": "tool_activity"}
        if has_keyword:
            return {"should_save": True, "reason": "durable_keyword"}
        if len(task_text) >= 80 or len(result_text) >= 300:
            return {"should_save": True, "reason": "long_task_or_result"}
        return {"should_save": False, "reason": "trivial_task"}

    def _extract_task_memory_items(
        self,
        task_id: str,
        timestamp: str,
        task: str,
        result: str,
        context: list[dict[str, Any]],
    ) -> dict[str, Any]:
        # 只有通过本地价值判断的任务才会走这里；每个任务最多一次模型请求。
        context_text = self._limit_text(json.dumps(context, ensure_ascii=False, default=str), 12000)
        user_prompt = json.dumps(
            {
                "task_id": task_id,
                "timestamp": timestamp,
                "task": task,
                "result": result,
                "context": context_text,
            },
            ensure_ascii=False,
            indent=2,
        )
        try:
            response = self._create_memory_completion(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": load_prompt("memory_extraction_system.md", prompts_dir=self.prompts_dir),
                    },
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=self.temperature,
            )
            extracted = json.loads(response.choices[0].message.content)
            return extracted if isinstance(extracted, dict) else {"should_save": False, "memory_items": []}
        except Exception:
            self._log_error("memory extraction model call failed", traceback.format_exc())
            return {"should_save": False, "reason": "model_call_failed", "memory_items": []}

    def _normalize_extracted_summary(
        self,
        task_id: str,
        timestamp: str,
        extraction: dict[str, Any],
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        summary = dict(fallback)
        summary.update(
            {
                "task_id": task_id,
                "timestamp": timestamp,
                "title": self._string_or_default(extraction.get("title"), fallback["title"], 120),
                "result_summary": self._string_or_default(
                    extraction.get("task_summary"),
                    fallback["result_summary"],
                    500,
                ),
                "tags": self._string_list(extraction.get("tags"), 12),
                "memory_status": "captured",
                "memory_item_count": len(extraction.get("memory_items") or []),
            }
        )
        return summary

    def _normalize_memory_items(
        self,
        task_id: str,
        timestamp: str,
        items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        normalized_items = []
        for item in items:
            if not isinstance(item, dict):
                continue
            content = self._string_or_default(item.get("content"), "", 500)
            if not content:
                continue
            topic = self._normalize_topic(item.get("topic"))
            normalized_items.append(
                {
                    "task_id": task_id,
                    "timestamp": timestamp,
                    "topic": topic,
                    "type": self._string_or_default(item.get("type"), "fact", 40),
                    "content": content,
                    "confidence": self._string_or_default(item.get("confidence"), "medium", 20),
                }
            )
        return normalized_items

    def _append_memory_items(self, items: list[dict[str, Any]]) -> None:
        for item in items:
            self._append_jsonl(self.memory_items_file, item)

    def _append_topic_items(self, items: list[dict[str, Any]]) -> None:
        with self._storage_lock:
            for item in items:
                topic_path = self._topic_file(item["topic"])
                line = (
                    f"- {item['timestamp']} | {item['type']} | {item['content']} "
                    f"(confidence: {item['confidence']}; source: {item['task_id']})"
                )
                with topic_path.open("a", encoding="utf-8") as file:
                    file.write(line + "\n")

    def _normalize_topic(self, topic: Any) -> str:
        normalized = re.sub(r"[^0-9a-zA-Z_\-]+", "_", str(topic or "").strip().lower()).strip("_")
        if normalized not in self.DEFAULT_TOPICS:
            return "learnings"
        return normalized

    def _topic_file(self, topic: str) -> Path:
        return self.topics_dir / f"{self._normalize_topic(topic)}.md"

    def _summarize_task_memory(
        self,
        task_id: str,
        timestamp: str,
        task: str,
        result: str,
        context: list[dict[str, Any]],
    ) -> dict[str, Any]:
        # 即使模型摘要失败，也保留一条最低限度可检索的任务记录。
        fallback = self._fallback_task_summary(task_id, timestamp, task, result)
        # 完整任务上下文可能很大，发送给摘要模型前先截断，避免记忆整理请求过长。
        context_text = self._limit_text(json.dumps(context, ensure_ascii=False, default=str), 12000)
        user_prompt = json.dumps(
            {
                "task_id": task_id,
                "timestamp": timestamp,
                "task": task,
                "result": result,
                "context": context_text,
            },
            ensure_ascii=False,
            indent=2,
        )
        try:
            response = self._create_memory_completion(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": load_prompt("memory_task_summary_system.md", prompts_dir=self.prompts_dir),
                    },
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=self.temperature,
            )
            content = response.choices[0].message.content
            summary = json.loads(content)
            if not isinstance(summary, dict):
                return fallback
            return self._normalize_task_summary(task_id, timestamp, summary, fallback)
        except Exception:
            self._log_error("task summary model call failed", traceback.format_exc())
            return fallback

    def _fallback_task_summary(self, task_id: str, timestamp: str, task: str, result: str) -> dict[str, Any]:
        return {
            "task_id": task_id,
            "timestamp": timestamp,
            "title": self._limit_text(task.replace("\n", " "), 80),
            "tags": [],
            "status": "completed",
            "result_summary": self._limit_text(str(result).replace("\n", " "), 240),
            "important_facts": [],
            "decisions": [],
            "changed_files": [],
            "followups": [],
            "keywords": [],
        }

    def _normalize_task_summary(
        self,
        task_id: str,
        timestamp: str,
        summary: dict[str, Any],
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        # JSON mode 只能保证大体是 JSON，不保证字段完整或类型正确；落盘前统一规整结构。
        normalized = dict(fallback)
        normalized.update(
            {
                "task_id": task_id,
                "timestamp": timestamp,
                "title": self._string_or_default(summary.get("title"), fallback["title"], 120),
                "status": self._string_or_default(summary.get("status"), fallback["status"], 40),
                "result_summary": self._string_or_default(
                    summary.get("result_summary"),
                    fallback["result_summary"],
                    500,
                ),
                "tags": self._string_list(summary.get("tags"), 12),
                "important_facts": self._string_list(summary.get("important_facts"), 20),
                "decisions": self._string_list(summary.get("decisions"), 20),
                "changed_files": self._string_list(summary.get("changed_files"), 30),
                "followups": self._string_list(summary.get("followups"), 20),
                "keywords": self._string_list(summary.get("keywords"), 20),
            }
        )
        return normalized

    def _build_index_record(
        self,
        task_id: str,
        timestamp: str,
        summary: dict[str, Any],
        full_context_path: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # 索引刻意保持轻量：system prompt 和后续 task_id 查找流程主要依赖它。
        metadata = metadata or {}
        record = {
            "task_id": task_id,
            "timestamp": timestamp,
            "title": summary.get("title", task_id),
            "tags": summary.get("tags", []),
            "status": summary.get("status", "completed"),
            "summary": summary.get("result_summary", ""),
            "memory_status": summary.get("memory_status", "captured"),
            "skip_reason": summary.get("skip_reason"),
            "memory_item_count": summary.get("memory_item_count", 0),
            "full_context_path": full_context_path,
        }
        if metadata.get("last_memory_checked_event_id"):
            record["last_memory_checked_event_id"] = metadata.get("last_memory_checked_event_id")
        if metadata.get("session_id"):
            record["session_id"] = metadata.get("session_id")
        if metadata.get("turn_id"):
            record["turn_id"] = metadata.get("turn_id")
        return record

    def compact_memory_index(self, force: bool = False, reason: str = "manual") -> dict[str, Any]:
        """
        手动或外部触发 MEMORY.md 整理；会先等待当前任务记忆写入完成。
        """
        self.wait_for_pending()
        return self._compact_memory_index_if_needed(reason=reason, force=force)

    def enqueue_compact_memory_index(self, reason: str = "session_end"):
        if self._shutdown:
            return None
        future = self._executor.submit(self._compact_memory_index_if_needed, reason, False)
        with self._futures_lock:
            self._futures.append(future)
        self._cleanup_finished_futures()
        return future

    def _compact_memory_index_if_needed(self, reason: str, force: bool = False) -> dict[str, Any]:
        with self._compaction_lock:
            self._ensure_memory_files()
            with self._storage_lock:
                state = self._read_memory_state()
                dirty_count = int(state.get("dirty_topic_item_count") or 0)
                line_count = self._memory_index_line_count()
                should_compact = (
                    line_count > self.memory_index_line_limit
                    or dirty_count >= self.topic_compact_threshold
                    or (reason == "session_end" and dirty_count > 0)
                    or (force and (dirty_count > 0 or line_count > self.memory_index_line_limit))
                )
                current_memory = self._limit_text(self.memory_index_file.read_text(encoding="utf-8"), 10000)
                topics_text = self._collect_topic_text()

            if not should_compact:
                result = {
                    "status": "skipped",
                    "reason": "no_worthy_memory",
                    "dirty_topic_item_count": dirty_count,
                    "memory_index_line_count": line_count,
                }
                if force or reason == "session_end":
                    self._log_maintenance("memory_compact_skipped", result)
                return result

            try:
                response = self._create_memory_completion(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": load_prompt("memory_index_compaction_system.md", prompts_dir=self.prompts_dir),
                        },
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "reason": reason,
                                    "current_memory_md": current_memory,
                                    "topic_memory": topics_text,
                                },
                                ensure_ascii=False,
                                indent=2,
                            ),
                        },
                    ],
                    temperature=self.temperature,
                )
                updated_memory = response.choices[0].message.content.strip()
            except Exception:
                self._log_error("memory index compaction model call failed", traceback.format_exc())
                updated_memory = self._fallback_memory_index(topics_text)

            with self._storage_lock:
                self.memory_index_file.write_text(updated_memory, encoding="utf-8")
                state = self._read_memory_state()
                current_dirty_count = int(state.get("dirty_topic_item_count") or 0)
                state["dirty_topic_item_count"] = max(0, current_dirty_count - dirty_count)
                state["last_compacted_at"] = datetime.now().isoformat(timespec="seconds")
                state["last_compact_reason"] = reason
                self._write_memory_state(state)
            result = {
                "status": "compacted",
                "reason": reason,
                "dirty_topic_item_count": dirty_count,
                "memory_index_line_count": line_count,
                "memory_index_path": str(self.memory_index_file),
            }
            self._log_maintenance("memory_compacted", result)
            return result

    def _rebuild_memory_index_from_topics(self, reason: str = "memory_rebuild") -> dict[str, Any]:
        """
        删除记忆后不能复用旧 MEMORY.md 作为输入，否则已删除内容可能被模型保留下来。
        这里只从剩余 topic 文件重建短索引，保证 system prompt 看到的是删除后的权威状态。
        """
        self._ensure_memory_files()
        topics_text = self._collect_topic_text()
        if not topics_text.strip():
            updated_memory = self._empty_memory_index()
        else:
            try:
                response = self._create_memory_completion(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": load_prompt("memory_index_compaction_system.md", prompts_dir=self.prompts_dir),
                        },
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "reason": reason,
                                    "current_memory_md": "",
                                    "topic_memory": topics_text,
                                },
                                ensure_ascii=False,
                                indent=2,
                            ),
                        },
                    ],
                    temperature=self.temperature,
                )
                updated_memory = response.choices[0].message.content.strip()
            except Exception:
                self._log_error("memory index rebuild model call failed", traceback.format_exc())
                updated_memory = self._fallback_memory_index(topics_text)

        with self._storage_lock:
            self.memory_index_file.write_text(updated_memory, encoding="utf-8")
            state = self._read_memory_state()
            state["dirty_topic_item_count"] = 0
            state["last_compacted_at"] = datetime.now().isoformat(timespec="seconds")
            state["last_compact_reason"] = reason
            self._write_memory_state(state)
        result = {
            "status": "rebuilt",
            "reason": reason,
            "memory_index_path": str(self.memory_index_file),
        }
        self._log_maintenance("memory_index_rebuilt", result)
        return result

    def _collect_topic_text(self, max_chars: int = 24000) -> str:
        chunks = []
        with self._storage_lock:
            for topic in sorted(self.DEFAULT_TOPICS):
                path = self._topic_file(topic)
                text = path.read_text(encoding="utf-8").strip() if path.exists() else ""
                if not text or text == f"# {topic}":
                    continue
                chunks.append(f"\n\n## {topic}\n{text}")
        return self._limit_text("".join(chunks).strip(), max_chars)

    def _fallback_memory_index(self, topics_text: str) -> str:
        if not topics_text.strip():
            return self._empty_memory_index()
        lines = ["# MEMORY", "", "以下内容由本地兜底整理生成，建议后续手动检查。", ""]
        for line in topics_text.splitlines():
            if line.startswith("## ") or line.startswith("- "):
                lines.append(line)
            if len(lines) >= self.memory_index_line_limit:
                break
        return "\n".join(lines).strip()

    @staticmethod
    def _empty_memory_index() -> str:
        return "# MEMORY\n\n暂无稳定长期记忆。"

    def _increase_dirty_topic_count(self, count: int) -> None:
        if count <= 0:
            return
        with self._storage_lock:
            state = self._read_memory_state()
            state["dirty_topic_item_count"] = int(state.get("dirty_topic_item_count") or 0) + count
            state["last_topic_item_added_at"] = datetime.now().isoformat(timespec="seconds")
            self._write_memory_state(state)

    def _memory_index_line_count(self) -> int:
        with self._storage_lock:
            if not self.memory_index_file.exists():
                return 0
            text = self.memory_index_file.read_text(encoding="utf-8")
        return len(text.splitlines())

    def _default_memory_state(self) -> dict[str, Any]:
        return {
            "dirty_topic_item_count": 0,
            "last_compacted_at": None,
            "last_compact_reason": None,
            "last_topic_item_added_at": None,
        }

    def _read_memory_state(self) -> dict[str, Any]:
        try:
            with self._storage_lock:
                state = json.loads(self.memory_state_file.read_text(encoding="utf-8"))
        except Exception:
            return self._default_memory_state()
        if not isinstance(state, dict):
            return self._default_memory_state()
        default = self._default_memory_state()
        default.update(state)
        return default

    def _write_memory_state(self, state: dict[str, Any]) -> None:
        with self._storage_lock:
            self.memory_state_file.parent.mkdir(parents=True, exist_ok=True)
            self.memory_state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def _log_maintenance(self, action: str, payload: dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "action": action,
            **payload,
        }
        self._append_jsonl(self.maintenance_log_file, record)

    def _memory_request_client(self):
        # 优先使用 OpenAI SDK 的 with_options，为后台记忆请求单独设置短 timeout 和禁用重试。
        with_options = getattr(self.client, "with_options", None)
        if not callable(with_options):
            return self.client
        try:
            return with_options(timeout=self.memory_request_timeout, max_retries=self.memory_max_retries)
        except TypeError:
            try:
                return with_options(timeout=self.memory_request_timeout)
            except TypeError:
                return self.client

    def _create_memory_completion(self, **kwargs):
        # 统一封装后台记忆模型请求，避免新增记忆流程时漏掉 timeout 控制。
        request_client = self._memory_request_client()
        if request_client is self.client:
            kwargs.setdefault("timeout", self.memory_request_timeout)
        try:
            return request_client.chat.completions.create(**kwargs)
        except TypeError:
            # 兼容少量测试替身或旧版兼容客户端不接受 timeout 参数的情况。
            kwargs.pop("timeout", None)
            return request_client.chat.completions.create(**kwargs)

    def _load_recent_tasks(self) -> list[dict[str, Any]]:
        # 最近任务索引用于提供时间邻近性，不把完整历史任务全部注入 prompt。
        if not self.task_index_file.exists():
            return []
        lines = self.task_index_file.read_text(encoding="utf-8").splitlines()
        records = []
        deleted_task_ids, deleted_turn_refs = self._load_deleted_memory_refs()
        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if self._is_deleted_memory_record(record, deleted_task_ids, deleted_turn_refs):
                continue
            records.append(record)
        return records[-self.recent_limit:]

    def _load_deleted_memory_refs(self) -> tuple[set[str], set[tuple[str | None, str]]]:
        """
        读取被软删除的长期记忆引用。
        :return: (task_id 集合, (session_id, turn_id) 集合)
        """
        if not self.deleted_tasks_file.exists():
            return set(), set()
        deleted_task_ids: set[str] = set()
        deleted_turn_refs: set[tuple[str | None, str]] = set()
        for line in self.deleted_tasks_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            task_id = record.get("task_id")
            turn_id = record.get("turn_id")
            if task_id:
                deleted_task_ids.add(str(task_id))
            if turn_id:
                deleted_turn_refs.add((record.get("session_id"), str(turn_id)))
        return deleted_task_ids, deleted_turn_refs

    def _is_deleted_memory_record(
        self,
        record: dict[str, Any],
        deleted_task_ids: set[str],
        deleted_turn_refs: set[tuple[str | None, str]],
    ) -> bool:
        """
        判断长期记忆索引是否已被软删除。
        :param record: task_index 记录
        :param deleted_task_ids: 被删除的 task_id
        :param deleted_turn_refs: 被删除的 session/turn 引用
        :return: 是否应过滤
        """
        task_id = record.get("task_id")
        if task_id and str(task_id) in deleted_task_ids:
            return True
        turn_id = record.get("turn_id")
        if not turn_id:
            return False
        session_id = record.get("session_id")
        return (session_id, str(turn_id)) in deleted_turn_refs or (None, str(turn_id)) in deleted_turn_refs

    def _read_jsonl_records(self, path: Path) -> list[dict[str, Any]]:
        with self._storage_lock:
            if not path.exists():
                return []
            records = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
        return records

    def _write_jsonl_records(self, path: Path, records: list[dict[str, Any]]) -> None:
        content = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
        if content:
            content += "\n"
        with self._storage_lock:
            path.write_text(content, encoding="utf-8")

    @staticmethod
    def _normalize_task_ids(task_ids: list[str] | set[str]) -> set[str]:
        return {
            str(task_id)
            for task_id in task_ids
            if task_id and re.fullmatch(r"task_\d{8}_\d{6}_[0-9a-f]{6}", str(task_id))
        }

    @staticmethod
    def _memory_record_matches(
            record: dict[str, Any],
            task_ids: set[str],
            session_ids: set[str],
            turn_ids: set[str] | None = None,
    ) -> bool:
        task_id = str(record.get("task_id") or "")
        session_id = str(record.get("session_id") or "")
        turn_id = str(record.get("turn_id") or "")
        if task_id and task_id in task_ids:
            return True
        if turn_ids:
            # 指定 turn_ids 时，session_ids 只作为作用域限制，不再表示删除整个 session。
            if turn_id not in turn_ids:
                return False
            return not session_ids or (session_id and session_id in session_ids)
        return bool(session_id and session_id in session_ids)

    def _delete_full_context_file(self, path_value: str) -> str:
        path = self._safe_full_context_path(path_value)
        if path is None:
            return "unsafe"
        if not path.exists():
            return "missing"
        # 每次只删除一个 MemoryManager 自己管理的明确完整上下文文件。
        path.unlink()
        return "deleted"

    def _remove_topic_items_for_tasks(self, task_ids: set[str]) -> int:
        if not task_ids:
            return 0
        removed_count = 0
        with self._storage_lock:
            for topic in self.DEFAULT_TOPICS:
                topic_path = self._topic_file(topic)
                if not topic_path.exists():
                    continue
                lines = topic_path.read_text(encoding="utf-8").splitlines()
                original_count = len(lines)
                kept_lines = [
                    line
                    for line in lines
                    if not any(f"source: {task_id}" in line for task_id in task_ids)
                ]
                removed_count += original_count - len(kept_lines)
                if kept_lines:
                    topic_path.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
                else:
                    topic_path.write_text(f"# {topic}\n", encoding="utf-8")
        return removed_count

    def _safe_full_context_path(self, path_value: str) -> Path | None:
        try:
            path = Path(path_value)
            if not path.is_absolute():
                path = self.project_root / path
            resolved_path = path.resolve()
            resolved_root = self.full_context_dir.resolve()
            resolved_path.relative_to(resolved_root)
        except (OSError, ValueError):
            return None
        if not re.fullmatch(r"task_\d{8}_\d{6}_[0-9a-f]{6}\.json", resolved_path.name):
            return None
        return resolved_path

    def _append_task_index(self, record: dict[str, Any]) -> None:
        self._append_jsonl(self.task_index_file, record)

    def _append_task_summary(self, summary: dict[str, Any]) -> None:
        self._append_jsonl(self.task_summaries_file, summary)

    def _append_jsonl(self, path: Path, record: dict[str, Any]) -> None:
        with self._storage_lock:
            with path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _cleanup_finished_futures(self):
        # 清理已完成的后台任务，避免 future 列表无限增长；异常统一写入 memory_errors.log。
        finished = []
        remaining = []
        with self._futures_lock:
            for future in self._futures:
                if future.done():
                    finished.append(future)
                else:
                    remaining.append(future)
            self._futures = remaining
        for future in finished:
            self._log_future_exception(future)

    def _log_future_exception(self, future):
        try:
            future.result()
        except Exception:
            self._log_error("memory future failed", traceback.format_exc())

    def _log_error(self, title: str, detail: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._storage_lock:
            with self.error_log_file.open("a", encoding="utf-8") as file:
                file.write(f"\n## {timestamp} {title}\n{detail}\n")

    def _limit_text(self, text: str, limit: int) -> str:
        text = text or ""
        if len(text) <= limit:
            return text
        return text[:limit] + "\n...[truncated]"

    def _string_or_default(self, value: Any, default: str, limit: int) -> str:
        if not isinstance(value, str) or not value.strip():
            return default
        return self._limit_text(value.strip(), limit)

    def _string_list(self, value: Any, limit: int) -> list[str]:
        if not isinstance(value, list):
            return []
        items = []
        for item in value[:limit]:
            if isinstance(item, str) and item.strip():
                items.append(self._limit_text(item.strip(), 240))
        return items
