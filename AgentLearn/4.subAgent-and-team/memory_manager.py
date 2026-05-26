# encoding: utf-8
import json
import re
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, wait
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any
from prompt_loader import load_prompt


class MemoryManager:
    """管理分层长期记忆，并把记忆整理流程隐藏在 Agent 运行时内部。"""

    def __init__(
        self,
        project_root: str | Path,
        client: Any,
        model: str,
        temperature: float = 0.1,
        recent_limit: int = 10,
        prompt_char_limit: int = 6000,
    ):
        self.project_root = Path(project_root)
        self.prompts_dir = self.project_root / "agent" / "prompts"
        self.client = client
        self.model = model
        self.temperature = temperature
        self.recent_limit = recent_limit
        self.prompt_char_limit = prompt_char_limit

        self.memory_dir = self.project_root / "agent" / "memory"
        self.full_context_dir = self.memory_dir / "full_context"
        self.global_summary_file = self.memory_dir / "global_summary.md"
        self.task_index_file = self.memory_dir / "task_index.jsonl"
        self.task_summaries_file = self.memory_dir / "task_summaries.jsonl"
        self.deleted_tasks_file = self.memory_dir / "deleted_tasks.jsonl"
        self.error_log_file = self.memory_dir / "memory_errors.log"

        # 只使用一个后台线程，保证 jsonl 追加和 global_summary.md 重写的顺序稳定。
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="memory-manager")
        self._futures = []
        self._shutdown = False
        self._ensure_memory_files()

    def _ensure_memory_files(self):
        # 启动时按需创建目录和空文件，避免新检出项目需要手动初始化记忆结构。
        self.full_context_dir.mkdir(parents=True, exist_ok=True)
        for file_path in (
            self.global_summary_file,
            self.task_index_file,
            self.task_summaries_file,
            self.deleted_tasks_file,
            self.error_log_file,
        ):
            file_path.touch(exist_ok=True)

    def enqueue(
        self,
        task: str,
        result: str,
        context: list[dict[str, Any]],
        session_id: str | None = None,
        turn_id: str | None = None,
    ):
        if self._shutdown:
            return None
        # Agent.chat() 结束后会清空当前任务上下文，因此这里先复制快照再交给后台线程。
        context_snapshot = deepcopy(context or [])
        metadata = {"session_id": session_id, "turn_id": turn_id}
        future = self._executor.submit(self._process_memory_update, task, result, context_snapshot, metadata)
        self._futures.append(future)
        self._cleanup_finished_futures()
        return future

    def has_pending(self) -> bool:
        self._cleanup_finished_futures()
        return any(not future.done() for future in self._futures)

    def wait_for_pending(self, timeout: float | None = None) -> bool:
        # Agent 退出前调用这里，等待用户无感的后台记忆整理任务完成。
        self._cleanup_finished_futures()
        if not self._futures:
            return True
        done, not_done = wait(self._futures, timeout=timeout)
        for future in done:
            self._log_future_exception(future)
        self._futures = list(not_done)
        return not not_done

    def shutdown(self):
        if self._shutdown:
            return
        self.wait_for_pending()
        self._executor.shutdown(wait=True)
        self._shutdown = True

    def load_prompt_memory_view(self) -> str:
        return self._load_prompt_memory_view()

    def _load_prompt_memory_view(self) -> str:
        # system prompt 只注入有长度上限的记忆视图，完整任务上下文只保存在磁盘上。
        self._ensure_memory_files()
        global_summary = self.global_summary_file.read_text(encoding="utf-8").strip()
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
            global_summary=global_summary or "No stable long-term memory recorded.",
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
        self.global_summary_file.write_text("", encoding="utf-8")
        self.task_index_file.write_text("", encoding="utf-8")
        self.task_summaries_file.write_text("", encoding="utf-8")
        self.deleted_tasks_file.write_text("", encoding="utf-8")
        self.error_log_file.write_text("", encoding="utf-8")

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
            # 先保存完整上下文，再写索引；这样索引里的 full_context_path 始终指向真实文件。
            metadata = metadata or {}
            task_id = self._create_task_id()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            full_context_path = self._save_task_full_context(task_id, timestamp, task, result, context, metadata)
            summary = self._summarize_task_memory(task_id, timestamp, task, result, context)
            index_record = self._build_index_record(task_id, timestamp, summary, full_context_path, metadata)
            self._append_task_index(index_record)
            self._append_task_summary(summary)
            self._update_global_memory_summary(summary)
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
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

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
            response = self.client.chat.completions.create(
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
            "full_context_path": full_context_path,
        }
        if metadata.get("session_id"):
            record["session_id"] = metadata.get("session_id")
        if metadata.get("turn_id"):
            record["turn_id"] = metadata.get("turn_id")
        return record

    def _update_global_memory_summary(self, task_summary: dict[str, Any]):
        # global_summary.md 由“旧汇总 + 新任务摘要”重新生成，只保留长期稳定的信息。
        current_summary = self.global_summary_file.read_text(encoding="utf-8").strip()
        user_prompt = json.dumps(
            {
                "current_global_summary": self._limit_text(current_summary, 8000),
                "new_task_summary": task_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": load_prompt("memory_global_summary_system.md", prompts_dir=self.prompts_dir),
                    },
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
            )
            updated_summary = response.choices[0].message.content.strip()
            self.global_summary_file.write_text(updated_summary, encoding="utf-8")
        except Exception:
            self._log_error("global summary model call failed", traceback.format_exc())

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

    def _append_task_index(self, record: dict[str, Any]) -> None:
        self._append_jsonl(self.task_index_file, record)

    def _append_task_summary(self, summary: dict[str, Any]) -> None:
        self._append_jsonl(self.task_summaries_file, summary)

    def _append_jsonl(self, path: Path, record: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _cleanup_finished_futures(self):
        # 清理已完成的后台任务，避免 future 列表无限增长；异常统一写入 memory_errors.log。
        remaining = []
        for future in self._futures:
            if future.done():
                self._log_future_exception(future)
            else:
                remaining.append(future)
        self._futures = remaining

    def _log_future_exception(self, future):
        try:
            future.result()
        except Exception:
            self._log_error("memory future failed", traceback.format_exc())

    def _log_error(self, title: str, detail: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
