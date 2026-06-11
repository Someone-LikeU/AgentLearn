# encoding: utf-8
import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


class SessionManager:
    """管理当前 Agent 会话的追加式 JSONL 日志。"""

    def __init__(self, project_root: str | Path, sessions_dir: str | Path | None = None):
        self.project_root = Path(project_root)
        # 默认把会话文件放在当前阶段目录的 sessions 下，测试时可传入临时目录。
        self.sessions_dir = Path(sessions_dir) if sessions_dir else self.project_root / "sessions"
        # 索引文件只记录会话元信息和路径，真实事件仍写入每个 session jsonl。
        self.index_file = self.sessions_dir / "session_index.jsonl"
        self.current_session_id: str | None = None
        self.current_session_path: Path | None = None
        self._session_closed = False
        self.current_session_title: str | None = None
        self.current_session_title_source: str | None = None
        self.current_session_model: str | None = None
        self.current_session_created_at: str | None = None
        # 会话事件可能由记忆保存回调写入，使用可重入锁保证单行 JSONL 追加不交错。
        self._write_lock = threading.RLock()
        self._ensure_sessions_root()

    def start_session(
        self,
        model: str,
        metadata: dict[str, Any] | None = None,
        title: str | None = None,
        title_source: str | None = None,
    ) -> str:
        """
        启动一个新会话，并按日期目录创建对应 jsonl 文件。
        :param model: 当前模型名
        :param metadata: 会话元信息
        :param title: 初始会话标题
        :param title_source: 初始标题来源
        :return: session_id
        """
        # 会话文件按日期归档，避免 sessions 根目录长期堆积过多文件。
        created_at = self._now()
        date_dir = self.sessions_dir / datetime.now().strftime("%Y") / datetime.now().strftime("%m") / datetime.now().strftime("%d")
        date_dir.mkdir(parents=True, exist_ok=True)

        # session_id 同时包含时间和随机后缀，便于人工定位且避免同秒冲突。
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        session_path = date_dir / f"{session_id}.jsonl"

        # 当前会话路径由 SessionManager 统一持有，Agent 不直接处理文件路径。
        self.current_session_id = session_id
        self.current_session_path = session_path
        self.current_session_title = self._normalize_title(title) or "未命名会话"
        self.current_session_title_source = title_source if title_source in {"user", "auto", "default"} else "default"
        self.current_session_model = model
        self.current_session_created_at = created_at
        self._session_closed = False

        # session_start 是会话文件的第一条事件，后续恢复时可从这里读取模型和元信息。
        session_start = {
            "event": "session_start",
            "event_id": self._new_event_id(),
            "session_id": session_id,
            "created_at": created_at,
            "timestamp": created_at,
            "model": model,
            "title": self.current_session_title,
            "title_source": self.current_session_title_source,
            "metadata": metadata or {},
        }
        self._append_jsonl(session_path, session_start)
        # 额外维护轻量索引，list_sessions 不需要扫描所有历史文件内容。
        self._append_jsonl(
            self.index_file,
            {
                "event": "session_index",
                "session_id": session_id,
                "created_at": created_at,
                "updated_at": created_at,
                "model": model,
                "title": self.current_session_title,
                "title_source": self.current_session_title_source,
                "path": str(session_path),
                "status": "active",
                "has_user_task": False,
                "metadata": metadata or {},
            },
        )
        return session_id

    def append_event(self, event: dict[str, Any]) -> dict[str, Any]:
        """
        向当前会话追加一条事件。
        :param event: 事件内容
        :return: 实际写入的事件
        """
        if self.current_session_path is None or self.current_session_id is None:
            raise RuntimeError("Session has not been started.")
        # 统一补齐事件元信息，调用方只需要关心业务字段。
        record = self._json_safe(dict(event))
        record.setdefault("event_id", self._new_event_id())
        record.setdefault("session_id", self.current_session_id)
        record.setdefault("timestamp", self._now())
        self._append_jsonl(self.current_session_path, record)
        return record

    def append_event_to_session(self, session_id: str, event: dict[str, Any]) -> dict[str, Any]:
        """
        向指定历史会话追加事件，不改变当前会话指针。
        """
        target_session_id = str(session_id or "").strip()
        path = self._resolve_session_path(target_session_id)
        if path is None or not path.exists():
            raise FileNotFoundError(f"Session not found: {target_session_id}")
        record = self._json_safe(dict(event))
        record.setdefault("event_id", self._new_event_id())
        record.setdefault("session_id", target_session_id)
        record.setdefault("timestamp", self._now())
        self._append_jsonl(path, record)
        return record

    def append_message(
        self,
        message: Any,
        turn_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        追加一条 OpenAI message，并转换为 session 事件。
        :param message: dict 或 OpenAI message 对象
        :param turn_id: 当前用户任务 id
        :param metadata: 仅写入 session 的辅助元信息
        :return: 实际写入的事件
        """
        # OpenAI SDK 对象、dict 和普通对象先统一转为可 JSON 序列化的 dict。
        normalized = self.normalize_message(message)
        role = normalized.get("role")
        content = normalized.get("content")
        # 所有上下文消息都写 message_id，后续做详情查看或审计时可以精确定位。
        base_event = {
            "message_id": self._new_message_id(),
            "turn_id": turn_id,
            "role": role,
            "content": content,
        }
        if metadata:
            base_event["metadata"] = metadata

        # assistant 的工具调用是模型输出的一部分，随 assistant_message 一起保存。
        if role == "assistant":
            base_event["event"] = "assistant_message"
            tool_calls = normalized.get("tool_calls")
            if tool_calls is not None:
                base_event["tool_calls"] = tool_calls
        # tool 返回需要保留 tool_call_id，恢复 messages 时才能和 assistant tool_calls 对齐。
        elif role == "tool":
            base_event["event"] = "tool_result"
            base_event["tool_call_id"] = normalized.get("tool_call_id")
        else:
            base_event["event"] = "message"

        return self.append_event(base_event)

    def record_model_stream_error(self, turn_id: str | None, partial_content: str, error: str) -> dict[str, Any]:
        """
        记录流式响应异常；正常流式片段不落盘。
        :param turn_id: 当前用户任务 id
        :param partial_content: 异常前已收到的内容
        :param error: 错误信息
        :return: 实际写入的事件
        """
        return self.append_event(
            {
                "event": "model_stream_error",
                "turn_id": turn_id,
                "partial_content": partial_content,
                "error": error,
            }
        )

    def record_response_usage(
        self,
        turn_id: str | None,
        usage: Any,
        response_kind: str,
        model: str | None = None,
        message_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        记录一次模型响应的真实 usage。
        :param turn_id: 当前用户任务 id
        :param usage: OpenAI usage 字典或对象
        :param response_kind: 响应类型
        :param model: 当前模型名
        :param message_id: 关联的 assistant message_id
        :param metadata: 额外元信息
        :return: 实际写入的事件
        """
        normalized_usage = self._normalize_usage(usage)
        if not normalized_usage:
            raise ValueError("Response usage cannot be empty.")
        event = {
            "event": "response_usage",
            "turn_id": turn_id,
            "response_kind": response_kind or "unknown",
            "usage": normalized_usage,
        }
        if model:
            event["model"] = model
        if message_id:
            event["message_id"] = message_id
        if metadata:
            event["metadata"] = metadata
        return self.append_event(event)

    def calculate_session_usage(self, session_id: str | None = None, include_deleted: bool = False) -> dict[str, Any]:
        """
        读取当前主会话上下文的真实模型 usage。
        :param session_id: 会话 id，默认当前会话
        :param include_deleted: 是否允许已软删除 turn 的 assistant_response 参与恢复
        :return: 最新 assistant_response 的 usage
        """
        target_session_id = session_id or self.current_session_id
        summary = self._empty_usage_summary()
        if not target_session_id:
            return summary
        events = self._events_after_latest_session_clear(self.load_session(target_session_id))
        if include_deleted:
            return self._assistant_usage_summary(events)

        deleted_turn_ids = self._deleted_turn_ids(events)
        latest_delete_index = self._latest_turn_deleted_index(events)
        if latest_delete_index < 0:
            return self._assistant_usage_summary(events, deleted_turn_ids=deleted_turn_ids)

        # 删除后如果已经发生新的主模型响应，该 usage 才代表当前重建后的上下文。
        post_delete_summary = self._assistant_usage_summary(
            events[latest_delete_index + 1:],
            deleted_turn_ids=deleted_turn_ids,
        )
        if post_delete_summary.get("has_real_usage"):
            return post_delete_summary

        task_turn_ids = self._task_entry_turn_ids(events)
        active_turn_ids = [turn_id for turn_id in task_turn_ids if turn_id not in deleted_turn_ids]
        if not active_turn_ids:
            return summary
        if task_turn_ids[:len(active_turn_ids)] != active_turn_ids:
            # 删除中间任务但保留后续任务时，旧 usage 对当前上下文不再可信。
            summary["is_stale"] = True
            return summary
        return self._assistant_usage_summary(events, allowed_turn_ids=set(active_turn_ids))

    def record_memory_saved(
        self,
        turn_id: str,
        task_id: str,
        full_context_path: str,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        记录长期记忆保存结果。
        :param turn_id: 当前用户任务 id
        :param task_id: MemoryManager 生成的任务 id
        :param full_context_path: 完整上下文文件路径
        :param metadata: 额外元信息
        :return: 实际写入的事件
        """
        event = {
            "event": "memory_saved",
            "turn_id": turn_id,
            "task_id": task_id,
            "full_context_path": full_context_path,
        }
        if metadata:
            event["metadata"] = metadata
        if session_id:
            return self.append_event_to_session(session_id, event)
        return self.append_event(event)

    def record_memory_checked(
        self,
        turn_id: str | None,
        last_memory_checked_event_id: str | None,
        status: str,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        记录长期记忆处理游标，避免继续旧 session 时重复扫描旧历史。
        """
        event = {
            "event": "memory_checked",
            "turn_id": turn_id,
            "last_memory_checked_event_id": last_memory_checked_event_id,
            "status": status or "unknown",
        }
        if task_id:
            event["task_id"] = task_id
        if metadata:
            event["metadata"] = metadata
        if session_id:
            return self.append_event_to_session(session_id, event)
        return self.append_event(event)

    def record_session_cleared(
            self,
            delete_memory: bool = True,
            memory_task_ids: list[str] | None = None,
            reason: str = "user_clear_current_session",
    ) -> dict[str, Any]:
        """
        记录当前会话被逻辑清空。
        :param delete_memory: 是否同步删除本会话关联的长期记忆
        :param memory_task_ids: 已处理的长期记忆 task_id
        :param reason: 清空原因
        :return: 实际写入的事件
        """
        if self.current_session_path is None or self.current_session_id is None:
            raise RuntimeError("Session has not been started.")
        cleared_at = self._now()
        event = {
            "event": "session_cleared",
            "cleared_at": cleared_at,
            "reason": reason,
            "delete_memory": bool(delete_memory),
        }
        normalized_task_ids = [task_id for task_id in (memory_task_ids or []) if task_id]
        if normalized_task_ids:
            event["memory_task_ids"] = normalized_task_ids
        record = self.append_event(event)
        # 逻辑清空后当前 session 仍然存在，但后续 history/continue 应只看清空后的任务。
        self._append_session_index_update(
            {
                "updated_at": cleared_at,
                "status": "active",
                "has_user_task": False,
            }
        )
        return record

    def session_memory_refs(self, session_id: str) -> list[dict[str, Any]]:
        """
        读取指定 session 已关联的长期记忆引用。
        :param session_id: 会话 id
        :return: 记忆引用列表
        """
        refs = []
        for event in self.load_session(session_id):
            if event.get("event") != "memory_saved":
                continue
            refs.append(
                {
                    "session_id": event.get("session_id") or session_id,
                    "turn_id": event.get("turn_id"),
                    "task_id": event.get("task_id"),
                    "full_context_path": event.get("full_context_path"),
                }
            )
        return refs

    def delete_session_file(self, session_id: str) -> dict[str, Any]:
        """
        硬删除一个明确 session 文件。
        :param session_id: 会话 id
        :return: 删除结果
        """
        target_session_id = str(session_id or "").strip()
        if not target_session_id:
            return {"session_id": target_session_id, "deleted": False, "error": "empty_session_id"}
        path = self._resolve_session_path(target_session_id)
        if path is None or not path.exists():
            return {"session_id": target_session_id, "deleted": False, "error": "session_not_found"}

        # 每次只删除一个明确路径的 session 文件，符合项目禁止批量删除的约束。
        path.unlink()
        deleted_at = self._now()
        if self.current_session_id == target_session_id:
            self.detach_current_session()
        self._append_jsonl(
            self.index_file,
            {
                "event": "session_index_update",
                "session_id": target_session_id,
                "updated_at": deleted_at,
                "deleted_at": deleted_at,
                "status": "deleted",
                "path": str(path),
                "has_user_task": False,
            },
        )
        return {"session_id": target_session_id, "deleted": True, "path": str(path)}

    def record_session_interrupted(self, reason: str, turn_id: str | None = None) -> dict[str, Any]:
        """
        记录会话被外部中断。
        :param reason: 中断原因
        :param turn_id: 中断时正在执行的用户任务 id
        :return: 实际写入的事件
        """
        event = {
            "event": "session_interrupted",
            "reason": reason or "unknown",
            "interrupted_at": self._now(),
        }
        if turn_id:
            event["turn_id"] = turn_id
        return self.append_event(event)

    def update_title(self, title: str, source: str = "user") -> dict[str, Any]:
        """
        更新当前会话标题。
        :param title: 新标题
        :param source: 标题来源，user 或 auto
        :return: 实际写入的事件；若自动标题被用户标题挡住，则返回 skipped 事件
        """
        if self.current_session_path is None or self.current_session_id is None:
            raise RuntimeError("Session has not been started.")
        normalized_title = self._normalize_title(title)
        if not normalized_title:
            raise ValueError("Session title cannot be empty.")
        source = source if source in {"user", "auto", "default"} else "user"

        with self._write_lock:
            # 用户手动标题优先，较晚完成的自动标题不能覆盖用户设置。
            if source == "auto" and self.current_session_title_source == "user":
                return {
                    "event": "session_title_update_skipped",
                    "session_id": self.current_session_id,
                    "title": self.current_session_title,
                    "source": source,
                    "reason": "user_title_exists",
                }

            updated_at = self._now()
            event = self.append_event(
                {
                    "event": "session_title_updated",
                    "title": normalized_title,
                    "source": source,
                    "updated_at": updated_at,
                }
            )
            self.current_session_title = normalized_title
            self.current_session_title_source = source
            self._append_session_index_update(
                {
                    "title": normalized_title,
                    "title_source": source,
                    "updated_at": updated_at,
                }
            )
            return event

    def get_current_session_info(self) -> dict[str, Any]:
        """
        获取当前会话元信息。
        :return: 当前会话信息
        """
        return {
            "session_id": self.current_session_id,
            "path": str(self.current_session_path) if self.current_session_path else None,
            "title": self.current_session_title,
            "title_source": self.current_session_title_source,
            "model": self.current_session_model,
            "created_at": self.current_session_created_at,
            "status": "ended" if self._session_closed else "active",
        }

    def switch_session(self, session_id: str) -> dict[str, Any]:
        """
        切换当前 SessionManager 指向的会话。
        :param session_id: 目标会话 id
        :return: 切换后的会话信息
        """
        target_session_id = str(session_id or "").strip()
        if not target_session_id:
            raise ValueError("Session id cannot be empty.")

        path = self._resolve_session_path(target_session_id)
        if path is None or not path.exists():
            raise FileNotFoundError(f"Session not found: {target_session_id}")

        events = self._read_jsonl(path)
        session_start = next((event for event in events if event.get("event") == "session_start"), None)
        if not session_start:
            raise ValueError(f"Invalid session file without session_start: {path}")

        title_info = self._latest_session_title_info(events)
        resumed_at = self._now()

        # 切换只更新内存指针和追加恢复事件，不重写历史 session 文件。
        self.current_session_id = session_start.get("session_id") or target_session_id
        self.current_session_path = path
        self.current_session_title = title_info["title"]
        self.current_session_title_source = title_info["source"]
        self.current_session_model = session_start.get("model")
        self.current_session_created_at = session_start.get("created_at") or session_start.get("timestamp")
        self._session_closed = False

        # 加载动作不作为模型上下文消息，但需要写入事件流，标记后续消息属于恢复后的继续对话。
        resume_event = self.append_event({"event": "session_resumed", "resumed_at": resumed_at})
        self._append_session_index_update(
            {
                "updated_at": resumed_at,
                "status": "active",
                "path": str(path),
                "title": self.current_session_title,
                "title_source": self.current_session_title_source,
                "model": self.current_session_model,
                "created_at": self.current_session_created_at,
            }
        )
        info = self.get_current_session_info()
        info["resume_event_id"] = resume_event.get("event_id")
        return info

    def mark_turn_deleted(
        self,
        turn_id: str,
        task_id: str | None = None,
        task_ids: list[str] | None = None,
        reason: str = "user_deleted_from_session",
    ) -> dict[str, Any]:
        """
        软删除一个用户任务对应的完整 turn。
        :param turn_id: 要删除的任务 id
        :param task_id: 关联的长期记忆任务 id
        :param task_ids: 关联的长期记忆任务 id 列表
        :param reason: 删除原因
        :return: 实际写入的事件
        """
        normalized_task_ids = [item for item in (task_ids or []) if item]
        if task_id and task_id not in normalized_task_ids:
            normalized_task_ids.insert(0, task_id)
        event = {
            "event": "turn_deleted",
            "turn_id": turn_id,
            "deleted_at": self._now(),
            "reason": reason,
        }
        if normalized_task_ids:
            event["task_id"] = normalized_task_ids[0]
            event["task_ids"] = normalized_task_ids
        return self.append_event(event)

    def end_session(self) -> None:
        """追加 session_end 事件，重复调用不会重复写入。"""
        # 退出路径可能被 exit 命令和 KeyboardInterrupt 同时触发，避免重复写 session_end。
        if self._session_closed or self.current_session_path is None:
            return
        ended_at = self._now()
        self.append_event({"event": "session_end", "ended_at": ended_at})
        self._append_session_index_update({"updated_at": ended_at, "ended_at": ended_at, "status": "ended"})
        self._session_closed = True

    def detach_current_session(self) -> None:
        """
        清除当前会话指针，不创建新文件。
        :return:
        """
        # session new 只切到待创建状态，下一条真实用户任务到来时再落盘。
        self.current_session_id = None
        self.current_session_path = None
        self.current_session_title = None
        self.current_session_title_source = None
        self.current_session_model = None
        self.current_session_created_at = None
        self._session_closed = False

    def touch_session(self, reason: str = "task_completed") -> None:
        """
        更新会话索引中的最近活跃时间。
        :param reason: 更新时间原因
        :return:
        """
        if not self.current_session_id:
            return
        changes = {"updated_at": self._now(), "reason": reason}
        if reason == "task_completed":
            # 任务完成后标记该 session 不是空会话，continue 可据此跳过无用户任务的历史文件。
            changes["has_user_task"] = True
        self._append_session_index_update(changes)

    def list_sessions(self, limit: int = 20, include_empty: bool = True) -> list[dict[str, Any]]:
        """
        列出最近会话。
        :param limit: 返回数量上限
        :param include_empty: 是否包含没有用户任务的空会话
        :return: 会话索引列表
        """
        records = self._read_jsonl(self.index_file)
        if not records:
            # 索引缺失或为空时，回退扫描 session 文件，保证历史会话仍可找回。
            records = self._scan_session_files()
        else:
            records = self._merge_session_index_records(records)
        # 索引是追加式日志，用户手动删除 session 文件后可能留下孤儿记录；列表展示只保留文件仍存在的会话。
        records = [record for record in records if self._session_record_file_exists(record)]
        if not records:
            records = self._scan_session_files()
        # 同一秒内可能创建多个 session，因此用读取顺序作为稳定的第二排序键。
        ordered_records = []
        for order, record in enumerate(records):
            item = dict(record)
            item.setdefault("_order", order)
            ordered_records.append(item)
        ordered_records.sort(
            key=lambda item: (item.get("updated_at") or item.get("created_at", ""), item.get("_order", 0)),
            reverse=True,
        )
        if not include_empty:
            ordered_records = [record for record in ordered_records if self._session_has_user_task(record)]
        for item in ordered_records:
            item.pop("_order", None)
        return ordered_records[:limit]

    def load_session(self, session_id: str) -> list[dict[str, Any]]:
        """
        读取指定会话的全部事件。
        :param session_id: 会话 id
        :return: 事件列表
        """
        path = self._resolve_session_path(session_id)
        if path is None or not path.exists():
            return []
        return self._read_jsonl(path)

    def latest_event_id(self, session_id: str | None = None) -> str | None:
        """
        返回指定 session 当前最后一条事件的 event_id。
        """
        target_session_id = session_id or self.current_session_id
        if not target_session_id:
            return None
        events = self.load_session(target_session_id)
        for event in reversed(events):
            event_id = event.get("event_id")
            if event_id:
                return str(event_id)
        return None

    def latest_memory_checked_event_id(self, session_id: str | None = None) -> str | None:
        """
        返回最新长期记忆处理游标。
        """
        target_session_id = session_id or self.current_session_id
        if not target_session_id:
            return None
        events = self.load_session(target_session_id)
        for event in reversed(events):
            if event.get("event") == "memory_checked" and event.get("last_memory_checked_event_id"):
                return str(event.get("last_memory_checked_event_id"))
        return None

    def rebuild_messages(self, session_id: str) -> list[dict[str, Any]]:
        """
        从 session 事件重建可发送给模型的 messages。
        :param session_id: 会话 id
        :return: OpenAI messages
        """
        events = self._events_after_latest_session_clear(self.load_session(session_id))
        deleted_turn_ids = self._deleted_turn_ids(events)
        messages: list[dict[str, Any]] = []
        for event in events:
            turn_id = event.get("turn_id")
            # 被软删除的用户任务整轮跳过，包含其 user、assistant 和 tool 消息。
            if turn_id in deleted_turn_ids:
                continue
            event_type = event.get("event")
            if event_type == "message":
                role = event.get("role")
                # 普通 message 只恢复 system/user；assistant/tool 使用更具体的事件恢复。
                if role in {"system", "user"}:
                    messages.append({"role": role, "content": event.get("content")})
            elif event_type == "assistant_message":
                message = {"role": "assistant", "content": event.get("content")}
                # 只有真实存在 tool_calls 字段时才回填，避免无工具回复多出 None 字段。
                if "tool_calls" in event:
                    message["tool_calls"] = event.get("tool_calls")
                messages.append(message)
            elif event_type == "tool_result":
                # tool 消息必须保留 tool_call_id，OpenAI 上下文校验依赖这个字段。
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": event.get("tool_call_id"),
                        "content": event.get("content"),
                    }
                )
        return messages

    def list_tasks(self, session_id: str | None = None) -> list[dict[str, Any]]:
        """
        按用户任务粒度列出 history 项。
        :param session_id: 会话 id，默认当前会话
        :return: 任务列表
        """
        target_session_id = session_id or self.current_session_id
        if not target_session_id:
            return []
        events = self._events_after_latest_session_clear(self.load_session(target_session_id))
        # history 以用户任务为粒度展示，但统计信息需要从同 turn 的全部事件聚合。
        deleted_turn_ids = self._deleted_turn_ids(events)
        stats = self._build_turn_stats(events)
        tasks = []
        for event in events:
            # 列表入口只取 user message，不直接展示 assistant/tool 底层消息。
            if event.get("event") != "message" or event.get("role") != "user":
                continue
            turn_id = event.get("turn_id")
            if not turn_id or turn_id in deleted_turn_ids:
                continue
            metadata = event.get("metadata") or {}
            # 计划模式等内部 user 消息不作为 history 任务入口展示。
            if metadata.get("is_task_entry", True) is not True:
                continue
            turn_stats = stats.get(turn_id, {})
            tasks.append(
                {
                    "index": len(tasks) + 1,
                    "turn_id": turn_id,
                    "timestamp": event.get("timestamp"),
                    "content": event.get("content"),
                    "assistant_message_count": turn_stats.get("assistant_message_count", 0),
                    "tool_result_count": turn_stats.get("tool_result_count", 0),
                    "tool_call_count": turn_stats.get("tool_call_count", 0),
                    "memory_task_ids": turn_stats.get("memory_task_ids", []),
                    "final_output": turn_stats.get("final_output", ""),
                }
            )
        return tasks

    @staticmethod
    def create_turn_id() -> str:
        # turn_id 表示一次用户任务，后续删除和记忆关联都以它为聚合键。
        return f"turn_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    @staticmethod
    def normalize_message(message: Any) -> dict[str, Any]:
        # dict 直接使用；OpenAI SDK 对象优先使用官方的 model_dump。
        if isinstance(message, dict):
            normalized = message
        elif hasattr(message, "model_dump"):
            normalized = message.model_dump()
        elif hasattr(message, "to_dict"):
            normalized = message.to_dict()
        else:
            # 兜底兼容 SimpleNamespace 或少量自定义消息对象。
            normalized = {
                "role": getattr(message, "role", None),
                "content": getattr(message, "content", None),
            }
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls is not None:
                normalized["tool_calls"] = tool_calls
            tool_call_id = getattr(message, "tool_call_id", None)
            if tool_call_id is not None:
                normalized["tool_call_id"] = tool_call_id
        # 通过 json 往返清理不可序列化对象，避免追加 JSONL 时失败。
        return SessionManager._json_safe(normalized)

    def _ensure_sessions_root(self) -> None:
        # 只创建 sessions 根目录和索引文件；具体日期目录在 start_session 时创建。
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.index_file.touch(exist_ok=True)

    def _append_jsonl(self, path: Path, record: dict[str, Any]) -> None:
        # 每条事件固定一行 JSON，便于追加写入和逐行恢复。
        line = json.dumps(self._json_safe(record), ensure_ascii=False)
        with self._write_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # 单行损坏时跳过该行，不影响其他事件恢复。
                continue
        return records

    @classmethod
    def _normalize_usage(cls, usage: Any) -> dict[str, int]:
        normalized = {}
        prompt_tokens = cls._normalize_token_value(cls._usage_value(usage, "prompt_tokens"))
        completion_tokens = cls._normalize_token_value(cls._usage_value(usage, "completion_tokens"))
        total_tokens = cls._normalize_token_value(cls._usage_value(usage, "total_tokens"))
        if total_tokens is None:
            prompt = prompt_tokens or 0
            completion = completion_tokens or 0
            total_tokens = prompt + completion if prompt + completion > 0 else None
        if prompt_tokens is not None:
            normalized["prompt_tokens"] = prompt_tokens
        if completion_tokens is not None:
            normalized["completion_tokens"] = completion_tokens
        if total_tokens is not None:
            normalized["total_tokens"] = total_tokens
        return normalized

    @staticmethod
    def _usage_value(usage: Any, key: str) -> Any:
        if isinstance(usage, dict):
            return usage.get(key)
        return getattr(usage, key, None)

    @staticmethod
    def _normalize_token_value(value: Any) -> int | None:
        if isinstance(value, bool) or value is None:
            return None
        try:
            number = int(value)
        except (TypeError, ValueError):
            return None
        return number if number >= 0 else None

    def _resolve_session_path(self, session_id: str) -> Path | None:
        # 当前会话优先返回内存中的路径，避免刚创建但索引尚未重读时查找失败。
        if self.current_session_id == session_id and self.current_session_path is not None:
            return self.current_session_path
        # 常规路径从索引文件解析。
        for record in self._read_jsonl(self.index_file):
            if record.get("session_id") == session_id and record.get("path"):
                path = Path(record.get("path"))
                if path.exists():
                    return path
        # 索引不完整时最后再扫描目录。
        return self._find_session_file_by_id(session_id)

    def _session_record_file_exists(self, record: dict[str, Any]) -> bool:
        path_value = record.get("path")
        if path_value:
            path = Path(path_value)
            if path.exists():
                return True
            session_id = record.get("session_id")
            repaired_path = self._find_session_file_by_id(session_id)
            if repaired_path is None:
                return False
            record["path"] = str(repaired_path)
            self._record_session_path_repair(session_id, repaired_path)
            return True
        session_id = record.get("session_id")
        if not session_id:
            return False
        # 兼容极早期没有 path 字段的索引记录，必要时按 session_id 回退扫描。
        repaired_path = self._find_session_file_by_id(session_id)
        if repaired_path is None:
            return False
        record["path"] = str(repaired_path)
        self._record_session_path_repair(session_id, repaired_path)
        return True

    def _find_session_file_by_id(self, session_id: str | None) -> Path | None:
        """
        按 session_id 在当前 sessions 目录下查找会话文件。
        :param session_id: 会话 id
        :return: 找到的会话文件路径
        """
        if not session_id:
            return None
        for path in self.sessions_dir.rglob(f"{session_id}.jsonl"):
            if path.name == self.index_file.name:
                continue
            return path
        return None

    def _record_session_path_repair(self, session_id: str | None, path: Path) -> None:
        """
        追加索引路径修复事件，不改变会话 updated_at 排序字段。
        :param session_id: 会话 id
        :param path: 当前可用的会话文件路径
        :return:
        """
        if not session_id:
            return
        self._append_jsonl(
            self.index_file,
            {
                "event": "session_index_update",
                "session_id": session_id,
                "path": str(path),
                "path_repaired_at": self._now(),
                "reason": "repair_missing_index_path",
            },
        )

    def _scan_session_files(self) -> list[dict[str, Any]]:
        records = []
        for path in self.sessions_dir.rglob("session_*.jsonl"):
            events = self._read_jsonl(path)
            # 只把包含 session_start 的文件视为有效 session 文件。
            start = next((event for event in events if event.get("event") == "session_start"), None)
            if start:
                title_info = self._latest_session_title_info(events)
                records.append(
                    {
                        "session_id": start.get("session_id"),
                        "created_at": start.get("created_at") or start.get("timestamp"),
                        "updated_at": self._latest_session_updated_at(events),
                        "model": start.get("model"),
                        "title": title_info["title"],
                        "title_source": title_info["source"],
                        "path": str(path),
                        "status": self._latest_session_status(events),
                        "has_user_task": self._events_have_user_task(events),
                        "metadata": start.get("metadata") or {},
                    }
                )
        return records

    def _deleted_turn_ids(self, events: list[dict[str, Any]]) -> set[str]:
        # 删除采用追加 turn_deleted 软删除事件，不改写历史消息。
        return {
            event.get("turn_id")
            for event in events
            if event.get("event") == "turn_deleted" and event.get("turn_id")
        }

    @staticmethod
    def _empty_usage_summary() -> dict[str, Any]:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "response_count": 0,
            "has_real_usage": False,
            "is_stale": False,
        }

    def _assistant_usage_summary(
            self,
            events: list[dict[str, Any]],
            allowed_turn_ids: set[str] | None = None,
            deleted_turn_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        """
        从事件列表中恢复最后一次主会话 assistant_response usage。
        :param events: 已按 session_cleared 截断后的事件
        :param allowed_turn_ids: 只允许参与恢复的 turn_id
        :param deleted_turn_ids: 需要排除的软删除 turn_id
        :return: usage 汇总
        """
        summary = self._empty_usage_summary()
        latest_usage = None
        assistant_response_count = 0
        deleted_turn_ids = deleted_turn_ids or set()
        for event in events:
            if event.get("event") != "response_usage":
                continue
            # 当前上下文 token 只来自主会话 assistant_response；completion check 等独立请求不参与 status。
            if event.get("response_kind") != "assistant_response":
                continue
            turn_id = event.get("turn_id")
            if allowed_turn_ids is not None and turn_id not in allowed_turn_ids:
                continue
            if turn_id in deleted_turn_ids:
                continue
            usage = self._normalize_usage(event.get("usage"))
            if not usage:
                continue
            latest_usage = usage
            assistant_response_count += 1
        if latest_usage:
            prompt_tokens = latest_usage.get("prompt_tokens", 0)
            completion_tokens = latest_usage.get("completion_tokens", 0)
            summary["prompt_tokens"] = prompt_tokens
            summary["completion_tokens"] = completion_tokens
            summary["total_tokens"] = latest_usage.get("total_tokens", prompt_tokens + completion_tokens)
            summary["response_count"] = assistant_response_count
            summary["has_real_usage"] = True
        return summary

    @staticmethod
    def _task_entry_turn_ids(events: list[dict[str, Any]]) -> list[str]:
        turn_ids = []
        seen_turn_ids = set()
        for event in events:
            if event.get("event") != "message" or event.get("role") != "user":
                continue
            turn_id = event.get("turn_id")
            if not turn_id or turn_id in seen_turn_ids:
                continue
            metadata = event.get("metadata") or {}
            if metadata.get("is_task_entry", True) is not True:
                continue
            seen_turn_ids.add(turn_id)
            turn_ids.append(turn_id)
        return turn_ids

    @staticmethod
    def _latest_turn_deleted_index(events: list[dict[str, Any]]) -> int:
        latest_index = -1
        for index, event in enumerate(events):
            if event.get("event") == "turn_deleted":
                latest_index = index
        return latest_index

    @staticmethod
    def _latest_session_cleared_index(events: list[dict[str, Any]]) -> int:
        latest_index = -1
        for index, event in enumerate(events):
            if event.get("event") == "session_cleared":
                latest_index = index
        return latest_index

    def _events_after_latest_session_clear(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        latest_clear_index = self._latest_session_cleared_index(events)
        if latest_clear_index < 0:
            return events
        return events[latest_clear_index + 1:]

    def _build_turn_stats(self, events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        # 统计信息用于 history 详情页，避免 UI 层反复扫描事件。
        stats: dict[str, dict[str, Any]] = {}
        for event in events:
            turn_id = event.get("turn_id")
            if not turn_id:
                continue
            item = stats.setdefault(
                turn_id,
                {
                    "assistant_message_count": 0,
                    "tool_result_count": 0,
                    "tool_call_count": 0,
                    "memory_task_ids": [],
                    "final_output": "",
                },
            )
            if event.get("event") == "assistant_message":
                item["assistant_message_count"] += 1
                item["tool_call_count"] += len(event.get("tool_calls") or [])
                content = event.get("content")
                if content:
                    # 每个 turn 可能有多次模型响应，history 详情展示最后一次有文本的模型输出。
                    item["final_output"] = str(content)
            elif event.get("event") == "tool_result":
                item["tool_result_count"] += 1
            elif event.get("event") == "memory_saved" and event.get("task_id"):
                # 一个 turn 理论上只有一个 task_id，但用列表兼容后续多记忆记录场景。
                item["memory_task_ids"].append(event.get("task_id"))
        return stats

    def _session_has_user_task(self, record: dict[str, Any]) -> bool:
        # 新索引会直接记录 has_user_task；旧索引缺失时回退扫描 session 文件。
        if record.get("has_user_task") is True:
            return True
        session_id = record.get("session_id")
        if not session_id:
            return False
        path_value = record.get("path")
        if path_value:
            events = self._read_jsonl(Path(path_value))
        else:
            events = self.load_session(session_id)
        return self._events_have_user_task(events)

    def _events_have_user_task(self, events: list[dict[str, Any]]) -> bool:
        # 空会话只包含 session_start/system/session_end，不应被 continue 当作上次会话。
        events = self._events_after_latest_session_clear(events)
        deleted_turn_ids = self._deleted_turn_ids(events)
        for event in events:
            if event.get("event") != "message" or event.get("role") != "user":
                continue
            turn_id = event.get("turn_id")
            if not turn_id or turn_id in deleted_turn_ids:
                continue
            metadata = event.get("metadata") or {}
            if metadata.get("is_task_entry", True) is True:
                return True
        return False

    def _append_session_index_update(self, changes: dict[str, Any]) -> None:
        # session_index_update 采用追加式写入，客户端列表读取时按 session_id 合并。
        if not self.current_session_id:
            return
        event = {
            "event": "session_index_update",
            "session_id": self.current_session_id,
            "updated_at": changes.get("updated_at") or self._now(),
        }
        event.update(changes)
        self._append_jsonl(self.index_file, event)

    def _merge_session_index_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # 兼容旧索引格式：没有 event 字段的历史记录视为 session_index。
        merged: dict[str, dict[str, Any]] = {}
        for order, record in enumerate(records):
            session_id = record.get("session_id")
            if not session_id:
                continue
            event_type = record.get("event") or "session_index"
            item = merged.setdefault(session_id, {"session_id": session_id, "_order": order})
            if event_type in {"session_index", "session_index_update"}:
                for key, value in record.items():
                    if key == "event" or value is None:
                        continue
                    item[key] = value
                item["_order"] = order
        result = []
        for item in merged.values():
            item.setdefault("title", "未命名会话")
            item.setdefault("title_source", "default")
            item.setdefault("updated_at", item.get("created_at"))
            result.append(item)
        return result

    def _latest_session_title_info(self, events: list[dict[str, Any]]) -> dict[str, str]:
        title = "未命名会话"
        source = "default"
        for event in events:
            if event.get("event") == "session_start":
                title = self._normalize_title(event.get("title")) or title
                source = event.get("title_source") or source
            elif event.get("event") == "session_title_updated":
                title = self._normalize_title(event.get("title")) or title
                source = event.get("source") or source
        return {"title": title, "source": source}

    def _latest_session_updated_at(self, events: list[dict[str, Any]]) -> str | None:
        updated_at = None
        for event in events:
            updated_at = event.get("updated_at") or event.get("timestamp") or updated_at
        return updated_at

    @staticmethod
    def _latest_session_status(events: list[dict[str, Any]]) -> str:
        status = "active"
        for event in events:
            if event.get("event") == "session_end":
                status = "ended"
        return status

    @staticmethod
    def _normalize_title(title: Any, limit: int = 60) -> str:
        # 标题只保留单行文本，避免会话列表渲染时被换行破坏。
        normalized = " ".join(str(title or "").strip().split())
        if len(normalized) > limit:
            return normalized[:limit]
        return normalized

    @staticmethod
    def _json_safe(value: Any) -> Any:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat(timespec="seconds")

    @staticmethod
    def _new_event_id() -> str:
        return f"event_{uuid.uuid4().hex}"

    @staticmethod
    def _new_message_id() -> str:
        return f"msg_{uuid.uuid4().hex}"
