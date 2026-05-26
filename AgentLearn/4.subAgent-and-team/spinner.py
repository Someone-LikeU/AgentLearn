# encoding : utf-8
import threading
import time
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.text import Text


class Spinner:
    """控制台等待动画。"""

    DEFAULT = "default"
    TOOL = "tool"
    WEB_SEARCH = "web_search"
    WEB_EXTRACT = "web_extract"
    WEB_SUMMARY = "web_summary"

    _DEFAULT_MESSAGES = [
        "正在思考中...",
        "正在整理上下文...",
        "正在组织输出结构...",
    ]
    _WEB_EXTRACT_MESSAGES = [
        "正在读取网页详情...",
        "正在清理网页正文...",
        "正在准备详情上下文...",
    ]
    _WEB_SUMMARY_MESSAGES = [
        "正在总结搜索结果...",
        "正在整合网页详情...",
        "正在生成引用链接...",
    ]

    def __init__(
            self,
            console: Console,
            messages: list[str] | None = None,
            *,
            preset: str = DEFAULT,
            refresh_interval: float = 0.12,
            **context: Any,
    ):
        self.console = console
        self.messages = messages or self.messages_for(preset, **context)
        self.refresh_interval = refresh_interval
        self._stop_event = threading.Event()
        self._thread = None
        self._live = None
        self._frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._frame_index = 0
        self._message_index = 0
        self._tick = 0

    @classmethod
    def messages_for(cls, preset: str = DEFAULT, **context: Any) -> list[str]:
        if preset == cls.TOOL:
            tool_name = str(context.get("tool_name") or "工具").strip()
            return [
                f"正在调用工具 {tool_name}...",
                "正在等待工具执行结果...",
                "正在整理工具返回内容...",
            ]
        if preset == cls.WEB_SEARCH:
            backend_name = str(context.get("backend_name") or "").strip()
            query = str(context.get("query") or "").strip()
            query_tip = query[:36] + "..." if len(query) > 36 else query
            backend_text = f"使用 {backend_name} " if backend_name else ""
            first_message = f"正在{backend_text}搜索网络"
            if query_tip:
                first_message += f"：{query_tip}"
            return [
                f"{first_message}...",
                "正在等待搜索引擎返回结果...",
            ]
        if preset == cls.WEB_EXTRACT:
            return cls._WEB_EXTRACT_MESSAGES.copy()
        if preset == cls.WEB_SUMMARY:
            return cls._WEB_SUMMARY_MESSAGES.copy()
        return cls._DEFAULT_MESSAGES.copy()

    def start(self):
        # 后台刷新动画，避免阻塞主线程里的模型或工具请求。
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        # 等待线程退出，避免后续控制台输出被 Live 动画干扰。
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self._thread = None

    def _run(self):
        # Live 负责单行刷新，transient=True 会在停止时清理等待提示。
        with Live(console=self.console, refresh_per_second=12, transient=True) as live:
            self._live = live
            while not self._stop_event.is_set():
                frame = self._frames[self._frame_index % len(self._frames)]
                message = self.messages[self._message_index % len(self.messages)]
                live.update(Text(f"{frame} {message}", style="dim"))
                self._frame_index += 1
                self._tick += 1
                if self._tick % 10 == 0:
                    self._message_index += 1
                time.sleep(self.refresh_interval)
