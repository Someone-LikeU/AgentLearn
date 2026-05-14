# encoding : utf-8
import threading
import time

from rich.console import Console
from rich.live import Live
from rich.text import Text


class Spinner:
    """控制台等待动画。"""

    def __init__(self, console: Console, messages: list[str] | None = None, refresh_interval: float = 0.12):
        # 轮播文案列表，后续可在此处扩展更多提示语。
        self.console = console
        self.messages = messages or [
            "正在思考中...",
            "正在整理上下文...",
            "正在组织输出结构...",
        ]
        self.refresh_interval = refresh_interval
        self._stop_event = threading.Event()
        self._thread = None
        self._live = None
        self._frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._frame_index = 0
        self._message_index = 0
        self._tick = 0

    def start(self):
        # 启动后台刷新线程，避免阻塞主线程模型请求。
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        # 停止动画并等待线程退出，确保后续输出不受干扰。
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self._thread = None

    def _run(self):
        # 通过 Live 实现单行滚动刷新提示。
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
