# encoding: utf-8
from dataclasses import dataclass
from typing import Any, Callable
import shutil
import sys
import unicodedata


DEFAULT_WINDOW_SIZE = 5
LIST_HINT_TEXT = "↑/↓: move   Enter: detail   d/Delete: delete task   q: quit"
DETAIL_HINT_TEXT = "↑/↓/PgUp/PgDn: scroll   Esc/Backspace: back   d/Delete: delete task   q: quit"
DELETE_CONFIRM_HINT_TEXT = "y: confirm delete   n/Esc/Backspace: cancel   q: quit"


@dataclass(frozen=True)
class TaskHistoryResult:
	"""history 交互结果。"""

	action: str
	task: dict[str, Any] | None = None


def calculate_window(selected_index: int, total_count: int, window_size: int = DEFAULT_WINDOW_SIZE) -> tuple[int, int]:
	"""
	计算滑动窗口范围。
	:param selected_index: 当前选中索引
	:param total_count: 任务总数
	:param window_size: 窗口大小
	:return: 左闭右开窗口范围
	"""
	if total_count <= 0:
		return 0, 0
	window_size = max(1, window_size)
	selected_index = min(max(0, selected_index), total_count - 1)
	window_start = max(0, selected_index - window_size + 1)
	window_end = min(total_count, window_start + window_size)
	return window_start, window_end


def format_task_time(timestamp: Any) -> str:
	"""
	格式化任务时间。
	:param timestamp: session 事件时间戳
	:return: YYYY-MM-DD HH:MM 或占位符
	"""
	text = str(timestamp or "").strip()
	if not text:
		return "---- -- -- --:--"
	date_part = text.split("T")[0].split(" ")[0]
	time_part = text.split("T")[-1].split(" ")[-1]
	parts = time_part.split(":")
	if len(date_part) == 10 and len(parts) >= 2:
		return f"{date_part} {parts[0]}:{parts[1]}"
	if len(parts) >= 2:
		return f"---- -- -- {parts[0]}:{parts[1]}"
	return "---- -- -- --:--"


def shorten_task_content(content: Any, limit: int = 42) -> str:
	"""
	压缩任务内容为单行展示文本。
	:param content: 用户任务内容
	:param limit: 最大展示长度
	:return: 单行摘要
	"""
	text = " ".join(str(content or "").split())
	if len(text) <= limit:
		return text
	return text[: max(1, limit - 3)] + "..."


class TaskHistoryViewer:
	"""展示当前会话的用户任务历史。"""

	def __init__(self, tasks: list[dict[str, Any]], window_size: int = DEFAULT_WINDOW_SIZE):
		self.tasks = tasks
		self.window_size = max(1, window_size)
		self.selected_index = max(0, len(tasks) - 1)
		self.window_start, _ = calculate_window(self.selected_index, len(tasks), self.window_size)

	def move(self, delta: int) -> None:
		"""
		移动选中任务。
		:param delta: 移动步长
		:return:
		"""
		if not self.tasks:
			self.selected_index = 0
			self.window_start = 0
			return
		self.selected_index = min(max(0, self.selected_index + delta), len(self.tasks) - 1)
		if self.selected_index < self.window_start:
			self.window_start = self.selected_index
		elif self.selected_index >= self.window_start + self.window_size:
			self.window_start = self.selected_index - self.window_size + 1
		self.window_start = min(max(0, self.window_start), max(0, len(self.tasks) - self.window_size))

	def selected_task(self) -> dict[str, Any] | None:
		"""
		获取当前选中的任务。
		:return: 任务信息
		"""
		if not self.tasks:
			return None
		return self.tasks[self.selected_index]

	def visible_tasks(self) -> list[tuple[int, dict[str, Any]]]:
		"""
		获取当前窗口内的任务。
		:return: (真实索引, 任务) 列表
		"""
		window_start = self.window_start
		window_end = min(len(self.tasks), window_start + self.window_size)
		return list(enumerate(self.tasks[window_start:window_end], start=window_start))

	def render_list(self) -> str:
		"""
		渲染任务列表。
		:return: 展示文本
		"""
		lines = [f"History  {len(self.tasks)} tasks", ""]
		for real_index, task in self.visible_tasks():
			prefix = ">" if real_index == self.selected_index else " "
			display_index = task.get("index") or real_index + 1
			time_text = format_task_time(task.get("timestamp"))
			content = shorten_task_content(task.get("content"))
			lines.append(f"{prefix} {display_index:02}  {time_text}  {content}")
		lines.append("")
		lines.append(LIST_HINT_TEXT)
		return "\n".join(lines)

	def render_list_fragments(self) -> list[tuple[str, str]]:
		"""
		渲染带样式的任务列表。
		:return: prompt_toolkit 格式化文本片段
		"""
		fragments: list[tuple[str, str]] = [("class:header", f"History  {len(self.tasks)} tasks"), ("", "\n\n")]
		for real_index, task in self.visible_tasks():
			prefix = ">" if real_index == self.selected_index else " "
			display_index = task.get("index") or real_index + 1
			time_text = format_task_time(task.get("timestamp"))
			content = shorten_task_content(task.get("content"))
			line = f"{prefix} {display_index:02}  {time_text}  {content}"
			style = "class:selected" if real_index == self.selected_index else ""
			fragments.append((style, line))
			fragments.append(("", "\n"))
		fragments.append(("", "\n"))
		fragments.append(("class:hint", LIST_HINT_TEXT))
		return fragments

	def render_detail(self, task: dict[str, Any] | None = None) -> str:
		"""
		渲染任务详情。
		:param task: 任务信息
		:return: 展示文本
		"""
		task = task or self.selected_task()
		if not task:
			return "No task selected."
		memory_task_ids = task.get("memory_task_ids") or []
		final_output = str(task.get("final_output") or "")
		lines = [
			f"Task {task.get('index')}",
			f"Time: {format_task_time(task.get('timestamp'))}",
			"",
			"User:",
			str(task.get("content") or ""),
			"",
			"Result:",
			final_output or "No assistant output recorded.",
			"",
			"Related:",
			f"- assistant messages: {task.get('assistant_message_count', 0)}",
			f"- tool calls: {task.get('tool_call_count', 0)}",
			f"- tool results: {task.get('tool_result_count', 0)}",
			f"- memory task_id: {', '.join(memory_task_ids) if memory_task_ids else 'none'}",
			"",
			DETAIL_HINT_TEXT,
		]
		return "\n".join(lines)

	def render_detail_fragments(self, task: dict[str, Any] | None = None) -> list[tuple[str, str]]:
		"""
		渲染带样式的任务详情。
		:param task: 任务信息
		:return: prompt_toolkit 格式化文本片段
		"""
		task = task or self.selected_task()
		if not task:
			return [("", "No task selected.")]
		memory_task_ids = task.get("memory_task_ids") or []
		final_output = str(task.get("final_output") or "")
		return [
			("class:header", f"Task {task.get('index')}"),
			("", "\n"),
			("class:section", f"Time: {format_task_time(task.get('timestamp'))}"),
			("", "\n\n"),
			("class:section", "User:"),
			("", "\n"),
			("", str(task.get("content") or "")),
			("", "\n\n"),
			("class:section", "Result:"),
			("", "\n"),
			("", final_output or "No assistant output recorded."),
			("", "\n\n"),
			("class:section", "Related:"),
			("", "\n"),
			("", f"- assistant messages: {task.get('assistant_message_count', 0)}\n"),
			("", f"- tool calls: {task.get('tool_call_count', 0)}\n"),
			("", f"- tool results: {task.get('tool_result_count', 0)}\n"),
			("", f"- memory task_id: {', '.join(memory_task_ids) if memory_task_ids else 'none'}"),
			("", "\n\n"),
			("class:hint", DETAIL_HINT_TEXT),
		]

	def render_detail_view_fragments(
		self,
		task: dict[str, Any] | None = None,
		scroll_offset: int = 0,
		viewport_height: int | None = None,
		viewport_width: int | None = None,
	) -> list[tuple[str, str]]:
		"""
		按当前滚动位置渲染详情页视口。
		:param task: 任务信息
		:param scroll_offset: 起始行偏移
		:param viewport_height: 可见内容行数
		:param viewport_width: 可见内容宽度
		:return: prompt_toolkit 格式化文本片段
		"""
		lines = self._fragment_display_lines(
			self._render_detail_body_fragments(task),
			viewport_width or self._default_detail_view_width(),
		)
		viewport_height = viewport_height or self._default_detail_view_height()
		viewport_height = max(3, viewport_height)
		max_offset = max(0, len(lines) - viewport_height)
		scroll_offset = min(max(0, scroll_offset), max_offset)
		visible_lines = lines[scroll_offset : scroll_offset + viewport_height]
		fragments = self._join_fragment_lines(visible_lines)
		fragments.append(("", "\n"))
		fragments.append(("class:hint", f"Lines {scroll_offset + 1}-{min(len(lines), scroll_offset + viewport_height)}/{len(lines)}   {DETAIL_HINT_TEXT}"))
		return fragments

	def detail_max_scroll(
		self,
		task: dict[str, Any] | None = None,
		viewport_height: int | None = None,
		viewport_width: int | None = None,
	) -> int:
		"""
		计算详情页最大滚动偏移。
		:param task: 任务信息
		:param viewport_height: 可见内容行数
		:param viewport_width: 可见内容宽度
		:return: 最大偏移
		"""
		lines = self._fragment_display_lines(
			self._render_detail_body_fragments(task),
			viewport_width or self._default_detail_view_width(),
		)
		viewport_height = viewport_height or self._default_detail_view_height()
		return max(0, len(lines) - max(3, viewport_height))

	def _render_detail_body_fragments(self, task: dict[str, Any] | None = None) -> list[tuple[str, str]]:
		"""
		渲染详情主体，底部操作提示由可滚动视口单独固定展示。
		:param task: 任务信息
		:return: prompt_toolkit 格式化文本片段
		"""
		fragments = self.render_detail_fragments(task)
		if fragments and fragments[-1][1] == DETAIL_HINT_TEXT:
			# 详情内容可能很长，提示固定在底部，避免滚到最下面才知道可用按键。
			return fragments[:-2] if len(fragments) >= 2 and fragments[-2][1] == "\n\n" else fragments[:-1]
		return fragments

	def _default_detail_view_height(self) -> int:
		"""
		根据终端高度估算详情内容视口。
		:return: 可见内容行数
		"""
		return max(6, shutil.get_terminal_size((100, 24)).lines - 4)

	def _default_detail_view_width(self) -> int:
		"""
		根据终端宽度估算详情内容视口。
		:return: 可见内容列数
		"""
		return max(20, shutil.get_terminal_size((100, 24)).columns - 1)

	@staticmethod
	def _fragment_lines(fragments: list[tuple[str, str]]) -> list[list[tuple[str, str]]]:
		"""
		把 prompt_toolkit 片段拆成行，便于自定义滚动。
		:param fragments: 格式化文本片段
		:return: 按行组织的片段
		"""
		lines: list[list[tuple[str, str]]] = [[]]
		for style, text in fragments:
			parts = str(text).split("\n")
			for index, part in enumerate(parts):
				if index > 0:
					lines.append([])
				if part:
					lines[-1].append((style, part))
		return lines

	@staticmethod
	def _join_fragment_lines(lines: list[list[tuple[str, str]]]) -> list[tuple[str, str]]:
		"""
		把按行组织的片段重新拼回 prompt_toolkit 文本。
		:param lines: 按行组织的片段
		:return: 格式化文本片段
		"""
		fragments: list[tuple[str, str]] = []
		for index, line in enumerate(lines):
			if index > 0:
				fragments.append(("", "\n"))
			if line:
				fragments.extend(line)
		return fragments

	@classmethod
	def _fragment_display_lines(cls, fragments: list[tuple[str, str]], width: int) -> list[list[tuple[str, str]]]:
		"""
		把 prompt_toolkit 片段按终端显示宽度拆成物理行。
		:param fragments: 格式化文本片段
		:param width: 最大显示列数
		:return: 按物理行组织的片段
		"""
		width = max(1, width)
		lines: list[list[tuple[str, str]]] = [[]]
		column = 0
		for style, text in fragments:
			for char in str(text):
				if char == "\n":
					lines.append([])
					column = 0
					continue
				display_text = "    " if char == "\t" else char
				display_width = cls._display_text_width(display_text)
				if column > 0 and column + display_width > width:
					lines.append([])
					column = 0
				lines[-1].append((style, display_text))
				column += display_width
		return lines

	@staticmethod
	def _display_text_width(text: str) -> int:
		"""
		估算文本在终端中的显示宽度。
		:param text: 待估算文本
		:return: 显示宽度
		"""
		width = 0
		for char in text:
			if unicodedata.combining(char):
				continue
			width += 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
		return width

	def render_delete_confirm(self, task: dict[str, Any] | None = None) -> str:
		"""
		渲染删除二次确认。
		:param task: 待删除任务
		:return: 展示文本
		"""
		task = task or self.selected_task()
		if not task:
			return "No task selected."
		return "\n".join(
			[
				f"Confirm delete task {task.get('index')}?",
				"",
				shorten_task_content(task.get("content"), limit=80),
				"",
				"Deleting will hide this user task and its related assistant/tool messages.",
				"",
				DELETE_CONFIRM_HINT_TEXT,
			]
		)

	def render_delete_confirm_fragments(self, task: dict[str, Any] | None = None) -> list[tuple[str, str]]:
		"""
		渲染带样式的删除二次确认。
		:param task: 待删除任务
		:return: prompt_toolkit 格式化文本片段
		"""
		task = task or self.selected_task()
		if not task:
			return [("", "No task selected.")]
		return [
			("class:danger", f"Confirm delete task {task.get('index')}?"),
			("", "\n\n"),
			("class:selected", shorten_task_content(task.get("content"), limit=80)),
			("", "\n\n"),
			("", "Deleting will hide this user task and its related assistant/tool messages."),
			("", "\n\n"),
			("class:hint", DELETE_CONFIRM_HINT_TEXT),
		]

	def run(self) -> TaskHistoryResult:
		"""
		运行交互式 history 窗口。
		:return: 交互结果
		"""
		if not self.tasks:
			return TaskHistoryResult(action="none")
		if _supports_prompt_toolkit():
			return self._run_prompt_toolkit()
		return self.run_fallback()

	def _run_prompt_toolkit(self) -> TaskHistoryResult:
		"""
		使用 prompt_toolkit 运行方向键交互。
		:return: 交互结果
		"""
		try:
			from prompt_toolkit import Application
			from prompt_toolkit.key_binding import KeyBindings
			from prompt_toolkit.layout import Layout
			from prompt_toolkit.layout.containers import HSplit, Window
			from prompt_toolkit.layout.controls import FormattedTextControl
			from prompt_toolkit.styles import Style
		except Exception:
			return self.run_fallback()

		show_detail = {"value": False}
		confirm_delete = {"value": False}
		detail_scroll = {"value": 0}
		result = {"value": TaskHistoryResult(action="quit")}

		def detail_view_height() -> int:
			return self._default_detail_view_height()

		def detail_view_width() -> int:
			return self._default_detail_view_width()

		def clamp_detail_scroll() -> None:
			detail_scroll["value"] = min(
				max(0, detail_scroll["value"]),
				self.detail_max_scroll(viewport_height=detail_view_height(), viewport_width=detail_view_width()),
			)

		def scroll_detail(delta: int) -> None:
			if not show_detail["value"] or confirm_delete["value"]:
				return
			detail_scroll["value"] += delta
			clamp_detail_scroll()

		def current_text():
			if confirm_delete["value"]:
				return self.render_delete_confirm_fragments()
			if show_detail["value"]:
				clamp_detail_scroll()
				return self.render_detail_view_fragments(
					scroll_offset=detail_scroll["value"],
					viewport_height=detail_view_height(),
					viewport_width=detail_view_width(),
				)
			return self.render_list_fragments()

		key_bindings = KeyBindings()

		@key_bindings.add("up")
		def _(event):
			if show_detail["value"] and not confirm_delete["value"]:
				scroll_detail(-1)
				event.app.invalidate()
			elif not confirm_delete["value"]:
				self.move(-1)
				event.app.invalidate()

		@key_bindings.add("down")
		def _(event):
			if show_detail["value"] and not confirm_delete["value"]:
				scroll_detail(1)
				event.app.invalidate()
			elif not confirm_delete["value"]:
				self.move(1)
				event.app.invalidate()

		@key_bindings.add("pageup")
		def _(event):
			scroll_detail(-max(1, detail_view_height() - 2))
			event.app.invalidate()

		@key_bindings.add("pagedown")
		def _(event):
			scroll_detail(max(1, detail_view_height() - 2))
			event.app.invalidate()

		@key_bindings.add("enter")
		def _(event):
			if confirm_delete["value"]:
				return
			# Enter 只展开详情，不退出 history；删除由第 7 步接管。
			show_detail["value"] = True
			detail_scroll["value"] = 0
			event.app.invalidate()

		@key_bindings.add("escape")
		@key_bindings.add("backspace")
		def _(event):
			if confirm_delete["value"]:
				confirm_delete["value"] = False
				event.app.invalidate()
				return
			show_detail["value"] = False
			detail_scroll["value"] = 0
			event.app.invalidate()

		@key_bindings.add("d")
		@key_bindings.add("delete")
		def _(event):
			# 先进入确认界面，避免用户误触 d/Delete 后直接软删除任务。
			confirm_delete["value"] = True
			event.app.invalidate()

		@key_bindings.add("y")
		def _(event):
			if not confirm_delete["value"]:
				return
			result["value"] = TaskHistoryResult(action="delete", task=self.selected_task())
			event.app.exit()

		@key_bindings.add("n")
		def _(event):
			if not confirm_delete["value"]:
				return
			confirm_delete["value"] = False
			event.app.invalidate()

		@key_bindings.add("q")
		@key_bindings.add("c-c")
		def _(event):
			result["value"] = TaskHistoryResult(action="quit")
			event.app.exit()

		control = FormattedTextControl(text=current_text)
		root = HSplit([Window(content=control, wrap_lines=False, always_hide_cursor=True)])
		style = Style.from_dict(
			{
				"header": "bold #00af87",
				"selected": "bold #ffffff bg:#005f87",
				"hint": "bold #00afd7",
				"section": "bold #ffd75f",
				"danger": "bold #ff5f5f",
			}
		)
		application = Application(
			layout=Layout(root),
			key_bindings=key_bindings,
			style=style,
			full_screen=False,
		)
		try:
			application.run()
		except (KeyboardInterrupt, EOFError):
			return TaskHistoryResult(action="quit")
		return result["value"]

	def run_fallback(
		self,
		input_func: Callable[[str], str] = input,
		output_func: Callable[[str], None] = print,
	) -> TaskHistoryResult:
		"""
		运行编号选择降级模式。
		:param input_func: 输入函数
		:param output_func: 输出函数
		:return: 交互结果
		"""
		if not self.tasks:
			return TaskHistoryResult(action="none")
		while True:
			output_func(self.render_list())
			try:
				command = input_func("history > ").strip()
			except (KeyboardInterrupt, EOFError):
				return TaskHistoryResult(action="quit")
			if not command:
				output_func(self.render_detail())
				continue
			parts = command.split(maxsplit=1)
			action = parts[0].lower()
			arg = parts[1].strip() if len(parts) > 1 else ""
			if action in {"q", "quit"}:
				return TaskHistoryResult(action="quit")
			if action in {"up", "u", "k", "w"}:
				self.move(-1)
				continue
			if action in {"down", "n", "j", "s"}:
				self.move(1)
				continue
			if action in {"d", "delete"}:
				task = self._task_by_display_index(arg) if arg else self.selected_task()
				if not task:
					output_func("未找到要删除的任务。")
					continue
				output_func(self.render_delete_confirm(task))
				try:
					confirm = input_func("confirm delete > ").strip().lower()
				except (KeyboardInterrupt, EOFError):
					return TaskHistoryResult(action="quit")
				if confirm in {"y", "yes", "是", "确认"}:
					return TaskHistoryResult(action="delete", task=task)
				output_func("已取消删除。")
				continue
			if action in {"detail", "show"}:
				task = self._task_by_display_index(arg) if arg else self.selected_task()
				output_func(self.render_detail(task))
				continue
			if action.isdigit():
				task = self._task_by_display_index(action)
				if task:
					self.selected_index = max(0, int(task.get("index", 1)) - 1)
					output_func(self.render_detail(task))
				continue
			output_func("可用输入：up/down、detail <序号>、d <序号>、q")

	def _task_by_display_index(self, value: str) -> dict[str, Any] | None:
		"""
		按展示序号查找任务。
		:param value: 展示序号
		:return: 任务信息
		"""
		if not str(value or "").isdigit():
			return None
		target_index = int(value)
		for task in self.tasks:
			if task.get("index") == target_index:
				return task
		return None


def _supports_prompt_toolkit() -> bool:
	"""
	判断当前终端是否适合 prompt_toolkit 交互。
	:return: 是否支持
	"""
	return bool(getattr(sys.stdin, "isatty", lambda: False)() and getattr(sys.stdout, "isatty", lambda: False)())


def open_task_history_viewer(tasks: list[dict[str, Any]], window_size: int = DEFAULT_WINDOW_SIZE) -> TaskHistoryResult:
	"""
	打开任务级 history 视图。
	:param tasks: 用户任务列表
	:param window_size: 窗口大小
	:return: 交互结果
	"""
	return TaskHistoryViewer(tasks=tasks, window_size=window_size).run()
