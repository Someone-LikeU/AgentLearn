# encoding: utf-8
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from task_history_viewer import TaskHistoryViewer, calculate_window, format_task_time, shorten_task_content


class TaskHistoryViewerTest(unittest.TestCase):
	def _tasks(self, count: int) -> list[dict]:
		return [
			{
				"index": index + 1,
				"turn_id": f"turn_{index + 1}",
				"timestamp": f"2026-05-21T14:{index:02}:00",
				"content": f"用户任务 {index + 1}",
				"assistant_message_count": index,
				"tool_call_count": index + 1,
				"tool_result_count": index + 2,
				"memory_task_ids": [f"task_{index + 1}"],
				"final_output": f"模型最终输出 {index + 1}",
			}
			for index in range(count)
		]

	def test_calculate_initial_window(self):
		self.assertEqual(calculate_window(9, 10, 5), (5, 10))
		self.assertEqual(calculate_window(0, 3, 5), (0, 3))
		self.assertEqual(calculate_window(0, 0, 5), (0, 0))

	def test_viewer_keeps_window_until_selection_crosses_start(self):
		viewer = TaskHistoryViewer(self._tasks(10), window_size=5)
		self.assertEqual([task["index"] for _, task in viewer.visible_tasks()], [6, 7, 8, 9, 10])
		self.assertEqual(viewer.selected_task()["index"], 10)

		for expected_selected in (9, 8, 7, 6):
			viewer.move(-1)
			self.assertEqual(viewer.selected_task()["index"], expected_selected)
			self.assertEqual([task["index"] for _, task in viewer.visible_tasks()], [6, 7, 8, 9, 10])

		viewer.move(-1)
		self.assertEqual(viewer.selected_task()["index"], 5)
		self.assertEqual([task["index"] for _, task in viewer.visible_tasks()], [5, 6, 7, 8, 9])

	def test_render_detail_contains_turn_stats(self):
		viewer = TaskHistoryViewer(self._tasks(2), window_size=5)
		detail = viewer.render_detail(viewer.tasks[1])
		self.assertIn("Time: 2026-05-21 14:01", detail)
		self.assertIn("User:", detail)
		self.assertIn("Result:", detail)
		self.assertIn("模型最终输出 2", detail)
		self.assertIn("assistant messages: 1", detail)
		self.assertIn("tool calls: 2", detail)
		self.assertIn("memory task_id: task_2", detail)
		self.assertIn("d/Delete: delete task", detail)

	def test_render_list_uses_colon_hints_and_selected_style(self):
		viewer = TaskHistoryViewer(self._tasks(3), window_size=5)
		text = viewer.render_list()
		fragments = viewer.render_list_fragments()

		self.assertIn("Enter: detail", text)
		self.assertIn("q: quit", text)
		self.assertTrue(any(style == "class:selected" and "用户任务 3" in value for style, value in fragments))
		self.assertTrue(any(style == "class:hint" and "Enter: detail" in value for style, value in fragments))

	def test_fallback_delete_requires_confirm(self):
		viewer = TaskHistoryViewer(self._tasks(3), window_size=5)
		inputs = iter(["d", "y"])

		result = viewer.run_fallback(input_func=lambda _: next(inputs), output_func=lambda _: None)

		self.assertEqual(result.action, "delete")
		self.assertEqual(result.task["index"], 3)

	def test_fallback_delete_cancel_returns_to_history(self):
		viewer = TaskHistoryViewer(self._tasks(3), window_size=5)
		inputs = iter(["d", "n", "q"])
		outputs = []

		result = viewer.run_fallback(input_func=lambda _: next(inputs), output_func=outputs.append)

		self.assertEqual(result.action, "quit")
		self.assertTrue(any("已取消删除" in output for output in outputs))

	def test_render_delete_confirm_contains_selected_task(self):
		viewer = TaskHistoryViewer(self._tasks(3), window_size=5)
		text = viewer.render_delete_confirm()

		self.assertIn("Confirm delete task 3?", text)
		self.assertIn("用户任务 3", text)
		self.assertIn("y: confirm delete", text)

	def test_detail_view_can_render_scrolled_content(self):
		viewer = TaskHistoryViewer(self._tasks(1), window_size=5)
		viewer.tasks[0]["final_output"] = "\n".join(f"结果行 {index}" for index in range(20))

		first_view = "".join(text for _, text in viewer.render_detail_view_fragments(scroll_offset=0, viewport_height=6))
		later_view = "".join(text for _, text in viewer.render_detail_view_fragments(scroll_offset=8, viewport_height=6))

		self.assertIn("User:", first_view)
		self.assertNotIn("结果行 19", first_view)
		self.assertIn("结果行", later_view)
		self.assertIn("Lines", later_view)
		self.assertGreater(viewer.detail_max_scroll(viewport_height=6), 0)

	def test_detail_view_wraps_long_lines_before_slicing(self):
		viewer = TaskHistoryViewer(self._tasks(1), window_size=5)
		viewer.tasks[0]["content"] = "生成一篇古代文言文风格的情书，不少于500字"
		viewer.tasks[0]["final_output"] = "卿卿之容，沉鱼落雁，闭月羞花；" * 20

		fragments = viewer.render_detail_view_fragments(scroll_offset=0, viewport_height=8, viewport_width=24)
		view_text = "".join(text for _, text in fragments)
		physical_lines = TaskHistoryViewer._fragment_lines(fragments)

		self.assertLessEqual(len(physical_lines), 9)
		self.assertIn("Lines 1-8/", view_text)
		self.assertIn("PgUp/PgDn: scroll", view_text)
		self.assertNotIn("Mouse", view_text)
		self.assertGreater(viewer.detail_max_scroll(viewport_height=8, viewport_width=24), 0)

	def test_format_helpers(self):
		self.assertEqual(format_task_time("2026-05-21T14:21:09"), "2026-05-21 14:21")
		self.assertEqual(format_task_time(""), "---- -- -- --:--")
		self.assertTrue(shorten_task_content("a" * 80, limit=10).endswith("..."))


if __name__ == "__main__":
	unittest.main()
