# encoding: utf-8
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str in sys.path:
    sys.path.remove(project_root_str)
sys.path.insert(0, project_root_str)

from token_tracker import TokenTracker


class TokenTrackerTest(unittest.TestCase):
    def test_update_from_response_prefers_total_tokens(self):
        tracker = TokenTracker(max_context_tokens=100)
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=35,
            )
        )

        usage = tracker.update_from_response(response)

        self.assertEqual(tracker.used_token, 35)
        self.assertEqual(tracker.total_token, 35)
        self.assertEqual(usage["total_tokens"], 35)

    def test_update_from_response_falls_back_to_prompt_and_completion(self):
        tracker = TokenTracker(max_context_tokens=100)
        response = SimpleNamespace(
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": None,
            }
        )

        tracker.update_from_response(response)

        self.assertEqual(tracker.used_token, 30)
        self.assertEqual(tracker.total_token, 30)

    def test_estimate_messages_includes_tools(self):
        tracker = TokenTracker(max_context_tokens=100)
        messages = [{"role": "user", "content": "hello"}]
        tools = [{"type": "function", "function": {"name": "READ_FILE"}}]

        without_tools = tracker.estimate_messages_tokens(messages)
        with_tools = tracker.estimate_messages_tokens(messages, tools)

        self.assertGreater(with_tools, without_tools)

    def test_should_compact_messages_uses_context_window(self):
        tracker = TokenTracker(max_context_tokens=10)
        messages = [{"role": "user", "content": "abcdefghijabcdefghij"}]

        self.assertTrue(tracker.should_compact_messages(messages, compact_trigger_ratio=0.5))

    def test_usage_ratio_and_bar_are_bounded(self):
        tracker = TokenTracker(max_context_tokens=100)

        self.assertEqual(tracker.calculate_usage_ratio(150), 1.0)
        self.assertEqual(tracker.calculate_usage_ratio(10, context_window=0), 0.0)
        self.assertIn("100.00%", tracker.render_usage_bar(1.5, width=5))

    def test_session_usage_summary_is_separate_from_context_estimate(self):
        tracker = TokenTracker(max_context_tokens=100)
        tracker.set_estimated_usage([{"role": "user", "content": "hello"}])
        tracker.set_session_usage_summary(
            {
                "prompt_tokens": 20,
                "completion_tokens": 5,
                "total_tokens": 25,
                "response_count": 2,
                "has_real_usage": True,
            }
        )

        summary = tracker.session_usage_summary()

        self.assertGreater(tracker.used_token, 0)
        self.assertEqual(summary["total_tokens"], 25)
        self.assertEqual(summary["response_count"], 2)
        self.assertTrue(summary["has_real_usage"])


if __name__ == "__main__":
    unittest.main()
