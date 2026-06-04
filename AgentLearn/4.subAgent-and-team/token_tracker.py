# encoding: utf-8
import json
from typing import Any


class TokenTracker:
    """集中管理模型 usage 解析、token 估算和状态展示计算。"""

    def __init__(self, max_context_tokens: int, default_context_window: int = 32768):
        self.default_context_window = default_context_window
        self.max_context_tokens = self._normalize_context_window(max_context_tokens)
        self.used_token = 0
        self.total_token = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_response_count = 0
        self.session_has_real_usage = False

    def update_context_window(self, max_context_tokens: int) -> int:
        self.max_context_tokens = self._normalize_context_window(max_context_tokens)
        return self.max_context_tokens

    def reset(self) -> None:
        self.used_token = 0
        self.total_token = 0
        self.clear_session_usage()

    def update_from_response(self, response: Any) -> dict[str, int] | None:
        """
        从 OpenAI 响应对象中提取 usage 并更新最近一次 token 使用统计。
        :param response: 非流式 response 或流式 chunk
        :return: 标准化后的 usage
        """
        usage = self.extract_usage(response)
        if usage is None:
            return None

        total_tokens = usage.get("total_tokens", 0)
        if total_tokens > 0:
            self.used_token = total_tokens
            self.total_token = total_tokens
        return usage

    def extract_usage(self, response_or_usage: Any) -> dict[str, int] | None:
        """
        从 OpenAI 响应对象或 usage 对象中提取标准 usage。
        :param response_or_usage: 响应对象或 usage 字典/对象
        :return: 标准 usage；不存在时返回 None
        """
        usage = getattr(response_or_usage, "usage", None)
        if usage is None and self._looks_like_usage(response_or_usage):
            usage = response_or_usage
        if usage is None:
            return None

        prompt_tokens = self._normalize_token_value(self._usage_value(usage, "prompt_tokens"))
        completion_tokens = self._normalize_token_value(self._usage_value(usage, "completion_tokens"))
        total_tokens = self._normalize_token_value(self._usage_value(usage, "total_tokens"))

        if total_tokens is None:
            prompt = prompt_tokens or 0
            completion = completion_tokens or 0
            total_tokens = prompt + completion if prompt + completion > 0 else None

        normalized = {}
        if prompt_tokens is not None:
            normalized["prompt_tokens"] = prompt_tokens
        if completion_tokens is not None:
            normalized["completion_tokens"] = completion_tokens
        if total_tokens is not None:
            normalized["total_tokens"] = total_tokens
        return normalized or None

    def set_session_usage_summary(self, summary: dict[str, Any] | None) -> None:
        """
        恢复当前主会话上下文的真实 usage。
        :param summary: SessionManager 计算出的最新 assistant_response usage
        :return:
        """
        summary = summary or {}
        self.session_prompt_tokens = int(summary.get("prompt_tokens") or 0)
        self.session_completion_tokens = int(summary.get("completion_tokens") or 0)
        self.session_total_tokens = int(summary.get("total_tokens") or 0)
        self.session_response_count = int(summary.get("response_count") or 0)
        self.session_has_real_usage = bool(summary.get("has_real_usage"))

    def clear_session_usage(self) -> None:
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_response_count = 0
        self.session_has_real_usage = False

    def session_usage_summary(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.session_prompt_tokens,
            "completion_tokens": self.session_completion_tokens,
            "total_tokens": self.session_total_tokens,
            "response_count": self.session_response_count,
            "has_real_usage": self.session_has_real_usage,
        }

    def estimate_text_tokens(self, text: str | None) -> int:
        text = text or ""
        ascii_count = sum(1 for char in text if ord(char) < 128)
        non_ascii_count = len(text) - ascii_count
        return max(1, ascii_count // 4 + non_ascii_count * 2)

    def estimate_messages_tokens(self, messages: list[Any], tools: list[dict[str, Any]] | None = None) -> int:
        total = 0
        for message in messages:
            total += 4 + self.estimate_text_tokens(self.message_text(message))
        if tools:
            total += self.estimate_text_tokens(json.dumps(tools, ensure_ascii=False, default=str))
        return total

    def should_compact_messages(
            self,
            messages: list[Any],
            tools: list[dict[str, Any]] | None = None,
            compact_trigger_ratio: float = 0.8,
    ) -> bool:
        used_tokens = self.estimate_messages_tokens(messages, tools)
        return used_tokens >= int(self.max_context_tokens * compact_trigger_ratio)

    def set_estimated_usage(self, messages: list[Any], tools: list[dict[str, Any]] | None = None) -> int:
        estimated_tokens = self.estimate_messages_tokens(messages, tools)
        self.used_token = estimated_tokens
        self.total_token = estimated_tokens
        return estimated_tokens

    def current_used_tokens(self, messages: list[Any], tools: list[dict[str, Any]] | None = None) -> int:
        return self.used_token or self.estimate_messages_tokens(messages, tools)

    def current_total_tokens(self) -> int:
        return self.total_token or self.max_context_tokens

    def calculate_usage_ratio(self, used_tokens: int, context_window: int | None = None) -> float:
        """
        计算 token 使用率。
        :param used_tokens: 已使用 token
        :param context_window: 上下文窗口 token 上限
        :return:
        """
        context_window = self.max_context_tokens if context_window is None else context_window
        # 边界统一归零，避免 status 命令展示时出现除零或负比例。
        if context_window <= 0:
            return 0.0
        ratio = max(0.0, used_tokens / context_window)
        return min(ratio, 1.0)

    def render_usage_bar(self, usage_ratio: float, width: int = 30) -> str:
        """
        渲染横向 token 使用柱状图。
        :param usage_ratio: 使用率（0~1）
        :param width: 柱状图宽度
        :return:
        """
        ratio = min(max(usage_ratio, 0.0), 1.0)
        filled = int(width * ratio)
        bar = "█" * filled + "░" * (width - filled)
        if ratio >= 0.9:
            color = "red"
        elif ratio >= 0.7:
            color = "yellow"
        else:
            color = "green"
        return f"[{color}]{bar}[/{color}] {ratio * 100:.2f}%"

    @staticmethod
    def message_text(message: Any) -> str:
        if isinstance(message, dict):
            parts = [str(message.get("role", "")), str(message.get("content", ""))]
            if message.get("tool_calls"):
                parts.append(json.dumps(message.get("tool_calls"), ensure_ascii=False, default=str))
            if message.get("tool_call_id"):
                parts.append(str(message.get("tool_call_id")))
            return "\n".join(part for part in parts if part)

        parts = [str(getattr(message, "role", "")), str(getattr(message, "content", ""))]
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            parts.append(json.dumps(tool_calls, ensure_ascii=False, default=str))
        return "\n".join(part for part in parts if part)

    @staticmethod
    def _usage_value(usage: Any, key: str) -> Any:
        if isinstance(usage, dict):
            return usage.get(key)
        return getattr(usage, key, None)

    @staticmethod
    def _looks_like_usage(value: Any) -> bool:
        if isinstance(value, dict):
            return any(key in value for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
        return any(hasattr(value, key) for key in ("prompt_tokens", "completion_tokens", "total_tokens"))

    @staticmethod
    def _normalize_token_value(value: Any) -> int | None:
        if isinstance(value, bool) or value is None:
            return None
        try:
            number = int(value)
        except (TypeError, ValueError):
            return None
        return number if number >= 0 else None

    def _normalize_context_window(self, value: int) -> int:
        try:
            context_window = int(value)
        except (TypeError, ValueError):
            return self.default_context_window
        return context_window if context_window > 0 else self.default_context_window
