import html
import json
import os
import re
import urllib.parse
import urllib.request
from typing import Any, Callable

from prompt_loader import load_prompt

try:
    from ddgs import DDGS
except ModuleNotFoundError:
    DDGS = None


class WebTool:
    """网络搜索工具实现，负责搜索、网页详情提取和结果总结。"""

    def __init__(
            self,
            *,
            client: Any,
            model: str,
            spinner_factory: Callable[..., Any] | None = None,
    ):
        self.client = client
        self.model = model
        self.spinner_factory = spinner_factory
        self.last_web_search_results: dict[str, Any] | None = None

    _ADULT_OR_SPAM_KEYWORDS = (
        "色情", "成人视频", "成人", "裸聊", "约炮", "博彩", "赌场", "澳门", "六合彩",
        "porn", "sex", "xxx", "casino", "betting",
    )

    def _run_with_spinner(self, preset: str, func: Callable[[], Any], **context: Any) -> Any:
        spinner = self.spinner_factory(preset=preset, **context) if callable(self.spinner_factory) else None
        try:
            return func()
        finally:
            if spinner is not None:
                spinner.stop()

    def _normalize_web_search_result(self, item: dict[str, Any], source: str) -> dict[str, str]:
        return {
            "title": str(item.get("title", "")).strip(),
            "body": str(item.get("body", "")).strip(),
            "href": str(item.get("href", "")).strip(),
            "source": source,
        }

    def _read_ddgs_text(self, query: str, max_results: int) -> list[dict[str, Any]]:
        if DDGS is None:
            raise RuntimeError("missing dependency ddgs")
        ddgs = DDGS()
        # 兼容新版 ddgs 的上下文管理器用法，同时保留无上下文管理器时的关闭兜底。
        if hasattr(ddgs, "__enter__"):
            with ddgs as client:
                return list(client.text(query, max_results=max_results,
                                        region="cn-zh", safesearch="on",
                                        backend="duckduckgo",))
        try:
            return list(ddgs.text(query, max_results=max_results))
        finally:
            close = getattr(ddgs, "close", None)
            if callable(close):
                close()

    def _search_with_duckduckgo(self, query: str, max_results: int) -> list[dict[str, str]]:
        raw_items = self._read_ddgs_text(query, max_results)
        return [
            self._normalize_web_search_result(item, "duckduckgo")
            for item in raw_items
        ]

    def _query_tokens(self, query: str) -> set[str]:
        tokens = set(re.findall(r"[A-Za-z0-9]+", query.lower()))
        tokens.update(token for token in re.findall(r"[\u4e00-\u9fff]{2,}", query) if len(token) >= 2)
        return tokens

    def _is_search_result_relevant(self, query: str, item: dict[str, str]) -> bool:
        text = f"{item.get('title', '')} {item.get('body', '')} {item.get('href', '')}".lower()
        if any(keyword in text for keyword in self._ADULT_OR_SPAM_KEYWORDS):
            return False
        tokens = self._query_tokens(query)
        if not tokens:
            return True
        matched = sum(1 for token in tokens if token.lower() in text)
        # 对短查询保守放行；对长查询要求至少两个关键词命中，避免 Bing 只按“2026”泛化返回。
        return matched >= min(2, len(tokens))

    def _filter_relevant_results(self, query: str, results: list[dict[str, str]]) -> list[dict[str, str]]:
        return [item for item in results if self._is_search_result_relevant(query, item)]

    def _strip_html_text(self, raw_html: str) -> str:
        text = re.sub(r"<script[\s\S]*?</script>", " ", raw_html, flags=re.IGNORECASE)
        text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"<noscript[\s\S]*?</noscript>", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"<svg[\s\S]*?</svg>", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"<(header|footer|nav|form|aside)[\s\S]*?</\1>", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = html.unescape(text)
        return re.sub(r"\s+", " ", text).strip()

    def _extract_html_title(self, page_html: str) -> str:
        match = re.search(r"<title[^>]*>(?P<title>[\s\S]*?)</title>", page_html, flags=re.IGNORECASE)
        return self._strip_html_text(match.group("title")) if match else ""

    def _search_with_bing(self, query: str, max_results: int) -> list[dict[str, str]]:
        params = urllib.parse.urlencode({"q": query, "count": max_results, "setlang": "zh-CN"})
        request = urllib.request.Request(
            f"https://www.bing.com/search?{params}",
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
                ),
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            },
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            page_html = response.read().decode("utf-8", errors="replace")

        results: list[dict[str, str]] = []
        for block_match in re.finditer(r"<li[^>]+class=[\"'][^\"']*\bb_algo\b[^\"']*[\"'][^>]*>[\s\S]*?</li>", page_html):
            block_html = block_match.group(0)
            link_match = re.search(
                r"<h2[^>]*>[\s\S]*?<a[^>]+href=[\"'](?P<href>[^\"']+)[\"'][^>]*>"
                r"(?P<title>[\s\S]*?)</a>[\s\S]*?</h2>",
                block_html,
                flags=re.IGNORECASE,
            )
            if not link_match:
                continue
            title = self._strip_html_text(link_match.group("title"))
            if not title:
                continue
            body_match = re.search(
                r"<p[^>]*>(?P<body>[\s\S]*?)</p>",
                block_html,
                flags=re.IGNORECASE,
            )
            body = self._strip_html_text(body_match.group("body"))[:300] if body_match else ""
            results.append(
                self._normalize_web_search_result(
                    {
                        "title": title,
                        "body": body,
                        "href": link_match.group("href"),
                    },
                    "bing",
                )
            )
            if len(results) >= max_results:
                break
        return results

    def _web_search_backend_order(self) -> list[tuple[str, Callable[[str, int], list[dict[str, str]]]]]:
        configured = os.environ.get("WEB_SEARCH_BACKEND", "bing,duckduckgo").strip().lower()
        if configured == "auto":
            configured = "bing,duckduckgo"
        backend_map = {
            "bing": self._search_with_bing,
            "duckduckgo": self._search_with_duckduckgo,
            "ddg": self._search_with_duckduckgo,
        }
        result = []
        for name in configured.split(","):
            backend_name = name.strip()
            search_func = backend_map.get(backend_name)
            if search_func:
                result.append(("duckduckgo" if backend_name == "ddg" else backend_name, search_func))
        return result or [("bing", self._search_with_bing), ("duckduckgo", self._search_with_duckduckgo)]

    def _fetch_url_detail(self, href: str, max_chars: int) -> dict[str, Any]:
        request_url = self._quote_url_for_request(href)
        request = urllib.request.Request(
            request_url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
                ),
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            },
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            content_type = response.headers.get("Content-Type", "")
            raw_bytes = response.read(2_000_000)
            charset_match = re.search(r"charset=([^;\s]+)", content_type, flags=re.IGNORECASE)
            charset = charset_match.group(1) if charset_match else "utf-8"
            page_html = raw_bytes.decode(charset, errors="replace")
        text = self._strip_html_text(page_html)
        return {
            "ok": bool(text),
            "href": href,
            "title": self._extract_html_title(page_html),
            "text": text[:max_chars],
            "text_length": len(text),
        }

    def _quote_url_for_request(self, href: str) -> str:
        parts = urllib.parse.urlsplit(href)
        netloc = parts.netloc.encode("idna").decode("ascii") if parts.netloc else ""
        path = urllib.parse.quote(parts.path, safe="/:%")
        query = urllib.parse.quote(parts.query, safe="=&?/:;%+")
        fragment = urllib.parse.quote(parts.fragment, safe="")
        return urllib.parse.urlunsplit((parts.scheme, netloc, path, query, fragment))

    def web_extract(
            self,
            search_results: list[dict[str, Any]] | str,
            max_pages: int = 5,
            max_chars_per_page: int = 6000,
    ) -> list[dict[str, Any]]:
        if isinstance(search_results, str):
            search_results = json.loads(search_results)
        pages: list[dict[str, Any]] = []
        seen_links: set[str] = set()
        for item in search_results:
            href = str(item.get("href", "")).strip()
            # 只读取搜索结果中的真实网页链接，避免重复链接和非 HTTP 链接污染详情上下文。
            if not href or href in seen_links:
                continue
            if not href.lower().startswith(("http://", "https://")):
                continue
            seen_links.add(href)
            try:
                pages.append(self._fetch_url_detail(href, max_chars_per_page))
            except Exception as error:
                pages.append(
                    {
                        "ok": False,
                        "href": href,
                        "title": str(item.get("title", "")).strip(),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
            if len(pages) >= max_pages:
                break
        return pages

    def _format_search_results(self, results: list[dict[str, str]]) -> str:
        lines = []
        for i, item in enumerate(results, 1):
            lines.append(
                f"{i}. 来源: {item.get('source', '').strip()}\n"
                f"   标题: {item.get('title', '').strip()}\n"
                f"   摘要: {item.get('body', '').strip()}\n"
                f"   链接: {item.get('href', '').strip()}\n"
            )
        return "\n".join(lines)

    def _format_extracted_pages(self, extracted_pages: list[dict[str, Any]]) -> str:
        lines = []
        for i, page in enumerate(extracted_pages, 1):
            if page.get("ok"):
                lines.append(
                    f"{i}. 链接: {page.get('href', '')}\n"
                    f"   标题: {page.get('title', '')}\n"
                    f"   正文详情: {page.get('text', '')}\n"
                )
            else:
                lines.append(
                    f"{i}. 链接: {page.get('href', '')}\n"
                    f"   标题: {page.get('title', '')}\n"
                    f"   读取失败: {page.get('error_type', '')}: {page.get('error', '')}\n"
                )
        return "\n".join(lines)

    def _summarize_web_search_results(
            self,
            query: str,
            results: list[dict[str, str]],
            extracted_pages: list[dict[str, Any]],
    ) -> str:
        def _summarize() -> str:
            summarization_prompt = load_prompt("web_search_summary_system.md")
            user_content = load_prompt(
                "web_search_summary_user.md",
                query=query,
                search_text=self._format_search_results(results),
                detail_text=self._format_extracted_pages(extracted_pages),
            )
            summary_response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": summarization_prompt}, {"role": "user", "content": user_content}],
                temperature=0,
            )
            summary = summary_response.choices[0].message.content
            return summary if summary else "未能生成总结。"

        return self._run_with_spinner(
            "web_summary",
            _summarize,
        )

    def web_search(self, query: str, max_results: int = 10) -> str:
        max_results = int(max_results) if max_results else 10
        max_results = max(1, min(max_results, 10))
        errors: list[str] = []
        attempts: list[dict[str, Any]] = []
        self.last_web_search_results = {
            "query": query,
            "max_results": max_results,
            "backend": None,
            "results": [],
            "extracted_pages": [],
            "errors": [],
            "attempts": attempts,
        }
        for backend_name, search_func in self._web_search_backend_order():
            try:
                # 搜索、详情提取、总结分阶段执行，便于在长耗时网络请求中展示明确等待状态。
                results = self._run_with_spinner(
                    "web_search",
                    lambda: search_func(query, max_results),
                    backend_name=backend_name,
                    query=query,
                )
                filtered_results = self._filter_relevant_results(query, results)
                attempts.append(
                    {
                        "backend": backend_name,
                        "ok": bool(results),
                        "result_count": len(results),
                        "filtered_result_count": len(filtered_results),
                    }
                )
                if filtered_results:
                    extracted_pages = self._run_with_spinner(
                        "web_extract",
                        lambda: self.web_extract(filtered_results, max_pages=min(max_results, 5)),
                    )
                    self.last_web_search_results = {
                        "query": query,
                        "max_results": max_results,
                        "backend": backend_name,
                        "results": filtered_results,
                        "raw_results": results,
                        "extracted_pages": extracted_pages,
                        "attempts": attempts,
                    }
                    return self._summarize_web_search_results(query, filtered_results, extracted_pages)
                errors.append(f"{backend_name}: no relevant results" if results else f"{backend_name}: no results")
            except Exception as e:
                attempts.append(
                    {
                        "backend": backend_name,
                        "ok": False,
                        "error_type": type(e).__name__,
                        "error": str(e),
                    }
                )
                errors.append(f"{backend_name}: {e}")
            self.last_web_search_results = {
                "query": query,
                "max_results": max_results,
                "backend": None,
                "results": [],
                "extracted_pages": [],
                "errors": errors,
                "attempts": attempts,
            }
        return f"WEB_SEARCH 执行失败: {'; '.join(errors)}"
