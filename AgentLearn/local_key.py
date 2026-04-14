import json
import logging
import logging.handlers
import os
import time
import uuid
import asyncio
import traceback
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

# ---------- 硬编码配置 ----------
HARDCODED_API_KEY = "ak_2Nu3Zp7IO0fa5M01Aa3xq6F66uh0k"
THINKING = "LongCat-Flash-Thinking-2601"
CHAT = "LongCat-Flash-Chat"
HARDCODED_MODEL = CHAT #TODO:THINKING模型似乎存在问题

TARGET_URL = "https://api.longcat.chat/anthropic"
RECORDS_DIR = Path("request_records")
RECORDS_DIR.mkdir(exist_ok=True)

# ---------- 创建日志目录 ----------
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)


# ---------- 详细的日志系统配置 ----------
def setup_detailed_logging():
    """配置极其详细的日志系统"""
    
    # 创建根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # 清除已有的处理器
    root_logger.handlers.clear()
    
    # 日志格式 - 详细版
    detailed_format = logging.Formatter(
        '%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)-20s | %(funcName)-25s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 简单格式（用于控制台）
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # 1. 控制台处理器 - INFO级别
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    # 2. 详细日志文件 - DEBUG级别（包含所有调试信息）
    detailed_handler = logging.handlers.RotatingFileHandler(
        LOGS_DIR / "proxy_detailed.log",
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=10,
        encoding='utf-8'
    )
    detailed_handler.setLevel(logging.DEBUG)
    detailed_handler.setFormatter(detailed_format)
    root_logger.addHandler(detailed_handler)
    
    # 3. 错误日志文件 - ERROR级别（仅错误和异常）
    error_handler = logging.handlers.RotatingFileHandler(
        LOGS_DIR / "proxy_errors.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_format)
    root_logger.addHandler(error_handler)
    
    # 4. 请求追踪日志 - INFO级别（记录所有请求的关键信息）
    request_handler = logging.handlers.RotatingFileHandler(
        LOGS_DIR / "proxy_requests.log",
        maxBytes=30 * 1024 * 1024,  # 30MB
        backupCount=10,
        encoding='utf-8'
    )
    request_handler.setLevel(logging.INFO)
    request_handler.setFormatter(detailed_format)
    # 添加过滤器，只记录包含请求ID的日志
    request_handler.addFilter(lambda record: '[request-' in record.getMessage())
    root_logger.addHandler(request_handler)
    
    # 5. 性能监控日志 - 记录耗时操作
    perf_handler = logging.handlers.RotatingFileHandler(
        LOGS_DIR / "proxy_performance.log",
        maxBytes=20 * 1024 * 1024,  # 20MB
        backupCount=5,
        encoding='utf-8'
    )
    perf_handler.setLevel(logging.DEBUG)
    perf_handler.setFormatter(detailed_format)
    perf_handler.addFilter(lambda record: 'PERF:' in record.getMessage())
    root_logger.addHandler(perf_handler)
    
    return root_logger


# 初始化日志系统
logger = setup_detailed_logging()

# 记录启动信息
logger.info("=" * 80)
logger.info("日志系统初始化完成")
logger.info(f"详细日志文件: {LOGS_DIR / 'proxy_detailed.log'}")
logger.info(f"错误日志文件: {LOGS_DIR / 'proxy_errors.log'}")
logger.info(f"请求日志文件: {LOGS_DIR / 'proxy_requests.log'}")
logger.info(f"性能日志文件: {LOGS_DIR / 'proxy_performance.log'}")
logger.info("=" * 80)

app = FastAPI(title="Anthropic API Proxy (Hardcoded Key & Model + Thinking Support)")


# ---------- HTTP 客户端 (跳过 SSL 验证) ----------
class LoggedAsyncClient(httpx.AsyncClient):
    """带详细日志的HTTP客户端"""
    
    async def request(self, *args, **kwargs):
        request_id = kwargs.pop('request_id', 'unknown')
        method = args[0] if args else kwargs.get('method', 'UNKNOWN')
        url = args[1] if len(args) > 1 else kwargs.get('url', 'UNKNOWN')
        
        logger.debug(f"[request-{request_id}] HTTP请求开始: {method} {url}")
        logger.debug(f"[request-{request_id}] 请求参数: {json.dumps(kwargs, default=str, indent=2)}")
        
        start_time = time.time()
        try:
            response = await super().request(*args, **kwargs)
            elapsed = (time.time() - start_time) * 1000
            logger.debug(f"[request-{request_id}] HTTP请求完成: {method} {url} - 状态码: {response.status_code} - 耗时: {elapsed:.2f}ms")
            logger.debug(f"[request-{request_id}] 响应头: {dict(response.headers)}")
            return response
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            logger.error(f"[request-{request_id}] HTTP请求失败: {method} {url} - 耗时: {elapsed:.2f}ms - 错误: {e}")
            logger.debug(f"[request-{request_id}] 错误详情: {traceback.format_exc()}")
            raise


client = LoggedAsyncClient(
    timeout=httpx.Timeout(120.0, connect=10.0),
    limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
    verify=False
)


class RequestRecorder:
    """请求记录器（仅记录非探测请求）"""

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.start_time = time.time()
        self.stage_times = {}  # 记录各阶段耗时
        self.request_data: Dict[str, Any] = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "request": {},
            "response": {},
            "error": None,
            "duration_ms": 0,
            "is_stream": False,
            "stage_times": {}
        }
        logger.debug(f"[request-{request_id}] 创建记录器")

    def record_stage(self, stage_name: str):
        """记录阶段耗时"""
        current_time = time.time()
        elapsed = (current_time - self.start_time) * 1000
        self.stage_times[stage_name] = elapsed
        logger.debug(f"[request-{self.request_id}] PERF: {stage_name} - {elapsed:.2f}ms")

    def record_request(self, method: str, url: str, headers: Dict[str, str], body: Any):
        self.record_stage("request_recorded")
        self.request_data["request"] = {
            "method": method,
            "url": url,
            "headers": self._sanitize_headers(headers),
            "body": body
        }
        if isinstance(body, dict) and body.get("stream") is True:
            self.request_data["is_stream"] = True
            logger.info(f"[request-{self.request_id}] 客户端请求流式，将模拟返回")
        
        # 详细记录请求信息
        logger.debug(f"[request-{self.request_id}] 请求方法: {method}")
        logger.debug(f"[request-{self.request_id}] 请求URL: {url}")
        logger.debug(f"[request-{self.request_id}] 请求头: {json.dumps(self._sanitize_headers(headers), indent=2)}")
        logger.debug(f"[request-{self.request_id}] 请求体: {json.dumps(body, ensure_ascii=False, indent=2) if body else 'None'}")

    def record_response(self, status_code: int, headers: Dict[str, str], body: Any):
        self.record_stage("response_recorded")
        self.request_data["response"] = {
            "status_code": status_code,
            "headers": dict(headers),
            "body": body
        }
        logger.debug(f"[request-{self.request_id}] 响应状态码: {status_code}")
        logger.debug(f"[request-{self.request_id}] 响应头: {json.dumps(dict(headers), indent=2)}")
        
        # 智能截断响应体日志（避免过大）
        body_str = json.dumps(body, ensure_ascii=False) if isinstance(body, dict) else str(body)
        if len(body_str) > 2000:
            logger.debug(f"[request-{self.request_id}] 响应体(截断): {body_str[:2000]}... [总长度: {len(body_str)}]")
        else:
            logger.debug(f"[request-{self.request_id}] 响应体: {body_str}")

    def record_error(self, error: str):
        self.record_stage("error_occurred")
        self.request_data["error"] = error
        logger.error(f"[request-{self.request_id}] 请求失败: {error}")
        logger.debug(f"[request-{self.request_id}] 错误堆栈: {traceback.format_exc()}")

    def finalize(self):
        self.record_stage("finalized")
        self.request_data["duration_ms"] = round((time.time() - self.start_time) * 1000, 2)
        self.request_data["stage_times"] = self.stage_times
        self._save_to_file()
        status = self.request_data["response"].get("status_code", "未知")
        logger.info(f"[request-{self.request_id}] 完成 - 状态码: {status} - 总耗时: {self.request_data['duration_ms']}ms")
        logger.debug(f"[request-{self.request_id}] PERF: 各阶段耗时 - {json.dumps(self.stage_times)}")

    def _sanitize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        sensitive = {"authorization", "x-api-key", "cookie", "api-key"}
        return {k: ("***REDACTED***" if k.lower() in sensitive else v) for k, v in headers.items()}

    def _save_to_file(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = RECORDS_DIR / f"request_{ts}_{self.request_id[:8]}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.request_data, f, ensure_ascii=False, indent=2)
        logger.debug(f"[request-{self.request_id}] 记录已保存到: {filename}")


def should_skip_logging(request: Request) -> bool:
    """判断是否为测试/探测请求，若是则跳过日志记录"""
    if request.method == "HEAD":
        logger.debug(f"跳过HEAD请求日志记录")
        return True
    ua = request.headers.get("user-agent", "")
    if "Bun" in ua:
        logger.debug(f"跳过Bun探测请求日志记录")
        return True
    return False


def simulate_sse_stream(full_response: dict, request_id: str) -> List[str]:
    """
    将完整响应转换为符合 Anthropic 规范的 SSE 事件列表。
    支持 text、tool_use、thinking、redacted_thinking 类型。
    """
    logger.debug(f"[request-{request_id}] 开始模拟SSE流，响应包含 {len(full_response.get('content', []))} 个内容块")
    
    events = []
    msg_id = full_response.get("id", str(uuid.uuid4()).replace("-", ""))
    model = full_response.get("model", "")
    content_list = full_response.get("content", [])
    stop_reason = full_response.get("stop_reason")
    usage = full_response.get("usage", {})
    
    logger.debug(f"[request-{request_id}] 消息ID: {msg_id}, 模型: {model}, 停止原因: {stop_reason}")

    # 1. message_start
    message_start_data = {
        'type': 'message_start',
        'message': {
            'id': msg_id,
            'type': 'message',
            'role': 'assistant',
            'model': model,
            'content': [],
            'stop_reason': None,
            'stop_sequence': None,
            'usage': {}
        }
    }
    events.append(
        "event:message_start\n"
        f"data:{json.dumps(message_start_data)}\n"
    )
    logger.debug(f"[request-{request_id}] SSE事件: message_start")

    # 2. 遍历每个 content 块
    for idx, block in enumerate(content_list):
        block_type = block.get("type")
        logger.debug(f"[request-{request_id}] 处理内容块 {idx}: 类型={block_type}")

        # ----- 文本块 -----
        if block_type == "text":
            text = block.get("text", "")
            logger.debug(f"[request-{request_id}] 文本块 {idx}: 长度={len(text)}")
            
            text_start_data = {
                'type': 'content_block_start',
                'index': idx,
                'content_block': {'type': 'text', 'text': ''}
            }
            events.append(
                "event:content_block_start\n"
                f"data:{json.dumps(text_start_data)}\n"
            )
            
            text_delta_data = {
                'type': 'content_block_delta',
                'index': idx,
                'delta': {'type': 'text_delta', 'text': text}
            }
            events.append(
                "event:content_block_delta\n"
                f"data:{json.dumps(text_delta_data)}\n"
            )
            
            text_stop_data = {'type': 'content_block_stop', 'index': idx}
            events.append(
                "event:content_block_stop\n"
                f"data:{json.dumps(text_stop_data)}\n"
            )
            logger.debug(f"[request-{request_id}] SSE事件: text块 {idx} 完成")

        # ----- 工具调用块 -----
        elif block_type == "tool_use":
            tool_id = block.get("id", f"call_{uuid.uuid4().hex[:12]}")
            tool_name = block.get("name", "")
            tool_input = block.get("input", {})
            logger.debug(f"[request-{request_id}] 工具调用块 {idx}: 工具名={tool_name}, ID={tool_id}")
            
            tool_start_data = {
                'type': 'content_block_start',
                'index': idx,
                'content_block': {
                    'type': 'tool_use',
                    'id': tool_id,
                    'name': tool_name,
                    'input': {}
                }
            }
            events.append(
                "event:content_block_start\n"
                f"data:{json.dumps(tool_start_data)}\n"
            )
            
            input_json_str = json.dumps(tool_input, ensure_ascii=False)
            logger.debug(f"[request-{request_id}] 工具输入JSON长度: {len(input_json_str)}")
            
            chunk_size = 2
            chunk_count = (len(input_json_str) + chunk_size - 1) // chunk_size
            for i in range(0, len(input_json_str), chunk_size):
                partial = input_json_str[i:i+chunk_size]
                tool_delta_data = {
                    'type': 'content_block_delta',
                    'index': idx,
                    'delta': {
                        'type': 'input_json_delta',
                        'partial_json': partial
                    }
                }
                events.append(
                    "event:content_block_delta\n"
                    f"data:{json.dumps(tool_delta_data)}\n"
                )
            
            tool_stop_data = {'type': 'content_block_stop', 'index': idx}
            events.append(
                "event:content_block_stop\n"
                f"data:{json.dumps(tool_stop_data)}\n"
            )
            logger.debug(f"[request-{request_id}] SSE事件: tool_use块 {idx} 完成，共 {chunk_count} 个分块")

        # ----- 思考块 -----
        elif block_type == "thinking":
            thinking_text = block.get("thinking", "")
            logger.debug(f"[request-{request_id}] 思考块 {idx}: 长度={len(thinking_text)}")
            
            thinking_start_data = {
                'type': 'content_block_start',
                'index': idx,
                'content_block': {'type': 'thinking', 'thinking': ''}
            }
            events.append(
                "event:content_block_start\n"
                f"data:{json.dumps(thinking_start_data)}\n"
            )
            
            thinking_delta_data = {
                'type': 'content_block_delta',
                'index': idx,
                'delta': {'type': 'thinking_delta', 'thinking': thinking_text}
            }
            events.append(
                "event:content_block_delta\n"
                f"data:{json.dumps(thinking_delta_data)}\n"
            )
            
            thinking_stop_data = {'type': 'content_block_stop', 'index': idx}
            events.append(
                "event:content_block_stop\n"
                f"data:{json.dumps(thinking_stop_data)}\n"
            )
            logger.debug(f"[request-{request_id}] SSE事件: thinking块 {idx} 完成")

        # ----- 省略的思考块 -----
        elif block_type == "redacted_thinking":
            data = block.get("data", "")
            logger.debug(f"[request-{request_id}] 省略的思考块 {idx}: 数据长度={len(data)}")
            
            redacted_start_data = {
                'type': 'content_block_start',
                'index': idx,
                'content_block': {'type': 'redacted_thinking', 'data': data}
            }
            events.append(
                "event:content_block_start\n"
                f"data:{json.dumps(redacted_start_data)}\n"
            )
            
            redacted_stop_data = {'type': 'content_block_stop', 'index': idx}
            events.append(
                "event:content_block_stop\n"
                f"data:{json.dumps(redacted_stop_data)}\n"
            )
            logger.debug(f"[request-{request_id}] SSE事件: redacted_thinking块 {idx} 完成")

        # ----- 未知类型 -----
        else:
            logger.warning(f"[request-{request_id}] 未知的 content 块类型: {block_type}，已跳过")
            logger.debug(f"[request-{request_id}] 未知块内容: {json.dumps(block, ensure_ascii=False, indent=2)}")

    # 3. message_delta
    message_delta_data = {
        'type': 'message_delta',
        'delta': {
            'stop_reason': stop_reason,
            'stop_sequence': None
        },
        'usage': usage
    }
    events.append(
        "event:message_delta\n"
        f"data:{json.dumps(message_delta_data)}\n"
    )
    logger.debug(f"[request-{request_id}] SSE事件: message_delta - usage={usage}")

    # 4. message_stop
    message_stop_data = {'type': 'message_stop'}
    events.append(
        "event:message_stop\n"
        f"data:{json.dumps(message_stop_data)}\n"
    )
    logger.debug(f"[request-{request_id}] SSE事件: message_stop")
    
    logger.debug(f"[request-{request_id}] SSE流模拟完成，共生成 {len(events)} 个事件")
    return events


# ---------- 路由 ----------
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_request(request: Request, path: str):
    request_id = str(uuid.uuid4())
    client_ip = request.client.host if request.client else "unknown"
    
    logger.info(f"[request-{request_id}] ========== 新请求开始 ==========")
    logger.info(f"[request-{request_id}] 客户端IP: {client_ip}")
    logger.info(f"[request-{request_id}] 请求方法: {request.method}")
    logger.info(f"[request-{request_id}] 请求路径: /{path}")
    logger.info(f"[request-{request_id}] 查询参数: {request.url.query if request.url.query else 'None'}")
    
    skip_log = should_skip_logging(request)

    if skip_log:
        logger.info(f"[request-{request_id}] 检测到探测请求，跳过详细日志记录")
        logger.info(f"[request-{request_id}] ========== 请求结束(跳过) ==========")

    recorder = RequestRecorder(request_id) if not skip_log else None

    try:
        method = request.method
        headers = dict(request.headers)
        
        logger.debug(f"[request-{request_id}] 原始请求头: {json.dumps({k: ('***' if k.lower() in ['authorization'] else v) for k, v in headers.items()}, indent=2)}")
        
        headers.pop("host", None)
        headers.pop("content-length", None)
        
        # 强制覆盖 Authorization 头
        original_auth = headers.get("authorization", "未提供")
        headers["authorization"] = f"Bearer {HARDCODED_API_KEY}"
        logger.info(f"[request-{request_id}] 已强制覆盖 Authorization: '{original_auth[:20] if original_auth != '未提供' else original_auth}...' -> 硬编码值")

        # 获取请求体
        body = None
        original_body = None
        try:
            body = await request.json()
            original_body = body.copy()
            logger.debug(f"[request-{request_id}] 成功解析JSON请求体")
        except Exception as e:
            logger.debug(f"[request-{request_id}] 无法解析JSON请求体: {e}")
            try:
                body_bytes = await request.body()
                if body_bytes:
                    body = body_bytes.decode('utf-8')
                    original_body = body
                    logger.debug(f"[request-{request_id}] 请求体作为文本解析，长度: {len(body)}")
            except Exception as e2:
                logger.debug(f"[request-{request_id}] 无法读取请求体: {e2}")
                body = None
                original_body = None

        # 强制覆盖 model 和 stream 字段
        client_wants_stream = False
        if isinstance(body, dict):
            client_wants_stream = body.get("stream", False)
            original_model = body.get("model", "未指定")
            body["model"] = HARDCODED_MODEL
            logger.info(f"[request-{request_id}] 已强制覆盖 model: '{original_model}' -> '{HARDCODED_MODEL}'")
            
            original_stream = body.get("stream", None)
            body["stream"] = False
            logger.info(f"[request-{request_id}] 已强制覆盖 stream: {original_stream} -> False")
            
            # 记录其他重要字段
            if "max_tokens" in body:
                logger.debug(f"[request-{request_id}] max_tokens: {body['max_tokens']}")
            if "temperature" in body:
                logger.debug(f"[request-{request_id}] temperature: {body['temperature']}")
            if "messages" in body:
                logger.debug(f"[request-{request_id}] 消息数量: {len(body['messages'])}")

        # 目标 URL
        target_url = f"{TARGET_URL}/{path}" if path else TARGET_URL
        if request.url.query:
            target_url += f"?{request.url.query}"
        
        logger.info(f"[request-{request_id}] 目标URL: {target_url}")

        # 记录请求
        if recorder:
            recorder.record_stage("request_parsed")
            record_body = original_body
            if isinstance(record_body, dict):
                record_body = record_body.copy()
                record_body["_note"] = "Authorization和Model已被代理强制覆盖为硬编码值"
            recorder.record_request(method, target_url, headers, record_body)

        # 发送非流式请求到上游
        logger.info(f"[request-{request_id}] 向上游发送非流式请求 (model={HARDCODED_MODEL})")
        if recorder:
            recorder.record_stage("upstream_request_start")
        
        upstream_start = time.time()
        response = await client.request(
            method=method,
            url=target_url,
            headers=headers,
            json=body if isinstance(body, dict) else None,
            content=body if not isinstance(body, dict) and body else None,
            request_id=request_id
        )
        upstream_elapsed = (time.time() - upstream_start) * 1000
        logger.info(f"[request-{request_id}] PERF: 上游请求耗时: {upstream_elapsed:.2f}ms")
        
        if recorder:
            recorder.record_stage("upstream_request_complete")

        # 获取完整响应文本
        response_text = response.text
        logger.debug(f"[request-{request_id}] 响应文本长度: {len(response_text)} 字节")
        logger.debug(f"[request-{request_id}] 响应Content-Type: {response.headers.get('content-type', 'unknown')}")

        # 尝试解析为 JSON
        try:
            response_json = json.loads(response_text)
            logger.debug(f"[request-{request_id}] 响应成功解析为JSON")
            if isinstance(response_json, dict):
                logger.debug(f"[request-{request_id}] 响应包含字段: {list(response_json.keys())}")
                if "usage" in response_json:
                    logger.info(f"[request-{request_id}] Token使用: {response_json['usage']}")
        except Exception as e:
            logger.debug(f"[request-{request_id}] 响应不是有效的JSON: {e}")
            response_json = response_text

        # 记录完整响应
        if recorder:
            recorder.record_stage("response_parsed")
            recorder.record_response(
                status_code=response.status_code,
                headers=dict(response.headers),
                body=response_json
            )
            recorder.finalize()

        # 根据客户端原始意图决定返回方式
        if client_wants_stream and isinstance(response_json, dict):
            logger.info(f"[request-{request_id}] 开始模拟流式返回")
            
            stream_start = time.time()
            event_count = 0
            total_bytes = 0

            async def sse_generator():
                nonlocal event_count, total_bytes
                events = simulate_sse_stream(response_json, request_id)
                for ev in events:
                    event_count += 1
                    data = (ev + "\n").encode('utf-8')
                    total_bytes += len(data)
                    logger.debug(f"[request-{request_id}] 发送SSE事件 {event_count}: {len(data)} 字节")
                    yield data
                    await asyncio.sleep(0.01)
                
                stream_elapsed = (time.time() - stream_start) * 1000
                logger.info(f"[request-{request_id}] PERF: 流式模拟完成 - 事件数: {event_count}, 总字节: {total_bytes}, 耗时: {stream_elapsed:.2f}ms")

            resp_headers = {
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
            logger.debug(f"[request-{request_id}] 流式响应头: {resp_headers}")
            logger.info(f"[request-{request_id}] ========== 请求结束(流式) ==========")
            return StreamingResponse(sse_generator(), headers=resp_headers)
        else:
            resp_headers = dict(response.headers)
            resp_headers.pop("content-length", None)
            resp_headers.pop("transfer-encoding", None)
            
            logger.debug(f"[request-{request_id}] 返回普通响应，状态码: {response.status_code}")
            logger.info(f"[request-{request_id}] ========== 请求结束(普通) ==========")
            
            return Response(
                content=response_text,
                status_code=response.status_code,
                headers=resp_headers,
                media_type=response.headers.get("content-type", "application/json")
            )

    except httpx.TimeoutException as e:
        logger.error(f"[request-{request_id}] 上游请求超时: {e}")
        logger.debug(f"[request-{request_id}] 超时详情: {traceback.format_exc()}")
        if recorder:
            recorder.record_error(f"上游请求超时: {str(e)}")
            recorder.finalize()
        return JSONResponse(
            status_code=504,
            content={"error": "上游服务器超时", "detail": str(e), "request_id": request_id}
        )
    except httpx.ReadError as e:
        logger.error(f"[request-{request_id}] 上游连接读取错误: {e}")
        logger.debug(f"[request-{request_id}] ReadError详情: {traceback.format_exc()}")
        if recorder:
            recorder.record_error(f"连接读取错误: {str(e)}")
            recorder.finalize()
        return JSONResponse(
            status_code=502,
            content={"error": "上游连接错误", "detail": str(e), "request_id": request_id}
        )
    except Exception as e:
        logger.error(f"[request-{request_id}] 代理错误: {type(e).__name__}: {e}")
        logger.debug(f"[request-{request_id}] 完整错误堆栈:\n{traceback.format_exc()}")
        if recorder:
            recorder.record_error(f"{type(e).__name__}: {str(e)}")
            recorder.finalize()
        logger.info(f"[request-{request_id}] ========== 请求结束(错误) ==========")
        return JSONResponse(
            status_code=500,
            content={"error": "代理服务器错误", "detail": str(e), "request_id": request_id}
        )


@app.on_event("startup")
async def startup():
    logger.info("=" * 80)
    logger.info("代理服务器启动")
    logger.info(f"版本: 详细日志增强版")
    logger.info(f"目标地址: {TARGET_URL}")
    logger.info(f"硬编码 API Key: {HARDCODED_API_KEY[:10]}...")
    logger.info(f"硬编码 Model: {HARDCODED_MODEL}")
    logger.info(f"请求记录目录: {RECORDS_DIR.absolute()}")
    logger.info(f"日志目录: {LOGS_DIR.absolute()}")
    logger.info("日志文件列表:")
    logger.info(f"  - 详细日志: {LOGS_DIR / 'proxy_detailed.log'}")
    logger.info(f"  - 错误日志: {LOGS_DIR / 'proxy_errors.log'}")
    logger.info(f"  - 请求日志: {LOGS_DIR / 'proxy_requests.log'}")
    logger.info(f"  - 性能日志: {LOGS_DIR / 'proxy_performance.log'}")
    logger.info("SSL 验证: 已禁用")
    logger.info("HTTP连接池: max_keepalive=5, max_connections=10")
    logger.info("=" * 80)


@app.on_event("shutdown")
async def shutdown():
    logger.info("=" * 80)
    logger.info("代理服务器正在关闭...")
    await client.aclose()
    logger.info("HTTP客户端已关闭")
    logger.info("代理服务器已关闭")
    logger.info("=" * 80)


@app.get("/health")
async def health():
    logger.debug("健康检查请求")
    return {
        "status": "healthy",
        "target": TARGET_URL,
        "hardcoded_model": HARDCODED_MODEL,
        "hardcoded_key_prefix": f"{HARDCODED_API_KEY[:10]}...",
        "logs_directory": str(LOGS_DIR.absolute())
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")