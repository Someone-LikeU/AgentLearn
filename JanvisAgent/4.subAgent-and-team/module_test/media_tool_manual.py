# encoding: utf-8
import functools
import http.server
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from tools.tool_manager import AgentToolHandlers, ToolManager, ToolManagerConfig
from tools.tool_names import ToolNameConstant


class DummyClient:
	pass



def _build_tool_manager() -> ToolManager:
	# 手动测试只验证工具实现，不依赖真实模型客户端。
	return ToolManager(
		config=ToolManagerConfig(
			project_root=str(PROJECT_ROOT),
			client=DummyClient(),
			model="manual-test",
			temperature=0,
			is_main_agent=False,
		),
		handlers=AgentToolHandlers(),
		mcp_client=None,
	)


def test_audio_transcribe_gaia_sample():
	audio_path = (
		"D:\\AgentLongTaskTest\\GAIA\\2023\\"
		"\u5206\u7c7bvalidation\\Level_1\\1f975693-876d-457b-a649-393859e79bf3.mp3"
	)
	manager = _build_tool_manager()

	result = json.loads(
		manager.available_functions[ToolNameConstant.AUDIO_TRANSCRIBE](
			path=audio_path,
			output_format="txt",
		)
	)
	print("音频解析结果：", json.dumps(result, ensure_ascii=False, indent=2, default=str))
	assert result.get("ok") is True
	assert result.get("content")


def test_image_download_local_http_sample():
	try:
		from PIL import Image, ImageDraw
	except Exception as error:
		pytest.skip(f"Pillow is required to generate a local sample image: {error}")

	with tempfile.TemporaryDirectory() as tmp:
		tmp_dir = Path(tmp)
		source_image = tmp_dir / "source.png"
		image = Image.new("RGB", (240, 80), "white")
		draw = ImageDraw.Draw(image)
		draw.text((16, 28), "Janvis image download test", fill="black")
		image.save(source_image)

		handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_dir))
		server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
		thread = threading.Thread(target=server.serve_forever, daemon=True)
		thread.start()
		try:
			url = f"http://127.0.0.1:{server.server_port}/source.png"
			manager = _build_tool_manager()
			result = json.loads(
				manager.available_functions[ToolNameConstant.IMAGE_DOWNLOAD](
					url=url,
					output_path="local_http_image.png",
					max_mb=5,
					timeout_seconds=30,
				)
			)
		finally:
			server.shutdown()
			server.server_close()

	print("本地HTTP图片下载结果：")
	print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
	assert result.get("ok") is True
	assert result.get("path")
	assert Path(result["path"]).exists()
	assert result.get("image", {}).get("width") == 240
	assert result.get("image", {}).get("height") == 80


def test_image_ocr_manual():
	# image_path = os.environ.get("JANVIS_TEST_OCR_IMAGE_PATH", "").strip()
	image_path = r"D:\AgentLongTaskTest\GAIA\2023\分类validation\Level_1\9318445f-fe6a-4e1b-acbf-c68228c9906a.png"
	languages = os.environ.get("JANVIS_TEST_OCR_LANGUAGES", "eng+chi_sim").strip() or "eng+chi_sim"
	if not image_path:
		pytest.skip("Fill image_path or set JANVIS_TEST_OCR_IMAGE_PATH before running this manual OCR test.")

	manager = _build_tool_manager()
	result = json.loads(
		manager.available_functions[ToolNameConstant.IMAGE_OCR](
			path=image_path,
			languages=languages,
		)
	)

	print("图片OCR结果：")
	print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
	assert result.get("ok") is True
	assert result.get("text")


def test_video_download_manual():
	# url = os.environ.get("JANVIS_TEST_VIDEO_URL", "https://www.youtube.com/watch?v=1htKBjuUWec").strip()
	url = "https://www.youtube.com/watch?v=2Njmx-UuU3M"
	if not url:
		pytest.skip("Fill VIDEO_DOWNLOAD_URL or set JANVIS_TEST_VIDEO_URL before running this manual video download test.")

	manager = _build_tool_manager()
	default_cookie_paths = [Path(r"D:\youtube_cookies.txt"), Path(r"D:\my_youtube_cookies.txt")]
	cookie_path = os.environ.get("JANVIS_TEST_VIDEO_COOKIE_PATH")
	if not cookie_path:
		for default_cookie_path in default_cookie_paths:
			if default_cookie_path.exists():
				cookie_path = str(default_cookie_path)
				break
	download_args = {
		"url": url,
		"max_mb": 500,
		"timeout_seconds": 120,
	}
	if cookie_path:
		download_args["cookie_path"] = cookie_path
	result = json.loads(
		manager.available_functions[ToolNameConstant.VIDEO_DOWNLOAD](**download_args)
	)

	print("下载执行结果：")
	print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
	assert result.get("ok") is True
	assert result.get("path")
	assert Path(result["path"]).exists()
	assert Path(result["path"]).stat().st_size > 0


def test_video_download_local_http_sample():
	ffmpeg = shutil.which("ffmpeg")
	if not ffmpeg:
		pytest.skip("ffmpeg is required to generate a local sample video.")

	with tempfile.TemporaryDirectory() as tmp:
		tmp_dir = Path(tmp)
		source_video = tmp_dir / "source.mp4"
		subprocess.run(
			[
				ffmpeg,
				"-y",
				"-f",
				"lavfi",
				"-i",
				"testsrc=size=128x72:rate=5:duration=1",
				"-an",
				"-pix_fmt",
				"yuv420p",
				str(source_video),
			],
			check=True,
			capture_output=True,
		)

		handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_dir))
		server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
		thread = threading.Thread(target=server.serve_forever, daemon=True)
		thread.start()
		try:
			url = f"http://127.0.0.1:{server.server_port}/source.mp4"
			manager = _build_tool_manager()
			result = json.loads(
				manager.available_functions[ToolNameConstant.VIDEO_DOWNLOAD](
					url=url,
					output_path="local_http_sample.mp4",
					max_mb=10,
					timeout_seconds=30,
				)
			)
		finally:
			server.shutdown()
			server.server_close()

	print("本地HTTP视频下载结果：")
	print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
	assert result.get("ok") is True
	assert result.get("path")
	assert Path(result["path"]).exists()
	assert result.get("video", {}).get("streams")
