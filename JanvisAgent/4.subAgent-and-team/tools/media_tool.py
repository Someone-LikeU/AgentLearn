from __future__ import annotations

import base64
import contextlib
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


class MediaTool:
    """音频、图片、视频处理工具集合。"""

    def __init__(self, *, project_root: str, mcp_client=None):
        self.project_root = Path(project_root).resolve()
        self.output_root = self.project_root / "runtime_output" / "media"
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.mcp_client = mcp_client

    def media_probe(self, path: str) -> str:
        try:
            file_path = self._resolve_input_path(path)
            stat = file_path.stat()
            payload: dict[str, Any] = {
                "ok": True,
                "path": str(file_path),
                "name": file_path.name,
                "suffix": file_path.suffix,
                "size_bytes": stat.st_size,
                "mime_type": mimetypes.guess_type(str(file_path))[0],
                "image": self._image_metadata(file_path),
                "ffprobe": self._ffprobe(file_path),
            }
            return self._json(payload)
        except Exception as error:
            return self._json(self._error_payload(error))

    def image_describe(
            self,
            path: str,
            prompt: str | None = None,
            model: str | None = None,
    ) -> str:
        try:
            image_path = self._resolve_input_path(path)
            prompt = prompt or "这张图片描述了什么？"
            selected_model = model or self._default_vision_model_name()
            return self._json(self._describe_with_ollama_vision(image_path, prompt=prompt, model=selected_model))
        except Exception as error:
            return self._json(self._error_payload(error))

    def get_vision_models(self) -> str:
        try:
            config = self._load_vision_model_config()
            models = self._vision_model_names(config)
            return self._json(
                {
                    "ok": True,
                    "default_model": config.get("default_model") or (models[0] if models else None),
                    "models": models,
                    "config_path": str(self._vision_model_config_path()),
                }
            )
        except Exception as error:
            return self._json(self._error_payload(error))

    def image_ocr(self, path: str, languages: str = "eng+chi_sim") -> str:
        try:
            image_path = self._resolve_input_path(path)
            timeout_seconds = int(os.environ.get("JANVIS_IMAGE_OCR_TIMEOUT_SECONDS", "300"))
            payload = self._run_image_ocr_worker(
                image_path=image_path,
                languages=languages,
                timeout_seconds=timeout_seconds,
            )
            payload["ok"] = bool(payload.get("text"))
            payload["backend"] = payload.get("ocr_backend")
            return self._json(payload)
        except Exception as error:
            return self._json(self._error_payload(error))

    def image_transform(
            self,
            path: str,
            output_path: str | None = None,
            format: str | None = None,
            max_width: int | None = None,
            max_height: int | None = None,
            rotate_degrees: float = 0,
            crop_box: list[int] | None = None,
            quality: int = 90,
    ) -> str:
        try:
            from PIL import Image

            image_path = self._resolve_input_path(path)
            image = Image.open(image_path)
            if crop_box:
                if len(crop_box) != 4:
                    raise ValueError("crop_box must contain [left, top, right, bottom].")
                image = image.crop(tuple(crop_box))
            if rotate_degrees:
                image = image.rotate(float(rotate_degrees), expand=True)
            if max_width or max_height:
                image.thumbnail((int(max_width or image.width), int(max_height or image.height)))

            target_format = (format or image.format or image_path.suffix.lstrip(".") or "png").upper()
            suffix = "." + ("jpg" if target_format == "JPEG" else target_format.lower())
            output = self._resolve_output_path(output_path, self._safe_stem(image_path.stem), suffix)
            if target_format in ("JPG", "JPEG") and image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
            image.save(output, format="JPEG" if target_format == "JPG" else target_format, quality=int(quality))
            return self._json({"ok": True, "path": str(output), "format": target_format, "image": self._image_metadata(output)})
        except Exception as error:
            return self._json(self._error_payload(error))

    def image_download(
            self,
            url: str,
            output_path: str | None = None,
            max_mb: int = 50,
            timeout_seconds: int = 60,
    ) -> str:
        try:
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme not in ("http", "https"):
                raise ValueError("Only http and https URLs are supported.")
            default_name = self._filename_from_url(parsed) or f"image_{self._now_stamp()}.jpg"
            suffix = Path(default_name).suffix or ".jpg"
            output = self._resolve_output_path(output_path or default_name, Path(default_name).stem, suffix)
            part_path = output.with_suffix(output.suffix + ".part")
            request = urllib.request.Request(url, headers={"User-Agent": "JanvisAgent/1.0"})
            max_bytes = int(max_mb) * 1024 * 1024
            downloaded = 0
            started = time.perf_counter()
            with urllib.request.urlopen(request, timeout=int(timeout_seconds)) as response, open(part_path, "wb") as file:
                content_type = response.headers.get("Content-Type", "")
                if content_type and not content_type.lower().startswith("image/"):
                    raise ValueError(f"URL did not return an image content type: {content_type}")
                while True:
                    chunk = response.read(512 * 1024)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > max_bytes:
                        return self._json(
                            {
                                "ok": False,
                                "error_type": "FileTooLarge",
                                "error": f"Download exceeded max_mb={max_mb}.",
                                "partial_path": str(part_path),
                                "downloaded_bytes": downloaded,
                            }
                        )
                    file.write(chunk)
            os.replace(part_path, output)
            return self._json(
                {
                    "ok": True,
                    "url": url,
                    "path": str(output),
                    "size_bytes": output.stat().st_size,
                    "mime_type": mimetypes.guess_type(str(output))[0],
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "image": self._image_metadata(output),
                }
            )
        except Exception as error:
            return self._json(self._error_payload(error))

    def audio_transcribe(
            self,
            path: str,
            output_format: str = "txt",
            model: str | None = None,
            language: str | None = None,
            beam_size: int = 5,
    ) -> str:
        try:
            audio_path = self._resolve_input_path(path)
            output_format = self._normalize_output_format(output_format)
            model_name = model or os.environ.get("JANVIS_AUDIO_TRANSCRIBE_MODEL") or "base"
            timeout_seconds = int(os.environ.get("JANVIS_AUDIO_TRANSCRIBE_TIMEOUT_SECONDS", "600"))
            payload = self._run_audio_transcribe_worker(
                audio_path=audio_path,
                output_format=output_format,
                model_name=model_name,
                language=language,
                beam_size=beam_size,
                timeout_seconds=timeout_seconds,
            )
            return self._json(payload)
        except Exception as error:
            return self._json(self._error_payload(error))

    def audio_convert(
            self,
            path: str,
            output_path: str | None = None,
            format: str | None = None,
            start_seconds: float | None = None,
            duration_seconds: float | None = None,
            sample_rate: int | None = None,
            channels: int | None = None,
    ) -> str:
        try:
            audio_path = self._resolve_input_path(path)
            suffix = "." + (format or audio_path.suffix.lstrip(".") or "wav")
            output = self._resolve_output_path(output_path, self._safe_stem(audio_path.stem), suffix)
            command = self._ffmpeg_base_command(start_seconds) + ["-i", str(audio_path)]
            if duration_seconds is not None:
                command += ["-t", str(duration_seconds)]
            if sample_rate:
                command += ["-ar", str(int(sample_rate))]
            if channels:
                command += ["-ac", str(int(channels))]
            command.append(str(output))
            result = self._run_command(command, timeout=300)
            result.update({"path": str(output), "format": suffix.lstrip(".")})
            return self._json(result)
        except Exception as error:
            return self._json(self._error_payload(error))

    def video_download(
            self,
            url: str,
            output_path: str | None = None,
            max_mb: int = 500,
            timeout_seconds: int = 120,
            cookie_path: str | None = None,
    ) -> str:
        try:
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme not in ("http", "https"):
                raise ValueError("Only http and https URLs are supported.")
            default_name = self._video_output_name(parsed)
            output = self._resolve_output_path(output_path or default_name, Path(default_name).stem, Path(default_name).suffix or ".mp4")
            cookie_file = self._resolve_video_cookie_path(cookie_path)
            if self._is_youtube_url(parsed):
                downloader_result = self._download_video_with_ytdlp(
                    url,
                    output,
                    max_mb=max_mb,
                    timeout_seconds=timeout_seconds,
                    cookie_path=cookie_file,
                )
                if downloader_result.get("ok") or downloader_result.get("requires_user_action"):
                    return self._json(downloader_result)
                direct_result = self._download_video_direct(url, output, max_mb=max_mb, timeout_seconds=timeout_seconds)
                if direct_result.get("ok"):
                    direct_result["fallback_from"] = "yt-dlp"
                    return self._json(direct_result)
                downloader_result["direct_fallback"] = self._compact_failed_tool_payload(direct_result)
                return self._json(downloader_result)

            if self._should_use_media_downloader_for_video_url(parsed):
                downloader_result = self._download_video_with_you_get(
                    url,
                    output,
                    max_mb=max_mb,
                    timeout_seconds=timeout_seconds,
                    cookie_path=cookie_file,
                )
                if downloader_result.get("ok") or downloader_result.get("requires_user_action"):
                    return self._json(downloader_result)
                direct_result = self._download_video_direct(url, output, max_mb=max_mb, timeout_seconds=timeout_seconds)
                if direct_result.get("ok"):
                    direct_result["fallback_from"] = "you-get"
                    return self._json(direct_result)
                downloader_result["direct_fallback"] = self._compact_failed_tool_payload(direct_result)
                return self._json(downloader_result)

            direct_result = self._download_video_direct(url, output, max_mb=max_mb, timeout_seconds=timeout_seconds)
            if direct_result.get("ok"):
                return self._json(direct_result)
            if direct_result.get("error_type") in {"NonVideoResponse", "InvalidVideoFile"}:
                downloader_result = self._download_video_with_you_get(
                    url,
                    output,
                    max_mb=max_mb,
                    timeout_seconds=timeout_seconds,
                    cookie_path=cookie_file,
                )
                if downloader_result.get("ok"):
                    downloader_result["fallback_from"] = direct_result.get("error_type")
                    return self._json(downloader_result)
                if downloader_result.get("requires_user_action"):
                    return self._json(downloader_result)
                direct_result["you_get_fallback"] = self._compact_failed_tool_payload(downloader_result)
            return self._json(direct_result)
        except Exception as error:
            return self._json(self._error_payload(error))

    def video_extract_frames(
            self,
            path: str,
            output_dir: str | None = None,
            timestamps: list[Any] | None = None,
            interval_seconds: float | None = None,
            max_frames: int = 20,
            image_format: str = "jpg",
    ) -> str:
        try:
            video_path = self._resolve_input_path(path)
            ffmpeg = self._require_command("ffmpeg")
            frame_dir = self._resolve_output_dir(output_dir, f"{self._safe_stem(video_path.stem)}_frames_{self._now_stamp()}")
            image_format = (image_format or "jpg").lstrip(".").lower()
            outputs: list[str] = []
            if timestamps:
                for index, timestamp in enumerate(timestamps, 1):
                    output = frame_dir / f"frame_{index:04d}.{image_format}"
                    command = [ffmpeg, "-y", "-ss", self._format_timestamp(timestamp), "-i", str(video_path), "-frames:v", "1", str(output)]
                    result = self._run_command(command, timeout=90)
                    if result.get("ok") and output.exists():
                        outputs.append(str(output))
            else:
                interval = max(float(interval_seconds or 5), 0.1)
                pattern = frame_dir / f"frame_%04d.{image_format}"
                command = [
                    ffmpeg,
                    "-y",
                    "-i",
                    str(video_path),
                    "-vf",
                    f"fps=1/{interval}",
                    "-vframes",
                    str(int(max_frames)),
                    str(pattern),
                ]
                result = self._run_command(command, timeout=300)
                if not result.get("ok"):
                    return self._json(result)
                outputs = [str(p) for p in sorted(frame_dir.glob(f"*.{image_format}"))]
            return self._json({"ok": True, "path": str(video_path), "output_dir": str(frame_dir), "frames": outputs})
        except Exception as error:
            return self._json(self._error_payload(error))

    def video_transcribe(
            self,
            path: str,
            output_format: str = "txt",
            model: str | None = None,
            language: str | None = None,
    ) -> str:
        try:
            video_path = self._resolve_input_path(path)
            audio_output = self._resolve_output_path(None, f"{self._safe_stem(video_path.stem)}_audio", ".wav")
            command = self._ffmpeg_base_command(None) + [
                "-i",
                str(video_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                str(audio_output),
            ]
            extract_result = self._run_command(command, timeout=300)
            if not extract_result.get("ok"):
                return self._json(extract_result)
            transcript = json.loads(
                self.audio_transcribe(
                    str(audio_output),
                    output_format=output_format,
                    model=model,
                    language=language,
                )
            )
            transcript["source_video"] = str(video_path)
            transcript["extracted_audio"] = str(audio_output)
            return self._json(transcript)
        except Exception as error:
            return self._json(self._error_payload(error))

    def video_convert(
            self,
            path: str,
            output_path: str | None = None,
            format: str | None = None,
            start_seconds: float | None = None,
            duration_seconds: float | None = None,
            max_width: int | None = None,
            crf: int = 23,
            preset: str = "medium",
            extract_audio: bool = False,
    ) -> str:
        try:
            video_path = self._resolve_input_path(path)
            suffix = "." + ("mp3" if extract_audio else (format or video_path.suffix.lstrip(".") or "mp4"))
            output = self._resolve_output_path(output_path, self._safe_stem(video_path.stem), suffix)
            command = self._ffmpeg_base_command(start_seconds) + ["-i", str(video_path)]
            if duration_seconds is not None:
                command += ["-t", str(duration_seconds)]
            if extract_audio:
                command += ["-vn"]
            else:
                if max_width:
                    command += ["-vf", f"scale='min({int(max_width)},iw)':-2"]
                command += ["-crf", str(int(crf)), "-preset", str(preset)]
            command.append(str(output))
            result = self._run_command(command, timeout=900)
            result.update({"path": str(output), "format": suffix.lstrip(".")})
            return self._json(result)
        except Exception as error:
            return self._json(self._error_payload(error))

    def _download_video_direct(self, url: str, output: Path, *, max_mb: int, timeout_seconds: int) -> dict[str, Any]:
        part_path = output.with_suffix(output.suffix + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "JanvisAgent/1.0"})
        max_bytes = int(max_mb) * 1024 * 1024
        downloaded = 0
        started = time.perf_counter()
        content_type = ""
        final_url = url
        try:
            with urllib.request.urlopen(request, timeout=int(timeout_seconds)) as response:
                content_type = response.headers.get("Content-Type", "")
                final_url = response.geturl()
                if self._is_obvious_non_video_content_type(content_type):
                    head = response.read(1024)
                    return {
                        "ok": False,
                        "error_type": "NonVideoResponse",
                        "error": f"URL returned non-video content type: {content_type or 'unknown'}.",
                        "url": url,
                        "final_url": final_url,
                        "content_type": content_type,
                        "response_head": self._decode_bytes(head)[:500],
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                    }
                with open(part_path, "wb") as file:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        downloaded += len(chunk)
                        if downloaded > max_bytes:
                            return {
                                "ok": False,
                                "error_type": "FileTooLarge",
                                "error": f"Download exceeded max_mb={max_mb}.",
                                "partial_path": str(part_path),
                                "downloaded_bytes": downloaded,
                                "elapsed_seconds": round(time.perf_counter() - started, 3),
                            }
                        file.write(chunk)
        except Exception as error:
            return self._error_payload(error, url=url, backend="direct_http", elapsed_seconds=round(time.perf_counter() - started, 3))

        validation = self._validate_downloaded_video(part_path)
        if not validation.get("ok"):
            validation.update(
                {
                    "url": url,
                    "final_url": final_url,
                    "content_type": content_type,
                    "partial_path": str(part_path),
                    "size_bytes": part_path.stat().st_size if part_path.exists() else 0,
                    "response_head": self._file_text_head(part_path),
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
            )
            return validation

        os.replace(part_path, output)
        return {
            "ok": True,
            "backend": "direct_http",
            "url": url,
            "final_url": final_url,
            "path": str(output),
            "size_bytes": output.stat().st_size,
            "content_type": content_type,
            "video": validation.get("video"),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }

    def _download_video_with_ytdlp(
            self,
            url: str,
            output: Path,
            *,
            max_mb: int,
            timeout_seconds: int,
            cookie_path: Path | None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        command = self._build_ytdlp_command(
            url,
            output,
            max_mb=max_mb,
            timeout_seconds=timeout_seconds,
            cookie_path=cookie_path,
        )
        result = self._run_command(command, timeout=max(10, int(timeout_seconds)), cwd=output.parent)
        downloaded_path = self._locate_downloaded_video(output)
        combined_output = "\n".join([result.get("stdout") or "", result.get("stderr") or ""])
        if not result.get("ok"):
            cookie_failure = self._cookie_failure_type(combined_output, cookie_path=cookie_path)
            payload = {
                "ok": False,
                "error_type": cookie_failure or "YtDlpFailed",
                "error": "yt-dlp failed to download the video.",
                "backend": "yt-dlp",
                "url": url,
                "returncode": result.get("returncode"),
                "stdout_tail": result.get("stdout_tail"),
                "stderr_tail": result.get("stderr_tail"),
                "cookie_path": str(cookie_path) if cookie_path else None,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
            if cookie_failure:
                payload.update(self._video_cookie_required_payload(cookie_failure, cookie_path))
            if downloaded_path.exists():
                payload["partial_path"] = str(downloaded_path)
            return payload
        if not downloaded_path.exists():
            return {
                "ok": False,
                "error_type": "DownloadedFileMissing",
                "error": "yt-dlp finished but no output file was found.",
                "backend": "yt-dlp",
                "url": url,
                "stdout_tail": result.get("stdout_tail"),
                "stderr_tail": result.get("stderr_tail"),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        if downloaded_path.stat().st_size > int(max_mb) * 1024 * 1024:
            return {
                "ok": False,
                "error_type": "FileTooLarge",
                "error": f"Download exceeded max_mb={max_mb}.",
                "backend": "yt-dlp",
                "path": str(downloaded_path),
                "size_bytes": downloaded_path.stat().st_size,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }

        validation = self._validate_downloaded_video(downloaded_path)
        if not validation.get("ok"):
            validation.update(
                {
                    "backend": "yt-dlp",
                    "url": url,
                    "path": str(downloaded_path),
                    "size_bytes": downloaded_path.stat().st_size if downloaded_path.exists() else 0,
                    "stdout_tail": result.get("stdout_tail"),
                    "stderr_tail": result.get("stderr_tail"),
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
            )
            return validation
        return {
            "ok": True,
            "backend": "yt-dlp",
            "url": url,
            "path": str(downloaded_path),
            "size_bytes": downloaded_path.stat().st_size,
            "video": validation.get("video"),
            "cookie_path": str(cookie_path) if cookie_path else None,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }

    def _build_ytdlp_command(
            self,
            url: str,
            output: Path,
            *,
            max_mb: int,
            timeout_seconds: int,
            cookie_path: Path | None,
    ) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "yt_dlp",
            "--no-playlist",
            "--force-overwrites",
            "--socket-timeout",
            str(max(5, int(timeout_seconds))),
            "--retries",
            "3",
            "--fragment-retries",
            "3",
            "-f",
            self._build_ytdlp_format_selector(max_mb),
            "-S",
            "res,ext:mp4:m4a",
            "--merge-output-format",
            "mp4",
            "-o",
            str(output),
        ]
        js_runtime = self._resolve_ytdlp_js_runtime()
        if js_runtime:
            command += ["--js-runtimes", js_runtime]
        if cookie_path:
            command += ["--cookies", str(cookie_path)]
        command.append(url)
        return command

    @staticmethod
    def _build_ytdlp_format_selector(max_mb: int) -> str:
        total_limit_mb = max(1, int(max_mb))
        video_limit_mb = max(1, int(total_limit_mb * 0.75))
        return (
            f"bv*[ext=mp4][filesize<{video_limit_mb}M]+ba[ext=m4a]/"
            f"b[ext=mp4][filesize<{total_limit_mb}M]/"
            f"bv*[filesize<{video_limit_mb}M]+ba/"
            f"b[filesize<{total_limit_mb}M]/best"
        )

    def _resolve_ytdlp_js_runtime(self) -> str | None:
        configured = os.environ.get("JANVIS_YTDLP_JS_RUNTIME")
        if configured:
            return configured

        for name in ("deno", "node"):
            resolved = shutil.which(name)
            if resolved:
                return f"{name}:{resolved}"

        home = Path.home()
        candidates = [
            ("deno", home / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages" / "DenoLand.Deno_Microsoft.Winget.Source_8wekyb3d8bbwe" / "deno.exe"),
            ("deno", home / ".deno" / "bin" / "deno.exe"),
            ("node", Path(os.environ.get("ProgramFiles", "C:\\Program Files")) / "nodejs" / "node.exe"),
            ("node", home / "AppData" / "Local" / "Programs" / "nodejs" / "node.exe"),
        ]
        for name, path in candidates:
            if path.exists():
                return f"{name}:{path}"
        return None

    def _download_video_with_you_get(
            self,
            url: str,
            output: Path,
            *,
            max_mb: int,
            timeout_seconds: int,
            cookie_path: Path | None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        command = self._build_you_get_command(url, output, timeout_seconds=timeout_seconds, cookie_path=cookie_path)
        result = self._run_command(command, timeout=max(10, int(timeout_seconds)), cwd=output.parent)
        downloaded_path = self._locate_downloaded_video(output)
        combined_output = "\n".join([result.get("stdout") or "", result.get("stderr") or ""])
        if not result.get("ok"):
            cookie_failure = self._you_get_cookie_failure_type(combined_output, cookie_path=cookie_path)
            payload = {
                "ok": False,
                "error_type": cookie_failure or "YouGetFailed",
                "error": "you-get failed to download the video.",
                "backend": "you-get",
                "url": url,
                "returncode": result.get("returncode"),
                "stdout_tail": result.get("stdout_tail"),
                "stderr_tail": result.get("stderr_tail"),
                "cookie_path": str(cookie_path) if cookie_path else None,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
            if cookie_failure:
                payload.update(self._video_cookie_required_payload(cookie_failure, cookie_path))
            if downloaded_path.exists():
                payload["partial_path"] = str(downloaded_path)
            return payload
        if not downloaded_path.exists():
            return {
                "ok": False,
                "error_type": "DownloadedFileMissing",
                "error": "you-get finished but no output file was found.",
                "backend": "you-get",
                "url": url,
                "stdout_tail": result.get("stdout_tail"),
                "stderr_tail": result.get("stderr_tail"),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        if downloaded_path.stat().st_size > int(max_mb) * 1024 * 1024:
            return {
                "ok": False,
                "error_type": "FileTooLarge",
                "error": f"Download exceeded max_mb={max_mb}.",
                "backend": "you-get",
                "path": str(downloaded_path),
                "size_bytes": downloaded_path.stat().st_size,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }

        validation = self._validate_downloaded_video(downloaded_path)
        if not validation.get("ok"):
            validation.update(
                {
                    "backend": "you-get",
                    "url": url,
                    "path": str(downloaded_path),
                    "size_bytes": downloaded_path.stat().st_size if downloaded_path.exists() else 0,
                    "stdout_tail": result.get("stdout_tail"),
                    "stderr_tail": result.get("stderr_tail"),
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
            )
            return validation
        return {
            "ok": True,
            "backend": "you-get",
            "url": url,
            "path": str(downloaded_path),
            "size_bytes": downloaded_path.stat().st_size,
            "video": validation.get("video"),
            "cookie_path": str(cookie_path) if cookie_path else None,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }

    def _build_you_get_command(
            self,
            url: str,
            output: Path,
            *,
            timeout_seconds: int,
            cookie_path: Path | None,
    ) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "you_get",
            "--force",
            "--timeout",
            str(max(5, int(timeout_seconds))),
            "--output-dir",
            str(output.parent),
            "--output-filename",
            output.stem,
        ]
        if cookie_path:
            command += ["--cookies", str(cookie_path)]
        command.append(url)
        return command

    @staticmethod
    def _you_get_cookie_failure_type(output: str, *, cookie_path: Path | None) -> str | None:
        return MediaTool._cookie_failure_type(output, cookie_path=cookie_path)

    @staticmethod
    def _cookie_failure_type(output: str, *, cookie_path: Path | None) -> str | None:
        lowered = (output or "").lower()
        markers = (
            "cookie",
            "cookies",
            "login",
            "sign in",
            "not a bot",
            "captcha",
            "forbidden",
            "403",
            "unauthorized",
            "verify",
            "需要登录",
            "请登录",
        )
        if not any(marker in lowered for marker in markers):
            return None
        return "CookieInvalidOrExpired" if cookie_path else "CookieRequired"

    def _video_cookie_required_payload(self, error_type: str, cookie_path: Path | None) -> dict[str, Any]:
        if cookie_path:
            message = (
                f"视频站点仍然拒绝访问，当前 cookie 可能已失效或不匹配：{cookie_path}。"
                "请重新导出有效 cookie 文件，并在下一次请求中提供新的 cookie_path。"
            )
        else:
            message = (
                "视频站点要求登录或 cookie，当前任务无法继续。"
                "请导出 cookie 到本地文件，并在下一次请求中提供 cookie_path，例如 D:\\my_youtube_cookies.txt。"
            )
        return {
            "requires_user_action": True,
            "retryable_with_cookie": True,
            "message": message,
            "hint": message,
        }

    def _validate_downloaded_video(self, file_path: Path) -> dict[str, Any]:
        if not file_path.exists():
            return {"ok": False, "error_type": "DownloadedFileMissing", "error": "Downloaded video file was not created."}
        if file_path.stat().st_size <= 0:
            return {"ok": False, "error_type": "EmptyFile", "error": "Downloaded video file is empty."}
        if self._file_starts_like_text(file_path):
            return {"ok": False, "error_type": "NonVideoResponse", "error": "Downloaded content looks like text or HTML, not a video file."}

        probe = self._ffprobe(file_path)
        if not probe.get("available"):
            return {"ok": True, "video": {"validation": "ffprobe_unavailable", "warning": probe.get("error")}}
        streams = probe.get("streams") or []
        if not any(stream.get("codec_type") == "video" for stream in streams if isinstance(stream, dict)):
            return {
                "ok": False,
                "error_type": "InvalidVideoFile",
                "error": "Downloaded file does not contain a video stream.",
                "ffprobe": self._compact_failed_tool_payload(probe),
            }
        return {"ok": True, "video": self._video_probe_summary(probe)}

    def _video_probe_summary(self, probe: dict[str, Any]) -> dict[str, Any]:
        format_info = probe.get("format") or {}
        streams = [
            {
                "codec_type": stream.get("codec_type"),
                "codec_name": stream.get("codec_name"),
                "width": stream.get("width"),
                "height": stream.get("height"),
                "duration": stream.get("duration"),
            }
            for stream in (probe.get("streams") or [])
            if isinstance(stream, dict) and stream.get("codec_type") in {"video", "audio"}
        ]
        return {
            "format_name": format_info.get("format_name"),
            "duration": format_info.get("duration"),
            "bit_rate": format_info.get("bit_rate"),
            "streams": streams,
        }

    def _describe_with_ollama_vision(self, image_path: Path, *, prompt: str, model: str) -> dict[str, Any]:
        import httpx
        from openai import OpenAI

        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        client = OpenAI(
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1/"),
            api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
            http_client=httpx.Client(trust_env=False, timeout=300),
        )
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": f"data:{mime_type};base64,{image_b64}",
                            },
                        ],
                    }
                ],
                temperature=0,
                timeout=300,
            )
            content = response.choices[0].message.content or ""
            return {
                "ok": True,
                "backend": "local_ollama",
                "model": model,
                "text": content,
                "metadata": self._image_metadata(image_path),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        finally:
            close_client = getattr(client, "close", None)
            if callable(close_client):
                close_client()
            self._unload_ollama_model(model)

    def _unload_ollama_model(self, model: str) -> None:
        try:
            import httpx

            # 本地视觉模型通常占用显存，调用后主动释放，避免后续 Agent 运行被显存挤占。
            base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1/").rstrip("/")
            if base_url.endswith("/v1"):
                base_url = base_url[:-3]
            with httpx.Client(trust_env=False, timeout=60) as unload_client:
                unload_client.post(
                    f"{base_url.rstrip('/')}/api/chat",
                    json={"model": model, "messages": [], "keep_alive": 0},
                )
        except Exception:
            pass

    def _vision_model_config_path(self) -> Path:
        return self.project_root / "agent" / "config" / "local_vision_model.json"

    def _load_vision_model_config(self) -> dict[str, Any]:
        config_path = self._vision_model_config_path()
        if not config_path.exists():
            return {"default_model": "gemma4:e4b", "models": [{"name": "qwen3.5:9b"}, {"name": "gemma4:e4b"}]}
        with open(config_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}

    def _vision_model_names(self, config: dict[str, Any]) -> list[str]:
        result: list[str] = []
        for item in config.get("models") or []:
            if isinstance(item, str):
                name = item.strip()
            elif isinstance(item, dict):
                name = str(item.get("name") or "").strip()
            else:
                name = ""
            if name and name not in result:
                result.append(name)
        return result

    def _default_vision_model_name(self) -> str:
        config = self._load_vision_model_config()
        default_model = str(config.get("default_model") or "").strip()
        if default_model:
            return default_model
        models = self._vision_model_names(config)
        return models[0] if models else "gemma4:e4b"

    def _describe_with_local_pytorch(
            self,
            image_path: Path,
            *,
            prompt: str,
            model: str | None,
            allow_download: bool,
            max_new_tokens: int,
    ) -> dict[str, Any]:
        model_name = model or os.environ.get("JANVIS_IMAGE_VISION_MODEL", "Salesforce/blip-image-captioning-base")
        try:
            import torch
        except Exception as error:
            return self._error_payload(error, backend="local_pytorch", model=model_name)
        try:
            if "qwen" in model_name.lower() and "vl" in model_name.lower():
                return self._describe_with_qwen_vl(
                    image_path,
                    prompt=prompt,
                    model_name=model_name,
                    allow_download=allow_download,
                    max_new_tokens=max_new_tokens,
                )
            if "llava" in model_name.lower():
                return self._describe_with_llava(
                    image_path,
                    prompt=prompt,
                    model_name=model_name,
                    allow_download=allow_download,
                    max_new_tokens=max_new_tokens,
                    torch=torch,
                )
            return self._describe_with_blip(
                image_path,
                prompt=prompt,
                model_name=model_name,
                allow_download=allow_download,
                max_new_tokens=max_new_tokens,
                torch=torch,
            )
        except Exception as error:
            return self._error_payload(error, backend="local_pytorch", model=model_name)

    def _describe_with_blip(
            self,
            image_path: Path,
            *,
            prompt: str,
            model_name: str,
            allow_download: bool,
            max_new_tokens: int,
            torch,
    ) -> dict[str, Any]:
        from PIL import Image
        from transformers import BlipForConditionalGeneration, BlipProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        local_files_only = not bool(allow_download)
        processor = BlipProcessor.from_pretrained(model_name, local_files_only=local_files_only)
        model = BlipForConditionalGeneration.from_pretrained(model_name, local_files_only=local_files_only).to(device)
        image = Image.open(image_path).convert("RGB")
        # BLIP 是图片描述模型，不是完整 VLM；prompt 只作为条件描述输入。
        inputs = processor(image, prompt, return_tensors="pt").to(device)
        generated = model.generate(**inputs, max_new_tokens=int(max_new_tokens))
        text = processor.decode(generated[0], skip_special_tokens=True).strip()
        return {
            "ok": True,
            "backend": "local_pytorch",
            "model": model_name,
            "device": device,
            "text": text,
            "metadata": self._image_metadata(image_path),
        }

    def _describe_with_qwen_vl(
            self,
            image_path: Path,
            *,
            prompt: str,
            model_name: str,
            allow_download: bool,
            max_new_tokens: int,
    ) -> dict[str, Any]:
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        local_files_only = not bool(allow_download)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            local_files_only=local_files_only,
        )
        processor = AutoProcessor.from_pretrained(model_name, local_files_only=local_files_only)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=int(max_new_tokens))
        trimmed_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return {
            "ok": True,
            "backend": "local_pytorch",
            "model": model_name,
            "device": "cuda" if torch.cuda.is_available() else "auto",
            "text": output_text.strip(),
            "metadata": self._image_metadata(image_path),
        }

    def _describe_with_llava(
            self,
            image_path: Path,
            *,
            prompt: str,
            model_name: str,
            allow_download: bool,
            max_new_tokens: int,
            torch,
    ) -> dict[str, Any]:
        from PIL import Image
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        cuda_available = torch.cuda.is_available()
        local_files_only = not bool(allow_download)
        dtype = torch.float16 if cuda_available else torch.float32
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=local_files_only,
        )
        if cuda_available:
            model = model.to(0)
        processor = AutoProcessor.from_pretrained(model_name, local_files_only=local_files_only)
        image = Image.open(image_path).convert("RGB")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            }
        ]
        # LLaVA 需要按模型 chat template 注入 <image> 占位，避免手写模板和模型版本耦合。
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=text_prompt, return_tensors="pt")
        if cuda_available:
            inputs = inputs.to(0, dtype)
        output = model.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=False)
        text = processor.decode(output[0], skip_special_tokens=True).strip()
        return {
            "ok": True,
            "backend": "local_pytorch",
            "model": model_name,
            "device": "cuda" if cuda_available else "cpu",
            "text": text,
            "metadata": self._image_metadata(image_path),
        }

    def _describe_with_mcp(self, image_path: Path, *, prompt: str) -> dict[str, Any]:
        if self.mcp_client is None:
            return {"ok": False, "error_type": "NoMCPClient", "error": "No MCP client is configured."}
        try:
            tools = self.mcp_client.list_tools()
        except Exception as error:
            return self._error_payload(error, backend="mcp")
        candidates = [tool for tool in tools if self._is_image_vision_mcp_tool(tool)]
        if not candidates:
            return {"ok": False, "error_type": "NoVisionMCPTool", "error": "No image vision MCP tool is available."}

        last_error: dict[str, Any] | None = None
        for tool in candidates:
            tool_name = tool.get("name")
            args = self._build_mcp_image_args(tool, image_path, prompt)
            try:
                result = self.mcp_client.call_tool(tool_name, args)
                if isinstance(result, dict) and result.get("ok") is False:
                    last_error = {"ok": False, "tool": tool_name, "error": result}
                    continue
                return {
                    "ok": True,
                    "backend": "mcp",
                    "mcp_tool": tool_name,
                    "text": result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str),
                    "raw_result": result,
                    "metadata": self._image_metadata(image_path),
                }
            except Exception as error:
                last_error = self._error_payload(error, backend="mcp", tool=tool_name)
        return last_error or {"ok": False, "error_type": "MCPError", "error": "All image vision MCP tools failed."}

    def _is_image_vision_mcp_tool(self, tool: dict[str, Any]) -> bool:
        text = f"{tool.get('name', '')} {tool.get('description', '')}".lower()
        positive = ("image", "vision", "visual", "ocr", "describe", "analyze", "multimodal")
        negative = ("generate", "create image", "text-to-image", "tts")
        return any(word in text for word in positive) and not any(word in text for word in negative)

    def _build_mcp_image_args(self, tool: dict[str, Any], image_path: Path, prompt: str) -> dict[str, Any]:
        params = tool.get("parameters") or {}
        properties = params.get("properties") or {}
        if not properties:
            return {"path": str(image_path), "prompt": prompt}

        image_base64: str | None = None
        args: dict[str, Any] = {}
        for key, schema in properties.items():
            lower = key.lower()
            description = str((schema or {}).get("description", "")).lower()
            if lower in ("prompt", "question", "query", "instruction", "text"):
                args[key] = prompt
            elif lower in ("path", "file_path", "image_path", "local_path"):
                args[key] = str(image_path)
            elif lower in ("image_paths", "paths", "files"):
                args[key] = [str(image_path)]
            elif "base64" in lower or "base64" in description:
                image_base64 = image_base64 or self._image_as_base64(image_path)
                args[key] = image_base64
            elif lower in ("image", "input_image") and "path" in description:
                args[key] = str(image_path)
        return args or {"path": str(image_path), "prompt": prompt}

    def _image_ocr_payload(self, image_path: Path, languages: str = "eng+chi_sim") -> dict[str, Any]:
        metadata = self._image_metadata(image_path)
        try:
            from PIL import Image
            import pytesseract

            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, lang=languages).strip()
            return {"path": str(image_path), "metadata": metadata, "ocr_backend": "pytesseract", "text": text}
        except Exception as first_error:
            try:
                import easyocr
                import numpy as np
                from PIL import Image

                reader = easyocr.Reader(self._easyocr_languages(languages), gpu=True)
                # easyocr 传路径时会走 OpenCV，Windows 中文路径可能读取失败；改用内存图像。
                image = Image.open(image_path)
                if image.mode == "RGBA":
                    background = Image.new("RGB", image.size, "white")
                    background.paste(image, mask=image.getchannel("A"))
                    image = background
                else:
                    image = image.convert("RGB")
                text = "\n".join(reader.readtext(np.array(image), detail=0)).strip()
                return {"path": str(image_path), "metadata": metadata, "ocr_backend": "easyocr", "text": text}
            except Exception as second_error:
                return {
                    "path": str(image_path),
                    "metadata": metadata,
                    "ocr_backend": None,
                    "text": "",
                    "ocr_error": f"{type(first_error).__name__}: {first_error}; {type(second_error).__name__}: {second_error}",
                }

    def _image_metadata(self, image_path: Path) -> dict[str, Any] | None:
        try:
            from PIL import ExifTags, Image

            image = Image.open(image_path)
            metadata: dict[str, Any] = {
                "format": image.format,
                "mode": image.mode,
                "width": image.width,
                "height": image.height,
            }
            exif = image.getexif()
            if exif:
                # EXIF 只返回前 30 项，避免工具结果被相机元数据撑大。
                metadata["exif"] = {
                    str(ExifTags.TAGS.get(tag, tag)): str(value)
                    for tag, value in list(exif.items())[:30]
                }
            return metadata
        except Exception:
            return None

    def _ffprobe(self, file_path: Path) -> dict[str, Any]:
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            return {"available": False, "error": "ffprobe not found in PATH."}
        result = self._run_command(
            [ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(file_path)],
            timeout=30,
        )
        if not result.get("ok"):
            return result
        try:
            return {"available": True, **json.loads(result.get("stdout", "{}") or "{}")}
        except json.JSONDecodeError:
            return {"available": True, "raw": result.get("stdout", "")}

    def _ffmpeg_base_command(self, start_seconds: float | None) -> list[str]:
        ffmpeg = self._require_command("ffmpeg")
        command = [ffmpeg, "-y"]
        if start_seconds is not None:
            command += ["-ss", str(start_seconds)]
        return command

    def _run_command(self, command: list[str], timeout: int, cwd: Path | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                timeout=timeout,
                cwd=str(cwd) if cwd else None,
            )
        except subprocess.TimeoutExpired as error:
            return {
                "ok": False,
                "error_type": "TimeoutExpired",
                "error": f"Command timed out after {timeout}s.",
                "stdout": self._decode_bytes(error.stdout or b""),
                "stderr": self._decode_bytes(error.stderr or b""),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        stdout = self._decode_bytes(result.stdout)
        stderr = self._decode_bytes(result.stderr)
        return {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-4000:],
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }

    def _format_transcript_payload(self, segments: list[dict[str, Any]], *, output_format: str, info: dict[str, Any]) -> dict[str, Any]:
        text = "\n".join(segment["text"] for segment in segments if segment.get("text"))
        if output_format == "json":
            content: Any = {"segments": segments, "info": info}
        elif output_format == "srt":
            content = self._segments_to_srt(segments)
        else:
            content = text
        return {
            "ok": True,
            "backend": "faster_whisper",
            "output_format": output_format,
            "content": content,
            "segments": segments if output_format != "json" else None,
            "info": info,
        }

    def _run_audio_transcribe_worker(
            self,
            *,
            audio_path: Path,
            output_format: str,
            model_name: str,
            language: str | None,
            beam_size: int,
            timeout_seconds: int,
    ) -> dict[str, Any]:
        payload = {
            "project_root": str(self.project_root),
            "path": str(audio_path),
            "output_format": output_format,
            "model": model_name,
            "language": language,
            "beam_size": int(beam_size),
        }
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["PYTHONPATH"] = str(self.project_root) + os.pathsep + env.get("PYTHONPATH", "")
        started = time.perf_counter()
        try:
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--audio-transcribe-worker"],
                input=json.dumps(payload, ensure_ascii=False),
                text=True,
                encoding="utf-8",
                capture_output=True,
                cwd=str(self.project_root),
                env=env,
                timeout=max(1, int(timeout_seconds)),
            )
        except subprocess.TimeoutExpired as error:
            return {
                "ok": False,
                "error_type": "TimeoutExpired",
                "error": f"AUDIO_TRANSCRIBE timed out after {timeout_seconds}s.",
                "model": model_name,
                "output_format": output_format,
                "stdout_tail": self._decode_text(error.stdout)[-2000:],
                "stderr_tail": self._decode_text(error.stderr)[-2000:],
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        if result.returncode != 0:
            return {
                "ok": False,
                "error_type": "WorkerFailed",
                "error": "Audio transcription worker failed.",
                "model": model_name,
                "output_format": output_format,
                "returncode": result.returncode,
                "stdout_tail": (result.stdout or "")[-2000:],
                "stderr_tail": (result.stderr or "")[-4000:],
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        try:
            payload = self._parse_worker_json_output(result.stdout or "{}")
        except json.JSONDecodeError:
            return {
                "ok": False,
                "error_type": "InvalidWorkerOutput",
                "error": "Audio transcription worker returned non-JSON output.",
                "model": model_name,
                "output_format": output_format,
                "stdout_tail": (result.stdout or "")[-4000:],
                "stderr_tail": (result.stderr or "")[-4000:],
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        if isinstance(payload, dict):
            payload.setdefault("elapsed_seconds", round(time.perf_counter() - started, 3))
            if result.stderr:
                payload["worker_stderr_tail"] = result.stderr[-2000:]
            return payload
        return {
            "ok": False,
            "error_type": "InvalidWorkerOutput",
            "error": "Audio transcription worker returned an unexpected JSON value.",
            "model": model_name,
            "output_format": output_format,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }

    def _parse_worker_json_output(self, stdout: str) -> Any:
        text = (stdout or "").strip()
        try:
            return json.loads(text or "{}")
        except json.JSONDecodeError:
            pass

        decoder = json.JSONDecoder()
        last_payload: Any = None
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                last_payload = payload
        if last_payload is not None:
            return last_payload
        raise json.JSONDecodeError("No JSON object found in worker stdout.", text, 0)

    def _run_image_ocr_worker(
            self,
            *,
            image_path: Path,
            languages: str,
            timeout_seconds: int,
    ) -> dict[str, Any]:
        payload = {
            "project_root": str(self.project_root),
            "path": str(image_path),
            "languages": languages,
        }
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["PYTHONPATH"] = str(self.project_root) + os.pathsep + env.get("PYTHONPATH", "")
        # Anaconda + torch/easyocr 在 Windows 上可能重复加载 OpenMP runtime，放在子进程里隔离。
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        started = time.perf_counter()
        try:
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--image-ocr-worker"],
                input=json.dumps(payload, ensure_ascii=False),
                text=True,
                encoding="utf-8",
                capture_output=True,
                cwd=str(self.project_root),
                env=env,
                timeout=max(1, int(timeout_seconds)),
            )
        except subprocess.TimeoutExpired as error:
            return {
                "path": str(image_path),
                "ocr_backend": None,
                "text": "",
                "ocr_error": f"IMAGE_OCR timed out after {timeout_seconds}s.",
                "stdout_tail": self._decode_text(error.stdout)[-2000:],
                "stderr_tail": self._decode_text(error.stderr)[-2000:],
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        if result.returncode != 0:
            return {
                "path": str(image_path),
                "ocr_backend": None,
                "text": "",
                "ocr_error": "Image OCR worker failed.",
                "returncode": result.returncode,
                "stdout_tail": (result.stdout or "")[-2000:],
                "stderr_tail": (result.stderr or "")[-4000:],
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        try:
            payload = self._parse_worker_json_output(result.stdout or "{}")
        except json.JSONDecodeError:
            return {
                "path": str(image_path),
                "ocr_backend": None,
                "text": "",
                "ocr_error": "Image OCR worker returned non-JSON output.",
                "stdout_tail": (result.stdout or "")[-4000:],
                "stderr_tail": (result.stderr or "")[-4000:],
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        if isinstance(payload, dict):
            payload.setdefault("elapsed_seconds", round(time.perf_counter() - started, 3))
            if result.stderr and not payload.get("text"):
                payload["worker_stderr_tail"] = result.stderr[-2000:]
            return payload
        return {
            "path": str(image_path),
            "ocr_backend": None,
            "text": "",
            "ocr_error": "Image OCR worker returned an unexpected JSON value.",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }

    def _audio_transcribe_in_process(
            self,
            *,
            audio_path: Path,
            output_format: str,
            model_name: str,
            language: str | None,
            beam_size: int,
    ) -> dict[str, Any]:
        try:
            from faster_whisper import WhisperModel
        except Exception as error:
            return {
                "ok": False,
                "error_type": type(error).__name__,
                "error": "Audio transcription dependency is not available.",
                "detail": str(error),
                "model": model_name,
                "output_format": output_format,
            }

        last_error: dict[str, Any] | None = None
        for device, compute_type in self._whisper_device_candidates():
            started = time.perf_counter()
            try:
                whisper_model = WhisperModel(model_name, device=device, compute_type=compute_type)
                segments_iter, info = whisper_model.transcribe(
                    str(audio_path),
                    language=language or None,
                    beam_size=int(beam_size),
                )
                segments = [
                    {"start": float(segment.start), "end": float(segment.end), "text": segment.text.strip()}
                    for segment in segments_iter
                ]
                return self._format_transcript_payload(
                    segments,
                    output_format=output_format,
                    info={
                        "language": getattr(info, "language", None),
                        "duration": getattr(info, "duration", None),
                        "model": model_name,
                        "device": device,
                        "compute_type": compute_type,
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                    },
                )
            except Exception as error:
                last_error = {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "model": model_name,
                    "output_format": output_format,
                    "device": device,
                    "compute_type": compute_type,
                }
                # CUDA 运行库缺失时自动降级到 CPU，避免 Agent 反复重试不同模型。
                if device == "cuda":
                    continue
                return last_error
        return last_error or {
            "ok": False,
            "error_type": "TranscriptionFailed",
            "error": "Audio transcription failed.",
            "model": model_name,
            "output_format": output_format,
        }

    def _whisper_device_candidates(self) -> list[tuple[str, str]]:
        requested_device = os.environ.get("JANVIS_WHISPER_DEVICE", "cpu").strip().lower()
        requested_compute_type = os.environ.get("JANVIS_WHISPER_COMPUTE_TYPE", "").strip()
        if requested_device == "cuda":
            return [("cuda", requested_compute_type or "float16"), ("cpu", "int8")]
        if requested_device == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    return [("cuda", requested_compute_type or "float16"), ("cpu", "int8")]
            except Exception:
                pass
        return [("cpu", requested_compute_type or "int8")]

    def _segments_to_srt(self, segments: list[dict[str, Any]]) -> str:
        blocks = []
        for index, segment in enumerate(segments, 1):
            blocks.append(
                f"{index}\n"
                f"{self._srt_time(segment['start'])} --> {self._srt_time(segment['end'])}\n"
                f"{segment['text']}\n"
            )
        return "\n".join(blocks)

    def _srt_time(self, seconds: float) -> str:
        millis = int(round(float(seconds) * 1000))
        hours, rest = divmod(millis, 3600_000)
        minutes, rest = divmod(rest, 60_000)
        secs, millis = divmod(rest, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _whisper_device(self) -> tuple[str, str]:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda", os.environ.get("JANVIS_WHISPER_COMPUTE_TYPE", "float16")
        except Exception:
            pass
        return "cpu", os.environ.get("JANVIS_WHISPER_COMPUTE_TYPE", "int8")

    def _resolve_input_path(self, raw_path: str) -> Path:
        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        return path

    def _resolve_optional_file_path(self, raw_path: str | None) -> Path | None:
        if not raw_path:
            return None
        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        return path

    def _resolve_video_cookie_path(self, raw_path: str | None) -> Path | None:
        # cookie 参数优先，其次兼容不同下载后端的环境变量。
        candidate = (
                raw_path
                or os.environ.get("JANVIS_VIDEO_COOKIE_FILE")
                or os.environ.get("JANVIS_YTDLP_COOKIE_FILE")
                or os.environ.get("JANVIS_YOU_GET_COOKIE_FILE")
        )
        return self._resolve_optional_file_path(candidate)

    def _resolve_output_path(self, raw_path: str | None, default_stem: str, suffix: str) -> Path:
        if raw_path:
            path = Path(str(raw_path)).expanduser()
            if not path.is_absolute():
                path = self.output_root / path
        else:
            path = self.output_root / f"{default_stem}_{self._now_stamp()}{suffix}"
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _resolve_output_dir(self, raw_dir: str | None, default_name: str) -> Path:
        path = Path(str(raw_dir)).expanduser() if raw_dir else self.output_root / default_name
        if not path.is_absolute():
            path = self.output_root / path
        path = path.resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _require_command(self, name: str) -> str:
        resolved = shutil.which(name)
        if not resolved:
            raise FileNotFoundError(f"{name} not found in PATH.")
        return resolved

    def _video_output_name(self, parsed: urllib.parse.ParseResult) -> str:
        name = self._filename_from_url(parsed)
        if name and Path(name).suffix.lower() in self._video_file_suffixes():
            return name
        return f"video_{self._now_stamp()}.mp4"

    def _should_use_media_downloader_for_video_url(self, parsed: urllib.parse.ParseResult) -> bool:
        host = (parsed.netloc or "").lower()
        path_suffix = Path(urllib.parse.unquote(parsed.path)).suffix.lower()
        known_page_hosts = (
            "youtube.com",
            "youtu.be",
            "vimeo.com",
            "bilibili.com",
            "twitter.com",
            "x.com",
            "tiktok.com",
            "facebook.com",
            "instagram.com",
        )
        if any(host == item or host.endswith(f".{item}") for item in known_page_hosts):
            return True
        if path_suffix in {".m3u8", ".mpd"}:
            return True
        return bool(parsed.path and path_suffix not in self._video_file_suffixes())

    @staticmethod
    def _video_file_suffixes() -> set[str]:
        return {".mp4", ".m4v", ".mov", ".webm", ".mkv", ".avi", ".flv", ".wmv", ".mpeg", ".mpg", ".ts"}

    @staticmethod
    def _is_youtube_url(parsed: urllib.parse.ParseResult) -> bool:
        host = (parsed.netloc or "").lower()
        return host == "youtu.be" or host.endswith(".youtu.be") or host == "youtube.com" or host.endswith(".youtube.com")

    @staticmethod
    def _is_obvious_non_video_content_type(content_type: str) -> bool:
        normalized = (content_type or "").split(";", 1)[0].strip().lower()
        if not normalized:
            return False
        if normalized.startswith("text/"):
            return True
        return normalized in {
            "application/json",
            "application/xml",
            "application/xhtml+xml",
            "application/vnd.apple.mpegurl",
            "application/x-mpegurl",
            "audio/mpegurl",
        }

    def _file_starts_like_text(self, file_path: Path) -> bool:
        head = file_path.read_bytes()[:512].lstrip()
        lowered = head[:128].lower()
        return (
            lowered.startswith(b"<!doctype html")
            or lowered.startswith(b"<html")
            or lowered.startswith(b"<?xml")
            or lowered.startswith(b"#extm3u")
        )

    def _file_text_head(self, file_path: Path, limit: int = 500) -> str:
        try:
            return self._decode_bytes(file_path.read_bytes()[:limit])
        except Exception:
            return ""

    def _locate_downloaded_video(self, expected_path: Path) -> Path:
        if expected_path.exists():
            return expected_path
        candidates = [
            path
            for path in expected_path.parent.glob(f"{expected_path.stem}*")
            if path.is_file() and path.suffix not in {".part", ".ytdl", ".temp"}
        ]
        if not candidates:
            return expected_path
        video_candidates = [path for path in candidates if path.suffix.lower() in self._video_file_suffixes()]
        return max(video_candidates or candidates, key=lambda path: path.stat().st_mtime)

    @staticmethod
    def _compact_failed_tool_payload(payload: dict[str, Any]) -> dict[str, Any]:
        keys = (
            "ok",
            "error_type",
            "error",
            "message",
            "returncode",
            "path",
            "partial_path",
            "content_type",
            "stdout_tail",
            "stderr_tail",
        )
        return {key: payload.get(key) for key in keys if key in payload and payload.get(key) is not None}

    def _filename_from_url(self, parsed: urllib.parse.ParseResult) -> str | None:
        name = Path(urllib.parse.unquote(parsed.path)).name
        if not name or "." not in name:
            return None
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)

    def _safe_stem(self, value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-") or "media"

    def _format_timestamp(self, value: Any) -> str:
        if isinstance(value, (int, float)):
            return str(float(value))
        return str(value)

    def _normalize_output_format(self, value: str | None) -> str:
        output_format = (value or "txt").strip().lower()
        if output_format not in ("txt", "json", "srt"):
            raise ValueError("output_format must be one of: txt, json, srt.")
        return output_format

    def _easyocr_languages(self, languages: str) -> list[str]:
        mapping = {"eng": "en", "en": "en", "chi_sim": "ch_sim", "ch_sim": "ch_sim", "zh": "ch_sim"}
        return [mapping.get(part.strip(), part.strip()) for part in re.split(r"[+,]", languages) if part.strip()]

    def _image_as_base64(self, image_path: Path) -> str:
        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _decode_text(self, value: bytes | str | None) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return self._decode_bytes(value)

    def _decode_bytes(self, value: bytes | str | None) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        for encoding in ("utf-8", "gb18030", "gbk"):
            try:
                return value.decode(encoding)
            except UnicodeDecodeError:
                continue
        return value.decode("utf-8", errors="replace")

    def _attempt_summary(self, backend: str, result: dict[str, Any]) -> dict[str, Any]:
        return {
            "backend": backend,
            "ok": bool(result.get("ok")),
            "error_type": result.get("error_type"),
            "error": result.get("error"),
            "model": result.get("model"),
            "tool": result.get("tool") or result.get("mcp_tool"),
        }

    def _error_payload(self, error: Exception, **extra: Any) -> dict[str, Any]:
        payload = {"ok": False, "error_type": type(error).__name__, "error": str(error)}
        payload.update(extra)
        return payload

    def _now_stamp(self) -> str:
        return time.strftime("%Y%m%d_%H%M%S")

    def _json(self, payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False, default=str)


def _audio_transcribe_worker_main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        tool = MediaTool(project_root=str(payload.get("project_root") or Path(__file__).resolve().parents[1]))
        # 第三方转写库可能把日志写到 stdout，worker 的 stdout 只保留最终 JSON。
        with contextlib.redirect_stdout(sys.stderr):
            result = tool._audio_transcribe_in_process(
                audio_path=Path(str(payload["path"])),
                output_format=str(payload.get("output_format") or "txt"),
                model_name=str(payload.get("model") or "base"),
                language=payload.get("language"),
                beam_size=int(payload.get("beam_size") or 5),
            )
    except Exception as error:
        result = {"ok": False, "error_type": type(error).__name__, "error": str(error)}
    sys.stdout.write(json.dumps(result, ensure_ascii=False, default=str))
    return 0


def _image_ocr_worker_main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        tool = MediaTool(project_root=str(payload.get("project_root") or Path(__file__).resolve().parents[1]))
        # OCR 框架会打印模型下载和初始化日志，stdout 只保留最终 JSON。
        with contextlib.redirect_stdout(sys.stderr):
            result = tool._image_ocr_payload(
                image_path=Path(str(payload["path"])),
                languages=str(payload.get("languages") or "eng+chi_sim"),
            )
    except Exception as error:
        result = {"path": "", "ocr_backend": None, "text": "", "ocr_error": f"{type(error).__name__}: {error}"}
    sys.stdout.write(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    if "--audio-transcribe-worker" in sys.argv:
        raise SystemExit(_audio_transcribe_worker_main())
    if "--image-ocr-worker" in sys.argv:
        raise SystemExit(_image_ocr_worker_main())
