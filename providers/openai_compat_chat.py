import asyncio
import base64
import binascii
import json
import re
from typing import Any, List, Union

from astrbot import logger

from .base import BaseProvider


class OpenAICompatChatProvider(BaseProvider):
    """兼容 OpenAI Chat Completions 及其第三方兼容端点。"""

    IMAGE_URL_RE = re.compile(r"https?://[^\s)'\"]+", re.IGNORECASE)
    MARKDOWN_IMAGE_RE = re.compile(r"!\[.*?\]\((.*?)\)")
    DATA_URI_RE = re.compile(
        r"data:(image|video)/([a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=\s_-]+)",
        re.IGNORECASE,
    )
    VIDEO_EXTENSIONS = (
        ".mp4",
        ".mov",
        ".webm",
        ".mkv",
        ".avi",
        ".m4v",
        ".gifv",
    )

    def _is_video_url(self, url: str) -> bool:
        lowered = url.lower().split("?", 1)[0]
        return lowered.endswith(self.VIDEO_EXTENSIONS) or "/video" in lowered

    def _decode_base64_payload(self, payload: str) -> bytes | None:
        cleaned = re.sub(r"\s+", "", payload)
        try:
            return base64.b64decode(cleaned, validate=True)
        except (binascii.Error, ValueError):
            return None

    def _extract_from_text(
        self, text: str
    ) -> tuple[bytes | None, str | dict[str, str] | None]:
        if not text:
            return None, None

        markdown_match = self.MARKDOWN_IMAGE_RE.search(text)
        if markdown_match:
            candidate = markdown_match.group(1).strip()
            if candidate.startswith("data:"):
                return self._extract_from_data_uri(candidate)
            if self._is_video_url(candidate):
                return None, {"type": "video", "url": candidate}
            return None, candidate

        data_match = self.DATA_URI_RE.search(text)
        if data_match:
            media_type = data_match.group(1).lower()
            raw = self._decode_base64_payload(data_match.group(3))
            if raw:
                if media_type == "video":
                    return (
                        None,
                        "接口返回了 base64 视频数据，当前插件暂不支持直接发送内嵌视频数据",
                    )
                return raw, None

        for url in self.IMAGE_URL_RE.findall(text):
            cleaned = url.rstrip(").,;\"'")
            if self._is_video_url(cleaned):
                return None, {"type": "video", "url": cleaned}
            return None, cleaned

        return None, None

    def _extract_from_data_uri(
        self, uri: str
    ) -> tuple[bytes | None, str | dict[str, str] | None]:
        match = self.DATA_URI_RE.search(uri)
        if not match:
            return None, None
        media_type = match.group(1).lower()
        raw = self._decode_base64_payload(match.group(3))
        if not raw:
            return None, "接口返回了无效的 base64 数据"
        if media_type == "video":
            return (
                None,
                "接口返回了 base64 视频数据，当前插件暂不支持直接发送内嵌视频数据",
            )
        return raw, None

    def _extract_from_message(
        self, message: dict[str, Any]
    ) -> tuple[bytes | None, str | dict[str, str] | None]:
        content = message.get("content", "")

        if isinstance(content, str):
            return self._extract_from_text(content)

        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type == "text":
                    text_parts.append(str(part.get("text", "")))
                elif part_type == "image_url":
                    image_url = part.get("image_url", {})
                    if isinstance(image_url, dict):
                        url = str(image_url.get("url", "")).strip()
                        if url.startswith("data:"):
                            raw, err = self._extract_from_data_uri(url)
                            if raw or err:
                                return raw, err
                        elif url:
                            return (
                                (None, {"type": "video", "url": url})
                                if self._is_video_url(url)
                                else (None, url)
                            )
                elif part_type == "output_text":
                    text_parts.append(str(part.get("text", "")))

            return self._extract_from_text("\n".join(p for p in text_parts if p))

        return None, None

    def _extract_from_json_response(
        self, payload: dict[str, Any]
    ) -> tuple[bytes | None, str | dict[str, str] | None]:
        choices = payload.get("choices") or []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message") or {}
            if isinstance(message, dict):
                raw, result = self._extract_from_message(message)
                if raw or result:
                    return raw, result

            delta = choice.get("delta") or {}
            if isinstance(delta, dict):
                raw, result = self._extract_from_message(delta)
                if raw or result:
                    return raw, result

        data_items = payload.get("data") or []
        if isinstance(data_items, list):
            for item in data_items:
                if not isinstance(item, dict):
                    continue
                if item.get("b64_json"):
                    raw = self._decode_base64_payload(str(item["b64_json"]))
                    if raw:
                        return raw, None
                if item.get("url"):
                    url = str(item["url"]).strip()
                    if self._is_video_url(url):
                        return None, {"type": "video", "url": url}
                    return None, url

        for key in ("output_text", "text", "response"):
            value = payload.get(key)
            if isinstance(value, str):
                raw, result = self._extract_from_text(value)
                if raw or result:
                    return raw, result

        return None, None

    def _extract_from_sse_response(
        self, response_text: str
    ) -> tuple[bytes | None, str | dict[str, str] | None]:
        text_fragments: list[str] = []
        for line in response_text.strip().splitlines():
            if not line.startswith("data: "):
                continue
            line_data = line[6:].strip()
            if line_data == "[DONE]":
                continue
            try:
                chunk = json.loads(line_data)
            except Exception:
                continue

            raw, result = self._extract_from_json_response(chunk)
            if raw or result:
                return raw, result

            for choice in chunk.get("choices", []):
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta", {})
                if isinstance(delta, dict):
                    content = delta.get("content", "")
                    if isinstance(content, str):
                        text_fragments.append(content)
                    elif isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text_fragments.append(str(part.get("text", "")))

        return self._extract_from_text("".join(text_fragments))

    async def generate(
        self, image_bytes_list: List[bytes], prompt: str
    ) -> Union[bytes, str, dict[str, str]]:
        api_url = self.node.get("api_url")
        model_name = self.node.get("model")
        if not api_url:
            return f"{self.name}: 配置错误 - 无 API URL"

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in image_bytes_list:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64.b64encode(img).decode('utf-8')}",
                        "detail": "high",
                    },
                }
            )

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": content}],
            "stream": self.node.get("stream", True),
        }

        last_err = "未知错误"
        for i in range(self.max_retry):
            api_key = await self._get_api_key()
            if not api_key:
                return f"{self.name}: 配置错误 - 无 API Key"

            try:
                async with self.iwf.session.post(
                    api_url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    proxy=self.proxy,
                    timeout=self.api_timeout,
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        last_err = f"HTTP {resp.status}: {body[:200]}"
                    else:
                        response_text = await resp.text()
                        raw_data: bytes | None = None
                        result: str | dict[str, str] | None = None

                        if "data: " in response_text:
                            raw_data, result = self._extract_from_sse_response(
                                response_text
                            )

                        if not raw_data and not result:
                            try:
                                response_json = json.loads(response_text)
                                raw_data, result = self._extract_from_json_response(
                                    response_json
                                )
                            except json.JSONDecodeError:
                                raw_data, result = self._extract_from_text(
                                    response_text
                                )

                        if raw_data:
                            return raw_data

                        if result:
                            if isinstance(result, dict):
                                return result
                            if result.startswith("http://") or result.startswith(
                                "https://"
                            ):
                                downloaded = await self.iwf._download_image(result)
                                if downloaded:
                                    return downloaded
                                last_err = f"下载返回资源失败: {result}"
                            else:
                                last_err = result
                        else:
                            last_err = "未找到可解析的图片或视频结果"

                        logger.debug(
                            "OpenAICompatChatProvider 未解析到结果，响应预览: %s",
                            response_text[:500],
                        )
            except Exception as e:
                last_err = str(e)

            # 付费API使用固定短间隔重试
            await asyncio.sleep(1)

        return f"OpenAI 兼容 Chat Completions 生成失败: {last_err}"


class Flow2APIProvider(OpenAICompatChatProvider):
    """旧名称兼容别名。"""
