import asyncio
import base64
from typing import Any

from astrbot import logger

from .base import BaseProvider


class VertexAIProvider(BaseProvider):
    """Vertex AI 官方 API（/v1/publishers/google/models/{model}:generateContent）。"""

    async def generate(self, image_bytes_list: list[bytes], prompt: str) -> bytes | str:
        api_url = self.node.get("api_url")
        model_name = self.node.get("model", "gemini-3.1-flash-image-preview")
        if not api_url:
            return f"{self.name}: 配置错误 - 未设置 API URL"

        last_err = "未知错误"
        for attempt in range(self.max_retry):
            api_key = await self._get_api_key()
            if not api_key:
                return f"{self.name}: 配置错误 - 无 API Key"

            url = f"{api_url}/{model_name}:generateContent?key={api_key}"
            context = self._build_context(image_bytes_list, prompt)

            try:
                async with self.iwf.session.post(
                    url,
                    json=context,
                    headers={"Content-Type": "application/json"},
                    proxy=self.proxy,
                    timeout=self.api_timeout,
                ) as resp:
                    data = await resp.json(content_type=None)

                    if resp.status != 200:
                        err_msg = (
                            data.get("error", {}).get("message", "未知原因")
                            if isinstance(data, dict)
                            else str(data)[:200]
                        )
                        last_err = f"API请求失败 (HTTP {resp.status}): {err_msg}"

                        if resp.status == 429:
                            logger.warning(
                                f"[VertexAI] 频率限制 (429)，退避重试 "
                                f"({attempt + 1}/{self.max_retry})"
                            )
                            await asyncio.sleep(3)
                            continue

                        logger.warning(f"[VertexAI] {last_err}")
                        await asyncio.sleep(1)
                        continue

                    parse_result = self._extract_image_or_error(data)
                    if isinstance(parse_result, bytes):
                        return parse_result
                    if isinstance(parse_result, str):
                        return parse_result
                    last_err = "响应中未包含图片数据"

            except asyncio.TimeoutError:
                last_err = f"请求超时 ({self.api_timeout}s)"
            except Exception as e:
                last_err = f"请求异常: {e}"

            logger.warning(
                f"[VertexAI] 调用失败 ({attempt + 1}/{self.max_retry}): {last_err}"
            )
            await asyncio.sleep(1)

        return f"Vertex AI 生成失败: {last_err}"

    def _build_context(
        self, image_bytes_list: list[bytes], prompt: str
    ) -> dict[str, object]:
        """构建 Vertex AI 官方接口请求体。"""
        parts: list[dict[str, object]] = [{"text": prompt}]
        for img in image_bytes_list:
            parts.append(
                {
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": base64.b64encode(img).decode("utf-8"),
                    }
                }
            )

        return {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"responseModalities": ["IMAGE"]},
        }

    def _extract_image_or_error(self, data: dict[str, Any]) -> bytes | str | None:
        """返回 bytes=成功；str=终止错误；None=未取到图片。"""
        prompt_feedback = data.get("promptFeedback", {})
        if isinstance(prompt_feedback, dict) and prompt_feedback.get("blockReason"):
            reason = prompt_feedback.get("blockReason", "未知")
            return f"请求被内容安全系统拦截: {reason}"

        for candidate in data.get("candidates", []):
            finish_reason = candidate.get("finishReason", "")

            if finish_reason and finish_reason != "STOP":
                if "SAFETY" in finish_reason or "BLOCK" in finish_reason:
                    return f"生成被安全策略拦截: {finish_reason}"
                return f"生成失败: {finish_reason}"

            for part in candidate.get("content", {}).get("parts", []):
                inline_data = part.get("inlineData")
                if inline_data and inline_data.get("data"):
                    img_b64 = inline_data["data"]
                    logger.info(f"[VertexAI] 成功提取图片数据 ({len(img_b64)} chars)")
                    return base64.b64decode(img_b64)

        return None
