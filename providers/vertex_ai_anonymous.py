import asyncio
import base64
import json
import random
import re
import string
from typing import List, Optional, Union
from urllib.parse import parse_qs, urlparse

import aiohttp
from bs4 import BeautifulSoup

from astrbot import logger

from .base import BaseProvider

try:
    from curl_cffi.requests import AsyncSession as CurlSession

    CURL_CFFI_AVAILABLE = True
except ImportError:
    CURL_CFFI_AVAILABLE = False


class VertexAIAnonymousProvider(BaseProvider):
    """
    Vertex AI 匿名接口（逆向）。
    无需用户提供 API Key，自带 Recaptcha 验证与浏览器指纹伪装。
    """

    def __init__(self, node_config: dict, workflow, global_config):
        super().__init__(node_config, workflow, global_config)
        self.impersonate_index = 0
        self.request_count = 0
        self._curl_session = None

    async def _get_curl_session(self):
        """获取或创建持久化的 curl_cffi 会话"""
        if not CURL_CFFI_AVAILABLE:
            return None
        if self._curl_session is None:
            imp_list = self.node.get(
                "impersonate_list", ["chrome131", "firefox135", "safari18_0"]
            )
            if not imp_list:
                imp_list = ["chrome131"]

            impersonate = imp_list[self.impersonate_index % len(imp_list)]
            self.impersonate_index += 1

            if self.node.get("verbose_logging", True):
                logger.info(
                    f"[VertexAI] 正在创建新会话 | 指纹: {impersonate} | 轮次: {self.impersonate_index}"
                )

            self._curl_session = CurlSession(impersonate=impersonate, proxy=self.proxy)
        return self._curl_session

    async def close(self):
        """关闭 curl_cffi 会话"""
        if self._curl_session:
            self._curl_session.close()
            self._curl_session = None

    async def generate(
        self, image_bytes_list: List[bytes], prompt: str
    ) -> Union[bytes, str]:
        # 会话老化机制：成功请求50次后主动轮换
        if self.request_count >= 50:
            if self.node.get("verbose_logging", True):
                logger.info("[VertexAI] 会话老化 (50次请求)，主动销毁并轮换指纹")
            await self.close()
            self.request_count = 0

        model_name = self.node.get("model", "gemini-3-pro-image-preview")

        parts = [{"text": prompt}]
        for img in image_bytes_list:
            parts.append(
                {
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": base64.b64encode(img).decode("utf-8"),
                    }
                }
            )

        # 清晰度处理
        image_size = self.node.get("image_size", "智能匹配")
        if image_size == "智能匹配":
            sz_match = re.search(r"\b([124]K)\b", prompt, re.IGNORECASE)
            image_size = sz_match.group(1).upper() if sz_match else "1K"

        context = {
            "model": model_name,
            "contents": [{"parts": parts, "role": "user"}],
            "generationConfig": {
                "temperature": 1,
                "topP": 0.95,
                "maxOutputTokens": 32768,
                "responseModalities": ["IMAGE"],
                "imageConfig": {
                    "imageOutputOptions": {"mimeType": "image/png"},
                    "personGeneration": "ALLOW_ALL",
                    "imageSize": image_size,
                },
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
            ],
            "region": "global",
        }

        system_prompt = self.node.get("system_prompt", "")
        if system_prompt:
            context["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        body = {
            "querySignature": "2/l8eCsMMY49imcDQ/lwwXyL8cYtTjxZBF2dNqy69LodY=",
            "operationName": "StreamGenerateContentAnonymous",
            "variables": context,
        }

        recaptcha_base = self.node.get("recaptcha_base_api", "https://www.google.com")
        vertex_base = self.node.get(
            "vertex_ai_base_api", "https://cloudconsole-pa.clients6.google.com"
        )
        api_timeout = self.api_timeout

        last_err = "未知错误"
        token = await self._get_recaptcha_token(recaptcha_base)
        if not token:
            return "Vertex AI 生成失败: 获取 recaptcha token 失败"

        # 当前 recaptcha token 的本地尝试次数：
        # 经验上在 Failed to verify action 场景下，第二次提交同 token 可能成功
        captcha_try_count = 0

        attempt = 0
        while attempt < self.max_retry:
            attempt += 1
            error_status_code = None

            body["variables"]["recaptchaToken"] = token
            api_url = (
                f"{vertex_base}/v3/entityServices/AiplatformEntityService"
                f"/schemas/AIPLATFORM_GRAPHQL:batchGraphql"
                f"?key=AIzaSyCI-zsRP85UVOi0DjtiCwWBwQ1djDy741g&prettyPrint=false"
            )

            # 使用 curl_cffi 以规避指纹风控
            if CURL_CFFI_AVAILABLE:
                try:
                    headers = {
                        "referer": "https://console.cloud.google.com/vertex-ai",
                        "Content-Type": "application/json",
                        "Origin": "https://console.cloud.google.com",
                    }
                    session = await self._get_curl_session()
                    resp = await session.post(
                        api_url, json=body, headers=headers, timeout=api_timeout
                    )

                    if resp.status_code != 200:
                        last_err = f"API请求失败 (HTTP {resp.status_code})"
                        if resp.status_code == 429:
                            last_err = "API返回错误: 频率限制 (Resource exhausted)"
                            logger.warning(
                                "Vertex AI 429 Limited. Retrying with backoff..."
                            )
                        else:
                            logger.warning(
                                f"Vertex AI API error detail (Status {resp.status_code}): {resp.text[:500]}"
                            )
                            await self.close()
                        continue

                    result = resp.json()
                    for elem in result:
                        for item in elem.get("results", []):
                            if item.get("errors"):
                                err_item = item["errors"][0]
                                err_msg = err_item.get("message", "")
                                error_status_code = (
                                    err_item.get("extensions", {})
                                    .get("status", {})
                                    .get("code")
                                )
                                last_err = f"API返回错误: {err_msg}"
                                if (
                                    "SAFETY" in last_err
                                    or "content" in last_err.lower()
                                ):
                                    return last_err
                                continue

                            for candidate in item.get("data", {}).get("candidates", []):
                                if candidate.get("finishReason") == "STOP":
                                    for part in candidate.get("content", {}).get(
                                        "parts", []
                                    ):
                                        if "inlineData" in part and part[
                                            "inlineData"
                                        ].get("data"):
                                            img_data = part["inlineData"]["data"]
                                            logger.info(
                                                f"Successfully extracted image data ({len(img_data)} chars)"
                                            )
                                            self.request_count += 1
                                            return base64.b64decode(img_data)
                                elif candidate.get("finishReason"):
                                    last_err = (
                                        f"生成中断: {candidate.get('finishReason')}"
                                    )
                except Exception as e:
                    last_err = f"curl_cffi 请求异常: {str(e)}"
                    await self.close()
            else:
                # 回退到 aiohttp 逻辑
                headers = {
                    "referer": "https://console.cloud.google.com/vertex-ai",
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                    "Origin": "https://console.cloud.google.com",
                    "Accept-Encoding": "identity",
                }
                client_timeout = aiohttp.ClientTimeout(
                    total=api_timeout, sock_read=api_timeout
                )
                try:
                    async with self.iwf.session.post(
                        api_url,
                        json=body,
                        headers=headers,
                        proxy=self.proxy,
                        timeout=client_timeout,
                    ) as resp:
                        if resp.status != 200:
                            last_err = f"API请求失败 (HTTP {resp.status})"
                            continue

                        chunks = []
                        async for chunk in resp.content.iter_any():
                            chunks.append(chunk)
                        content_bytes = b"".join(chunks)

                        try:
                            result = json.loads(content_bytes)
                        except Exception:
                            content_str = content_bytes.decode("utf-8", errors="ignore")
                            last_brace = content_str.rfind("}")
                            if last_brace != -1:
                                result = json.loads(content_str[: last_brace + 1] + "]")
                            else:
                                last_err = (
                                    f"无法解析响应 (收到 {len(content_bytes)} 字节)"
                                )
                                continue

                        for elem in result:
                            for item in elem.get("results", []):
                                if item.get("errors"):
                                    err_item = item["errors"][0]
                                    err_msg = err_item.get("message", "")
                                    error_status_code = (
                                        err_item.get("extensions", {})
                                        .get("status", {})
                                        .get("code")
                                    )
                                    last_err = f"API返回错误: {err_msg}"
                                    if (
                                        "SAFETY" in last_err
                                        or "content" in last_err.lower()
                                    ):
                                        return last_err
                                    continue
                                for candidate in item.get("data", {}).get(
                                    "candidates", []
                                ):
                                    if candidate.get("finishReason") == "STOP":
                                        for part in candidate.get("content", {}).get(
                                            "parts", []
                                        ):
                                            if "inlineData" in part and part[
                                                "inlineData"
                                            ].get("data"):
                                                return base64.b64decode(
                                                    part["inlineData"]["data"]
                                                )
                except Exception as e:
                    last_err = f"aiohttp 请求异常: {str(e)}"

            if (
                error_status_code == 3
                and "Failed to verify action" in last_err
                and captcha_try_count < 1
            ):
                captcha_try_count += 1
                logger.info(
                    "[VertexAI] 命中 Failed to verify action，复用同一 token 重试一次"
                )
                attempt -= 1
                continue

            if error_status_code == 3:
                token = await self._get_recaptcha_token(recaptcha_base)
                if not token:
                    last_err = "获取 recaptcha token 失败"
                    await self.close()
                    continue
                captcha_try_count = 0

            logger.warning(
                f"Vertex AI API 调用失败，重试 ({attempt}/{self.max_retry}): {last_err}"
            )
            backoff_time = 2 + random.uniform(0, 3)
            if "Resource exhausted" in last_err:
                backoff_time += 5
            await asyncio.sleep(backoff_time)

        return f"Vertex AI 生成失败: {last_err}"

    async def _get_recaptcha_token_curl(self, base_url: str) -> Optional[str]:
        """使用 curl_cffi 获取 Recaptcha Token，保持指纹一致性"""
        session = await self._get_curl_session()
        if not session:
            return None

        headers = {"Referer": "https://console.cloud.google.com/vertex-ai"}

        try:
            for _ in range(3):
                cb = "".join(random.choices(string.ascii_letters + string.digits, k=10))
                anchor_url = (
                    f"{base_url}/recaptcha/enterprise/anchor?ar=1"
                    f"&k=6LdCjtspAAAAAMcV4TGdWLJqRTEk1TfpdLqEnKdj"
                    f"&co=aHR0cHM6Ly9jb25zb2xlLmNsb3VkLmdvb2dsZS5jb206NDQz"
                    f"&hl=zh-CN&v=jdMmXeCQEkPbnFDy9T04NbgJ"
                    f"&size=invisible&anchor-ms=20000&execute-ms=15000&cb={cb}"
                )

                resp = await session.get(anchor_url, headers=headers)
                if resp.status_code != 200:
                    continue

                soup = BeautifulSoup(resp.text, "html.parser")
                token_input = soup.find("input", {"id": "recaptcha-token"})
                if not token_input:
                    continue
                base_token = token_input.get("value")

                reload_url = f"{base_url}/recaptcha/enterprise/reload?k=6LdCjtspAAAAAMcV4TGdWLJqRTEk1TfpdLqEnKdj"
                parsed = urlparse(anchor_url)
                query_params = parse_qs(parsed.query)
                post_payload = {
                    "v": query_params["v"][0],
                    "reason": "q",
                    "k": query_params["k"][0],
                    "c": base_token,
                    "co": query_params["co"][0],
                    "hl": query_params["hl"][0],
                    "size": "invisible",
                    "vh": "6581054572",
                    "chr": "",
                    "bg": "",
                }

                resp = await session.post(
                    reload_url,
                    data=post_payload,
                    headers={
                        "Content-Type": "application/x-www-form-urlencoded",
                        "Origin": "https://console.cloud.google.com",
                        **headers,
                    },
                )
                if resp.status_code != 200:
                    continue

                token_match = re.search(r'rresp","(.*?)"', resp.text)
                if token_match:
                    return token_match.group(1)
        except Exception as e:
            logger.error(f"curl_cffi 获取 recaptcha token 异常: {str(e)}")
            await self.close()
        return None

    async def _get_recaptcha_token(self, base_url: str) -> Optional[str]:
        # 优先使用 curl_cffi 路径
        if CURL_CFFI_AVAILABLE:
            return await self._get_recaptcha_token_curl(base_url)

        try:
            for _ in range(3):
                cb = "".join(random.choices(string.ascii_letters + string.digits, k=10))
                anchor_url = (
                    f"{base_url}/recaptcha/enterprise/anchor?ar=1"
                    f"&k=6LdCjtspAAAAAMcV4TGdWLJqRTEk1TfpdLqEnKdj"
                    f"&co=aHR0cHM6Ly9jb25zb2xlLmNsb3VkLmdvb2dsZS5jb206NDQz"
                    f"&hl=zh-CN&v=jdMmXeCQEkPbnFDy9T04NbgJ"
                    f"&size=invisible&anchor-ms=20000&execute-ms=15000&cb={cb}"
                )

                async with self.iwf.session.get(anchor_url, proxy=self.proxy) as resp:
                    if resp.status != 200:
                        continue
                    html = await resp.text()
                    soup = BeautifulSoup(html, "html.parser")
                    token_input = soup.find("input", {"id": "recaptcha-token"})
                    if not token_input:
                        continue
                    base_token = token_input.get("value")

                reload_url = f"{base_url}/recaptcha/enterprise/reload?k=6LdCjtspAAAAAMcV4TGdWLJqRTEk1TfpdLqEnKdj"
                parsed = urlparse(anchor_url)
                query_params = parse_qs(parsed.query)
                post_payload = {
                    "v": query_params["v"][0],
                    "reason": "q",
                    "k": query_params["k"][0],
                    "c": base_token,
                    "co": query_params["co"][0],
                    "hl": query_params["hl"][0],
                    "size": "invisible",
                    "vh": "6581054572",
                    "chr": "",
                    "bg": "",
                }

                async with self.iwf.session.post(
                    reload_url, data=post_payload, proxy=self.proxy
                ) as resp:
                    if resp.status != 200:
                        continue
                    text = await resp.text()
                    token_match = re.search(r'rresp","(.*?)"', text)
                    if token_match:
                        return token_match.group(1)
        except Exception as e:
            logger.error(f"获取 recaptcha token 过程中发生异常: {str(e)}")
        return None
