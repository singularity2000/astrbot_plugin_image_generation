import asyncio
import base64
import json
import re
import string
import random
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from urllib.parse import parse_qs, urlparse
import aiohttp

try:
    from curl_cffi.requests import AsyncSession as CurlSession

    CURL_CFFI_AVAILABLE = True
except ImportError:
    CURL_CFFI_AVAILABLE = False
from bs4 import BeautifulSoup
from astrbot import logger
from astrbot.core import AstrBotConfig
from .workflow import ImageWorkflow


# ---------------------------------------------------------------------------
# 基类：所有 API 提供商的公共接口
# ---------------------------------------------------------------------------


class BaseProvider(ABC):
    """API 提供商基类。每个子类实现一种 API 的调用逻辑。"""

    def __init__(
        self, node_config: dict, workflow: ImageWorkflow, global_config: AstrBotConfig
    ):
        self.node = node_config
        self.iwf = workflow
        self.conf = global_config
        self.key_index = 0
        self.key_lock = asyncio.Lock()

    @property
    def name(self) -> str:
        """供日志和错误信息使用的提供商名称。"""
        return self.__class__.__name__

    @property
    def enabled(self) -> bool:
        return self.node.get("enabled", True)

    @property
    def max_retry(self) -> int:
        return self.node.get("max_retry", 3)

    @property
    def api_timeout(self) -> int:
        return self.node.get("api_timeout", 300)

    @property
    def proxy(self) -> Optional[str]:
        """节点级代理。留空则不使用代理。"""
        p = self.node.get("proxy", "")
        return p if p else None

    async def _get_api_key(self) -> Optional[str]:
        keys = self.node.get("api_keys", [])
        if not keys:
            return None
        async with self.key_lock:
            key = keys[self.key_index % len(keys)]
            self.key_index = (self.key_index + 1) % len(keys)
            return key

    def get_api_keys(self) -> list:
        """返回此节点的 key 列表引用（供 main.py 管理命令使用）。"""
        return self.node.get("api_keys", [])

    @abstractmethod
    async def generate(
        self, image_bytes_list: List[bytes], prompt: str
    ) -> Union[bytes, str]:
        """
        执行生图调用。
        返回 bytes 表示成功（图片数据），返回 str 表示失败（错误信息）。
        """
        ...

    async def close(self):
        """可选的资源清理。子类按需覆写。"""
        pass


# ---------------------------------------------------------------------------
# 通用 API 提供商（硅基流动 / 智谱 AI 等标准 images/generations 端点）
# ---------------------------------------------------------------------------


class GenericImageProvider(BaseProvider):
    """
    通用的图像生成 API（兼容 siliconflow / bigmodel 等 /images/generations 端点）。
    根据 __template_key 自动决定响应解析字段。
    """

    # 不同提供商的响应数据字段映射
    DATA_FORM = {"siliconflow": "images", "bigmodel": "data"}

    async def generate(
        self, image_bytes_list: List[bytes], prompt: str
    ) -> Union[bytes, str]:
        api_url = self.node.get("api_url")
        if not api_url:
            return f"{self.name}: 配置错误 - 未设置 API URL"

        model_name = self.node.get("model")
        payload: Dict[str, Any] = {"model": model_name, "prompt": prompt}

        if image_bytes_list:
            for i, img in enumerate(image_bytes_list[:3]):
                key = "image" if i == 0 else f"image{i + 1}"
                payload[key] = (
                    f"data:image/png;base64,{base64.b64encode(img).decode('utf-8')}"
                )

        template_key = self.node.get("__template_key", "")
        data_form = self.DATA_FORM.get(template_key, "images")

        last_err = "未知错误"
        for i in range(self.max_retry):
            api_key = await self._get_api_key()
            if not api_key:
                return f"{self.name}: 配置错误 - 无 API Key"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

            try:
                async with self.iwf.session.post(
                    api_url,
                    json=payload,
                    headers=headers,
                    proxy=self.proxy,
                    timeout=self.api_timeout,
                ) as resp:
                    if resp.status != 200:
                        last_err = f"API请求失败 (HTTP {resp.status})"
                    else:
                        data = await resp.json()
                        try:
                            url = data[data_form][0]["url"]
                            if url.startswith("data:image/"):
                                return base64.b64decode(url.split(",", 1)[1])
                            return await self.iwf._download_image(url) or "下载失败"
                        except Exception:
                            last_err = f"解析响应失败: {str(data)[:200]}"
            except Exception as e:
                last_err = f"错误: {e}"

            # 付费API使用固定短间隔重试，无需拟人化抖动
            await asyncio.sleep(1)

        return f"{self.name} 生成失败: {last_err}"


# ---------------------------------------------------------------------------
# OpenAI Responses API
# ---------------------------------------------------------------------------


class OpenAIResponsesProvider(BaseProvider):
    """兼容 OpenAI /v1/responses 端点。"""

    async def generate(
        self, image_bytes_list: List[bytes], prompt: str
    ) -> Union[bytes, str]:
        api_url = self.node.get("api_url")
        model_name = self.node.get("model")

        content_items = [{"type": "input_text", "text": prompt}]
        for img in image_bytes_list:
            content_items.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{base64.b64encode(img).decode('utf-8')}",
                }
            )

        payload = {
            "model": model_name,
            "input": [{"role": "user", "content": content_items}],
            "tools": [{"type": "image_generation"}],
            "tool_choice": {"type": "image_generation"},
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
                    headers={"Authorization": f"Bearer {api_key}"},
                    proxy=self.proxy,
                ) as resp:
                    data = await resp.json()
                    if resp.status != 200:
                        last_err = f"API错误: {data}"
                    else:
                        b64 = None
                        if data.get("object") == "response":
                            for item in data.get("output", []):
                                if item.get("type") == "image_generation_call":
                                    b64 = item.get("result")
                                    break
                        if not b64:
                            for item in data.get("data", []):
                                if item.get("type") == "image_result":
                                    b64 = item.get("b64_json")
                                    break

                        if b64:
                            if "base64," in b64:
                                b64 = b64.split("base64,", 1)[1]
                            return base64.b64decode(b64)
                        last_err = "未找到图像数据"
            except Exception as e:
                last_err = str(e)

            # 付费API使用固定短间隔重试
            await asyncio.sleep(1)

        return f"OpenAI Responses 生成失败: {last_err}"


# ---------------------------------------------------------------------------
# Flow2API（Chat Completions 中转）
# ---------------------------------------------------------------------------


class Flow2APIProvider(BaseProvider):
    """兼容 Flow2API Chat Completions 端点。"""

    async def generate(
        self, image_bytes_list: List[bytes], prompt: str
    ) -> Union[bytes, str]:
        api_url = self.node.get("api_url")
        model_name = self.node.get("model")
        if not api_url:
            return f"{self.name}: 配置错误 - 无 API URL"

        content = [{"type": "text", "text": prompt}]
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
            "stream": True,
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
                    headers={"Authorization": f"Bearer {api_key}"},
                    proxy=self.proxy,
                ) as resp:
                    if resp.status != 200:
                        last_err = f"HTTP {resp.status}"
                    else:
                        response_text = await resp.text()
                        full_content = ""
                        for line in response_text.strip().split("\n"):
                            if line.startswith("data: ") and "[DONE]" not in line:
                                try:
                                    chunk = json.loads(line[6:])
                                    full_content += (
                                        chunk["choices"][0]
                                        .get("delta", {})
                                        .get("content", "")
                                    )
                                except:
                                    pass

                        match = re.search(r"https?://[^\s)]+", full_content)
                        if match:
                            url = match.group(0)
                            return await self.iwf._download_image(url) or "下载失败"
                        last_err = "未找到图片URL"
            except Exception as e:
                last_err = str(e)

            # 付费API使用固定短间隔重试
            await asyncio.sleep(1)

        return f"Flow2API 生成失败: {last_err}"


# ---------------------------------------------------------------------------
# Vertex AI 匿名（逆向 API，自带 Key 和专属逻辑）
# ---------------------------------------------------------------------------


class VertexAIAnonymousProvider(BaseProvider):
    """
    Vertex AI 匿名接口（逆向）。
    无需用户提供 API Key，自带 Recaptcha 验证与浏览器指纹伪装。
    """

    def __init__(
        self, node_config: dict, workflow: ImageWorkflow, global_config: AstrBotConfig
    ):
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
        for i in range(self.max_retry):
            token = await self._get_recaptcha_token(recaptcha_base)
            if not token:
                last_err = "获取 recaptcha token 失败"
                await self.close()
                continue

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
                                last_err = (
                                    f"API返回错误: {item['errors'][0].get('message')}"
                                )
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
                                    last_err = f"API返回错误: {item['errors'][0].get('message')}"
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

            logger.warning(
                f"Vertex AI API 调用失败，重试 ({i + 1}/{self.max_retry}): {last_err}"
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


# ---------------------------------------------------------------------------
# 提供商工厂
# ---------------------------------------------------------------------------

PROVIDER_MAP: Dict[str, type] = {
    "openai_responses": OpenAIResponsesProvider,
    "siliconflow": GenericImageProvider,
    "bigmodel": GenericImageProvider,
    "flow2api": Flow2APIProvider,
    "vertex_ai_anonymous": VertexAIAnonymousProvider,
}


def create_provider(
    node_config: dict, workflow: ImageWorkflow, global_config: AstrBotConfig
) -> Optional[BaseProvider]:
    """根据 template_key 创建对应的 Provider 实例。"""
    template_key = node_config.get("__template_key", "")
    cls = PROVIDER_MAP.get(template_key)
    if not cls:
        logger.warning(f"未知的 API 提供商模板: {template_key}，跳过")
        return None
    return cls(node_config, workflow, global_config)


# ---------------------------------------------------------------------------
# 管线调度器（Pipeline）
# ---------------------------------------------------------------------------


class ImageGenPipeline:
    """
    管线调度器：按顺序依次调用 enabled 的 Provider，第一个成功即返回，
    失败则自动回退到下一个。
    """

    def __init__(self, global_config: AstrBotConfig, workflow: ImageWorkflow):
        self.conf = global_config
        self.iwf = workflow
        self.providers: List[BaseProvider] = []
        self.api_call_lock = asyncio.Lock()
        self.last_api_call_time: Optional[datetime] = None

    def build(self, pipeline_config: list):
        """从配置列表构建 Provider 链。"""
        self.providers.clear()
        for node in pipeline_config:
            provider = create_provider(node, self.iwf, self.conf)
            if provider:
                self.providers.append(provider)
        enabled_names = [p.name for p in self.providers if p.enabled]
        logger.info(
            f"API 管线构建完成: {enabled_names} "
            f"({len(self.providers)} 个节点, {len(enabled_names)} 个已启用)"
        )

    async def check_rate_limit(self) -> Optional[str]:
        rate_limit = self.conf.get("rate_limit_seconds", 120)
        if rate_limit <= 0:
            return None
        async with self.api_call_lock:
            now = datetime.now()
            if self.last_api_call_time:
                elapsed = (now - self.last_api_call_time).total_seconds()
                if elapsed < rate_limit:
                    return f"⏳ 操作太频繁，请在 {int(rate_limit - elapsed)} 秒后再试。"
            self.last_api_call_time = now
        return None

    async def execute(
        self, image_bytes_list: List[bytes], prompt: str
    ) -> Union[bytes, str]:
        """
        依次调用管线中已启用的 Provider。
        返回 bytes 表示成功（图片数据），返回 str 表示全部失败（汇总错误信息）。
        """
        errors: List[str] = []
        for provider in self.providers:
            if not provider.enabled:
                continue
            logger.info(f"[Pipeline] 尝试: {provider.name}")
            result = await provider.generate(image_bytes_list, prompt)
            if isinstance(result, bytes):
                return result
            logger.warning(f"[Pipeline] {provider.name} 失败: {result}")
            errors.append(f"{provider.name}: {result}")

        if not errors:
            return "API 管线为空或无已启用的提供商，请在配置页面添加至少一个 API 节点。"
        return "所有 API 均失败:\n" + "\n".join(errors)

    def get_first_keyed_provider(self) -> Optional[BaseProvider]:
        """找到管线中第一个拥有 api_keys 配置的 Provider（供 Key 管理命令使用）。"""
        for p in self.providers:
            if p.enabled and "api_keys" in p.node:
                return p
        return None

    async def close(self):
        """关闭所有 Provider 的资源。"""
        for p in self.providers:
            await p.close()
