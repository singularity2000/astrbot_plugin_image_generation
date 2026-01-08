import asyncio
import base64
import json
import re
import string
import random
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

class ImageGenAPI:
    def __init__(self, config: AstrBotConfig, workflow: ImageWorkflow):
        self.conf = config
        self.iwf = workflow
        self.key_index = 0
        self.key_lock = asyncio.Lock()
        self.api_call_lock = asyncio.Lock()
        self.last_api_call_time: Optional[datetime] = None
        
        self.form = {"siliconflow": "images", "bigmodel": "data"}
        self.data_form = self.form.get(self.conf.get("api_from"), "images")
        self._curl_session = None

    async def _get_curl_session(self):
        """获取或创建持久化的 curl_cffi 会话"""
        if not CURL_CFFI_AVAILABLE:
            return None
        if self._curl_session is None:
            self._curl_session = CurlSession(impersonate="chrome131", proxy=self.iwf.proxy)
        return self._curl_session

    async def close(self):
        """关闭会话"""
        if self._curl_session:
            self._curl_session.close()
            self._curl_session = None

    async def _get_api_key(self) -> Optional[str]:
        keys = self.conf.get("api_keys", [])
        if not keys: return None
        async with self.key_lock:
            key = keys[self.key_index]
            self.key_index = (self.key_index + 1) % len(keys)
            return key

    def _sanitize_for_log(self, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: self._sanitize_for_log(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_for_log(item) for item in data]
        elif isinstance(data, str) and len(data) > 200:
            return data[:200] + "..."
        return data

    async def check_rate_limit(self) -> Optional[str]:
        rate_limit = self.conf.get("rate_limit_seconds", 120)
        if rate_limit <= 0: return None
        async with self.api_call_lock:
            now = datetime.now()
            if self.last_api_call_time:
                elapsed = (now - self.last_api_call_time).total_seconds()
                if elapsed < rate_limit:
                    return f"⏳ 操作太频繁，请在 {int(rate_limit - elapsed)} 秒后再试。"
            self.last_api_call_time = now
        return None

    async def call_api(self, image_bytes_list: List[bytes], prompt: str) -> Union[bytes, str]:
        api_from = self.conf.get("api_from")
        if api_from == "OpenAI-responses":
            return await self._call_openai_responses_api(image_bytes_list, prompt)
        if api_from == "Flow2API":
            return await self._call_flow2api_api(image_bytes_list, prompt)
        if api_from == "Vertex_AI_Anonymous":
            return await self._call_vertex_ai_anonymous_api(image_bytes_list, prompt)

        # 通用 API 逻辑
        api_url = self.conf.get("api_url")
        api_key = await self._get_api_key()
        if not api_url or not api_key: return "API 配置错误或无 Key"
        
        model_name = self.conf.get("model")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        payload = {"model": model_name, "prompt": prompt}

        if image_bytes_list:
            for i, img in enumerate(image_bytes_list[:3]):
                key = "image" if i == 0 else f"image{i+1}"
                payload[key] = f"data:image/png;base64,{base64.b64encode(img).decode('utf-8')}"

        try:
            timeout = self.conf.get("api_timeout", 180)
            async with self.iwf.session.post(api_url, json=payload, headers=headers, proxy=self.iwf.proxy, timeout=timeout) as resp:
                if resp.status != 200: return f"API请求失败 (HTTP {resp.status})"
                data = await resp.json()
                self.data_form = self.form.get(api_from, "images")
                
                try:
                    url = data[self.data_form][0]["url"]
                    if url.startswith("data:image/"):
                        return base64.b64decode(url.split(",", 1)[1])
                    return await self.iwf._download_image(url) or "下载失败"
                except Exception:
                    return f"解析响应失败: {str(data)[:200]}"
        except Exception as e:
            return f"错误: {e}"

    async def _call_flow2api_api(self, image_bytes_list: List[bytes], prompt: str) -> Union[bytes, str]:
        api_url = self.conf.get("api_url")
        api_key = await self._get_api_key()
        model_name = self.conf.get("model")
        if not api_url or not api_key: return "配置错误"

        content = [{"type": "text", "text": prompt}]
        for img in image_bytes_list:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(img).decode('utf-8')}", "detail": "high"}})
        
        payload = {"model": model_name, "messages": [{"role": "user", "content": content}], "stream": True}
        
        try:
            async with self.iwf.session.post(api_url, json=payload, headers={"Authorization": f"Bearer {api_key}"}, proxy=self.iwf.proxy) as resp:
                if resp.status != 200: return f"HTTP {resp.status}"
                response_text = await resp.text()
                full_content = ""
                for line in response_text.strip().split('\n'):
                    if line.startswith("data: ") and "[DONE]" not in line:
                        try:
                            chunk = json.loads(line[6:])
                            full_content += chunk["choices"][0].get("delta", {}).get("content", "")
                        except: pass
                
                match = re.search(r'https?://[^\s)]+', full_content)
                if match:
                    url = match.group(0)
                    return await self.iwf._download_image(url) or "下载失败"
                return "未找到图片URL"
        except Exception as e: return str(e)

    async def _call_openai_responses_api(self, image_bytes_list: List[bytes], prompt: str) -> Union[bytes, str]:
        api_url = self.conf.get("api_url")
        api_key = await self._get_api_key()
        model_name = self.conf.get("model")
        
        content_items = [{"type": "input_text", "text": prompt}]
        for img in image_bytes_list:
            content_items.append({"type": "input_image", "image_url": f"data:image/png;base64,{base64.b64encode(img).decode('utf-8')}"})
        
        payload = {
            "model": model_name,
            "input": [{"role": "user", "content": content_items}],
            "tools": [{"type": "image_generation"}],
            "tool_choice": {"type": "image_generation"},
        }

        try:
            async with self.iwf.session.post(api_url, json=payload, headers={"Authorization": f"Bearer {api_key}"}, proxy=self.iwf.proxy) as resp:
                data = await resp.json()
                if resp.status != 200: return f"API错误: {data}"
                
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
                    if "base64," in b64: b64 = b64.split("base64,", 1)[1]
                    return base64.b64decode(b64)
                return "未找到图像数据"
        except Exception as e: return str(e)

    async def _call_vertex_ai_anonymous_api(self, image_bytes_list: List[bytes], prompt: str) -> Union[bytes, str]:
        model_name = self.conf.get("model", "gemini-3-pro-image-preview")
            
        parts = [{"text": prompt}]
        for img in image_bytes_list:
            parts.append({"inlineData": {"mimeType": "image/png", "data": base64.b64encode(img).decode('utf-8')}})

        # 清晰度处理
        image_size = self.conf.get("vertex_ai_image_size", "智能匹配")
        if image_size == "智能匹配":
            # 按照从左到右的顺序匹配第一个出现的 1K, 2K, 或 4K
            match = re.search(r'\b([124]K)\b', prompt, re.IGNORECASE)
            image_size = match.group(1).upper() if match else "1K"

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
                    "imageSize": image_size
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
        
        system_prompt = self.conf.get("vertex_ai_system_prompt", "")
        if system_prompt:
            context["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        body = {
            "querySignature": "2/l8eCsMMY49imcDQ/lwwXyL8cYtTjxZBF2dNqy69LodY=",
            "operationName": "StreamGenerateContentAnonymous",
            "variables": context,
        }

        max_retry = self.conf.get("vertex_ai_max_retry", 10)
        recaptcha_base = self.conf.get("recaptcha_base_api", "https://www.google.com")
        vertex_base = self.conf.get("vertex_ai_base_api", "https://cloudconsole-pa.clients6.google.com")
        api_timeout = self.conf.get("api_timeout", 180)
        
        last_err = "未知错误"
        for i in range(max_retry):
            token = await self._get_recaptcha_token(recaptcha_base)
            if not token:
                last_err = "获取 recaptcha token 失败"
                continue
            
            body["variables"]["recaptchaToken"] = token
            url = f"{vertex_base}/v3/entityServices/AiplatformEntityService/schemas/AIPLATFORM_GRAPHQL:batchGraphql?key=AIzaSyCI-zsRP85UVOi0DjtiCwWBwQ1djDy741g&prettyPrint=false"
            
            # 使用 curl_cffi 以规避指纹风控
            if CURL_CFFI_AVAILABLE:
                try:
                    # 仅保留最基础的业务 Headers，指纹相关 Headers 由 impersonate 自动接管
                    headers = {
                        "referer": "https://console.cloud.google.com/", 
                        "Content-Type": "application/json",
                    }
                    session = await self._get_curl_session()
                    resp = await session.post(url, json=body, headers=headers, timeout=api_timeout)
                    
                    if resp.status_code != 200:
                        last_err = f"API请求失败 (HTTP {resp.status_code})"
                        # 识别 429
                        if resp.status_code == 429:
                            last_err = "API返回错误: 频率限制 (Resource exhausted)"
                        logger.warning(f"Vertex AI API error detail (Status {resp.status_code}): {resp.text[:500]}")
                        continue
                    
                    result = resp.json()
                    # 核心解析逻辑
                    for elem in result:
                        for item in elem.get("results", []):
                            if item.get("errors"):
                                last_err = f"API返回错误: {item['errors'][0].get('message')}"
                                if "SAFETY" in last_err or "content" in last_err.lower(): return last_err
                                continue
                            
                            for candidate in item.get("data", {}).get("candidates", []):
                                if candidate.get("finishReason") == "STOP":
                                    for part in candidate.get("content", {}).get("parts", []):
                                        if "inlineData" in part and part["inlineData"].get("data"):
                                            img_data = part["inlineData"]["data"]
                                            logger.info(f"Successfully extracted image data ({len(img_data)} chars)")
                                            return base64.b64decode(img_data)
                                elif candidate.get("finishReason"):
                                    last_err = f"生成中断: {candidate.get('finishReason')}"
                except Exception as e:
                    last_err = f"curl_cffi 请求异常: {str(e)}"
            else:
                # 回退到 aiohttp 逻辑 (保持基本的指纹优化)
                headers = {
                    "referer": "https://console.cloud.google.com/", 
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                    "Origin": "https://console.cloud.google.com",
                    "Accept-Encoding": "identity"
                }
                client_timeout = aiohttp.ClientTimeout(total=api_timeout, sock_read=api_timeout)
                try:
                    async with self.iwf.session.post(url, json=body, headers=headers, proxy=self.iwf.proxy, timeout=client_timeout) as resp:
                        if resp.status != 200:
                            last_err = f"API请求失败 (HTTP {resp.status})"
                            continue
                        
                        chunks = []
                        async for chunk in resp.content.iter_any():
                            chunks.append(chunk)
                        content_bytes = b"".join(chunks)
                        
                        try:
                            result = json.loads(content_bytes)
                        except:
                            # 尝试救灾解析截断的 JSON
                            content_str = content_bytes.decode('utf-8', errors='ignore')
                            last_brace = content_str.rfind('}')
                            if last_brace != -1:
                                result = json.loads(content_str[:last_brace+1] + ']')
                            else:
                                last_err = f"无法解析响应 (收到 {len(content_bytes)} 字节)"
                                continue

                        for elem in result:
                            for item in elem.get("results", []):
                                if item.get("errors"):
                                    last_err = f"API返回错误: {item['errors'][0].get('message')}"
                                    if "SAFETY" in last_err or "content" in last_err.lower(): return last_err
                                    continue
                                for candidate in item.get("data", {}).get("candidates", []):
                                    if candidate.get("finishReason") == "STOP":
                                        for part in candidate.get("content", {}).get("parts", []):
                                            if "inlineData" in part and part["inlineData"].get("data"):
                                                return base64.b64decode(part["inlineData"]["data"])
                except Exception as e:
                    last_err = f"aiohttp 请求异常: {str(e)}"
            
            logger.warning(f"Vertex AI API 调用失败，重试 ({i+1}/{max_retry}): {last_err}")
            # 引入随机退避重试，模拟真人行为
            backoff_time = 2 + random.uniform(1, 4)
            if "Resource exhausted" in last_err:
                backoff_time += 5 # 遇到 429 额外多等一会儿
            await asyncio.sleep(backoff_time)
            
        return f"Vertex AI 生成失败: {last_err}"

    async def _get_recaptcha_token_curl(self, base_url: str) -> Optional[str]:
        """使用 curl_cffi 获取 Recaptcha Token，保持指纹一致性"""
        session = await self._get_curl_session()
        if not session: return None
        
        try:
            for _ in range(3):
                cb = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
                anchor_url = f"{base_url}/recaptcha/enterprise/anchor?ar=1&k=6LdCjtspAAAAAMcV4TGdWLJqRTEk1TfpdLqEnKdj&co=aHR0cHM6Ly9jb25zb2xlLmNsb3VkLmdvb2dsZS5jb206NDQz&hl=zh-CN&v=jdMmXeCQEkPbnFDy9T04NbgJ&size=invisible&anchor-ms=20000&execute-ms=15000&cb={cb}"
                
                # 1. GET Anchor
                resp = await session.get(anchor_url)
                if resp.status_code != 200: continue
                
                soup = BeautifulSoup(resp.text, "html.parser")
                token_input = soup.find("input", {"id": "recaptcha-token"})
                if not token_input: continue
                base_token = token_input.get("value")
                
                # 2. POST Reload
                reload_url = f"{base_url}/recaptcha/enterprise/reload?k=6LdCjtspAAAAAMcV4TGdWLJqRTEk1TfpdLqEnKdj"
                parsed = urlparse(anchor_url)
                query_params = parse_qs(parsed.query)
                payload = {
                    "v": query_params["v"][0], "reason": "q", "k": query_params["k"][0], "c": base_token,
                    "co": query_params["co"][0], "hl": query_params["hl"][0], "size": "invisible",
                    "vh": "6581054572", "chr": "", "bg": "",
                }
                
                resp = await session.post(reload_url, data=payload, headers={"Content-Type": "application/x-www-form-urlencoded"})
                if resp.status_code != 200: continue
                
                match = re.search(r'rresp","(.*?)"', resp.text)
                if match: return match.group(1)
        except Exception as e:
            logger.error(f"curl_cffi 获取 recaptcha token 异常: {str(e)}")
        return None

    async def _get_recaptcha_token(self, base_url: str) -> Optional[str]:
        # 优先使用 curl_cffi 路径
        if CURL_CFFI_AVAILABLE:
            return await self._get_recaptcha_token_curl(base_url)
            
        try:
            for _ in range(3):
                cb = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
                anchor_url = f"{base_url}/recaptcha/enterprise/anchor?ar=1&k=6LdCjtspAAAAAMcV4TGdWLJqRTEk1TfpdLqEnKdj&co=aHR0cHM6Ly9jb25zb2xlLmNsb3VkLmdvb2dsZS5jb206NDQz&hl=zh-CN&v=jdMmXeCQEkPbnFDy9T04NbgJ&size=invisible&anchor-ms=20000&execute-ms=15000&cb={cb}"
                
                async with self.iwf.session.get(anchor_url, proxy=self.iwf.proxy) as resp:
                    if resp.status != 200: continue
                    html = await resp.text()
                    soup = BeautifulSoup(html, "html.parser")
                    token_input = soup.find("input", {"id": "recaptcha-token"})
                    if not token_input: continue
                    base_token = token_input.get("value")
                
                reload_url = f"{base_url}/recaptcha/enterprise/reload?k=6LdCjtspAAAAAMcV4TGdWLJqRTEk1TfpdLqEnKdj"
                parsed = urlparse(anchor_url)
                query_params = parse_qs(parsed.query)
                payload = {
                    "v": query_params["v"][0], "reason": "q", "k": query_params["k"][0], "c": base_token,
                    "co": query_params["co"][0], "hl": query_params["hl"][0], "size": "invisible",
                    "vh": "6581054572", "chr": "", "bg": "",
                }
                
                async with self.iwf.session.post(reload_url, data=payload, proxy=self.iwf.proxy) as resp:
                    if resp.status != 200: continue
                    text = await resp.text()
                    match = re.search(r'rresp","(.*?)"', text)
                    if match: return match.group(1)
        except Exception as e:
            logger.error(f"获取 recaptcha token 过程中发生异常: {str(e)}")
        return None
