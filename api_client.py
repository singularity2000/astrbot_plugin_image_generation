import asyncio
import base64
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import aiohttp
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