import asyncio
import base64
import json
import re
from typing import List, Union

from .base import BaseProvider


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
                                except Exception:
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
