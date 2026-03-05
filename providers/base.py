import asyncio
from abc import ABC, abstractmethod
from typing import List, Optional, Union

from astrbot.core import AstrBotConfig

from ..workflow import ImageWorkflow


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
