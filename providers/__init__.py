from typing import Dict, Optional

from astrbot import logger
from astrbot.core import AstrBotConfig

from ..workflow import ImageWorkflow
from .base import BaseProvider
from .flow2api import Flow2APIProvider
from .gemini import GeminiProvider
from .generic import GenericImageProvider
from .openai_responses import OpenAIResponsesProvider
from .vertex_ai_anonymous import VertexAIAnonymousProvider


PROVIDER_MAP: Dict[str, type] = {
    "gemini": GeminiProvider,
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
