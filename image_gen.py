from astrbot.api import FunctionTool
from astrbot.api.event import AstrMessageEvent
from dataclasses import dataclass, field

@dataclass
class ImageGenTool(FunctionTool):
    name: str = "image_generation"
    description: str = ""
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The detailed prompt for image generation. Expand the user's request into a professional prompt including style, lighting, and details.",
                }
            },
            "required": ["prompt"],
        }
    )

    def __init__(self, plugin_instance):
        self.plugin = plugin_instance
        self.description = plugin_instance.conf.get("llm_tool_description", "这是一个高级图片生成工具。主要功能为文生图、图生图。理解用户意图，仅当用户需要你画图，或修改图片内容时，才调用此工具，并智能决定调用文生图还是图生图。你可以根据用户的描述和意图对提示词进行扩充，使其更加详细（例如扩充为包含风格、光影、细节的专业提示词）。")

    async def run(self, event: AstrMessageEvent, prompt: str):
        # 检测上下文中是否有图片
        img_bytes_list = await self.plugin.iwf.get_images(event)
        
        # 智能决策：有图则图生图，无图则文生图
        is_i2i = len(img_bytes_list) > 0
        
        # 调用插件核心逻辑
        async for result in self.plugin.handle_image_gen_logic(event, prompt, is_i2i=is_i2i):
            await event.send(result)
        
        return "Image generation request processed."