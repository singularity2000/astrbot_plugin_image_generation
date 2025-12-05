import asyncio
import base64
import functools
import io
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import aiohttp
from PIL import Image as PILImage

from astrbot import logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.core import AstrBotConfig
from astrbot.core.message.components import At, Image, Reply, Plain
from astrbot.core.platform.astr_message_event import AstrMessageEvent


@register(
    "astrbot_plugin_image_generation",
    "MoonShadow1976",
    "通过第三方api进行图像生成的插件，支持图生图和文生图，可自定义预设指令。",
    "2.0.0", 
    "https://github.com/MoonShadow1976/astrbot_plugin_image_generation",
)
class FigurineProPlugin(Star):
    class ImageWorkflow:
        def __init__(self, proxy_url: str | None = None):
            if proxy_url: logger.info(f"ImageWorkflow 使用代理: {proxy_url}")
            self.session = aiohttp.ClientSession()
            self.proxy = proxy_url

        async def _download_image(self, url: str) -> bytes | None:
            logger.info(f"正在尝试下载图片: {url}")
            try:
                async with self.session.get(url, proxy=self.proxy, timeout=30) as resp:
                    resp.raise_for_status()
                    return await resp.read()
            except aiohttp.ClientResponseError as e:
                logger.error(f"图片下载失败: HTTP状态码 {e.status}, URL: {url}, 原因: {e.message}")
                return None
            except asyncio.TimeoutError:
                logger.error(f"图片下载失败: 请求超时 (30s), URL: {url}")
                return None
            except Exception as e:
                logger.error(f"图片下载失败: 发生未知错误, URL: {url}, 错误类型: {type(e).__name__}, 错误: {e}",
                             exc_info=True)
                return None

        async def _get_avatar(self, user_id: str) -> bytes | None:
            if not user_id.isdigit(): logger.warning(f"无法获取非 QQ 平台或无效 QQ 号 {user_id} 的头像。"); return None
            avatar_url = f"https://q1.qlogo.cn/g?b=qq&nk={user_id}&s=640"
            return await self._download_image(avatar_url)

        def _extract_first_frame_sync(self, raw: bytes) -> bytes:
            img_io = io.BytesIO(raw)
            try:
                with PILImage.open(img_io) as img:
                    if getattr(img, "is_animated", False):
                        logger.info("检测到动图, 将抽取第一帧进行生成")
                        img.seek(0)
                        first_frame = img.convert("RGBA")
                        out_io = io.BytesIO()
                        first_frame.save(out_io, format="PNG")
                        return out_io.getvalue()
            except Exception as e:
                logger.warning(f"抽取图片帧时发生错误, 将返回原始数据: {e}", exc_info=True)
            return raw

        async def _load_bytes(self, src: str) -> bytes | None:
            raw: bytes | None = None
            loop = asyncio.get_running_loop()
            if Path(src).is_file():
                raw = await loop.run_in_executor(None, Path(src).read_bytes)
            elif src.startswith("http"):
                raw = await self._download_image(src)
            elif src.startswith("base64://"):
                raw = await loop.run_in_executor(None, base64.b64decode, src[9:])
            if not raw: return None
            return await loop.run_in_executor(None, self._extract_first_frame_sync, raw)

        async def get_images(self, event: AstrMessageEvent) -> List[bytes]:
            img_bytes_list: List[bytes] = []
            at_user_ids: List[str] = []

            for seg in event.message_obj.message:
                if isinstance(seg, Reply) and seg.chain:
                    for s_chain in seg.chain:
                        if isinstance(s_chain, Image):
                            if s_chain.url and (img := await self._load_bytes(s_chain.url)):
                                img_bytes_list.append(img)
                            elif s_chain.file and (img := await self._load_bytes(s_chain.file)):
                                img_bytes_list.append(img)

            for seg in event.message_obj.message:
                if isinstance(seg, Image):
                    if seg.url and (img := await self._load_bytes(seg.url)):
                        img_bytes_list.append(img)
                    elif seg.file and (img := await self._load_bytes(seg.file)):
                        img_bytes_list.append(img)
                elif isinstance(seg, At):
                    at_user_ids.append(str(seg.qq))

            if img_bytes_list:
                return img_bytes_list

            if at_user_ids:
                for user_id in at_user_ids:
                    if avatar := await self._get_avatar(user_id):
                        img_bytes_list.append(avatar)
                return img_bytes_list

            if avatar := await self._get_avatar(event.get_sender_id()):
                img_bytes_list.append(avatar)

            return img_bytes_list

        async def terminate(self):
            if self.session and not self.session.closed: await self.session.close()

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.conf = config
        self.plugin_data_dir = StarTools.get_data_dir()
        self.user_counts_file = self.plugin_data_dir / "user_counts.json"
        self.user_counts: Dict[str, int] = {}
        self.group_counts_file = self.plugin_data_dir / "group_counts.json"
        self.group_counts: Dict[str, int] = {}

        self.user_daily_counts_file = self.plugin_data_dir / "user_daily_counts.json"
        self.user_daily_counts: Dict[str, Dict[str, Any]] = {}  # { "user_id": {"date": "YYYY-MM-DD", "count": N} }
        self.group_daily_counts_file = self.plugin_data_dir / "group_daily_counts.json"
        self.group_daily_counts: Dict[str, Dict[str, Any]] = {}  # { "group_id": {"date": "YYYY-MM-DD", "count": N} }

        self.user_checkin_file = self.plugin_data_dir / "user_checkin.json"
        self.user_checkin_data: Dict[str, str] = {}
        self.prompt_map: Dict[str, str] = {}
        self.key_index = 0
        self.key_lock = asyncio.Lock()
        self.last_api_call_time: Optional[datetime] = None
        self.api_call_lock = asyncio.Lock()
        self.iwf: Optional[FigurineProPlugin.ImageWorkflow] = None

        # 区别不同风格的响应格式
        # 硅基流动的响应格式: {"images": [{"url": "..."}]}
        # 智谱AI的响应格式: {"data": [{"url": "..."}]}
        self.form: dict[str, str] = {
            "siliconflow": "images",
            "bigmodel": "data",
        }
        self.data_form: str = self.form.get(self.conf.get("api_from"), "images")

    async def initialize(self):
        use_proxy = self.conf.get("use_proxy", False)
        proxy_url = self.conf.get("proxy_url") if use_proxy else None
        self.iwf = self.ImageWorkflow(proxy_url)
        await self._load_prompt_map()
        await self._load_user_counts()
        await self._load_group_counts()
        await self._load_user_daily_counts()
        await self._load_group_daily_counts()
        await self._load_user_checkin_data()
        logger.info("FigurinePro 插件已加载 (lmarena 风格)")
        if not self.conf.get("api_keys"):
            logger.warning("FigurinePro: 未配置任何 API 密钥，插件可能无法工作")

    async def _load_prompt_map(self):
        logger.info("正在加载 prompts...")
        self.prompt_map.clear()
        prompt_list = self.conf.get("prompt_list", [])
        for item in prompt_list:
            try:
                if ":" in item:
                    key, value = item.split(":", 1)
                    self.prompt_map[key.strip()] = value.strip()
                else:
                    logger.warning(f"跳过格式错误的 prompt (缺少冒号): {item}")
            except ValueError:
                logger.warning(f"跳过格式错误的 prompt: {item}")
        logger.info(f"加载了 {len(self.prompt_map)} 个 prompts。")

    @filter.event_message_type(filter.EventMessageType.ALL, priority=5)
    async def on_figurine_request(self, event: AstrMessageEvent):
        if self.conf.get("prefix", True) and not event.is_at_or_wake_command:
            return
        text = event.message_str.strip()
        if not text: return
        cmd = text.split()[0].strip()
        bnn_command = self.conf.get("extra_prefix", "bnn")
        user_prompt = ""
        is_bnn = False
        if cmd == bnn_command:
            user_prompt = text.removeprefix(cmd).strip()
            is_bnn = True
            if not user_prompt: return
        elif cmd in self.prompt_map:
            user_prompt = self.prompt_map.get(cmd)
        else:
            return
        sender_id = event.get_sender_id()
        group_id = event.get_group_id()
        is_master = self.is_global_admin(event)
        if not is_master:
            if sender_id in self.conf.get("user_blacklist", []): return
            if group_id and group_id in self.conf.get("group_blacklist", []): return
            if self.conf.get("user_whitelist", []) and sender_id not in self.conf.get("user_whitelist", []): return
            if group_id and self.conf.get("group_whitelist", []) and group_id not in self.conf.get("group_whitelist",
                                                                                                   []): return
            user_count = self._get_user_count(sender_id)
            group_count = self._get_group_count(group_id) if group_id else 0
            user_limit_on = self.conf.get("enable_user_limit", True)
            group_limit_on = self.conf.get("enable_group_limit", False) and group_id
            has_group_count = not group_limit_on or group_count > 0
            has_user_count = not user_limit_on or user_count > 0
            if group_id:
                if not has_group_count and not has_user_count:
                    yield event.plain_result("❌ 本群次数与您的个人次数均已用尽。");
                    return
            elif not has_user_count:
                yield event.plain_result("❌ 您的使用次数已用完。");
                return

            if error_msg := await self._check_and_update_rate_limit():
                yield event.plain_result(error_msg)
                return
                
        if not self.iwf or not (img_bytes_list := await self.iwf.get_images(event)):
            if not is_bnn:
                yield event.plain_result("请发送或引用一张图片。");
                return
        images_to_process = []
        display_cmd = cmd
        if is_bnn:
            MAX_IMAGES = 5
            original_count = len(img_bytes_list)
            if original_count > MAX_IMAGES:
                images_to_process = img_bytes_list[:MAX_IMAGES]
                yield event.plain_result(f"🎨 检测到 {original_count} 张图片，已选取前 {MAX_IMAGES} 张…")
            else:
                images_to_process = img_bytes_list
            display_cmd = user_prompt[:10] + '...' if len(user_prompt) > 10 else user_prompt
            yield event.plain_result(f"🎨 检测到 {len(images_to_process)} 张图片，正在生成 [{display_cmd}]...")
        else:
            if not img_bytes_list:
                yield event.plain_result("请发送或引用一张图片。");
                return
            images_to_process = [img_bytes_list[0]]
            yield event.plain_result(f"🎨 收到请求，正在生成 [{cmd}]...")
        start_time = datetime.now()
        res = await self._call_api(images_to_process, user_prompt)
        elapsed = (datetime.now() - start_time).total_seconds()
        if isinstance(res, bytes):
            if not is_master:
                if self.conf.get("enable_group_limit", False) and group_id and self._get_group_count(group_id) > 0:
                    await self._decrease_group_count(group_id)
                elif self.conf.get("enable_user_limit", True) and self._get_user_count(sender_id) > 0:
                    await self._decrease_user_count(sender_id)
            caption_parts = [f"✅ 生成成功 ({elapsed:.2f}s)", f"预设: {display_cmd}"]
            if is_master:
                caption_parts.append("剩余次数: ∞")
            else:
                if self.conf.get("enable_user_limit", True): caption_parts.append(
                    f"个人剩余: {self._get_user_count(sender_id)}")
                if self.conf.get("enable_group_limit", False) and group_id: caption_parts.append(
                    f"本群剩余: {self._get_group_count(group_id)}")
            yield event.chain_result([Image.fromBytes(res), Plain(" | ".join(caption_parts))])
        else:
            yield event.plain_result(f"❌ 生成失败 ({elapsed:.2f}s)\n原因: {res}")
        event.stop_event()

    @filter.command("文生图", prefix_optional=True)
    async def on_text_to_image_request(self, event: AstrMessageEvent):
        prompt = event.message_str.strip()
        if not prompt:
            yield event.plain_result("请提供文生图的描述。用法: #文生图 <描述>")
            return

        sender_id = event.get_sender_id()
        group_id = event.get_group_id()
        is_master = self.is_global_admin(event)

        # --- 权限和次数检查 ---
        if not is_master:
            if sender_id in self.conf.get("user_blacklist", []): return
            if group_id and group_id in self.conf.get("group_blacklist", []): return
            if self.conf.get("user_whitelist", []) and sender_id not in self.conf.get("user_whitelist", []): return
            if group_id and self.conf.get("group_whitelist", []) and group_id not in self.conf.get("group_whitelist",
                                                                                                   []): return
            user_count = self._get_user_count(sender_id)
            group_count = self._get_group_count(group_id) if group_id else 0
            user_limit_on = self.conf.get("enable_user_limit", True)
            group_limit_on = self.conf.get("enable_group_limit", False) and group_id
            has_group_count = not group_limit_on or group_count > 0
            has_user_count = not user_limit_on or user_count > 0
            if group_id:
                if not has_group_count and not has_user_count:
                    yield event.plain_result("❌ 本群次数与您的个人次数均已用尽。");
                    return
            elif not has_user_count:
                yield event.plain_result("❌ 您的使用次数已用完。");
                return

            if error_msg := await self._check_and_update_rate_limit():
                yield event.plain_result(error_msg)
                return

        display_prompt = prompt[:20] + '...' if len(prompt) > 20 else prompt
        yield event.plain_result(f"🎨 收到文生图请求，正在生成 [{display_prompt}]...")

        start_time = datetime.now()
        # 调用通用API，但传入空的图片列表
        res = await self._call_api([], prompt)
        elapsed = (datetime.now() - start_time).total_seconds()

        if isinstance(res, bytes):
            if not is_master:
                # 扣除次数
                if self.conf.get("enable_group_limit", False) and group_id and self._get_group_count(group_id) > 0:
                    await self._decrease_group_count(group_id)
                elif self.conf.get("enable_user_limit", True) and self._get_user_count(sender_id) > 0:
                    await self._decrease_user_count(sender_id)

            caption_parts = [f"✅ 生成成功 ({elapsed:.2f}s)"]
            if is_master:
                caption_parts.append("剩余次数: ∞")
            else:
                if self.conf.get("enable_user_limit", True): caption_parts.append(
                    f"个人剩余: {self._get_user_count(sender_id)}")
                if self.conf.get("enable_group_limit", False) and group_id: caption_parts.append(
                    f"本群剩余: {self._get_group_count(group_id)}")
            yield event.chain_result([Image.fromBytes(res), Plain(" | ".join(caption_parts))])
        else:
            yield event.plain_result(f"❌ 生成失败 ({elapsed:.2f}s)\n原因: {res}")
        event.stop_event()

    @filter.command("lm添加", aliases={"lma"}, prefix_optional=True)
    async def add_lm_prompt(self, event: AstrMessageEvent):
        if not self.is_global_admin(event): return
        raw = event.message_str.strip()
        if ":" not in raw:
            yield event.plain_result('格式错误, 正确示例:\n#lm添加 姿势表:为这幅图创建一个姿势表, 摆出各种姿势')
            return

        key, new_value = map(str.strip, raw.split(":", 1))
        prompt_list = self.conf.get("prompt_list", [])
        found = False
        for idx, item in enumerate(prompt_list):
            if item.strip().startswith(key + ":"):
                prompt_list[idx] = f"{key}:{new_value}"
                found = True
                break
        if not found: prompt_list.append(f"{key}:{new_value}")

        await self.conf.set("prompt_list", prompt_list)
        await self._load_prompt_map()
        yield event.plain_result(f"已保存LM生图提示语:\n{key}:{new_value}")

    @filter.command("lm帮助", aliases={"lmh", "画图帮助"}, prefix_optional=True)
    async def on_prompt_help(self, event: AstrMessageEvent):
        keyword = event.message_str.strip()
        if not keyword:
            msg = "图生图预设指令: \n"
            msg += "、".join(self.prompt_map.keys())
            msg += "\n\n纯文本生图指令: \n#文生图 <你的描述>"
            msg += "\n\n发送图片 + 预设指令 或 @用户 + 预设指令 来进行图生图。"
            yield event.plain_result(msg)
            return

        prompt = self.prompt_map.get(keyword)
        if not prompt:
            yield event.plain_result("未找到此预设指令")
            return
        yield event.plain_result(f"预设 [{keyword}] 的内容:\n{prompt}")

    def is_global_admin(self, event: AstrMessageEvent) -> bool:
        admin_ids = self.context.get_config().get("admins_id", [])
        return event.get_sender_id() in admin_ids

    async def _load_user_counts(self):
        if not self.user_counts_file.exists(): self.user_counts = {}; return
        loop = asyncio.get_running_loop()
        try:
            content = await loop.run_in_executor(None, self.user_counts_file.read_text, "utf-8")
            data = await loop.run_in_executor(None, json.loads, content)
            if isinstance(data, dict): self.user_counts = {str(k): v for k, v in data.items()}
        except Exception as e:
            logger.error(f"加载用户次数文件时发生错误: {e}", exc_info=True);
            self.user_counts = {}

    async def _save_user_counts(self):
        loop = asyncio.get_running_loop()
        try:
            json_data = await loop.run_in_executor(None,
                                                   functools.partial(json.dumps, self.user_counts, ensure_ascii=False,
                                                                     indent=4))
            await loop.run_in_executor(None, self.user_counts_file.write_text, json_data, "utf-8")
        except Exception as e:
            logger.error(f"保存用户次数文件时发生错误: {e}", exc_info=True)

    def _get_user_count(self, user_id: str) -> int:
        permanent_count = self.user_counts.get(str(user_id), 0)
        
        daily_fixed_quota = self.conf.get("user_daily_fixed_quota", 0)
        if daily_fixed_quota <= 0:
            return permanent_count

        today_str = datetime.now().strftime("%Y-%m-%d")
        user_daily_data = self.user_daily_counts.get(str(user_id), {})
        
        used_today = 0
        if user_daily_data.get("date") == today_str:
            used_today = user_daily_data.get("count", 0)
        
        remaining_daily = max(0, daily_fixed_quota - used_today)
        return permanent_count + remaining_daily

    async def _decrease_user_count(self, user_id: str):
        user_id_str = str(user_id)
        permanent_count = self.user_counts.get(user_id_str, 0)

        if permanent_count > 0:
            self.user_counts[user_id_str] = permanent_count - 1
            await self._save_user_counts()
            return

        daily_fixed_quota = self.conf.get("user_daily_fixed_quota", 0)
        if daily_fixed_quota > 0:
            today_str = datetime.now().strftime("%Y-%m-%d")
            user_daily_data = self.user_daily_counts.get(user_id_str, {})
            
            used_today = 0
            if user_daily_data.get("date") == today_str:
                used_today = user_daily_data.get("count", 0)

            if used_today < daily_fixed_quota:
                self.user_daily_counts[user_id_str] = {"date": today_str, "count": used_today + 1}
                await self._save_user_daily_counts()

    def _get_group_count(self, group_id: str) -> int:
        permanent_count = self.group_counts.get(str(group_id), 0)

        daily_fixed_quota = self.conf.get("group_daily_fixed_quota", 0)
        if daily_fixed_quota <= 0:
            return permanent_count

        today_str = datetime.now().strftime("%Y-%m-%d")
        group_daily_data = self.group_daily_counts.get(str(group_id), {})

        used_today = 0
        if group_daily_data.get("date") == today_str:
            used_today = group_daily_data.get("count", 0)

        remaining_daily = max(0, daily_fixed_quota - used_today)
        return permanent_count + remaining_daily

    async def _decrease_group_count(self, group_id: str):
        group_id_str = str(group_id)
        permanent_count = self.group_counts.get(group_id_str, 0)

        if permanent_count > 0:
            self.group_counts[group_id_str] = permanent_count - 1
            await self._save_group_counts()
            return

        daily_fixed_quota = self.conf.get("group_daily_fixed_quota", 0)
        if daily_fixed_quota > 0:
            today_str = datetime.now().strftime("%Y-%m-%d")
            group_daily_data = self.group_daily_counts.get(group_id_str, {})

            used_today = 0
            if group_daily_data.get("date") == today_str:
                used_today = group_daily_data.get("count", 0)

            if used_today < daily_fixed_quota:
                self.group_daily_counts[group_id_str] = {"date": today_str, "count": used_today + 1}
                await self._save_group_daily_counts()

    async def _load_user_checkin_data(self):
        if not self.user_checkin_file.exists(): self.user_checkin_data = {}; return
        loop = asyncio.get_running_loop()
        try:
            content = await loop.run_in_executor(None, self.user_checkin_file.read_text, "utf-8")
            data = await loop.run_in_executor(None, json.loads, content)
            if isinstance(data, dict): self.user_checkin_data = {str(k): v for k, v in data.items()}
        except Exception as e:
            logger.error(f"加载用户签到文件时发生错误: {e}", exc_info=True);
            self.user_checkin_data = {}

    async def _save_user_checkin_data(self):
        loop = asyncio.get_running_loop()
        try:
            json_data = await loop.run_in_executor(None, functools.partial(json.dumps, self.user_checkin_data,
                                                                           ensure_ascii=False, indent=4))
            await loop.run_in_executor(None, self.user_checkin_file.write_text, json_data, "utf-8")
        except Exception as e:
            logger.error(f"保存用户签到文件时发生错误: {e}", exc_info=True)

    @filter.command("画图签到", prefix_optional=True)
    async def on_checkin(self, event: AstrMessageEvent):
        if not self.conf.get("enable_checkin", False):
            yield event.plain_result("📅 本机器人未开启签到功能。")
            return
        user_id = event.get_sender_id()
        today_str = datetime.now().strftime("%Y-%m-%d")
        if self.user_checkin_data.get(user_id) == today_str:
            yield event.plain_result(f"您今天已经签到过了。\n剩余次数: {self._get_user_count(user_id)}")
            return
        
        reward = 0
        if str(self.conf.get("enable_random_checkin", False)).lower() == 'true':
            max_reward = max(1, int(self.conf.get("checkin_random_reward_max", 5)))
            reward = random.randint(1, max_reward)
        else:
            reward = int(self.conf.get("checkin_fixed_reward", 3))

        # 签到奖励只增加永久次数
        current_permanent_count = self.user_counts.get(user_id, 0)
        new_permanent_count = current_permanent_count + reward
        self.user_counts[user_id] = new_permanent_count
        await self._save_user_counts()

        self.user_checkin_data[user_id] = today_str
        await self._save_user_checkin_data()

        # 回复时显示总次数
        new_total_count = self._get_user_count(user_id)
        yield event.plain_result(f"🎉 签到成功！获得 {reward} 次（永久），当前总剩余: {new_total_count} 次。")

    @filter.command("画图增加用户次数", prefix_optional=True)
    async def on_add_user_counts(self, event: AstrMessageEvent):
        if not self.is_global_admin(event): return
        cmd_text = event.message_str.strip()
        at_seg = next((s for s in event.message_obj.message if isinstance(s, At)), None)
        target_qq, count = None, 0
        if at_seg:
            target_qq = str(at_seg.qq)
            match = re.search(r"(\d+)\s*$", cmd_text)
            if match: count = int(match.group(1))
        else:
            match = re.search(r"(\d+)\s+(\d+)", cmd_text)
            if match: target_qq, count = match.group(1), int(match.group(2))
        if not target_qq or count <= 0:
            yield event.plain_result(
                '格式错误:\n#画图增加用户次数 @用户 <次数>\n或 #画图增加用户次数 <QQ号> <次数>')
            return
            
        # 管理员增加的是永久次数
        current_permanent_count = self.user_counts.get(str(target_qq), 0)
        new_permanent_count = current_permanent_count + count
        self.user_counts[str(target_qq)] = new_permanent_count
        await self._save_user_counts()

        # 回复时显示总次数
        new_total_count = self._get_user_count(target_qq)
        yield event.plain_result(f"✅ 已为用户 {target_qq} 增加 {count} 次（永久），TA当前总剩余 {new_total_count} 次。")

    @filter.command("画图增加群组次数", prefix_optional=True)
    async def on_add_group_counts(self, event: AstrMessageEvent):
        if not self.is_global_admin(event): return
        match = re.search(r"(\d+)\s+(\d+)", event.message_str.strip())
        if not match:
            yield event.plain_result('格式错误: #画图增加群组次数 <群号> <次数>')
            return
        target_group, count = match.group(1), int(match.group(2))
        
        # 管理员增加的是永久次数
        current_permanent_count = self.group_counts.get(str(target_group), 0)
        new_permanent_count = current_permanent_count + count
        self.group_counts[str(target_group)] = new_permanent_count
        await self._save_group_counts()

        # 回复时显示总次数
        new_total_count = self._get_group_count(target_group)
        yield event.plain_result(f"✅ 已为群组 {target_group} 增加 {count} 次（永久），该群当前总剩余 {new_total_count} 次。")

    @filter.command("画图查询次数", prefix_optional=True)
    async def on_query_counts(self, event: AstrMessageEvent):
        user_id_to_query = event.get_sender_id()
        if self.is_global_admin(event):
            at_seg = next((s for s in event.message_obj.message if isinstance(s, At)), None)
            if at_seg:
                user_id_to_query = str(at_seg.qq)
            else:
                match = re.search(r"(\d+)", event.message_str)
                if match: user_id_to_query = match.group(1)
        user_count = self._get_user_count(user_id_to_query)
        reply_msg = f"用户 {user_id_to_query} 个人剩余次数为: {user_count}"
        if user_id_to_query == event.get_sender_id(): reply_msg = f"您好，您当前个人剩余次数为: {user_count}"
        if group_id := event.get_group_id(): reply_msg += f"\n本群共享剩余次数为: {self._get_group_count(group_id)}"
        yield event.plain_result(reply_msg)

    @filter.command("画图添加key", prefix_optional=True)
    async def on_add_key(self, event: AstrMessageEvent):
        if not self.is_global_admin(event): return
        new_keys = event.message_str.strip().split()
        if not new_keys: yield event.plain_result("格式错误，请提供要添加的Key。"); return
        api_keys = self.conf.get("api_keys", [])
        added_keys = [key for key in new_keys if key not in api_keys]
        api_keys.extend(added_keys)
        await self.conf.set("api_keys", api_keys)
        yield event.plain_result(f"✅ 操作完成，新增 {len(added_keys)} 个Key，当前共 {len(api_keys)} 个。")

    @filter.command("画图key列表", prefix_optional=True)
    async def on_list_keys(self, event: AstrMessageEvent):
        if not self.is_global_admin(event): return
        api_keys = self.conf.get("api_keys", [])
        if not api_keys: yield event.plain_result("📝 暂未配置任何 API Key。"); return
        key_list_str = "\n".join(f"{i + 1}. {key[:8]}...{key[-4:]}" for i, key in enumerate(api_keys))
        yield event.plain_result(f"🔑 API Key 列表:\n{key_list_str}")

    @filter.command("画图删除key", prefix_optional=True)
    async def on_delete_key(self, event: AstrMessageEvent):
        if not self.is_global_admin(event): return
        param = event.message_str.strip()
        api_keys = self.conf.get("api_keys", [])
        if param.lower() == "all":
            await self.conf.set("api_keys", [])
            yield event.plain_result(f"✅ 已删除全部 {len(api_keys)} 个 Key。")
        elif param.isdigit() and 1 <= int(param) <= len(api_keys):
            removed_key = api_keys.pop(int(param) - 1)
            await self.conf.set("api_keys", api_keys)
            yield event.plain_result(f"✅ 已删除 Key: {removed_key[:8]}...")
        else:
            yield event.plain_result("格式错误，请使用 #画图删除key <序号|all>")

    async def _check_and_update_rate_limit(self) -> Optional[str]:
        """Checks the rate limit. If not passed, returns an error string. If passed, updates the last call time and returns None."""
        rate_limit = self.conf.get("rate_limit_seconds", 120)
        if rate_limit <= 0:
            return None

        async with self.api_call_lock:
            now = datetime.now()
            if self.last_api_call_time:
                elapsed = (now - self.last_api_call_time).total_seconds()
                if elapsed < rate_limit:
                    return f"⏳ 操作太频繁或其他用户正在生图，请在 {int(rate_limit - elapsed)} 秒后再试。"
            # The check passed, update the time
            self.last_api_call_time = now
        return None

    async def _get_api_key(self) -> str | None:
        keys = self.conf.get("api_keys", [])
        if not keys: return None
        async with self.key_lock:
            key = keys[self.key_index]
            self.key_index = (self.key_index + 1) % len(keys)
            return key


    def _extract_image_url_from_response(self, data: Dict[str, Any]) -> str | None:
        """
        从 API 响应中提取图片 URL。
        """
        try:
            url = data[self.data_form][0]["url"]
            logger.info(f"成功从 API 响应中提取到 URL: {url[:50]}...")
            return url
        except (IndexError, TypeError, KeyError):
            logger.warning(f"未能在响应中找到 '{self.data_form}[0].url'，原始响应 (截断): {str(data)[:200]}")
            return None

    def _sanitize_for_log(self, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: self._sanitize_for_log(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_for_log(item) for item in data]
        elif isinstance(data, str) and len(data) > 200:
            return data[:200] + "..."
        else:
            return data

    async def _call_openai_responses_api(self, image_bytes_list: List[bytes], prompt: str) -> bytes | str:
        api_url = self.conf.get("api_url")
        if not api_url: return "API URL 未配置"
        api_key = await self._get_api_key()
        if not api_key: return "无可用的 API Key"
        
        model_name = self.conf.get("model")
        if not model_name: return "模型名称 (model) 未配置"

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        
        # This structure is based on the provided ttp.py for Gemini-FastAPI
        content_items = [{"type": "input_text", "text": prompt}]
        if image_bytes_list:
            for img_bytes in image_bytes_list:
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                content_items.append({"type": "input_image", "image_url": f"data:image/png;base64,{img_b64}"})
        
        payload = {
            "model": model_name,
            "input": [{"role": "user", "content": content_items}],
            "tools": [{"type": "image_generation"}],
            "tool_choice": {"type": "image_generation"},
        }
        
        logger.info(f"发送到 OpenAI-responses API: URL={api_url}, Model={model_name}, HasImage={bool(image_bytes_list)}")
        # Use the sanitize helper for logging the payload
        logger.debug(f"Payload (sanitized): {json.dumps(self._sanitize_for_log(payload), ensure_ascii=False)}")

        try:
            if not self.iwf: return "ImageWorkflow 未初始化"
            timeout = aiohttp.ClientTimeout(total=180)
            async with self.iwf.session.post(api_url, json=payload, headers=headers, proxy=self.iwf.proxy, timeout=timeout) as resp:
                try:
                    data = await resp.json()
                    # Use the sanitize helper for logging the response
                    sanitized_data = self._sanitize_for_log(data)
                    logger.info(f"完整API响应 (sanitized): {json.dumps(sanitized_data, indent=2, ensure_ascii=False)}")
                except Exception as e:
                    raw_text = await resp.text()
                    logger.error(f"解析API响应JSON失败: {e}")
                    logger.info(f"原始响应文本: {raw_text[:500]}...")
                    return f"API响应解析失败: {raw_text[:200]}"

                if resp.status != 200:
                    error_message = data.get("error", {}).get("message", str(data))
                    logger.error(f"API 请求失败: HTTP {resp.status}, 响应: {error_message}")
                    return f"API请求失败 (HTTP {resp.status}): {error_message[:200]}"

                base64_string = None
                
                # Parsing logic from ttp.py
                if data.get("object") == "response" and "output" in data:
                    output_list = data.get("output", [])
                    if isinstance(output_list, list):
                        for item in output_list:
                            if isinstance(item, dict) and item.get("type") == "image_generation_call":
                                result = item.get("result")
                                if isinstance(result, str) and result:
                                    base64_string = result
                                    logger.info("在响应 `output` 字段中找到 Base64 图像数据 (image_generation_call)")
                                    break
                
                if not base64_string:
                    response_data = data.get("data", [])
                    if isinstance(response_data, list):
                        for item in response_data:
                            if isinstance(item, dict) and item.get("type") == "image_result":
                                b64_json = item.get("b64_json")
                                if isinstance(b64_json, str) and b64_json:
                                    base64_string = b64_json
                                    logger.info("在响应 `data` 字段中找到 Base64 图像数据 (b64_json)")
                                    break
                
                if not base64_string:
                    choices = data.get("choices", [])
                    if choices:
                        message = choices[0].get("message", {})
                        content_list = message.get("content", [])
                        if isinstance(content_list, list):
                            for item in content_list:
                                if isinstance(item, dict) and item.get("type") == "image_result":
                                    base64_data = item.get("image")
                                    if isinstance(base64_data, str) and base64_data:
                                        base64_string = base64_data
                                        logger.info("在响应 `choices` 中找到 Base64 图像数据")
                                        break
                
                if not base64_string:
                    error_msg = f"API响应中未找到有效的图像数据"
                    logger.error(f"{error_msg}. Response: {self._sanitize_for_log(data)}")
                    return error_msg

                # Remove potential data URI prefix
                if "base64," in base64_string:
                    base64_string = base64_string.split("base64,", 1)[1]
                
                return base64.b64decode(base64_string)

        except asyncio.TimeoutError:
            logger.error("API 请求超时");
            return "请求超时"
        except Exception as e:
            logger.error(f"调用 OpenAI-responses API 时发生未知错误: {e}", exc_info=True);
            return f"发生未知错误: {e}"

    async def _call_api(self, image_bytes_list: List[bytes], prompt: str) -> bytes | str:
        api_from = self.conf.get("api_from")

        if api_from == "OpenAI-responses":
            return await self._call_openai_responses_api(image_bytes_list, prompt)

        # Existing logic for other API types
        api_url = self.conf.get("api_url")
        if not api_url: return "API URL 未配置"
        api_key = await self._get_api_key()
        if not api_key: return "无可用的 API Key"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

        model_name = self.conf.get("model")
        if not model_name:
            return "模型名称 (model) 未配置"

        payload: Dict[str, Any] = {
            "model": model_name,
            "prompt": prompt,
        }

        if image_bytes_list:
            try:
                img_b64 = base64.b64encode(image_bytes_list[0]).decode("utf-8")
                payload["image"] = f"data:image/png;base64,{img_b64}"
                if len(image_bytes_list) > 1:
                    img_b64_2 = base64.b64encode(image_bytes_list[1]).decode("utf-8")
                    payload["image2"] = f"data:image/png;base64,{img_b64_2}"
                if len(image_bytes_list) > 2:
                    img_b64_3 = base64.b64encode(image_bytes_list[2]).decode("utf-8")
                    payload["image3"] = f"data:image/png;base64,{img_b64_3}"
            except Exception as e:
                logger.error(f"Base64 编码图片时出错: {e}", exc_info=True)
                return f"图片编码失败: {e}"

        logger.info(f"发送到 {api_from} API: URL={api_url}, Model={model_name}, HasImage={bool(image_bytes_list)}")
        
        try:
            if not self.iwf: return "ImageWorkflow 未初始化"
            async with self.iwf.session.post(api_url, json=payload, headers=headers, proxy=self.iwf.proxy,
                                             timeout=120) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"API 请求失败: HTTP {resp.status}, 响应: {error_text}")
                    return f"API请求失败 (HTTP {resp.status}): {error_text[:200]}"

                data = await resp.json()
                
                # Dynamically update data_form based on current config
                self.data_form = self.form.get(api_from, "images")

                if self.data_form not in data or not data[self.data_form]:
                    error_msg = f"API响应中未找到图片数据: {str(data)[:500]}..."
                    logger.error(f"API响应中未找到图片数据: {data}")
                    if "error" in data:
                        return data["error"].get("message", json.dumps(data["error"]))
                    return error_msg

                gen_image_url = self._extract_image_url_from_response(data)

                if not gen_image_url:
                    error_msg = f"API响应解析失败: {str(data)[:500]}..."
                    logger.error(f"API响应解析失败: {data}")
                    return error_msg

                if gen_image_url.startswith("data:image/"):
                    b64_data = gen_image_url.split(",", 1)[1]
                    return base64.b64decode(b64_data)
                else:
                    return await self.iwf._download_image(gen_image_url) or "下载生成的图片失败"
        except asyncio.TimeoutError:
            logger.error("API 请求超时");
            return "请求超时"
        except Exception as e:
            logger.error(f"调用 API 时发生未知错误: {e}", exc_info=True);
            return f"发生未知错误: {e}"

    async def terminate(self):
        if self.iwf: await self.iwf.terminate()
        logger.info("[FigurinePro] 插件已终止")
