"""
AstrBot Gemini 图像生成插件主文件
支持 Google 官方 API 和 OpenAI 兼容格式 API，提供生图和改图功能，支持智能头像参考
"""

from __future__ import annotations

import asyncio
import base64
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp
import yaml
from astrbot.api import logger
from astrbot.api.all import Image, Reply
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

from .tl import create_zip, split_image
from .tl.enhanced_prompts import (
    enhance_prompt_for_figure,
    get_auto_modification_prompt,
    get_avatar_prompt,
    get_card_prompt,
    get_figure_prompt,
    get_generation_prompt,
    get_mobile_prompt,
    get_modification_prompt,
    get_poster_prompt,
    get_sticker_prompt,
    get_style_change_prompt,
    get_wallpaper_prompt,
)
from .tl.tl_api import (
    APIClient,
    APIError,
    ApiRequestConfig,
    get_api_client,
)
from .tl.tl_utils import (
    AvatarManager,
    cleanup_old_images,
    download_qq_avatar,
    send_file,
)


@register(
    "astrbot_plugin_gemini_image_generation",
    "piexian",
    "Gemini图像生成插件，支持生图和改图，可以自动获取头像作为参考",
    "v1.5.2",
)
class GeminiImageGenerationPlugin(Star):
    def __init__(self, context: Context, config: dict[str, Any]):
        super().__init__(context)
        self.config = config
        self.api_client: APIClient | None = None
        self.avatar_manager = AvatarManager()
        self._cleanup_task: asyncio.Task | None = None

        # 加载配置
        self._load_config()

        # 启动定时清理任务
        self._start_cleanup_task()

    def _start_cleanup_task(self):
        """启动定时清理任务"""
        if self._cleanup_task and not self._cleanup_task.done():
            return

        async def cleanup_loop():
            while True:
                try:
                    await cleanup_old_images()
                    # 每30分钟执行一次
                    await asyncio.sleep(1800)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.warning(f"清理任务异常: {e}")
                    await asyncio.sleep(300)

        self._cleanup_task = asyncio.create_task(cleanup_loop())
        logger.debug("定时清理任务已启动")

    async def terminate(self):
        """插件卸载/重载时调用"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            logger.debug("定时清理任务已停止")
        logger.info("🎨 Gemini 图像生成插件已卸载")

    def get_tool_timeout(self, event: AstrMessageEvent | None = None) -> int:
        """获取当前聊天环境的 tool_call_timeout 配置"""
        try:
            # 如果提供了事件，尝试获取特定聊天环境的配置
            if event:
                umo = event.unified_msg_origin
                chat_config = self.context.get_config(umo=umo)
                return chat_config.get("provider_settings", {}).get(
                    "tool_call_timeout", 60
                )

            # 否则使用默认配置
            default_config = self.context.get_config()
            return default_config.get("provider_settings", {}).get(
                "tool_call_timeout", 60
            )
        except Exception as e:
            logger.warning(f"获取 tool_call_timeout 配置失败: {e}，使用默认值 60 秒")
            return 60

    async def get_avatar_reference(self, event: AstrMessageEvent) -> list[str]:
        """获取头像作为参考图像，支持群头像和用户头像（直接HTTP下载）"""
        avatar_images = []
        download_tasks = []

        try:
            # 检查是否需要获取群头像
            if hasattr(event, "group_id") and event.group_id:
                group_id = str(event.group_id)
                prompt = event.wessage_str.lower()

                # 群头像获取的几种情况：
                # 1. 明确提到群相关关键词
                # 2. 在群聊中且启用了自动头像参考且触发了生图指令
                group_avatar_keywords = [
                    "群头像",
                    "本群",
                    "我们的群",
                    "这个群",
                    "群标志",
                    "群图标",
                ]
                explicit_group_request = any(
                    keyword in prompt for keyword in group_avatar_keywords
                )

                # 判断是否应该获取群头像
                should_get_group_avatar = explicit_group_request or (
                    self.auto_avatar_reference
                    and any(
                        keyword in prompt
                        for keyword in [
                            "生图",
                            "绘图",
                            "画图",
                            "生成图片",
                            "制作图片",
                            "改图",
                            "修改",
                        ]
                    )
                )

                if should_get_group_avatar:
                    if explicit_group_request:
                        logger.info(
                            f"检测到明确的群头像关键词，准备获取群 {group_id} 的头像"
                        )
                    else:
                        logger.info(
                            f"群聊中生图指令触发，自动获取群 {group_id} 的头像作为参考"
                        )

                    # 群头像暂时跳过，因为QQ群头像需要特殊API
                    logger.info("群头像功能暂未实现，跳过")

            # 获取头像逻辑
            # 获取头像：优先获取@用户头像，如果无@用户则获取发送者头像
            mentioned_users = await self.parse_mentions(event)

            if mentioned_users:
                # 有@用户：只获取被@用户的头像
                for user_id in mentioned_users:
                    logger.info(f"[AVATAR] 获取@用户头像: {user_id}")
                    download_tasks.append(
                        download_qq_avatar(str(user_id), f"mentioned_{user_id}")
                    )
            else:
                # 无@用户：获取发送者头像
                if (
                    hasattr(event, "message_obj")
                    and hasattr(event.message_obj, "sender")
                    and hasattr(event.message_obj.sender, "user_id")
                ):
                    sender_id = str(event.message_obj.sender.user_id)
                    logger.info(f"[AVATAR] 获取发送者头像: {sender_id}")
                    download_tasks.append(
                        download_qq_avatar(sender_id, f"sender_{sender_id}")
                    )

            # 执行下载任务
            if download_tasks:
                logger.info(
                    f"[AVATAR_DEBUG] 开始并发下载 {len(download_tasks)} 个头像..."
                )
                try:
                    # 设置总体超时时间为8秒，避免单个下载拖慢整体
                    results = await asyncio.wait_for(
                        asyncio.gather(*download_tasks, return_exceptions=True),
                        timeout=8.0,
                    )

                    # 处理结果
                    for result in results:
                        if isinstance(result, str) and result:
                            avatar_images.append(result)
                        elif isinstance(result, Exception):
                            logger.warning(f"头像下载任务失败: {result}")

                    logger.info(
                        f"头像下载完成，成功获取 {len(avatar_images)} 个头像，即将返回"
                    )

                except asyncio.TimeoutError:
                    logger.warning("头像下载总体超时，跳过剩余头像下载")
                except Exception as e:
                    logger.error(f"并发下载头像时发生错误: {e}")

        except Exception as e:
            logger.error(f"获取头像参考失败: {e}")

        return avatar_images

    async def should_use_avatar(self, event: AstrMessageEvent) -> bool:
        """判断是否应该使用头像作为参考（只有在有@用户时才使用）"""
        logger.info(
            f"[AVATAR_DEBUG] 检查auto_avatar_reference: {self.auto_avatar_reference}"
        )
        if not self.auto_avatar_reference:
            return False

        # 检查是否有@用户
        mentioned_users = await self.parse_mentions(event)
        logger.info(f"[AVATAR_DEBUG] @用户数量: {len(mentioned_users)}")

        # 只有当有@用户时才获取头像
        return len(mentioned_users) > 0

    async def parse_mentions(self, event: AstrMessageEvent) -> list[int]:
        """解析消息中的@用户，返回用户ID列表"""
        mentioned_users = []

        try:
            # 使用框架提供的方法获取消息组件
            messages = event.get_messages()

            for msg_component in messages:
                # 检查是否是@组件
                if hasattr(msg_component, "qq") and str(msg_component.qq) != str(
                    event.get_self_id()
                ):
                    mentioned_users.append(int(msg_component.qq))
                    self.log_debug(f"解析到@用户: {msg_component.qq}")

        except Exception as e:
            logger.warning(f"解析@用户失败: {e}")

        return mentioned_users

    def _load_config(self):
        """从配置加载所有设置"""
        self.api_keys = self.config.get("openrouter_api_keys", [])
        if not isinstance(self.api_keys, list):
            self.api_keys = [self.api_keys] if self.api_keys else []

        api_settings = self.config.get("api_settings", {})
        self.api_type = api_settings.get("api_type", "google")
        self.api_base = api_settings.get("custom_api_base", "")
        self.model = api_settings.get("model", "gemini-3-pro-image-preview")

        image_settings = self.config.get("image_generation_settings", {})
        self.resolution = image_settings.get("resolution", "1K")
        self.aspect_ratio = image_settings.get("aspect_ratio", "1:1")
        self.enable_grounding = image_settings.get("enable_grounding", False)
        self.max_reference_images = image_settings.get("max_reference_images", 6)
        self.enable_text_response = image_settings.get("enable_text_response", False)
        self.enable_sticker_split = image_settings.get("enable_sticker_split", True)
        self.enable_sticker_zip = image_settings.get("enable_sticker_zip", False)
        self.preserve_reference_image_size = image_settings.get(
            "preserve_reference_image_size", False
        )
        # 从配置中读取强制分辨率设置，默认为False
        self.force_resolution = image_settings.get("force_resolution", False)

        retry_settings = self.config.get("retry_settings", {})
        self.max_attempts_per_key = retry_settings.get("max_attempts_per_key", 3)
        self.enable_smart_retry = retry_settings.get("enable_smart_retry", True)
        self.total_timeout = retry_settings.get("total_timeout", 120)

        service_settings = self.config.get("service_settings", {})
        self.nap_server_address = service_settings.get(
            "nap_server_address", "localhost"
        )
        self.nap_server_port = service_settings.get("nap_server_port", 3658)
        self.auto_avatar_reference = service_settings.get(
            "auto_avatar_reference", False
        )
        self.verbose_logging = service_settings.get("verbose_logging", False)
        limit_settings = self.config.get("limit_settings", {})
        raw_mode = str(limit_settings.get("group_limit_mode", "none")).lower()
        if raw_mode not in {"none", "whitelist", "blacklist"}:
            raw_mode = "none"
        self.group_limit_mode: str = raw_mode

        raw_group_list = limit_settings.get("group_limit_list", []) or []
        # 统一使用字符串形式保存群号，便于与 NapCat/QQ 等平台的群 ID 对齐
        self.group_limit_list: set[str] = {
            str(group_id).strip()
            for group_id in raw_group_list
            if str(group_id).strip()
        }

        self.enable_rate_limit: bool = bool(
            limit_settings.get("enable_rate_limit", False)
        )
        # 限流周期与次数做基础防御，避免异常配置导致错误
        period = limit_settings.get("rate_limit_period", 60)
        max_requests = limit_settings.get("max_requests_per_group", 5)
        try:
            self.rate_limit_period: int = max(int(period), 1)
        except (TypeError, ValueError):
            self.rate_limit_period = 60
        try:
            self.max_requests_per_group: int = max(int(max_requests), 1)
        except (TypeError, ValueError):
            self.max_requests_per_group = 5

        # 内部限流状态：按群维度统计请求时间戳
        self._rate_limit_buckets: dict[str, list[float]] = {}
        self._rate_limit_lock = asyncio.Lock()

        if self.api_keys:
            self.api_client = get_api_client(self.api_keys)
            logger.info("✓ API 客户端已初始化")
            logger.info(f"  - 类型: {self.api_type}")
            logger.info(f"  - 模型: {self.model}")
            logger.info(f"  - 密钥数量: {len(self.api_keys)}")
            if self.api_base:
                logger.info(f"  - 自定义 API Base: {self.api_base}")
        else:
            logger.warning("✗ 未配置 API 密钥")

    def log_info(self, message: str):
        """根据配置输出info或debug级别日志"""
        if self.verbose_logging:
            logger.info(message)
        else:
            logger.debug(message)

    def log_debug(self, message: str):
        """输出debug级别日志"""
        logger.debug(message)

    @staticmethod
    def _is_valid_base64_image_str(value: str) -> bool:
        """粗略判断字符串是否为有效的 base64 图像数据或 data URL"""
        if not value:
            return False

        if value.startswith("data:image/"):
            return ";base64," in value

        try:
            base64.b64decode(value, validate=True)
            return True
        except Exception:
            return False

    @staticmethod
    def _clean_text_content(text: str) -> str:
        """清理文本内容，移除 markdown 图片链接等不可发送的内容"""
        if not text:
            return text

        import re

        text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
        text = text.strip()

        return text

    def _filter_valid_reference_images(
        self, images: list[str] | None, source: str
    ) -> list[str]:
        """
        过滤出合法的 base64 / data URL 参考图像。

        NapCat 等平台的图片 file_id（例如 D127D0...jpg）会在这里被过滤掉，
        避免传给 Gemini 导致 Base64 解码错误。
        """
        if not images:
            return []

        valid: list[str] = []
        for img in images:
            if not isinstance(img, str) or not img:
                self.log_debug(f"跳过非字符串参考图像({source}): {type(img)}")
                continue

            if self._is_valid_base64_image_str(img):
                valid.append(img)
            else:
                self.log_debug(f"跳过非 base64 格式参考图像({source}): {img[:64]}...")

        return valid

    def _get_group_id_from_event(self, event: AstrMessageEvent) -> str | None:
        """从事件中解析群ID，仅在群聊场景下返回"""
        try:
            if hasattr(event, "group_id") and event.group_id:
                return str(event.group_id)
            message_obj = getattr(event, "message_obj", None)
            if message_obj and getattr(message_obj, "group_id", ""):
                return str(message_obj.group_id)
        except Exception as e:
            self.log_debug(f"获取群ID失败: {e}")
        return None

    async def _check_and_consume_limit(
        self, event: AstrMessageEvent
    ) -> tuple[bool, str | None]:
        """
        检查当前事件是否通过群聊黑/白名单和限流校验。

        返回:
            (是否允许继续执行, 不允许时的提示消息)
        """
        group_id = self._get_group_id_from_event(event)

        if not group_id:
            return True, None

        if self.group_limit_mode == "whitelist":
            if self.group_limit_list and group_id not in self.group_limit_list:
                return False, None
        elif self.group_limit_mode == "blacklist":
            if self.group_limit_list and group_id in self.group_limit_list:
                return False, None

        if not self.enable_rate_limit:
            return True, None

        now = time.monotonic()
        window_start = now - self.rate_limit_period

        async with self._rate_limit_lock:
            bucket = self._rate_limit_buckets.get(group_id, [])
            bucket = [ts for ts in bucket if ts >= window_start]

            if len(bucket) >= self.max_requests_per_group:
                earliest = bucket[0]
                retry_after = int(earliest + self.rate_limit_period - now)
                if retry_after < 0:
                    retry_after = 0

                self._rate_limit_buckets[group_id] = bucket
                return (
                    False,
                    f"⏱️ 本群在最近 {self.rate_limit_period} 秒内的生图请求次数已达上限（{self.max_requests_per_group} 次），请约 {retry_after} 秒后再试。",
                )

            bucket.append(now)
            self._rate_limit_buckets[group_id] = bucket

        return True, None

    async def initialize(self):
        """插件初始化"""
        if self.api_client:
            logger.info("🎨 Gemini 图像生成插件已加载")
        else:
            logger.error("✗ API 客户端初始化失败，请检查配置")

    async def _collect_reference_images(self, event: AstrMessageEvent) -> list[str]:
        """从消息和回复中提取参考图片，并转换为base64格式"""
        reference_images = []
        max_images = self.max_reference_images

        if not hasattr(event, "message_obj") or not event.message_obj:
            return reference_images

        message_chain = event.message_obj.message
        if not message_chain:
            return reference_images

        async def convert_to_base64(img_source: str) -> str | None:
            """将图片源转换为base64格式"""
            try:
                if img_source.startswith(("http://", "https://")):
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            img_source, timeout=aiohttp.ClientTimeout(total=10)
                        ) as response:
                            if response.status == 200:
                                image_data = await response.read()
                                return base64.b64encode(image_data).decode("utf-8")
                            else:
                                logger.warning(f"下载图片失败: HTTP {response.status}")
                                return None
                elif img_source.startswith("data:image/"):
                    return img_source
                elif self._is_valid_base64_image_str(img_source):
                    return img_source
                else:
                    logger.debug(f"跳过非HTTP/base64格式的图片源: {img_source[:64]}...")
                    return None
            except Exception as e:
                import traceback

                logger.warning(
                    f"转换图片为base64失败: {repr(e)} | Source: {str(img_source)[:100]}"
                )
                logger.debug(traceback.format_exc())
                return None

        for component in message_chain:
            if isinstance(component, Image) and len(reference_images) < max_images:
                try:
                    img_source = None
                    if hasattr(component, "url") and component.url:
                        img_source = component.url
                    elif (
                        hasattr(component, "file")
                        and component.file
                        and isinstance(component.file, str)
                    ):
                        img_source = component.file

                    if img_source:
                        base64_img = await convert_to_base64(img_source)
                        if base64_img:
                            reference_images.append(base64_img)
                            logger.debug(
                                f"✓ 从当前消息提取图片 (当前: {len(reference_images)}/{max_images})"
                            )
                except Exception as e:
                    logger.warning(f"✗ 提取图片失败: {e}")

        for component in message_chain:
            if isinstance(component, Reply) and component.chain:
                for reply_comp in component.chain:
                    if (
                        isinstance(reply_comp, Image)
                        and len(reference_images) < max_images
                    ):
                        try:
                            img_source = None
                            if hasattr(reply_comp, "url") and reply_comp.url:
                                img_source = reply_comp.url
                            elif (
                                hasattr(reply_comp, "file")
                                and reply_comp.file
                                and isinstance(reply_comp.file, str)
                            ):
                                img_source = reply_comp.file

                            if img_source:
                                base64_img = await convert_to_base64(img_source)
                                if base64_img:
                                    reference_images.append(base64_img)
                                    self.log_debug("✓ 从回复消息提取图片")
                        except Exception as e:
                            logger.warning(f"✗ 提取回复图片失败: {e}")

        logger.info(f"📸 共收集到 {len(reference_images)} 张参考图片")
        return reference_images

    async def _generate_image_core_internal(
        self,
        event: AstrMessageEvent,
        prompt: str,
        reference_images: list[str],
        avatar_reference: list[str],
        is_modification: bool = False,
    ) -> tuple[bool, tuple[str, str, str | None] | str]:
        """
        内部核心图像生成方法，不发送消息，只返回结果

        Returns:
            tuple[bool, tuple[str, str, str | None] | str]: (是否成功, (图片路径, 文本内容, 思维签名) 或错误消息)
        """
        if not self.api_client:
            return False, "❌ 错误: API 客户端未初始化，请联系管理员配置 API 密钥"

        valid_msg_images = self._filter_valid_reference_images(
            reference_images, source="消息图片"
        )
        valid_avatar_images = self._filter_valid_reference_images(
            avatar_reference, source="头像"
        )
        all_reference_images = valid_msg_images + valid_avatar_images

        if (
            all_reference_images
            and len(all_reference_images) > self.max_reference_images
        ):
            logger.warning(
                f"参考图片数量 ({len(all_reference_images)}) 超过限制 ({self.max_reference_images})，将截取前 {self.max_reference_images} 张"
            )
            all_reference_images = all_reference_images[: self.max_reference_images]

        # 计算截断后的数量
        final_msg_count = min(len(valid_msg_images), len(all_reference_images))
        final_avatar_count = len(all_reference_images) - final_msg_count

        if final_avatar_count > 0:
            prompt += f"""

[System Note]
The last {final_avatar_count} image(s) provided are User Avatars (marked as optional reference). You may use them for character consistency if needed, but they are NOT mandatory if they conflict with the requested style."""

        response_modalities = "TEXT_IMAGE" if self.enable_text_response else "IMAGE"
        effective_resolution = self.resolution
        effective_aspect_ratio = self.aspect_ratio

        if (
            self.preserve_reference_image_size
            and is_modification
            and all_reference_images
        ):
            effective_resolution = None
            effective_aspect_ratio = None
            self.log_debug("保留参考图尺寸，不覆盖分辨率/比例")

        request_config = ApiRequestConfig(
            model=self.model,
            prompt=prompt,
            api_type=self.api_type,
            api_base=self.api_base,
            resolution=effective_resolution,
            aspect_ratio=effective_aspect_ratio,
            enable_grounding=self.enable_grounding,
            response_modalities=response_modalities,
            reference_images=all_reference_images if all_reference_images else None,
            enable_smart_retry=self.enable_smart_retry,
            enable_text_response=self.enable_text_response,
            force_resolution=self.force_resolution,
        )

        logger.info("🎨 图像生成请求:")
        logger.info(f"  模型: {self.model}")
        logger.info(f"  API 类型: {self.api_type}")
        logger.info(
            f"  参考图片: {len(all_reference_images) if all_reference_images else 0} 张"
        )

        try:
            logger.info("🚀 开始调用API生成图像...")
            start_time = asyncio.get_event_loop().time()

            tool_timeout = self.get_tool_timeout(event)
            per_retry_timeout = min(self.total_timeout, tool_timeout)
            max_total_time = tool_timeout
            logger.info(
                f"[TIMEOUT] tool_call_timeout={tool_timeout}s, per_retry_timeout={per_retry_timeout}s, max_retries={self.max_attempts_per_key}, max_total_time={max_total_time}s"
            )

            (
                image_url,
                image_path,
                text_content,
                thought_signature,
            ) = await self.api_client.generate_image(
                config=request_config,
                max_retries=self.max_attempts_per_key,
                per_retry_timeout=per_retry_timeout,
                max_total_time=max_total_time,
            )

            end_time = asyncio.get_event_loop().time()
            api_duration = end_time - start_time
            logger.info(f"✅ API调用完成，耗时: {api_duration:.2f}秒")

            if thought_signature:
                logger.debug(f"🧠 思维签名: {thought_signature[:50]}...")

            if image_path and Path(image_path).exists():
                if self.nap_server_address and self.nap_server_address != "localhost":
                    logger.info("📤 检测到远程服务器配置，开始文件传输...")

                    try:
                        remote_path = await asyncio.wait_for(
                            send_file(
                                image_path,
                                host=self.nap_server_address,
                                port=self.nap_server_port,
                            ),
                            timeout=10.0,
                        )
                        if remote_path:
                            image_path = remote_path
                    except asyncio.TimeoutError:
                        logger.warning("⚠️ 文件传输超时，使用本地文件")
                    except Exception as e:
                        logger.warning(f"⚠️ 文件传输失败: {e}，将使用本地文件")

                logger.info("📨 图像生成完成，准备返回结果...")
                return True, (image_path, text_content, thought_signature)
            else:
                error_msg = f"❌ 图像文件不存在或路径无效: {image_path}"
                logger.error(error_msg)
                return False, error_msg

        except APIError as e:
            error_msg = f"❌ 图像生成失败: {e.message}"
            if e.status_code == 429:
                error_msg += "\n💡 可能原因：API 速率限制或额度耗尽"
            elif e.status_code == 402:
                error_msg += "\n💡 可能原因：API 额度不足"
            elif e.status_code == 403:
                error_msg += "\n💡 可能原因：API 密钥无效或权限不足"
            logger.error(error_msg)
            return False, error_msg

        except Exception as e:
            logger.error(f"生成图像时发生未预期的错误: {e}", exc_info=True)
            return False, f"❌ 生成图像时发生错误: {str(e)}"

    async def _quick_generate_image(
        self,
        event: AstrMessageEvent,
        prompt: str,
        use_avatar: bool = False,
        skip_figure_enhance: bool = False,
    ):
        """快捷图像生成"""
        if not self.api_client:
            yield event.plain_result("❌ API 客户端未初始化")
            return

        try:
            ref_images = await self._collect_reference_images(event)
            self.log_debug(f"[MODIFY_DEBUG] 收集到 {len(ref_images)} 张参考图片")

            avatars = []
            if use_avatar:
                avatars = await self.get_avatar_reference(event)
                self.log_debug(f"[MODIFY_DEBUG] 收集到 {len(avatars)} 个头像")

            all_ref_images: list[str] = []
            all_ref_images.extend(
                self._filter_valid_reference_images(ref_images, source="消息图片")
            )
            all_ref_images.extend(
                self._filter_valid_reference_images(avatars, source="头像")
            )

            self.log_debug(f"[MODIFY_DEBUG] 有效参考图片总数: {len(all_ref_images)}")

            # 改图提示词增强 - 检测是否包含修改意图关键词
            modify_keywords = [
                "修改",
                "改图",
                "改成",
                "变成",
                "调整",
                "优化",
                "重做",
                "更换",
                "替换",
                "删除",
                "添加",
            ]
            is_modification_request = any(
                keyword in prompt for keyword in modify_keywords
            )
            self.log_debug(f"[MODIFY_DEBUG] 修改关键词匹配: {is_modification_request}")

            figure_keywords = ["手办", "figure", "模型", "手办化", "手办模型"]
            if (not skip_figure_enhance) and any(
                keyword in prompt.lower() for keyword in figure_keywords
            ):
                enhanced_prompt = enhance_prompt_for_figure(prompt)
                self.log_debug("[MODIFY_DEBUG] 使用手办化提示词增强")
            elif is_modification_request:
                # 对于改图请求，进一步强化提示词
                enhanced_prompt = get_auto_modification_prompt(prompt)
                self.log_debug("[MODIFY_DEBUG] 使用改图提示词增强")
            else:
                enhanced_prompt = prompt

            effective_resolution = self.resolution
            effective_aspect_ratio = self.aspect_ratio

            if (
                self.preserve_reference_image_size
                and is_modification_request
                and all_ref_images
            ):
                effective_resolution = None
                effective_aspect_ratio = None
                self.log_debug("[MODIFY_DEBUG] 保留参考图尺寸，不覆盖分辨率/比例")

            config = ApiRequestConfig(
                model=self.model,
                prompt=enhanced_prompt,
                api_type=self.api_type,
                api_base=self.api_base if self.api_base else None,
                resolution=effective_resolution,
                aspect_ratio=effective_aspect_ratio,
                enable_grounding=self.enable_grounding,
                reference_images=all_ref_images if all_ref_images else None,
                enable_smart_retry=self.enable_smart_retry,
                enable_text_response=self.enable_text_response,
            )

            # 记录改图请求的详细信息
            self.log_debug("[MODIFY_DEBUG] API请求配置:")
            self.log_debug(f"  - 提示词: {enhanced_prompt[:100]}...")
            self.log_debug(
                f"  - 参考图片数量: {len(all_ref_images) if all_ref_images else 0}"
            )
            self.log_debug(f"  - 是否改图请求: {is_modification_request}")
            self.log_debug(f"  - 模型: {self.model}")

            yield event.plain_result("🎨 生成中...")

            (
                image_url,
                image_path,
                text_content,
                thought_signature,
            ) = await self.api_client.generate_image(
                config=config,
                max_retries=self.max_attempts_per_key,
                per_retry_timeout=self.total_timeout,
                max_total_time=self.total_timeout * 2,
            )

            if image_url and image_path:
                logger.debug(
                    f"准备发送图像: image_path类型={type(image_path)}, 值={image_path}"
                )

                result_chain = []
                if text_content and self.enable_text_response:
                    cleaned_text = self._clean_text_content(text_content)
                    if cleaned_text:
                        result_chain.append(event.plain_result(f"📝 {cleaned_text}"))

                result_chain.append(event.image_result(image_path))

                for res in result_chain:
                    yield res

                if thought_signature:
                    logger.debug(f"🧠 思维签名: {thought_signature[:50]}...")
            else:
                yield event.plain_result("❌ 生成失败")

        except Exception as e:
            logger.error(f"快捷生成失败: {e}", exc_info=True)
            yield event.plain_result(f"❌ 错误: {str(e)}")
        finally:
            try:
                await self.avatar_manager.cleanup_used_avatars()
            except Exception as e:
                logger.warning(f"清理头像缓存失败: {e}")

    def _enhance_prompt_for_figure(self, prompt: str) -> str:
        """手办化提示词增强（已废弃，保留兼容性）"""
        return enhance_prompt_for_figure(prompt)

    @filter.command("生图")
    async def generate_image(self, event: AstrMessageEvent, prompt: str):
        """
        生图指令

        Args:
            prompt: 图像描述
        """
        allowed, limit_message = await self._check_and_consume_limit(event)
        if not allowed:
            if limit_message:
                yield event.plain_result(limit_message)
            return

        use_avatar = await self.should_use_avatar(event)

        generation_prompt = get_generation_prompt(prompt)

        yield event.plain_result("🎨 开始生成图像...")

        async for result in self._quick_generate_image(
            event, generation_prompt, use_avatar
        ):
            yield result

    async def _handle_quick_mode(
        self,
        event: AstrMessageEvent,
        prompt: str,
        resolution: str,
        aspect_ratio: str,
        mode_name: str,
        prompt_func: Any = None,
        **kwargs,
    ):
        """处理快速模式的通用逻辑"""
        allowed, limit_message = await self._check_and_consume_limit(event)
        if not allowed:
            if limit_message:
                yield event.plain_result(limit_message)
            return

        yield event.plain_result(f"🎨 使用{mode_name}模式生成图像...")

        old_resolution = self.resolution
        old_aspect_ratio = self.aspect_ratio

        try:
            self.resolution = resolution
            self.aspect_ratio = aspect_ratio

            # 使用新提示词函数
            if prompt_func:
                full_prompt = prompt_func(prompt)
            else:
                full_prompt = prompt

            use_avatar = await self.should_use_avatar(event)

            async for result in self._quick_generate_image(
                event, full_prompt, use_avatar, **kwargs
            ):
                yield result

        finally:
            self.resolution = old_resolution
            self.aspect_ratio = old_aspect_ratio

    @filter.command_group("快速")
    def quick_mode_group(self):
        """快速模式指令组"""
        pass

    @quick_mode_group.command("头像")
    async def quick_avatar(self, event: AstrMessageEvent, prompt: str):
        """头像快速模式 - 1K分辨率，1:1比例"""
        async for result in self._handle_quick_mode(
            event, prompt, "1K", "1:1", "头像", get_avatar_prompt
        ):
            yield result

    @quick_mode_group.command("海报")
    async def quick_poster(self, event: AstrMessageEvent, prompt: str):
        """海报快速模式 - 2K分辨率，16:9比例"""
        async for result in self._handle_quick_mode(
            event, prompt, "2K", "16:9", "海报", get_poster_prompt
        ):
            yield result

    @quick_mode_group.command("壁纸")
    async def quick_wallpaper(self, event: AstrMessageEvent, prompt: str):
        """壁纸快速模式 - 4K分辨率，16:9比例"""
        async for result in self._handle_quick_mode(
            event, prompt, "4K", "16:9", "壁纸", get_wallpaper_prompt
        ):
            yield result

    @quick_mode_group.command("卡片")
    async def quick_card(self, event: AstrMessageEvent, prompt: str):
        """卡片快速模式 - 1K分辨率，3:2比例"""
        async for result in self._handle_quick_mode(
            event, prompt, "1K", "3:2", "卡片", get_card_prompt
        ):
            yield result

    @quick_mode_group.command("手机")
    async def quick_mobile(self, event: AstrMessageEvent, prompt: str):
        """手机快速模式 - 2K分辨率，9:16比例"""
        async for result in self._handle_quick_mode(
            event, prompt, "2K", "9:16", "手机", get_mobile_prompt
        ):
            yield result

    @quick_mode_group.command("手办化")
    async def quick_figure(self, event: AstrMessageEvent, prompt: str):
        """手办化快速模式 - 树脂收藏级手办效果"""
        # 解析参数
        style_type = 1
        clean_prompt = prompt

        if prompt:
            p_lower = prompt.lower()
            if p_lower.startswith("1") or "pvc" in p_lower:
                style_type = 1
                clean_prompt = prompt.replace("1", "", 1).replace("pvc", "", 1).strip()
            elif p_lower.startswith("2") or "gk" in p_lower:
                style_type = 2
                clean_prompt = prompt.replace("2", "", 1).replace("gk", "", 1).strip()

        full_prompt = get_figure_prompt(clean_prompt, style_type)

        async for result in self._handle_quick_mode(
            event,
            full_prompt,
            "2K",
            "3:2",
            "手办化",
            None,
            skip_figure_enhance=True,
        ):
            yield result

    @quick_mode_group.command("表情包")
    async def quick_sticker(self, event: AstrMessageEvent, prompt: str = ""):
        """表情包快速模式 - 4K分辨率，16:9比例，Q版LINE风格

        功能受配置文件控制：
        - enable_sticker_split: 是否自动切割图片
        - enable_sticker_zip: 是否打包发送（如果发送失败则使用合并转发）
        """
        allowed, limit_message = await self._check_and_consume_limit(event)
        if not allowed:
            if limit_message:
                yield event.plain_result(limit_message)
            return

        yield event.plain_result("🎨 使用表情包模式生成图像...")

        # 检查是否包含参考图
        reference_images = await self._collect_reference_images(event)
        if not reference_images:
            yield event.plain_result("❌ 表情包模式需要参考图，请至少附带一张图片作为角色参考。")
            return

        # 如果没有开启切割功能，直接使用默认逻辑
        if not self.enable_sticker_split:
            full_prompt = get_sticker_prompt(prompt)
            old_resolution = self.resolution
            old_aspect_ratio = self.aspect_ratio

            try:
                self.resolution = "4K"
                self.aspect_ratio = "16:9"
                use_avatar = await self.should_use_avatar(event)
                async for result in self._quick_generate_image(
                    event, full_prompt, use_avatar
                ):
                    yield result
            finally:
                self.resolution = old_resolution
                self.aspect_ratio = old_aspect_ratio
            return

        # 开启了切割功能，执行自定义逻辑
        full_prompt = get_sticker_prompt(prompt)
        old_resolution = self.resolution
        old_aspect_ratio = self.aspect_ratio

        try:
            self.resolution = "4K"
            self.aspect_ratio = "16:9"

            use_avatar = await self.should_use_avatar(event)

            # 调用生图核心逻辑，但截获结果不直接发送
            reference_images = await self._collect_reference_images(event)
            avatar_reference = []
            if use_avatar:
                avatar_reference = await self.get_avatar_reference(event)

            sent_success = False
            split_files: list[str] = []
            image_path = None

            success, result_data = await self._generate_image_core_internal(
                event=event,
                prompt=full_prompt,
                reference_images=reference_images,
                avatar_reference=avatar_reference,
            )

            if not success or not isinstance(result_data, tuple):
                error_msg = result_data if isinstance(result_data, str) else "❌ 图像生成失败，请稍后重试"
                yield event.plain_result(error_msg)
                return

            image_path, text_content, thought_signature = result_data

            # 1. 切割图片
            yield event.plain_result("✂️ 正在切割图片...")
            try:
                split_files = await asyncio.to_thread(
                    split_image, image_path, rows=6, cols=4
                )
            except Exception as e:
                logger.error(f"切割图片时发生异常: {e}")
                split_files = []

            if not split_files:
                yield event.plain_result("❌ 图片切割失败")
                yield event.image_result(image_path)
                return

            # 2. 准备发送逻辑

            # 如果开启了ZIP，优先尝试发送ZIP
            if self.enable_sticker_zip:
                zip_path = await asyncio.to_thread(create_zip, split_files)
                if zip_path:
                    try:
                        from astrbot.api.message_components import File

                        file_comp = File(
                            file=zip_path, name=os.path.basename(zip_path)
                        )
                        yield event.chain_result([file_comp])
                        sent_success = True

                        yield event.image_result(image_path)
                    except Exception as e:
                        logger.warning(f"发送ZIP失败: {e}")
                        yield event.plain_result(
                            "⚠️ 压缩包发送失败，降级使用合并转发"
                        )
                        sent_success = False
                else:
                    yield event.plain_result("❌ 压缩包创建失败，降级使用合并转发")
                    sent_success = False

            # 3. 如果没开启ZIP或者ZIP发送失败，发送合并转发
            if not sent_success:
                from astrbot.api.message_components import Image as AstrImage
                from astrbot.api.message_components import Node, Plain

                # 构造节点内容：原图 + 所有小图
                node_content = []
                node_content.append(Plain("原图预览：\n"))
                node_content.append(AstrImage.fromFileSystem(image_path))
                node_content.append(Plain("\n\n表情包切片：\n"))

                for file_path in split_files:
                    node_content.append(AstrImage.fromFileSystem(file_path))

                # 构造单个节点，包含所有图片
                node = Node(
                    uin=event.message_obj.self_id,
                    name="Gemini表情包生成",
                    content=node_content,
                )

                yield event.chain_result([node])

        finally:
            self.resolution = old_resolution
            self.aspect_ratio = old_aspect_ratio
            try:
                await self.avatar_manager.cleanup_used_avatars()
            except Exception:
                pass

    @filter.command("生图帮助")
    async def show_help(self, event: AstrMessageEvent):
        """显示插件使用帮助"""
        group_id = self._get_group_id_from_event(event)
        if group_id and self.group_limit_list:
            if (
                self.group_limit_mode == "blacklist"
                and group_id in self.group_limit_list
            ):
                return
            if (
                self.group_limit_mode == "whitelist"
                and group_id not in self.group_limit_list
            ):
                return

        grounding_status = "✓ 启用" if self.enable_grounding else "✗ 禁用"
        smart_retry_status = "✓ 启用" if self.enable_smart_retry else "✗ 禁用"
        avatar_status = "✓ 启用" if self.auto_avatar_reference else "✗ 禁用"

        limit_settings = self.config.get("limit_settings", {})
        enable_rate_limit = limit_settings.get("enable_rate_limit", False)
        rate_limit_period = limit_settings.get("rate_limit_period", 60)
        max_requests = limit_settings.get("max_requests_per_group", 5)
        rate_limit_status = (
            f"✓ {max_requests}次/{rate_limit_period}秒"
            if enable_rate_limit
            else "✗ 禁用"
        )

        tool_timeout = self.get_tool_timeout(event)
        timeout_warning = ""
        if tool_timeout < 90:
            timeout_warning = (
                f"⚠️ LLM工具超时时间较短({tool_timeout}秒)，建议设置为90-120秒"
            )

        try:
            metadata_path = os.path.join(os.path.dirname(__file__), "metadata.yaml")
            with open(metadata_path, encoding="utf-8") as f:
                metadata = yaml.safe_load(f)
                version = metadata.get("version", "v1.3.0")
        except Exception:
            version = "v1.3.0"

        try:
            # 获取主题配置
            service_settings = self.config.get("service_settings", {})
            theme_settings = service_settings.get("theme_settings", {})

            # 解析配置
            mode = theme_settings.get("mode", "cycle")
            cycle_config = theme_settings.get("cycle_config", {})
            single_config = theme_settings.get("single_config", {})

            # 确定要使用的模板文件名
            template_filename = "help_template_light"  # 默认值

            if mode == "single":
                # 单独模式
                template_filename = single_config.get(
                    "template_name", "help_template_light"
                )
            else:
                # 循环模式 (默认)
                day_start = cycle_config.get("day_start", 6)
                day_end = cycle_config.get("day_end", 18)
                day_template = cycle_config.get("day_template", "help_template_light")
                night_template = cycle_config.get(
                    "night_template", "help_template_dark"
                )

                current_hour = datetime.now().hour
                if day_start <= current_hour < day_end:
                    template_filename = day_template
                else:
                    template_filename = night_template

            # 自动补全 .html 后缀
            if not template_filename.endswith(".html"):
                template_filename += ".html"

            # 构建模板路径
            template_path = os.path.join(
                os.path.dirname(__file__), "templates", template_filename
            )

            # 检查文件是否存在，不存在则回退
            if not os.path.exists(template_path):
                logger.warning(f"模板文件不存在: {template_path}，将回退到默认模板")
                template_filename = "help_template_light.html"
                template_path = os.path.join(
                    os.path.dirname(__file__), "templates", template_filename
                )

                # 如果默认模板也不存在（极端情况），抛出异常让外层处理
                if not os.path.exists(template_path):
                    raise FileNotFoundError(f"找不到模板文件: {template_path}")

            # 准备模板数据
            template_data = {
                "title": f"Gemini 图像生成插件 {version}",
                # 以下字段是为了兼容可能使用了旧变量的模板，虽然新设计应该由css控制
                "model": self.model,
                "api_type": self.api_type,
                "resolution": self.resolution,
                "aspect_ratio": self.aspect_ratio or "默认",
                "api_keys_count": len(self.api_keys),
                "grounding_status": grounding_status,
                "avatar_status": avatar_status,
                "smart_retry_status": smart_retry_status,
                "tool_timeout": tool_timeout,
                "rate_limit_status": rate_limit_status,
                "timeout_warning": timeout_warning if timeout_warning else "",
                "enable_sticker_split": self.enable_sticker_split,
            }

            # 读取模板文件
            with open(template_path, encoding="utf-8") as f:
                jinja2_template = f.read()

            # 使用AstrBot的html_render方法
            html_image_url = await self.html_render(jinja2_template, template_data)
            logger.info(f"HTML帮助图片生成成功 (使用模板: {template_filename})")
            yield event.image_result(html_image_url)

        except Exception as e:
            logger.error(f"HTML帮助图片生成失败: {e}")
            fallback_help = f"""🎨 Gemini 图像生成插件 {version}

基础指令:
• /生图 [描述] - 生成图像
• /快速 [预设] [描述] - 快速模式
• /改图 [描述] - 修改图像
• /换风格 [风格] - 风格转换
• /生图帮助 - 显示帮助

预设选项: 头像/海报/壁纸/卡片/手机/手办化

当前配置:
• 模型: {self.model}
• 分辨率: {self.resolution}
• API密钥: {len(self.api_keys)}个
• LLM工具超时: {tool_timeout}秒

系统状态:
• 搜索接地: {grounding_status}
• 自动头像: {avatar_status}
• 智能重试: {smart_retry_status}

⚠️ HTML渲染失败，使用文本模式显示

错误信息: {str(e)}"""
            yield event.plain_result(fallback_help)

    @filter.command("改图")
    async def modify_image(self, event: AstrMessageEvent, prompt: str):
        """
        根据提示词修改或重做图像（默认命令）

        Args:
            prompt: 修改描述，如"把头发改成红色"、"换个背景"、"画成动漫风格"等
        """
        allowed, limit_message = await self._check_and_consume_limit(event)
        if not allowed:
            if limit_message:
                yield event.plain_result(limit_message)
            return

        # 构造改图专用提示词，确保修改意图明确
        modification_prompt = get_modification_prompt(prompt)

        yield event.plain_result("🎨 开始修改图像...")

        # 根据配置决定是否使用头像参考
        use_avatar = await self.should_use_avatar(event)

        async for result in self._quick_generate_image(
            event, modification_prompt, use_avatar
        ):
            yield result

    @filter.command("换风格")
    async def change_style(self, event: AstrMessageEvent, style: str, prompt: str = ""):
        """
        改变图像风格

        Args:
            style: 风格描述，如"动漫"、"写实"、"水彩"、"油画"等
            prompt: 额外的修改要求（可选）
        """
        allowed, limit_message = await self._check_and_consume_limit(event)
        if not allowed:
            if limit_message:
                yield event.plain_result(limit_message)
            return

        full_prompt = get_style_change_prompt(style, prompt)

        reference_images = await self._collect_reference_images(event)

        # 根据配置决定是否使用头像参考
        avatar_reference = []
        if await self.should_use_avatar(event):
            avatar_reference = await self.get_avatar_reference(event)

        yield event.plain_result("🎨 开始转换风格...")

        success, result_data = await self._generate_image_core_internal(
            event=event,
            prompt=full_prompt,
            reference_images=reference_images,
            avatar_reference=avatar_reference,
            is_modification=True,
        )

        if success and result_data:
            image_path, text_content, thought_signature = result_data

            result_chain = []
            if text_content and self.enable_text_response:
                cleaned_text = self._clean_text_content(text_content)
                if cleaned_text:
                    result_chain.append(event.plain_result(f"📝 {cleaned_text}"))

            result_chain.append(event.image_result(image_path))

            for res in result_chain:
                yield res

            if thought_signature:
                logger.debug(f"🧠 思维签名: {thought_signature[:50]}...")
        else:
            yield event.plain_result(result_data)

    @filter.llm_tool(name="gemini_image_generation")
    async def generate_image_tool(
        self,
        event: AstrMessageEvent,
        prompt: str,
        use_reference_images: str,
        include_user_avatar: str = "false",
        **kwargs,
    ):
        """
        使用 Gemini 模型生成或修改图像

        当用户请求图像生成、绘画、改图、换风格或手办化时调用此函数。

        判断逻辑：
        - 用户说"改成"、"变成"、"基于"、"修改"、"改图"等词时，设置 use_reference_images="true"
        - 用户说"根据我"、"我的头像"或@某人时，设置 use_reference_images="true" 和 include_user_avatar="true"
        - 用户消息中包含图片且明确要求"修改这张图"时，设置 use_reference_images="true"

        Args:
            prompt(string): 图像生成或修改的详细描述
            use_reference_images(string): 是否使用上下文中的参考图片，true或false。当用户意图是修改、变换或基于现有图片时设置为true
            include_user_avatar(string): 是否包含用户头像作为参考图像，true或false。当用户说"根据我"、"我的头像"或@某人时设置为true
        """
        allowed, limit_message = await self._check_and_consume_limit(event)
        if not allowed:
            if limit_message:
                yield event.plain_result(limit_message)
            return

        if not self.api_client:
            yield event.plain_result(
                "❌ 错误: API 客户端未初始化，请联系管理员配置 API 密钥"
            )
            return

        reference_images = []
        is_modification_request = False
        if str(use_reference_images).lower() in {"true", "1", "yes", "y", "是"}:
            reference_images = await self._collect_reference_images(event)
            is_modification_request = bool(reference_images)

        avatar_reference = []

        avatar_value = str(include_user_avatar).lower()
        logger.info(f"[AVATAR_DEBUG] include_user_avatar参数: {avatar_value}")

        if avatar_value in {"true", "1", "yes", "y", "是"}:
            logger.info("[AVATAR_DEBUG] Gemini API建议获取头像，开始获取...")
            try:
                avatar_reference = await self.get_avatar_reference(event)
                logger.info(
                    f"[AVATAR_DEBUG] 头像获取完成，返回结果: {len(avatar_reference) if avatar_reference else 0} 个"
                )
            except Exception as e:
                logger.error(f"头像获取失败: {e}", exc_info=True)
                avatar_reference = []

            if avatar_reference:
                logger.info(f"成功获取 {len(avatar_reference)} 个头像作为参考图像")
                for i, avatar in enumerate(avatar_reference):
                    logger.info(f"  - 头像{i + 1}: {avatar[:50]}...")
            else:
                logger.info("未能获取头像，继续使用其他参考图像或纯文本生成")
        else:
            logger.info("[AVATAR_DEBUG] Gemini API未建议获取头像，跳过头像获取")

        success, result_data = await self._generate_image_core_internal(
            event=event,
            prompt=prompt,
            reference_images=reference_images,
            avatar_reference=avatar_reference,
            is_modification=is_modification_request,
        )

        try:
            await self.avatar_manager.cleanup_cache()
        except Exception as e:
            logger.warning(f"清理头像缓存失败: {e}")

        if success and result_data:
            image_path, text_content, thought_signature = result_data

            result_chain = []
            if text_content and self.enable_text_response:
                cleaned_text = self._clean_text_content(text_content)
                if cleaned_text:
                    result_chain.append(event.plain_result(cleaned_text))

            result_chain.append(event.image_result(image_path))

            for res in result_chain:
                yield res

            if thought_signature:
                logger.debug(f"🧠 思维签名: {thought_signature[:50]}...")
        else:
            yield event.plain_result(result_data)

