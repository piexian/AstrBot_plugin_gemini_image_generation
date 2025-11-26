"""
AstrBot Gemini 图像生成插件主文件
支持 Google 官方 API 和 OpenAI 兼容格式 API，提供生图和改图功能，支持智能头像参考
"""

from __future__ import annotations

import asyncio
import base64
import os
import time
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.api.all import Image, Reply
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

from .tl.enhanced_prompts import enhance_prompt_for_figure
from .tl.tl_api import (
    APIClient,
    APIError,
    ApiRequestConfig,
    get_api_client,
)
from .tl.tl_utils import AvatarManager, download_qq_avatar, send_file


@register(
    "astrbot_plugin_gemini_image_generation",
    "piexian",
    "Gemini图像生成插件，支持生图和改图，可以自动获取头像作为参考",
    "v1.4.1",
)
class GeminiImageGenerationPlugin(Star):
    def __init__(self, context: Context, config: dict[str, Any]):
        super().__init__(context)
        self.config = config
        self.api_client: APIClient | None = None
        self.avatar_manager = AvatarManager()

        # 加载配置
        self._load_config()

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
                    import aiohttp

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
                logger.warning(f"转换图片为base64失败: {e}")
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
    ) -> tuple[bool, tuple[str, str, str | None] | str]:
        """
        内部核心图像生成方法，不发送消息，只返回结果

        Returns:
            tuple[bool, tuple[str, str, str | None] | str]: (是否成功, (图片路径, 文本内容, 思维签名) 或错误消息)
        """
        if not self.api_client:
            return False, "❌ 错误: API 客户端未初始化，请联系管理员配置 API 密钥"

        all_reference_images: list[str] = []
        all_reference_images.extend(
            self._filter_valid_reference_images(reference_images, source="消息图片")
        )
        all_reference_images.extend(
            self._filter_valid_reference_images(avatar_reference, source="头像")
        )

        if (
            all_reference_images
            and len(all_reference_images) > self.max_reference_images
        ):
            logger.warning(
                f"参考图片数量 ({len(all_reference_images)}) 超过限制 ({self.max_reference_images})，将截取前 {self.max_reference_images} 张"
            )
            all_reference_images = all_reference_images[: self.max_reference_images]

        response_modalities = "TEXT_IMAGE" if self.enable_text_response else "IMAGE"
        request_config = ApiRequestConfig(
            model=self.model,
            prompt=prompt,
            api_type=self.api_type,
            api_base=self.api_base,
            resolution=self.resolution,
            aspect_ratio=self.aspect_ratio,
            enable_grounding=self.enable_grounding,
            response_modalities=response_modalities,
            reference_images=all_reference_images if all_reference_images else None,
            enable_smart_retry=self.enable_smart_retry,
            enable_text_response=self.enable_text_response,
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
                enhanced_prompt = f"""图像修改任务：{prompt}

请严格按照用户要求修改参考图像，确保：
1. 必须基于提供的参考图像进行修改
2. 保持主要对象和构图，只修改用户要求的部分
3. 修改后的图像要与原图有明显区别
4. 不要返回完全相同的原图
5. 修改要自然、合理，保持图像质量

重要：这是一项图像修改任务，不是生成新图像，必须基于参考图像进行修改！"""
                self.log_debug("[MODIFY_DEBUG] 使用改图提示词增强")
            else:
                enhanced_prompt = prompt

            config = ApiRequestConfig(
                model=self.model,
                prompt=enhanced_prompt,
                api_type=self.api_type,
                api_base=self.api_base if self.api_base else None,
                resolution=self.resolution,
                aspect_ratio=self.aspect_ratio,
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

                if text_content and self.enable_text_response:
                    cleaned_text = self._clean_text_content(text_content)
                    if cleaned_text:
                        yield event.plain_result(f"📝 {cleaned_text}")

                yield event.image_result(image_path)

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
        """手办化提示词增强"""
        figure_keywords = ["手办", "figure", "模型", "手办化", "手办模型"]
        if any(keyword in prompt.lower() for keyword in figure_keywords):
            return f"""请将此照片中的主要对象精确转换为写实的、杰作级别的 1/7 比例 PVC 手办。
在手办旁边应放置一个盒子：盒子正面应有一个大型清晰的透明窗口，印有主要艺术作品、产品名称、品牌标志、条形码，以及一个小规格或真伪验证面板。盒子的角落还必须贴有小价签。同时，在后方放置一个电脑显示器，显示器屏幕需要显示该手办的 ZBrush 建模过程。
在包装盒前方，手办应放置在圆形塑料底座上。手办必须有 3D 立体感和真实感，PVC 材质的纹理需要清晰表现。

{prompt}

质量要求：
- 修复任何缺失部分时，必须没有执行不佳的元素
- 人体部位必须自然，动作必须协调，所有部位比例必须合理
- 如果原始照片不是全身照，请尝试补充手办使其成为全身版本
- 人物表情和动作必须与照片完全一致
- 手办头部不应显得太大，腿部不应显得太短，手办不应看起来矮胖（除非明确是Q版设计）
- 对于动物手办，应减少毛发的真实感和细节层次，使其更像手办而不是真实的原始生物
- 不应有外轮廓线，手办绝不能是平面的
- 注意近大远小的透视关系"""

        return prompt

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

        # 构造生图专用提示词，确保生成意图明确
        generation_prompt = f"""图像生成任务：{prompt}

重要要求：
- 根据用户的描述生成全新的原创图像
- 生成图像要完全符合用户的描述要求

重要：这是一项图像生成任务，请根据描述创建全新的图像！"""

        yield event.plain_result("🎨 开始生成图像...")

        async for result in self._quick_generate_image(
            event, generation_prompt, use_avatar
        ):
            yield result

    @filter.command_group("快速")
    def quick_mode_group(self):
        """快速模式指令组"""
        pass

    @quick_mode_group.command("头像")
    async def quick_avatar(self, event: AstrMessageEvent, prompt: str):
        """头像快速模式 - 1K分辨率，1:1比例"""
        allowed, limit_message = await self._check_and_consume_limit(event)
        if not allowed:
            if limit_message:
                yield event.plain_result(limit_message)
            return

        yield event.plain_result("🎨 使用头像模式生成图像...")

        old_resolution = self.resolution
        old_aspect_ratio = self.aspect_ratio

        try:
            self.resolution = "1K"
            self.aspect_ratio = "1:1"

            use_avatar = await self.should_use_avatar(event)

            async for result in self._quick_generate_image(event, prompt, use_avatar):
                yield result

        finally:
            self.resolution = old_resolution
            self.aspect_ratio = old_aspect_ratio

    @quick_mode_group.command("海报")
    async def quick_poster(self, event: AstrMessageEvent, prompt: str):
        """海报快速模式 - 2K分辨率，16:9比例"""
        allowed, limit_message = await self._check_and_consume_limit(event)
        if not allowed:
            if limit_message:
                yield event.plain_result(limit_message)
            return

        yield event.plain_result("🎨 使用海报模式生成图像...")

        old_resolution = self.resolution
        old_aspect_ratio = self.aspect_ratio

        try:
            self.resolution = "2K"
            self.aspect_ratio = "16:9"

            use_avatar = await self.should_use_avatar(event)

            async for result in self._quick_generate_image(event, prompt, use_avatar):
                yield result

        finally:
            self.resolution = old_resolution
            self.aspect_ratio = old_aspect_ratio

    @quick_mode_group.command("壁纸")
    async def quick_wallpaper(self, event: AstrMessageEvent, prompt: str):
        """壁纸快速模式 - 4K分辨率，16:9比例"""
        allowed, limit_message = await self._check_and_consume_limit(event)
        if not allowed:
            if limit_message:
                yield event.plain_result(limit_message)
            return

        yield event.plain_result("🎨 使用壁纸模式生成图像...")

        old_resolution = self.resolution
        old_aspect_ratio = self.aspect_ratio

        try:
            self.resolution = "4K"
            self.aspect_ratio = "16:9"

            use_avatar = await self.should_use_avatar(event)

            async for result in self._quick_generate_image(event, prompt, use_avatar):
                yield result

        finally:
            self.resolution = old_resolution
            self.aspect_ratio = old_aspect_ratio

    @quick_mode_group.command("卡片")
    async def quick_card(self, event: AstrMessageEvent, prompt: str):
        """卡片快速模式 - 1K分辨率，3:2比例"""
        allowed, limit_message = await self._check_and_consume_limit(event)
        if not allowed:
            if limit_message:
                yield event.plain_result(limit_message)
            return

        yield event.plain_result("🎨 使用卡片模式生成图像...")

        old_resolution = self.resolution
        old_aspect_ratio = self.aspect_ratio

        try:
            self.resolution = "1K"
            self.aspect_ratio = "3:2"

            use_avatar = await self.should_use_avatar(event)

            async for result in self._quick_generate_image(event, prompt, use_avatar):
                yield result

        finally:
            self.resolution = old_resolution
            self.aspect_ratio = old_aspect_ratio

    @quick_mode_group.command("手机")
    async def quick_mobile(self, event: AstrMessageEvent, prompt: str):
        """手机快速模式 - 2K分辨率，9:16比例"""
        allowed, limit_message = await self._check_and_consume_limit(event)
        if not allowed:
            if limit_message:
                yield event.plain_result(limit_message)
            return

        yield event.plain_result("🎨 使用手机模式生成图像...")

        old_resolution = self.resolution
        old_aspect_ratio = self.aspect_ratio

        try:
            self.resolution = "2K"
            self.aspect_ratio = "9:16"

            use_avatar = await self.should_use_avatar(event)

            async for result in self._quick_generate_image(event, prompt, use_avatar):
                yield result

        finally:
            self.resolution = old_resolution
            self.aspect_ratio = old_aspect_ratio

    @quick_mode_group.command("手办化")
    async def quick_figure(self, event: AstrMessageEvent, prompt: str):
        """手办化快速模式 - 树脂收藏级手办效果"""
        allowed, limit_message = await self._check_and_consume_limit(event)
        if not allowed:
            if limit_message:
                yield event.plain_result(limit_message)
            return

        yield event.plain_result("🎨 使用手办化模式生成图像...")

        base_prompt = "将画面中的角色重塑为顶级收藏级树脂手办，全身动态姿势，置于角色主题底座，高精度材质，手工涂装，肌肤纹理与服装材质真实分明。戏剧性硬光为主光源，凸显立体感，无过曝；强效补光消除死黑，细节完整可见。背景为窗边景深模糊，侧后方隐约可见产品包装盒。博物馆级摄影质感，全身细节无损，面部结构精准。禁止：任何2D元素或照搬原图、塑料感、面部模糊、五官错位、细节丢失。"
        full_prompt = base_prompt if not prompt else f"{base_prompt}\n{prompt}"

        old_resolution = self.resolution
        old_aspect_ratio = self.aspect_ratio

        try:
            self.resolution = "2K"
            self.aspect_ratio = "3:2"

            use_avatar = await self.should_use_avatar(event)

            async for result in self._quick_generate_image(
                event, full_prompt, use_avatar, True
            ):
                yield result
        finally:
            self.resolution = old_resolution
            self.aspect_ratio = old_aspect_ratio

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
            import yaml

            metadata_path = os.path.join(os.path.dirname(__file__), "metadata.yaml")
            with open(metadata_path, encoding="utf-8") as f:
                metadata = yaml.safe_load(f)
                version = metadata.get("version", "v1.3.0")
        except Exception:
            version = "v1.3.0"

        try:
            from datetime import datetime

            # 获取主题配置
            service_settings = self.config.get("service_settings", {})
            theme_settings = service_settings.get("theme_settings", {})
            enable_auto_theme = theme_settings.get("enable_auto_theme", True)
            day_theme_start = theme_settings.get("day_theme_start", 6)
            day_theme_end = theme_settings.get("day_theme_end", 18)
            manual_theme = theme_settings.get("manual_theme", "light")

            # 判断当前应该使用的主题
            if enable_auto_theme:
                # 自动主题切换
                current_hour = datetime.now().hour
                is_daytime = day_theme_start <= current_hour < day_theme_end
                use_light_theme = is_daytime
            else:
                # 手动指定主题
                use_light_theme = manual_theme == "light"

            if use_light_theme:
                # 白色主题配置
                template_data = {
                    "title": f"Gemini 图像生成插件 {version}",
                    "background_color": "#E6F3FF",
                    "text_color": "#1a5490",
                    "container_bg": "rgba(255, 255, 255, 0.95)",
                    "border_color": "#4a90e2",
                    "box_shadow": "0 0 20px rgba(74, 144, 226, 0.3)",
                    "scanline_color": "rgba(74, 144, 226, 0)",
                    "header_color": "#2c5aa0",
                    "header_shadow": "rgba(44, 90, 160, 0.3)",
                    "version_color": "#4a90e2",
                    "section_border": "#4a90e2",
                    "section_title_color": "#2c5aa0",
                    "section_title_shadow": "rgba(44, 90, 160, 0.2)",
                    "status_text_color": "#2c5aa0",
                    "status_ok_color": "#28a745",
                    "status_warning_color": "#ffc107",
                    "status_error_color": "#dc3545",
                    "command_border": "rgba(74, 144, 226, 0.2)",
                    "command_name_color": "#2c5aa0",
                    "command_desc_color": "#1a5490",
                    "example_color": "#6c757d",
                    "feature_text_color": "#1a5490",
                    "feature_icon_color": "#4a90e2",
                    "tip_text_color": "#2c5aa0",
                    "tip_icon_color": "#6f42c1",
                    "warning_text_color": "#856404",
                    "warning_bg_color": "#fff3cd",
                    "warning_border_color": "#ffeaa7",
                    "scanlines_enabled": True,
                    "pulse_enabled": True,
                    "flicker_enabled": True,
                }
            else:
                # 黑色主题配置
                template_data = {
                    "title": f"Gemini 图像生成插件 {version}",
                    "background_color": "#1a1a1a",
                    "text_color": "#e0e0e0",
                    "container_bg": "rgba(40, 40, 40, 0.95)",
                    "border_color": "#00bcd4",
                    "box_shadow": "0 0 20px rgba(0, 188, 212, 0.3)",
                    "scanline_color": "rgba(0, 188, 212, 0)",
                    "header_color": "#00bcd4",
                    "header_shadow": "rgba(0, 188, 212, 0.3)",
                    "version_color": "#00acc1",
                    "section_border": "#00bcd4",
                    "section_title_color": "#00bcd4",
                    "section_title_shadow": "rgba(0, 188, 212, 0.2)",
                    "status_text_color": "#e0e0e0",
                    "status_ok_color": "#4caf50",
                    "status_warning_color": "#ff9800",
                    "status_error_color": "#f44336",
                    "command_border": "rgba(0, 188, 212, 0.2)",
                    "command_name_color": "#00bcd4",
                    "command_desc_color": "#b0bec5",
                    "example_color": "#757575",
                    "feature_text_color": "#b0bec5",
                    "feature_icon_color": "#00bcd4",
                    "tip_text_color": "#e0e0e0",
                    "tip_icon_color": "#9c27b0",
                    "warning_text_color": "#ff9800",
                    "warning_bg_color": "#4e342e",
                    "warning_border_color": "#ff6f00",
                    "scanlines_enabled": True,
                    "pulse_enabled": True,
                    "flicker_enabled": True,
                }

            # 添加通用数据
            template_data.update(
                {
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
                }
            )

            # 读取模板文件
            template_path = os.path.join(
                os.path.dirname(__file__), "templates", "help_template.html"
            )
            with open(template_path, encoding="utf-8") as f:
                jinja2_template = f.read()

            # 使用AstrBot的html_render方法
            html_image_url = await self.html_render(jinja2_template, template_data)
            logger.info("HTML帮助图片生成成功")
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
        modification_prompt = f"""请根据参考图像进行以下修改：{prompt}

重要要求：
- 必须基于提供的参考图像进行修改，不能忽略原图
- 保持图像的整体构图和主要对象
- 严格按照用户要求进行修改，不要返回原图
- 如果修改涉及颜色、风格或背景，必须有明显变化
- 确保修改后的图像与原图有可区分的差异"""

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

        full_prompt = f"将参考图像改为{style}风格"
        if prompt:
            full_prompt += f"，{prompt}"

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
        )

        if success and result_data:
            image_path, text_content, thought_signature = result_data

            if text_content and self.enable_text_response:
                cleaned_text = self._clean_text_content(text_content)
                if cleaned_text:
                    yield event.plain_result(f"📝 {cleaned_text}")

            yield event.image_result(image_path)

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
        if str(use_reference_images).lower() in {"true", "1", "yes", "y", "是"}:
            reference_images = await self._collect_reference_images(event)

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
        )

        try:
            await self.avatar_manager.cleanup_cache()
        except Exception as e:
            logger.warning(f"清理头像缓存失败: {e}")

        if success and result_data:
            image_path, text_content, thought_signature = result_data

            if text_content and self.enable_text_response:
                cleaned_text = self._clean_text_content(text_content)
                if cleaned_text:
                    yield event.plain_result(cleaned_text)

            yield event.image_result(image_path)

            if thought_signature:
                logger.debug(f"🧠 思维签名: {thought_signature[:50]}...")
        else:
            yield event.plain_result(result_data)

    async def terminate(self):
        """插件卸载时清理资源"""
        logger.info("🎨 Gemini 图像生成插件已卸载")
