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
from astrbot.api.message_components import Plain, Image as ImageComponent
from astrbot.api.star import Context, Star, register

from .tl.tl_api import (
    APIClient,
    APIError,
    ApiRequestConfig,
    get_api_client,
)
from .tl.tl_utils import AvatarManager, download_qq_avatar, send_file
from .tl.enhanced_prompts import enhance_prompt_for_figure


@register(
    "astrbot_plugin_gemini_image_generation",
    "piexian",
    "Gemini图像生成插件，支持生图和改图，可以自动获取头像作为参考",
    "v1.1.0",
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
            logger.warning(f"获取 tool_call_timeout 配置失败: {e}，使用默认值 b'y'g 秒")
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
        # API 密钥列表
        self.api_keys = self.config.get("openrouter_api_keys", [])
        if not isinstance(self.api_keys, list):
            self.api_keys = [self.api_keys] if self.api_keys else []

        # API 设置
        api_settings = self.config.get("api_settings", {})
        self.api_type = api_settings.get("api_type", "google")
        self.api_base = api_settings.get("custom_api_base", "")
        self.model = api_settings.get("model", "gemini-3-pro-image-preview")

        # 图像生成设置
        image_settings = self.config.get("image_generation_settings", {})
        self.resolution = image_settings.get("resolution", "1K")
        self.aspect_ratio = image_settings.get("aspect_ratio", "1:1")
        self.enable_grounding = image_settings.get("enable_grounding", False)
        self.max_reference_images = image_settings.get("max_reference_images", 6)
        self.enable_text_response = image_settings.get("enable_text_response", False)

        # 重试设置
        retry_settings = self.config.get("retry_settings", {})
        self.max_attempts_per_key = retry_settings.get("max_attempts_per_key", 3)
        self.enable_smart_retry = retry_settings.get("enable_smart_retry", True)
        self.total_timeout = retry_settings.get("total_timeout", 120)

        # 服务设置
        service_settings = self.config.get("service_settings", {})
        self.nap_server_address = service_settings.get(
            "nap_server_address", "localhost"
        )
        self.nap_server_port = service_settings.get("nap_server_port", 3658)
        self.auto_avatar_reference = service_settings.get(
            "auto_avatar_reference", False
        )

        # 日志设置
        self.verbose_logging = service_settings.get("verbose_logging", False)

        # 限制/限流设置
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

        # 初始化 API 客户端
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

        # 已经是 data URL 的情况，要求包含 ';base64,'
        if value.startswith("data:image/"):
            return ";base64," in value

        # 纯 base64 数据：尝试严格解码
        try:
            base64.b64decode(value, validate=True)
            return True
        except Exception:
            return False

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
                # 只打一点点前缀，避免日志过长
                self.log_debug(
                    f"跳过非 base64 格式参考图像({source}): {img[:64]}..."
                )

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

        # 仅对群聊应用黑/白名单与限流，私聊不做限制
        if not group_id:
            return True, None

        # 群限制模式：None / Whitelist / Blacklist
        if self.group_limit_mode == "whitelist":
            # 仅允许列表内的群可用（静默处理未在白名单中的群）
            if self.group_limit_list and group_id not in self.group_limit_list:
                return False, None
        elif self.group_limit_mode == "blacklist":
            # 禁止列表内的群使用（静默处理，不返回任何消息）
            if self.group_limit_list and group_id in self.group_limit_list:
                return False, None

        # 未开启限流则直接通过
        if not self.enable_rate_limit:
            return True, None

        now = time.monotonic()
        window_start = now - self.rate_limit_period

        async with self._rate_limit_lock:
            bucket = self._rate_limit_buckets.get(group_id, [])
            # 清理过期的时间戳
            bucket = [ts for ts in bucket if ts >= window_start]

            if len(bucket) >= self.max_requests_per_group:
                # 估算距离窗口重置的剩余时间
                earliest = bucket[0]
                retry_after = int(earliest + self.rate_limit_period - now)
                if retry_after < 0:
                    retry_after = 0

                self._rate_limit_buckets[group_id] = bucket
                return (
                    False,
                    f"⏱️ 本群在最近 {self.rate_limit_period} 秒内的生图请求次数已达上限（{self.max_requests_per_group} 次），请约 {retry_after} 秒后再试。",
                )

            # 记录本次请求
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

        # 从当前消息提取图片
        for component in message_chain:
            if isinstance(component, Image) and len(reference_images) < max_images:
                try:
                    # 直接使用 file 属性（v1.0.0 方式）
                    if hasattr(component, "file") and component.file and isinstance(component.file, str):
                        reference_images.append(component.file)
                        logger.debug(
                            f"✓ 从当前消息提取图片 (当前: {len(reference_images)}/{max_images})"
                        )
                except Exception as e:
                    logger.warning(f"✗ 提取图片失败: {e}")

        # 从回复消息提取图片
        for component in message_chain:
            if isinstance(component, Reply) and component.chain:
                for reply_comp in component.chain:
                    if (
                        isinstance(reply_comp, Image)
                        and len(reference_images) < max_images
                    ):
                        try:
                            # 直接使用 file 属性（v1.0.0 方式）
                            if hasattr(reply_comp, "file") and reply_comp.file and isinstance(reply_comp.file, str):
                                reference_images.append(reply_comp.file)
                                self.log_debug("✓ 从回复消息提取图片")
                        except Exception as e:
                            logger.warning(f"✗ 提取回复图片失败: {e}")

        logger.info(f"📸 共收集到 {len(reference_images)} 张参考图片")
        return reference_images

    async def _send_image_with_fallback(self, image_path: str) -> Image:
        """发送图片，优先使用 callback_api_base（优化版本，避免网络阻塞）"""
        callback_api_base = self.context.get_config().get("callback_api_base")

        if not callback_api_base:
            self.log_debug("未配置 callback_api_base，使用本地文件发送")
            return Image.fromFileSystem(image_path)

        try:
            # 尝试生成网络链接，但设置超时控制
            image_component = Image.fromFileSystem(image_path)
            download_url = await asyncio.wait_for(
                image_component.convert_to_web_link(),
                timeout=5.0,  # 5秒超时
            )
            self.log_debug("成功生成下载链接")
            return Image.fromURL(download_url)
        except asyncio.TimeoutError:
            logger.warning("生成下载链接超时，退回到本地文件")
            return Image.fromFileSystem(image_path)
        except (OSError, ConnectionError, TimeoutError) as e:
            logger.warning(f"网络/文件操作失败: {e}，退回到本地文件")
            return Image.fromFileSystem(image_path)
        except Exception as e:
            logger.error(f"发送图片出错: {e}，退回到本地文件")
            return Image.fromFileSystem(image_path)

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

        # 合并所有参考图片，确保只包含 base64 / data URL 字符串
        all_reference_images: list[str] = []
        all_reference_images.extend(
            self._filter_valid_reference_images(reference_images, source="消息图片")
        )
        all_reference_images.extend(
            self._filter_valid_reference_images(avatar_reference, source="头像")
        )

        # 限制参考图片数量
        if (
            all_reference_images
            and len(all_reference_images) > self.max_reference_images
        ):
            logger.warning(
                f"参考图片数量 ({len(all_reference_images)}) 超过限制 ({self.max_reference_images})，将截取前 {self.max_reference_images} 张"
            )
            all_reference_images = all_reference_images[: self.max_reference_images]

        # 构建请求配置
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

        # 日志记录
        logger.info("🎨 图像生成请求:")
        logger.info(f"  模型: {self.model}")
        logger.info(f"  API 类型: {self.api_type}")
        logger.info(
            f"  参考图片: {len(all_reference_images) if all_reference_images else 0} 张"
        )

        # 发送请求
        try:
            logger.info("🚀 开始调用API生成图像...")
            start_time = asyncio.get_event_loop().time()

            # 计算合理的超时时间，每次重试都应有完整的超时时间
            tool_timeout = self.get_tool_timeout(event)
            # 每次重试的超时时间，不超过配置的单次超时时间
            per_retry_timeout = min(self.total_timeout, tool_timeout)
            # 计算总的最大时间（所有重试加起来不能超过框架超时）
            max_total_time = tool_timeout
            logger.info(
                f"[TIMEOUT] tool_call_timeout={tool_timeout}s, per_retry_timeout={per_retry_timeout}s, max_retries={self.max_attempts_per_key}, max_total_time={max_total_time}s"
            )

            image_url, image_path, text_content, thought_signature = await self.api_client.generate_image(
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
                # 文件传输（如果需要）
                if self.nap_server_address and self.nap_server_address != "localhost":
                    logger.info("📤 检测到远程服务器配置，开始文件传输...")

                    try:
                        remote_path = await asyncio.wait_for(
                            send_file(
                                image_path,
                                HOST=self.nap_server_address,
                                PORT=self.nap_server_port,
                            ),
                            timeout=10.0,
                        )
                        if remote_path:
                            image_path = remote_path
                    except asyncio.TimeoutError:
                        logger.warning("⚠️ 文件传输超时，使用本地文件")
                    except Exception as e:
                        logger.warning(f"⚠️ 文件传输失败: {e}，将使用本地文件")

                # 返回结果数据（不发送消息）
                logger.info("📨 图像生成完成，准备返回结果...")

                # 如果启用了远程文件传输，检查是否需要传输文件
                final_image_path = image_path
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
                            final_image_path = remote_path
                    except asyncio.TimeoutError:
                        logger.warning("⚠️ 文件传输超时，使用本地文件")
                    except Exception as e:
                        logger.warning(f"⚠️ 文件传输失败: {e}，将使用本地文件")

                return True, (final_image_path, text_content, thought_signature)
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

    # 快捷指令处理方法
    async def _quick_generate_image(
        self, event: AstrMessageEvent, prompt: str, use_avatar: bool = False
    ):
        """快捷图像生成"""
        if not self.api_client:
            yield event.plain_result("❌ API 客户端未初始化")
            return

        try:
            # 收集参考图片
            ref_images = await self._collect_reference_images(event)

            # 获取头像
            avatars = []
            if use_avatar:
                avatars = await self.get_avatar_reference(event)

            # 合并参考图片和头像，确保只包含base64字符串
            all_ref_images: list[str] = []
            all_ref_images.extend(
                self._filter_valid_reference_images(ref_images, source="消息图片")
            )
            all_ref_images.extend(
                self._filter_valid_reference_images(avatars, source="头像")
            )

            # 检测是否需要手办化增强
            figure_keywords = ["手办", "figure", "模型", "手办化", "手办模型"]
            if any(keyword in prompt.lower() for keyword in figure_keywords):
                enhanced_prompt = enhance_prompt_for_figure(prompt)
            else:
                enhanced_prompt = prompt  # 直接使用用户提示词，不添加额外风格

            # 构建配置
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

            yield event.plain_result("🎨 生成中...")

            # 生成图像
            image_url, image_path, text_content, thought_signature = await self.api_client.generate_image(
                config=config,
                max_retries=self.max_attempts_per_key,
                per_retry_timeout=self.total_timeout,
                max_total_time=self.total_timeout * 2,
            )

            if image_url and image_path:
                logger.debug(f"准备发送图像: image_path类型={type(image_path)}, 值={image_path}")

                # 使用参考插件的方式：chain_result
                img_component = Image.fromFileSystem(image_path)
                chain = [img_component]

                if text_content and self.enable_text_response:
                    chain.append(Plain(f"📝 {text_content}"))

                if thought_signature:
                    logger.debug(f"🧠 思维签名: {thought_signature[:50]}...")

                yield event.chain_result(chain)
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

        # 判断是否需要头像（只有在有@用户时才使用）
        use_avatar = await self.should_use_avatar(event)

        yield event.plain_result("🎨 开始生成图像...")

        async for result in self._quick_generate_image(event, prompt, use_avatar):
            yield result

    # 快速模式指令组
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

        # 临时修改配置
        old_resolution = self.resolution
        old_aspect_ratio = self.aspect_ratio

        try:
            self.resolution = "1K"
            self.aspect_ratio = "1:1"

            # 判断是否需要头像（只有在有@用户时才使用）
            use_avatar = await self.should_use_avatar(event)

            # 调用快速生成方法
            async for result in self._quick_generate_image(event, prompt, use_avatar):
                yield result

        finally:
            # 恢复原始配置
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

        # 临时修改配置
        old_resolution = self.resolution
        old_aspect_ratio = self.aspect_ratio

        try:
            self.resolution = "2K"
            self.aspect_ratio = "16:9"

            # 判断是否需要头像（只有在有@用户时才使用）
            use_avatar = await self.should_use_avatar(event)

            # 调用快速生成方法
            async for result in self._quick_generate_image(event, prompt, use_avatar):
                yield result

        finally:
            # 恢复原始配置
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

        # 临时修改配置
        old_resolution = self.resolution
        old_aspect_ratio = self.aspect_ratio

        try:
            self.resolution = "4K"
            self.aspect_ratio = "16:9"

            # 判断是否需要头像（只有在有@用户时才使用）
            use_avatar = await self.should_use_avatar(event)

            # 调用快速生成方法
            async for result in self._quick_generate_image(event, prompt, use_avatar):
                yield result

        finally:
            # 恢复原始配置
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

        # 临时修改配置
        old_resolution = self.resolution
        old_aspect_ratio = self.aspect_ratio

        try:
            self.resolution = "1K"
            self.aspect_ratio = "3:2"

            # 判断是否需要头像（只有在有@用户时才使用）
            use_avatar = await self.should_use_avatar(event)

            # 调用快速生成方法
            async for result in self._quick_generate_image(event, prompt, use_avatar):
                yield result

        finally:
            # 恢复原始配置
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

        # 临时修改配置
        old_resolution = self.resolution
        old_aspect_ratio = self.aspect_ratio

        try:
            self.resolution = "2K"
            self.aspect_ratio = "9:16"

            # 判断是否需要头像（只有在有@用户时才使用）
            use_avatar = await self.should_use_avatar(event)

            # 调用快速生成方法
            async for result in self._quick_generate_image(event, prompt, use_avatar):
                yield result

        finally:
            # 恢复原始配置
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

        base_prompt = (
            "将画面中的角色重塑为顶级收藏级树脂手办，全身动态姿势，置于角色主题底座，高精度材质，手工涂装，"
            "肌肤纹理与服装材质真实分明。戏剧性硬光为主光源，凸显立体感，无过曝；强效补光消除死黑，细节完整可见。"
            "背景为窗边景深模糊，侧后方隐约可见产品包装盒。博物馆级摄影质感，全身细节无损，面部结构精准。"
            "禁止：任何2D元素或照搬原图、塑料感、面部模糊、五官错位、细节丢失。"
        )
        full_prompt = base_prompt if not prompt else f"{base_prompt}\n{prompt}"

        # 临时调整分辨率与比例以适配手办展示效果
        old_resolution = self.resolution
        old_aspect_ratio = self.aspect_ratio

        try:
            self.resolution = "2K"
            self.aspect_ratio = "3:2"

            # 使用与其他快速模式一致的头像逻辑
            use_avatar = await self.should_use_avatar(event)

            async for result in self._quick_generate_image(
                event, full_prompt, use_avatar
            ):
                yield result
        finally:
            self.resolution = old_resolution
            self.aspect_ratio = old_aspect_ratio

    @filter.command("生图帮助")
    async def show_help(self, event: AstrMessageEvent):
        """显示插件使用帮助"""
        # 黑名单模式下的群需要静默处理，直接返回
        group_id = self._get_group_id_from_event(event)
        if group_id and self.group_limit_list:
            # 黑名单模式：列表内静默
            if (
                self.group_limit_mode == "blacklist"
                and group_id in self.group_limit_list
            ):
                return
            # 白名单模式：不在列表内静默
            if (
                self.group_limit_mode == "whitelist"
                and group_id not in self.group_limit_list
            ):
                return

        grounding_status = "✓ 启用" if self.enable_grounding else "✗ 禁用"
        smart_retry_status = "✓ 启用" if self.enable_smart_retry else "✗ 禁用"
        avatar_status = "✓ 启用" if self.auto_avatar_reference else "✗ 禁用"

        # 获取当前聊天环境的超时配置
        tool_timeout = self.get_tool_timeout(event)
        timeout_warning = ""
        if tool_timeout < 90:
            timeout_warning = f"⚠ 超时时间较短({tool_timeout}秒)，建议设置为90-120秒"

        # 获取插件版本
        try:
            import yaml

            metadata_path = os.path.join(os.path.dirname(__file__), "metadata.yaml")
            with open(metadata_path, encoding="utf-8") as f:
                metadata = yaml.safe_load(f)
                version = metadata.get("version", "v1.1.0")
        except Exception:
            version = "v1.1.0"

        markdown_content = rf"""# 🎨 Gemini 图像生成插件 {version}

## 系统状态

- **模型**: `{self.model}`
- **API类型**: `{self.api_type}`
- **分辨率**: `{self.resolution}`
- **长宽比**: `{self.aspect_ratio or "默认"}`
- **API密钥**: `{len(self.api_keys)}个`
- **搜索接地**: {grounding_status}
- **自动头像**: {avatar_status}
- **智能重试**: {smart_retry_status}
- **超时时间**: `{tool_timeout}秒`
- **端点**: `{self.api_base or "默认"}`"""

        # 添加警告信息（如果存在）
        if timeout_warning:
            markdown_content += f"\n\n> ⚠️ 警告: {timeout_warning}"

        # 添加剩余内容
        markdown_content += """

## 🚀 指令使用

```
/生图 [描述]
```
> 基础图像生成功能
> 示例: `/生图 一只可爱的橙色小猫，动漫风格，高清细节`

```
/快速 [预设] [描述]
```
> 使用预设参数快速生成图像
> 预设: 头像/海报/壁纸/卡片/手机/手办化
> 示例: `/快速 头像 生成专业的个人头像`

```
/改图 [描述]
```
> 修改或重做图像（需要提供参考图片）
> 示例: 发送图片 + `/改图 把头发改成红色`

```
/换风格 [风格] [描述]
```
> 改变图像风格
> 示例: 发送图片 + `/换风格 动漫`
> 示例: 发送图片 + `/换风格 油画 古典艺术风格`

```
/生图帮助
```
> 显示此帮助信息

## ⭐ 进阶功能

- **引用图片**: 回复或引用图片自动作为参考图使用
- **@用户**: @某人会使用该用户头像作为参考（需要先获取头像权限）
- **关键词触发**: 包含"我"、"头像"、"自己"等关键词自动获取发送者头像
- **多风格支持**: 支持动漫、写实、水彩、油画等多种风格
- **智能重试**: 生成失败时自动重试，提高成功率

## 💡 使用技巧

- 提示词越详细，生成效果越好
- 生成高质量图像需要时间，请耐心等待
- 建议添加多个API密钥以提高成功率
- 快速模式预设了最佳分辨率和长宽比
- 工具超时时间建议设置为90-120秒

---

> 🤖 *由 Gemini AI 驱动的图像生成插件*"""

        # 生成帮助图片
        try:
            logger.info("开始生成HTML帮助图片...")

            # 构建Jinja2模板内容 - 淡蓝色主题
            jinja2_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{{ title }}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

        body {
            background-color: #E6F3FF;
            font-family: 'Share Tech Mono', 'Consolas', 'Courier New', monospace;
            color: #1a5490;
            padding: 20px;
            line-height: 1.6;
            margin: 0;
        }

        .container {
            max-width: 900px;
            margin: 0 auto;
            background-color: rgba(255, 255, 255, 0.95);
            border: 2px solid #4a90e2;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 0 20px rgba(74, 144, 226, 0.3);
        }

        .header {
            color: #2c5aa0;
            border-bottom: 2px solid #4a90e2;
            padding-bottom: 15px;
            margin-bottom: 25px;
            text-align: center;
        }

        .header h1 {
            margin: 0;
            font-size: 24px;
            text-shadow: 0 0 3px rgba(44, 90, 160, 0.2);
        }

        .section {
            margin: 20px 0;
            padding: 15px;
            border-left: 3px solid #4a90e2;
            background-color: rgba(230, 243, 255, 0.3);
            border-radius: 0 5px 5px 0;
        }

        .section h2 {
            color: #2c5aa0;
            margin-top: 0;
            margin-bottom: 15px;
            font-size: 20px;
            text-shadow: 0 0 3px rgba(44, 90, 160, 0.2);
        }

        .section h3 {
            color: #4a90e2;
            margin-top: 15px;
            margin-bottom: 8px;
            font-size: 16px;
        }

        .command {
            color: #2c5aa0;
            background-color: rgba(74, 144, 226, 0.1);
            padding: 4px 8px;
            border-radius: 4px;
            border: 1px solid #4a90e2;
            font-weight: bold;
            display: inline-block;
        }

        .example {
            color: #6c757d;
            font-style: italic;
            margin: 8px 0;
            padding-left: 15px;
            border-left: 2px solid #6c757d;
        }

        .feature {
            color: #4a90e2;
            font-weight: bold;
        }

        .status {
            background-color: rgba(230, 243, 255, 0.5);
            border: 1px solid #4a90e2;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }

        .status ul {
            margin: 0;
            padding-left: 20px;
        }

        .status li {
            margin: 8px 0;
            color: #1a5490;
        }

        .status li strong {
            color: #2c5aa0;
        }

        .warning {
            color: #856404;
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-left: 4px solid #ffc107;
            padding: 12px;
            border-radius: 4px;
            margin: 15px 0;
        }

        .warning strong {
            color: #856404;
        }

        .footer {
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #4a90e2;
            color: #6c757d;
        }

        ul, ol {
            margin: 10px 0;
            padding-left: 25px;
        }

        li {
            margin: 8px 0;
        }

        p {
            margin: 10px 0;
        }

        strong {
            color: #2c5aa0;
        }

        hr {
            border: none;
            border-top: 1px solid #4a90e2;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎨 Gemini 图像生成插件 {{ version }}</h1>
        </div>

        <div class="section">
            <h2>▶ 系统状态</h2>
            <div class="status">
                <ul>
                    <li><strong>模型</strong>: {{ model }}</li>
                    <li><strong>API类型</strong>: {{ api_type }}</li>
                    <li><strong>分辨率</strong>: {{ resolution }}</li>
                    <li><strong>长宽比</strong>: {{ aspect_ratio }}</li>
                    <li><strong>API密钥</strong>: {{ api_keys_count }}个</li>
                    <li><strong>搜索接地</strong>: {{ grounding_status }}</li>
                    <li><strong>自动头像</strong>: {{ avatar_status }}</li>
                    <li><strong>智能重试</strong>: {{ smart_retry_status }}</li>
                    <li><strong>超时时间</strong>: {{ tool_timeout }}秒</li>
                    <li><strong>端点</strong>: {{ api_base }}</li>
                </ul>
            </div>
            {% if timeout_warning %}
            <div class="warning">
                <strong>⚠️ 警告</strong>: {{ timeout_warning }}
            </div>
            {% endif %}
        </div>

        <div class="section">
            <h2>🚀 指令使用</h2>

            <h3><span class="command">/生图 [描述]</span></h3>
            <p>基础图像生成功能</p>
            <p class="example">示例: /生图 一只可爱的橙色小猫，动漫风格，高清细节</p>

            <h3><span class="command">/快速 [预设] [描述]</span></h3>
            <p>使用预设参数快速生成图像</p>
            <p class="example">预设: 头像/海报/壁纸/卡片/手机/手办化</p>
            <p class="example">示例: /快速 头像 生成专业的个人头像</p>

            <h3><span class="command">/改图 [描述]</span></h3>
            <p>修改或重做图像（需要提供参考图片）</p>
            <p class="example">示例: 发送图片 + /改图 把头发改成红色</p>

            <h3><span class="command">/换风格 [风格] [描述]</span></h3>
            <p>改变图像风格</p>
            <p class="example">示例: 发送图片 + /换风格 动漫</p>
            <p class="example">示例: 发送图片 + /换风格 油画 古典艺术风格</p>

            <h3><span class="command">/生图帮助</span></h3>
            <p>显示此帮助信息</p>
        </div>

        <div class="section">
            <h2>⭐ 进阶功能</h2>
            <ul>
                <li><span class="feature">引用图片</span>: 回复或引用图片自动作为参考图使用</li>
                <li><span class="feature">@用户</span>: @某人会使用该用户头像作为参考（需要先获取头像权限）</li>
                <li><span class="feature">关键词触发</span>: 包含"我"、"头像"、"自己"等关键词自动获取发送者头像</li>
                <li><span class="feature">多风格支持</span>: 支持动漫、写实、水彩、油画等多种风格</li>
                <li><span class="feature">智能重试</span>: 生成失败时自动重试，提高成功率</li>
            </ul>
        </div>

        <div class="section">
            <h2>💡 使用技巧</h2>
            <ul>
                <li>提示词越详细，生成效果越好</li>
                <li>生成高质量图像需要时间，请耐心等待</li>
                <li>建议添加多个API密钥以提高成功率</li>
                <li>快速模式预设了最佳分辨率和长宽比</li>
                <li>工具超时时间建议设置为90-120秒</li>
            </ul>
        </div>

        <div class="footer">
            <p>🤖 由 Gemini AI 驱动的图像生成插件</p>
        </div>
    </div>
</body>
</html>"""

            # 准备模板数据
            template_data = {
                "title": f"Gemini 图像生成插件 {version}",
                "version": version,
                "model": self.model,
                "api_type": self.api_type,
                "resolution": self.resolution,
                "aspect_ratio": self.aspect_ratio or "默认",
                "api_keys_count": len(self.api_keys),
                "grounding_status": grounding_status,
                "avatar_status": avatar_status,
                "smart_retry_status": smart_retry_status,
                "tool_timeout": tool_timeout,
                "api_base": self.api_base or "默认",
                "timeout_warning": timeout_warning if timeout_warning else ""
            }

            # 使用正确的HTML渲染API
            help_image_url = await self.html_render(jinja2_template, template_data)
            logger.info("HTML帮助图片生成成功")
            yield event.image_result(help_image_url)

        except Exception as e:
            # 如果图片生成失败，记录错误并回退到文本模式
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

        # 收集参考图片
        ref_images = await self._collect_reference_images(event)

        # 获取头像（只有在有@用户时才使用）
        avatars = await self.get_avatar_reference(event)
        if avatars:
            ref_images.extend(avatars)

        # 使用新的快捷生成方法
        async for result in self._quick_generate_image(
            event, f"根据参考图像修改：{prompt}", False
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
        avatar_reference = (
            await self.get_avatar_reference(event) if self.auto_avatar_reference else []
        )

        success, error_msg = await self._generate_image_core(
            event=event,
            prompt=full_prompt,
            reference_images=reference_images,
            avatar_reference=avatar_reference,
        )

        if not success and error_msg:
            yield event.plain_result(error_msg)

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
        使用 Gemini 模型生成或修改图像的高级工具

        当用户请求图像生成或绘画时，调用此函数。

        **重要判断逻辑：**

        1. **用户使用以下词语时，强烈建议设置 use_reference_images="true" 和 include_user_avatar="true"**：
           - "改成", "改为", "变成", "换成", "替换", "调整", "修改", "优化", "重做", "重新", "改图", "换风格"
           - "基于", "根据", "按照", "参考", "依照", "以...为基础", "以...为参考"
           - 当句子中出现"我的"、"我的头发"、"我的脸"等描述时，表示需要用户本人作为参考
           - 示例："把我的头发改成黑色", "把图片变成动漫风格", "根据我的头像生成图片", "让我的眼睛变大"

        2. **当用户说"按照我", "根据我", "基于我", "参考我", "我的头像", "我的"时**：
           - 必须设置 use_reference_images="true" 和 include_user_avatar="true"
           - 会自动获取当前用户的头像作为参考

        3. **当用户@某个用户时**：
           - 必须设置 use_reference_images="true" 和 include_user_avatar="true"
           - 会自动获取被@用户的头像作为参考

        4. **当用户消息中包含图片时**：
           - 如果用户明确说"基于这张图片", "修改这张图"等，设置 use_reference_images="true"
           - 图片可以是用户直接上传的，也可以是引用回复的消息中的图片

        **提示词优化指南：**

        1. **手办模型生成**：
        "请将此照片中的主要对象精确转换为写实的、杰作级别的 1/7 比例 PVC 手办。
        在手办旁边应放置一个盒子：盒子正面应有一个大型清晰的透明窗口，印有主要艺术作品、产品名称、品牌标志、条形码，以及一个小规格或真伪验证面板。盒子的角落还必须贴有小价签。同时，在后方放置一个电脑显示器，显示器屏幕需要显示该手办的 ZBrush 建模过程。
        在包装盒前方，手办应放置在圆形塑料底座上。手办必须有 3D 立体感和真实感，PVC 材质的纹理需要清晰表现。如果背景可以设置为室内场景，效果会更好。"

        2. **Q版手办模型**：
        "请将此照片中的主要对象精确转换为写实的、杰作级别的 1/7 比例 PVC 手办。
        在此手办的一侧后方，应放置一个盒子：在盒子正面，显示我输入的原始图像，带有主题艺术作品、产品名称、品牌标志、条形码，以及一个小规格或真伪验证面板。盒子的一个角落还必须贴有小价签。同时，在后方放置一个电脑显示器，显示器屏幕需要显示该手办的 ZBrush 建模过程。
        在包装盒前方，手办应放置在圆形塑料底座上。手办必须有 3D 立体感和真实感，PVC 材质的纹理需要清晰表现。如果背景可以设置为室内场景，效果会更好。"

        **质量要求：**
        - 修复任何缺失部分时，必须没有执行不佳的元素
        - 修复人体手办时（如适用），身体部位必须自然，动作必须协调，所有部位比例必须合理
        - 如果原始照片不是全身照，请尝试补充手办使其成为全身版本
        - 人物表情和动作必须与照片完全一致
        - 手办头部不应显得太大，腿部不应显得太短，手办不应看起来矮胖（除非明确是Q版设计）
        - 对于动物手办，应减少毛发的真实感和细节层次，使其更像手办而不是真实的原始生物
        - 不应有外轮廓线，手办绝不能是平面的
        - 注意近大远小的透视关系

        Args:
            prompt(string): 图像生成或修改的描述
            use_reference_images(string): 是否使用上下文中的参考图片（true/false）。当用户意图是"修改"、"变成"、"基于"、"改成"等时，必须设置为"true"
            include_user_avatar(string): 是否包含用户头像作为参考图像（true/false）。当用户说"根据我"、"我的头像"或明显需要用户本人图像时，设置为"true"
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

        # 收集参考图片（从消息中提取的图片，包括当前消息和引用回复中的图片）
        reference_images = []
        if str(use_reference_images).lower() in {"true", "1", "yes", "y", "是"}:
            reference_images = await self._collect_reference_images(event)

        # 自动获取头像作为参考
        avatar_reference = []

        # 直接信任Gemini API的判断
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

        # 调用核心生成方法，但不使用yield，而是直接处理结果
        success, result_data = await self._generate_image_core_internal(
            event=event,
            prompt=prompt,
            reference_images=reference_images,
            avatar_reference=avatar_reference,
        )

        if success and result_data:
            # 直接发送图片和文本结果
            image_path, text_content, thought_signature = result_data
            message_chain = []

            if text_content:
                message_chain.append(Plain(text_content))

            if image_path:
                message_chain.append(ImageComponent.fromFileSystem(image_path))

            if thought_signature:
                logger.debug(f"🧠 思维签名: {thought_signature[:50]}...")

            if message_chain:
                yield event.chain_result(message_chain)
        elif not success:
            # 发送错误消息
            yield event.plain_result(result_data)

        # 清理使用的头像缓存
        try:
            await self.avatar_manager.cleanup_cache()
        except Exception as e:
            logger.warning(f"清理头像缓存失败: {e}")

    async def terminate(self):
        """插件卸载时清理资源"""
        logger.info("🎨 Gemini 图像生成插件已卸载")
