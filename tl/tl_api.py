"""
API客户端模块 y
提供Google Gemini和OpenAI兼容API的客户端实现
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

from astrbot.api import logger

# 导入本地模块
try:
    from .tl_utils import get_plugin_data_dir, save_base64_image, save_image_data
except ImportError:
    # 如果tl_utils不存在，先创建简单的占位符
    async def save_base64_image(base64_data: str, image_format: str = "png") -> str | None:
        """占位符函数"""
        return None

    async def save_image_data(image_data: bytes, image_format: str = "png") -> str | None:
        """占位符函数"""
        return None


@dataclass
class ApiRequestConfig:
    """API 请求配置（基于 Gemini 官方文档）"""

    model: str
    prompt: str
    api_type: str = "openai"
    api_base: str | None = None
    api_key: str | None = None
    resolution: str | None = None
    aspect_ratio: str | None = None
    enable_grounding: bool = False
    response_modalities: str = "TEXT_IMAGE"  # 默认同时返回文本和图像
    max_tokens: int = 1000
    reference_images: list[str] | None = None
    response_text: str | None = None  # 存储文本响应
    enable_smart_retry: bool = True  # 智能重试开关
    enable_text_response: bool = False  # 文本响应开关

    # 官方文档推荐参数
    temperature: float = 0.7  # 控制生成随机性，0.0-1.0
    seed: int | None = None  # 固定种子以确保一致性
    safety_settings: dict | None = None  # 安全设置


class APIError(Exception):
    """API 错误基类"""

    def __init__(self, message: str, status_code: int = None, error_type: str = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type


class GeminiAPIClient:
    """遵循官方 API 规范的 Gemini API 客户端

    特性：
    - 支持 Google 官方 API 和 OpenRouter API
    - 支持自定义 API Base URL（反代）
    - 支持任意模型名称
    - 遵循官方 Gemini API 规范
    """

    # Google 官方 API 默认地址
    GOOGLE_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

    # OpenRouter API 默认地址
    OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

    def __init__(self, api_keys: list[str]):
        """
        初始化 API 客户端

        Args:
            api_keys: API 密钥列表
        """
        self.api_keys = api_keys or []
        self.current_key_index = 0
        self._lock = asyncio.Lock()
        logger.debug(f"API 客户端已初始化，支持 {len(self.api_keys)} 个 API 密钥")

    async def get_next_api_key(self) -> str:
        """获取下一个 API 密钥"""
        async with self._lock:
            if not self.api_keys:
                raise ValueError("API 密钥列表不能为空")
            key = self.api_keys[self.current_key_index % len(self.api_keys)]
            return key

    async def rotate_api_key(self):
        """轮换到下一个 API 密钥"""
        async with self._lock:
            if len(self.api_keys) > 1:
                self.current_key_index = (self.current_key_index + 1) % len(
                    self.api_keys
                )
                logger.debug(
                    f"已轮换到下一个 API 密钥，当前索引: {self.current_key_index}"
                )

    @staticmethod
    def _prepare_google_payload(config: ApiRequestConfig) -> dict[str, Any]:
        """准备 Google 官方 API 请求负载（遵循官方规范）"""
        parts = [{"text": config.prompt}]

        if config.reference_images:
            for base64_image in config.reference_images[:14]:
                mime_type, data = GeminiAPIClient._normalize_image_input(base64_image)
                if not data:
                    logger.warning(f"跳过无法识别/读取的参考图像: {type(base64_image)}")
                    continue

                parts.append({"inlineData": {"mimeType": mime_type, "data": data}})

        contents = [{"role": "user", "parts": parts}]

        generation_config = {"responseModalities": ["TEXT", "IMAGE"]}

        # 根据官方文档，图像生成必须同时包含 TEXT 和 IMAGE modalities
        # 这样可以确保返回图像而不是纯文本
        modalities_map = {
            "TEXT": ["TEXT"],
            "IMAGE": ["IMAGE"],
            "TEXT_IMAGE": ["TEXT", "IMAGE"],
        }

        # 获取配置的模态
        modalities = modalities_map.get(config.response_modalities, ["TEXT", "IMAGE"])

        # 确保包含图像模态
        if "IMAGE" not in modalities:
            logger.warning("配置中缺少 IMAGE modality，自动添加以支持图像生成")
            modalities.append("IMAGE")

        # 确保包含文本模态
        if "TEXT" not in modalities:
            logger.debug("添加 TEXT modality 以提供更好的兼容性")
            modalities.append("TEXT")

        generation_config["responseModalities"] = modalities
        logger.debug(f"响应模态: {modalities}")

        image_config = {}

        # 根据官方文档设置图像尺寸
        if config.resolution:
            resolution = config.resolution.upper()

            if resolution in ["1K", "1024x1024"]:
                image_config["image_size"] = "1K"
                logger.debug("设置图像尺寸: 1K")
            elif resolution in ["2K", "2048x2048"]:
                image_config["image_size"] = "2K"
                logger.debug("设置图像尺寸: 2K")
            elif resolution in ["4K", "4096x4096"]:
                image_config["image_size"] = "4K"
                logger.debug("设置图像尺寸: 4K")
            else:
                # 默认使用1K
                image_config["image_size"] = "1K"
                logger.warning(f"不支持的分辨率: {config.resolution}，使用默认尺寸 1K")

        # 设置长宽比
        if config.aspect_ratio and ":" in config.aspect_ratio:
            # 将长宽比转换为标准格式
            ratio_map = {
                "1:1": "1:1",
                "16:9": "16:9",
                "9:16": "9:16",
                "3:2": "3:2",
                "4:3": "4:3"
            }
            ratio = ratio_map.get(config.aspect_ratio, config.aspect_ratio)
            image_config["aspect_ratio"] = ratio
            logger.debug(f"设置长宽比: {ratio}")
        elif config.aspect_ratio:
            logger.warning(f"不支持的长宽比格式: {config.aspect_ratio}，将使用默认长宽比")

        if image_config:
            generation_config["image_config"] = image_config

        # 添加官方文档推荐参数
        if config.temperature is not None:
            generation_config["temperature"] = config.temperature
        if config.seed is not None:
            generation_config["seed"] = config.seed
        if config.safety_settings:
            generation_config["safetySettings"] = config.safety_settings

        tools = []
        if config.enable_grounding:
            tools.append({"google_search": {}})

        payload = {"contents": contents, "generationConfig": generation_config}

        if tools:
            payload["tools"] = tools

        # 调试：记录 image_config
        if "image_config" in generation_config:
            logger.debug(f"实际发送的 image_config: {generation_config['image_config']}")

        return payload

    @staticmethod
    def _prepare_openrouter_payload(config: ApiRequestConfig) -> dict[str, Any]:
        """准备 OpenRouter API 请求负载"""
        message_content = [
            {"type": "text", "text": f"Generate an image: {config.prompt}"}
        ]

        if config.reference_images:
            for base64_image in config.reference_images[:6]:
                mime_type, data = GeminiAPIClient._normalize_image_input(base64_image)
                if not data:
                    logger.warning(f"跳过无法识别/读取的参考图像: {type(base64_image)}")
                    continue

                image_str = f"data:{mime_type};base64,{data}"
                message_content.append(
                    {"type": "image_url", "image_url": {"url": image_str}}
                )

        # OpenAI 兼容接口下：
        # - 使用 chat/completions
        # - modalities: ["image", "text"]
        # - image_config: {aspect_ratio, image_size}
        # - tools: [{google_search:{}}]（当启用搜索接地时）
        payload: dict[str, Any] = {
            "model": config.model,
            "messages": [{"role": "user", "content": message_content}],
            "max_tokens": config.max_tokens,
            "temperature": 0.7,
            "modalities": ["image", "text"],
        }

        # image_config 与 Gemini 3 Pro Image 模型相关的配置
        image_config: dict[str, Any] = {}

        if config.aspect_ratio:
            image_config["aspect_ratio"] = config.aspect_ratio

        # 仅在 Gemini 3 Pro Image 系列模型下传递 image_size
        model_name = (config.model or "").lower()
        is_gemini_image_model = "gemini-3-pro-image" in model_name

        if is_gemini_image_model and config.resolution:
            # 前端 router 侧直接传递 "1K"/"2K"/"4K"，这里保持一致
            image_config["image_size"] = config.resolution

        if image_config:
            payload["image_config"] = image_config

        # 与前端 router 一致：启用搜索接地时，通过 tools.google_search 控制
        if is_gemini_image_model and config.enable_grounding:
            payload["tools"] = [{"google_search": {}}]

        return payload

    @staticmethod
    def _normalize_image_input(image_input: Any) -> tuple[str | None, str | None]:
        """
        将参考图像输入规范化为 (mime_type, base64_data)。
        支持 data URI、纯/宽松 base64 字符串、本地文件路径、file://、http/https URL。
        """
        try:
            if image_input is None:
                return None, None

            image_str = str(image_input).strip()
            if "&amp;" in image_str:
                image_str = image_str.replace("&amp;", "&")
            if not image_str:
                return None, None

            # data URI
            if image_str.startswith("data:image/") and ";base64," in image_str:
                header, data = image_str.split(";base64,", 1)
                mime_type = header.replace("data:", "")
                return mime_type, data

            # file:// 路径
            if image_str.startswith("file://"):
                parsed = urllib.parse.urlparse(image_str)
                image_path = Path(parsed.path)
                if image_path.exists() and image_path.is_file():
                    suffix = image_path.suffix.lower().lstrip(".") or "png"
                    mime_type = f"image/{suffix}"
                    with open(image_path, "rb") as f:
                        data_bytes = f.read()
                    data = base64.b64encode(data_bytes).decode("utf-8")
                    return mime_type, data
                else:
                    logger.warning(f"file:// 路径不存在: {image_str}")

            # http(s) URL -> 下载并转base64
            if image_str.startswith("http://") or image_str.startswith("https://"):
                try:
                    with urllib.request.urlopen(image_str, timeout=8) as resp:
                        content_type = resp.headers.get("Content-Type", "image/png")
                        mime_type = content_type.split(";")[0] if content_type else "image/png"
                        data_bytes = resp.read()
                        if data_bytes:
                            data = base64.b64encode(data_bytes).decode("utf-8")
                            return mime_type, data
                except Exception as e:
                    logger.warning(f"下载参考图失败: {e}")

            # 尝试解析为裸/宽松 base64 数据（在文件路径之前，避免长字符串导致 "File name too long"）
            if len(image_str) > 255 or not any(char in image_str for char in ["/", "\\", "."]):
                try:
                    cleaned = image_str.replace("\n", "").replace(" ", "")
                    decoded = base64.b64decode(cleaned, validate=False)
                    if decoded and len(decoded) > 100:
                        normalized = base64.b64encode(decoded).decode("utf-8")
                        return "image/png", normalized
                except (binascii.Error, ValueError):
                    pass

            # 本地文件路径（仅当字符串长度合理时尝试）
            if len(image_str) <= 255:
                candidate_paths = [
                    Path(image_str),
                    get_plugin_data_dir() / image_str,
                    Path.cwd() / image_str,
                ]
                for image_path in candidate_paths:
                    try:
                        if image_path.exists() and image_path.is_file():
                            suffix = image_path.suffix.lower().lstrip(".") or "png"
                            mime_type = f"image/{suffix}"
                            with open(image_path, "rb") as f:
                                data_bytes = f.read()
                            data = base64.b64encode(data_bytes).decode("utf-8")
                            return mime_type, data
                    except OSError:
                        continue

            return None, None
        except Exception as e:
            logger.warning(f"参考图像规范化失败: {e}")
            return None, None

    def _get_api_url(
        self, config: ApiRequestConfig
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        """
        根据配置获取 API URL、请求头和负载

        支持自定义 API Base URL（反代）
        """
        # 确定 API 基础地址（支持反代）
        if config.api_base:
            api_base = config.api_base.rstrip("/")
            logger.debug(f"使用自定义 API Base: {api_base}")
        else:
            if config.api_type == "google":
                api_base = self.GOOGLE_API_BASE
            else:  # openai 兼容格式
                api_base = self.OPENROUTER_API_BASE

            logger.debug(f"使用默认 API Base ({config.api_type}): {api_base}")

        # 准备请求
        if config.api_type == "google":
            url = f"{api_base}/models/{config.model}:generateContent"
            payload = self._prepare_google_payload(config)
            headers = {
                "x-goog-api-key": config.api_key,
                "Content-Type": "application/json",
            }
        else:
            url = f"{api_base}/chat/completions"
            payload = self._prepare_openrouter_payload(config)
            headers = {
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/astrbot",
                "X-Title": "AstrBot Gemini Image Advanced",
            }

        logger.debug(f"准备请求到: {url}")

        return url, headers, payload

    async def generate_image(
        self, config: ApiRequestConfig, max_retries: int = 3, total_timeout: int = 120, per_retry_timeout: int = None, max_total_time: int = None
    ) -> tuple[str | None, str | None, str | None, str | None]:
        """
        生成图像

        Args:
            config: 请求配置
            max_retries: 最大重试次数
            total_timeout: 总超时时间（秒）

        Returns:
            (image_url, image_path, text_content, thought_signature) 或 (None, None, None, None) 如果失败
        """
        if not self.api_keys:
            raise ValueError("未配置 API 密钥")

        if not config.api_key:
            config.api_key = await self.get_next_api_key()

        # 获取请求信息
        url, headers, payload = self._get_api_url(config)

        logger.debug(f"使用 {config.model} (通过 {config.api_type}) 生成图像")
        logger.debug(f"API 端点: {url[:80]}...")

        if config.resolution or config.aspect_ratio:
            logger.debug(
                f"分辨率: {config.resolution or '默认'}, 长宽比: {config.aspect_ratio or '默认'}"
            )

        if config.api_base:
            logger.debug(f"使用自定义 API Base: {config.api_base}")

        return await self._make_request(
            url=url,
            payload=payload,
            headers=headers,
            api_type=config.api_type,
            model=config.model,
            max_retries=max_retries,
            total_timeout=total_timeout,
        )

    async def _make_request(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        api_type: str,
        model: str,
        max_retries: int,
        total_timeout: int = 120,
    ) -> tuple[str | None, str | None, str | None, str | None]:
        """执行 API 请求并处理响应，每个重试有独立的超时控制"""

        current_retry = 0
        last_error = None

        while current_retry < max_retries:
            try:
                # 每个重试使用独立的超时控制，不共享总超时时间
                timeout = aiohttp.ClientTimeout(total=total_timeout, sock_read=total_timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    logger.debug(f"发送请求（重试 {current_retry}/{max_retries - 1}）")
                    return await self._perform_request(session, url, payload, headers, api_type, model)

            except asyncio.CancelledError:
                # 只有框架取消才不重试（这是最顶层的超时）
                logger.debug("请求被框架取消（工具调用总超时），不再重试")
                timeout_msg = "图像生成时间过长，超出了框架限制。请尝试简化图像描述或在框架配置中增加 tool_call_timeout 到 90-120 秒。"
                raise APIError(timeout_msg, None, "cancelled")
            except Exception as e:
                error_msg = str(e)
                error_type = self._classify_error(e, error_msg)

                # 判断是否可重试的错误
                if self._is_retryable_error(error_type, e):
                    last_error = APIError(error_msg, None, error_type)
                    logger.warning(f"可重试错误 (重试 {current_retry + 1}/{max_retries}): {error_msg}")

                    current_retry += 1
                    if current_retry < max_retries:
                        # 指数退避延迟：2秒、4秒、8秒……最大10秒
                        delay = min(2 ** (current_retry + 1), 10)
                        logger.debug(f"等待 {delay} 秒后重试...")
                        await asyncio.sleep(delay)
                        continue  # 继续下一次重试
                    else:
                        logger.error(f"达到最大重试次数 ({max_retries})，生成失败")
                else:
                    # 不可重试的错误，立即抛出
                    logger.error(f"不可重试错误: {error_msg}")
                    raise APIError(error_msg, None, error_type)

        # 如果都失败了，返回最后一次错误
        if last_error:
            raise last_error

        return None, None, None, None

    def _classify_error(self, exception: Exception, error_msg: str) -> str:
        """分类错误类型"""
        if isinstance(exception, asyncio.TimeoutError):
            return "timeout"
        elif "timeout" in error_msg.lower():
            return "timeout"
        elif "connection" in error_msg.lower():
            return "network"
        elif isinstance(exception, aiohttp.ClientError):
            return "network"
        else:
            return "unknown"

    def _is_retryable_error(self, error_type: str, exception: Exception) -> bool:
        """判断错误是否可重试"""
        # 可重试的错误：超时、网络错误、服务器错误
        if error_type in ["timeout", "network"]:
            return True

        # HTTP 状态码判断
        if hasattr(exception, "status"):
            status = exception.status
            # 可重试：408, 500, 502, 503, 504
            # 不可重试：401, 402, 403, 422, 429（速率限制）
            if status in [408, 500, 502, 503, 504]:
                return True
            elif status in [401, 402, 403, 422, 429]:
                return False

        return True  # 默认重试未知错误

    async def _perform_request(
        self,
        session: aiohttp.ClientSession,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        api_type: str,
        model: str,
    ) -> tuple[str | None, str | None, str | None, str | None]:
        """执行实际的HTTP请求"""
        logger.debug(f"发送请求到: {url[:100]}...")

        async with session.post(url, json=payload, headers=headers) as response:
            logger.debug(f"响应状态: {response.status}")
            response_text = await response.text()

            # 解析 JSON 响应，添加错误处理
            try:
                response_data = json.loads(response_text) if response_text else {}
            except json.JSONDecodeError as e:
                logger.error(f"JSON 解析失败: {e}")
                logger.error(f"响应内容前500字符: {response_text[:500]}")
                raise APIError(f"API 返回了无效的 JSON 响应: {e}", response.status)

            if response.status == 200:
                logger.debug("API 调用成功")
                if api_type == "google":
                    return await self._parse_gresponse(response_data, session)
                else:  # openai 兼容格式
                    return await self._parse_openrouter_response(response_data, session)
            elif response.status in [429, 402, 403]:
                error_msg = response_data.get("error", {}).get(
                    "message", f"HTTP {response.status}"
                )
                logger.warning(f"API 配额/权限问题: {error_msg}")
                raise APIError(error_msg, response.status, "quota")
            else:
                error_msg = response_data.get("error", {}).get(
                    "message", f"HTTP {response.status}"
                )
                logger.warning(f"API 错误: {error_msg}")
                raise APIError(error_msg, response.status)

    async def _parse_gresponse(
        self, response_data: dict, session: aiohttp.ClientSession
    ) -> tuple[str | None, str | None, str | None, str | None]:
        """解析 Google 官方 API 响应"""
        import asyncio

        parse_start = asyncio.get_event_loop().time()
        logger.debug("🔍 开始解析API响应数据...")

        if "candidates" not in response_data or not response_data["candidates"]:
            if "promptFeedback" in response_data:
                feedback = response_data["promptFeedback"]
                logger.warning(f"请求被阻止: {feedback}")
            else:
                logger.error(f"响应中没有 candidates: {response_data}")
            return None, None, None, None

        candidate = response_data["candidates"][0]
        logger.debug(f"📝 找到 {len(response_data['candidates'])} 个候选结果")

        if "finishReason" in candidate and candidate["finishReason"] in [
            "SAFETY",
            "RECITATION",
        ]:
            logger.warning(f"生成被阻止: {candidate['finishReason']}")
            return None, None, None, None

        if "content" not in candidate or "parts" not in candidate["content"]:
            logger.error("响应格式不正确")
            return None, None, None, None

        parts = candidate["content"]["parts"]
        logger.debug(f"📋 响应包含 {len(parts)} 个部分")

        # 查找图像、文本和思维签名
        image_url = None
        image_path = None
        text_content = None
        thought_signature = None

        logger.debug(f"🖼️ 搜索图像数据... (共 {len(parts)} 个part)")
        for i, part in enumerate(parts):
            try:
                logger.debug(f"检查第 {i} 个part: {list(part.keys())}")

                # 提取思维签名
                if "thoughtSignature" in part:
                    thought_signature = part["thoughtSignature"]
                    logger.debug(f"🧠 找到思维签名: {thought_signature[:50]}...")

                # 兼容 camelCase 与 snake_case 的图像返回字段
                inline_data = part.get("inlineData") or part.get("inline_data")
                if inline_data and not part.get("thought", False):
                    mime_type = inline_data.get("mimeType") or inline_data.get("mime_type") or "image/png"
                    base64_data = inline_data.get("data", "")

                    logger.debug(
                        f"🎯 找到图像数据 (第{i + 1}部分): {mime_type}, 大小: {len(base64_data)} 字符"
                    )

                    if base64_data:
                        image_format = (
                            mime_type.split("/")[1] if "/" in mime_type else "png"
                        )

                        logger.debug("💾 开始保存图像文件...")
                        save_start = asyncio.get_event_loop().time()

                        image_path = await save_base64_image(base64_data, image_format)

                        save_end = asyncio.get_event_loop().time()
                        logger.debug(
                            f"✅ 图像保存完成，耗时: {save_end - save_start:.2f}秒"
                        )

                        if image_path:
                            # 直接使用文件路径，不使用 file:// URI（根据 AstrBot 文档要求）
                            image_url = image_path
                    else:
                        logger.warning(f"第 {i} 个part有inlineData但data为空")
                elif "thought" in part and part.get("thought", False):
                    logger.debug(f"第 {i} 个part是思考内容")
                else:
                    logger.debug(f"第 {i} 个part不是图像也不是思考: {list(part.keys())}")
            except Exception as e:
                logger.error(f"处理第 {i} 个part时出错: {e}", exc_info=True)

        # 查找文本内容
        logger.debug("📝 搜索文本内容...")
        text_parts = [
            p for p in parts if "text" in p and not p.get("thought", False)
        ]
        if text_parts:
            text_content = " ".join([p["text"] for p in text_parts])
            logger.debug(f"🎯 找到文本内容: {text_content[:100]}...")

        # 如果找到了图像或文本，返回结果
        if image_url or text_content:
            parse_end = asyncio.get_event_loop().time()
            logger.debug(f"🎉 API响应解析完成，总耗时: {parse_end - parse_start:.2f}秒")
            return image_url, image_path, text_content, thought_signature

        # 检查是否只有文本响应（没有图像）
        if text_parts and len(text_parts) == len(
            [p for p in parts if not p.get("thought", False)]
        ):
            # 所有非思考part都是文本，没有图像
            text_content = " ".join([p["text"] for p in text_parts])
            logger.error("API只返回了文本响应，未生成图像")
            logger.error(f"文本内容: {text_content[:200]}...")
            raise APIError(
                "图像生成失败：API只返回了文本响应。请检查模型名称是否正确，可能需要使用支持图像生成的模型（如 gemini-3-pro-image-preview）",
                None,
                "no_image",
            )

        logger.error("未在响应中找到图像数据")
        raise APIError(
            "图像生成失败：响应格式异常，未找到有效的图像数据", None, "invalid_response"
        )

    async def _parse_openrouter_response(
        self, response_data: dict, session: aiohttp.ClientSession
    ) -> tuple[str | None, str | None, str | None, str | None]:
        """解析 OpenRouter API 响应"""

        image_url = None
        image_path = None
        text_content = None
        thought_signature = None

        if "choices" in response_data:
            choice = response_data["choices"][0]
            message = choice.get("message", {})
            content = message.get("content", "")


            text_chunks: list[str] = []
            image_candidates: list[str] = []

            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue

                    part_type = part.get("type")
                    if part_type == "text" and "text" in part:
                        text_chunks.append(str(part.get("text", "")))
                    elif part_type == "image_url":
                        image_obj = part.get("image_url") or {}
                        if isinstance(image_obj, dict):
                            url_val = image_obj.get("url")
                            if url_val:
                                image_candidates.append(url_val)
            elif isinstance(content, str):
                text_chunks.append(content)

            # 标准 images 字段（兼容 Gemini/OpenAI 混合格式）
            if "images" in message and message["images"]:
                for image_item in message["images"]:
                    if not isinstance(image_item, dict):
                        continue

                    # 典型格式：{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
                    image_obj = image_item.get("image_url")
                    if isinstance(image_obj, dict):
                        url_val = image_obj.get("url")
                        if isinstance(url_val, str) and url_val:
                            image_candidates.append(url_val)
                    elif isinstance(image_obj, str) and image_obj:
                        image_candidates.append(image_obj)
                    # 退化格式：{"url": "..."}
                    elif isinstance(image_item.get("url"), str):
                        image_candidates.append(image_item["url"])

            # 组装文本内容
            if text_chunks:
                text_content = " ".join([t for t in text_chunks if t]).strip() or None

            # 按顺序处理图像候选
            for candidate_url in image_candidates:
                if isinstance(candidate_url, str) and candidate_url.startswith("data:image/"):
                    image_url, image_path = await self._parse_data_uri(candidate_url)
                elif isinstance(candidate_url, str):
                    # 对于可访问的 http(s) 链接，直接返回 URL，避免重复下载占用带宽
                    if candidate_url.startswith("http://") or candidate_url.startswith("https://"):
                        return candidate_url, None, text_content, thought_signature
                    image_url, image_path = await self._download_image(candidate_url, session)
                else:
                    logger.warning(f"跳过非字符串类型的图像URL: {type(candidate_url)}")
                    continue

                if image_url or image_path:
                    return image_url, image_path, text_content, thought_signature

            # content 中查找内联 data URI（文本里）
            if isinstance(content, str):
                extracted_url, extracted_path = await self._extract_from_content(content)
            elif text_content:
                extracted_url, extracted_path = await self._extract_from_content(text_content)
            else:
                extracted_url, extracted_path = (None, None)

            if extracted_url or extracted_path:
                return extracted_url, extracted_path, text_content, thought_signature

        # OpenAI 格式
        elif "data" in response_data and response_data["data"]:
            for image_item in response_data["data"]:
                if "url" in image_item:
                    image_url, image_path = await self._download_image(image_item["url"], session)
                    return image_url, image_path, text_content, thought_signature
                elif "b64_json" in image_item:
                    image_path = await save_base64_image(image_item["b64_json"], "png")
                    if image_path:
                        # 直接使用文件路径，不使用 file:// URI（根据 AstrBot 文档要求）
                        image_url = image_path
                        return image_url, image_path, text_content, thought_signature

        # 如果只有文本内容，也返回
        if text_content:
            return None, None, text_content, thought_signature

        logger.warning("OpenRouter 响应格式不支持或未找到图像数据")
        return None, None, None, None

    async def _parse_data_uri(self, data_uri: str) -> tuple[str | None, str | None]:
        """解析 data URI 格式的图像"""
        try:
            if ";base64," not in data_uri:
                logger.error("无效的 data URI 格式")
                return None, None

            header, base64_data = data_uri.split(";base64,", 1)
            mime_type = header.replace("data:", "")
            format_type = mime_type.split("/")[1] if "/" in mime_type else "png"

            image_path = await save_base64_image(base64_data, format_type)
            if image_path:
                # 直接使用文件路径，不使用 file:// URI（根据 AstrBot 文档要求）
                image_url = image_path
                return image_url, image_path
        except Exception as e:
            logger.error(f"解析 data URI 失败: {e}")

        return None, None

    async def _extract_from_content(self, content: str) -> tuple[str | None, str | None]:
        """从文本内容中提取图像"""
        pattern = r"data:image/([^;]+);base64,([A-Za-z0-9+/=\s]+)"
        matches = re.findall(pattern, content)

        if matches:
            image_format, base64_string = matches[0]
            image_path = await save_base64_image(base64_string, image_format)
            if image_path:
                # 直接使用文件路径，不使用 file:// URI（根据 AstrBot 文档要求）
                image_url = image_path
                return image_url, image_path

        return None, None

    async def _download_image(
        self, image_url: str, session: aiohttp.ClientSession
    ) -> tuple[str | None, str | None]:
        """下载并保存图像"""
        try:
            logger.debug(f"正在下载图像: {image_url[:100]}...")

            async with session.get(
                image_url, timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    logger.error(f"下载图像失败: HTTP {response.status}")
                    return None, None

                image_data = await response.read()
                content_type = response.headers.get("Content-Type", "")

                if "/" in content_type:
                    image_format = content_type.split("/")[1]
                else:
                    image_format = "png"

                image_path = await save_image_data(image_data, image_format)
                if image_path:
                    # 直接使用文件路径，不使用 file:// URI（根据 AstrBot 文档要求）
                    image_url_local = image_path
                    return image_url_local, image_path
        except Exception as e:
            logger.error(f"下载图像失败: {e}")

        return None, None


# 为了兼容性，创建APIClient别名
APIClient = GeminiAPIClient

# 全局 API 客户端实例
_api_client: GeminiAPIClient | None = None


def get_api_client(api_keys: list[str]) -> GeminiAPIClient:
    """获取或创建 API 客户端实例"""
    global _api_client
    if _api_client is None:
        _api_client = GeminiAPIClient(api_keys)
    return _api_client


def clear_api_client():
    """清除全局 API 客户端实例（用于测试）"""
    global _api_client
    _api_client = None
