"""
工具函数模块
提供头像管理、文件传输和图像处理功能
"""

import asyncio
import base64
import os
import struct
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp

from astrbot.api import logger


def get_plugin_data_dir() -> Path:
    """获取插件数据目录"""
    # 使用AstrBot的StarTools获取标准数据目录
    from astrbot.api.star import StarTools

    return StarTools.get_data_dir("astrbot_plugin_gemini_image_generation")


async def save_base64_image(base64_data: str, image_format: str = "png") -> str | None:
    """
    保存base64图像数据到文件

    Args:
        base64_data: base64编码的图像数据
        image_format: 图像格式 (png, jpg, jpeg等)

    Returns:
        保存的文件路径，失败返回None
    """
    try:
        # 创建images目录
        images_dir = get_plugin_data_dir() / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"gemini_advanced_image_{timestamp}.{image_format}"
        file_path = images_dir / filename

        # 解码base64数据
        image_bytes = base64.b64decode(base64_data)

        # 写入文件
        with open(file_path, "wb") as f:
            f.write(image_bytes)

        logger.debug(f"图像已保存: {file_path}")
        return str(file_path)

    except Exception as e:
        logger.error(f"保存图像失败: {e}")
        return None


async def save_image_data(image_data: bytes, image_format: str = "png") -> str | None:
    """
    保存图像字节数据到文件

    Args:
        image_data: 图像字节数据
        image_format: 图像格式

    Returns:
        保存的文件路径，失败返回None
    """
    try:
        # 创建images目录
        images_dir = get_plugin_data_dir() / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"gemini_advanced_image_{timestamp}.{image_format}"
        file_path = images_dir / filename

        # 写入文件
        with open(file_path, "wb") as f:
            f.write(image_data)

        logger.debug(f"图像已保存: {file_path}")
        return str(file_path)

    except Exception as e:
        logger.error(f"保存图像失败: {e}")
        return None


async def cleanup_old_images(images_dir: Path | None = None):
    """
    清理超过15分钟的图像文件

    Args:
        images_dir (Path): images 目录路径，如果为None则使用默认路径
    """
    try:
        # 默认路径：插件根目录下的 images 文件夹
        if images_dir is None:
            images_dir = get_plugin_data_dir() / "images"

        if not images_dir.exists():
            return

        current_time = datetime.now()
        cutoff_time = current_time - timedelta(minutes=15)

        # 查找 images 目录下的所有图像文件（支持新旧两种命名格式）
        image_patterns = [
            "gemini_image_*.png",  # 旧格式
            "gemini_image_*.jpg",
            "gemini_image_*.jpeg",
            "gemini_advanced_image_*.png",  # 新格式
            "gemini_advanced_image_*.jpg",
            "gemini_advanced_image_*.jpeg",
        ]

        cleaned_count = 0
        for pattern in image_patterns:
            for file_path in images_dir.glob(pattern):
                try:
                    # 获取文件的修改时间
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)

                    # 如果文件超过15分钟，删除它
                    if file_mtime < cutoff_time:
                        file_path.unlink()
                        cleaned_count += 1
                        logger.debug(f"已清理过期图像: {file_path.name}")

                except Exception as e:
                    logger.warning(f"清理文件 {file_path} 时出错: {e}")

        if cleaned_count > 0:
            logger.debug(f"共清理 {cleaned_count} 个过期图像文件")

    except Exception as e:
        logger.error(f"图像清理过程出错: {e}")


async def download_qq_avatar(
    user_id: str, cache_name: str, images_dir: Path | None = None
) -> str | None:
    """
    下载QQ头像并转换为base64格式

    Args:
        user_id (str): QQ用户ID
        cache_name (str): 缓存文件名前缀
        images_dir (Path): images目录路径，如果为None则使用默认路径

    Returns:
        str: base64格式的头像数据，失败返回None
    """
    try:
        # 默认路径
        if images_dir is None:
            images_dir = get_plugin_data_dir() / "images"

        cache_dir = images_dir / "avatar_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        avatar_file = cache_dir / f"{cache_name}_avatar.jpg"

        # 检查缓存
        if avatar_file.exists() and avatar_file.stat().st_size > 1000:
            with open(avatar_file, "rb") as f:
                cached_data = f.read()
            base64_data = base64.b64encode(cached_data).decode("utf-8")
            logger.debug(f"使用缓存的头像: {cache_name}")
            return base64_data

        # 下载头像
        avatar_url = f"https://q1.qlogo.cn/g?b=qq&nk={user_id}&s=640"

        async with aiohttp.ClientSession() as session:
            async with session.get(
                avatar_url, timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    avatar_data = await response.read()

                    # 检查是否是有效的图片（不是默认头像）
                    if len(avatar_data) > 1000:  # 默认头像通常很小
                        with open(avatar_file, "wb") as f:
                            f.write(avatar_data)

                        base64_data = base64.b64encode(avatar_data).decode("utf-8")
                        logger.debug(f"头像下载成功: {cache_name}")
                        return base64_data
                    else:
                        logger.warning(f"用户 {user_id} 可能使用默认头像，跳过缓存")
                        return None
                else:
                    logger.error(f"下载头像失败: HTTP {response.status}")
                    return None

    except Exception as e:
        logger.error(f"下载头像 {cache_name} 失败: {e}")
        return None


async def send_file(filename: str, host: str, port: int):
    """
    发送文件到远程服务器

    Args:
        filename: 要发送的文件路径
        host: 远程主机地址
        port: 远程主机端口

    Returns:
        str: 远程文件路径，失败返回None
    """
    reader = None
    writer = None

    async def recv_all(reader, size):
        """接收指定大小的数据"""
        data = b""
        while len(data) < size:
            chunk = await reader.read(size - len(data))
            if not chunk:
                break
            data += chunk
        return data

    try:
        # 添加连接超时控制
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=5.0,  # 5秒连接超时
        )

        file_name = os.path.basename(filename)
        file_name_bytes = file_name.encode("utf-8")

        # 发送文件名长度和文件名
        writer.write(struct.pack(">I", len(file_name_bytes)))
        writer.write(file_name_bytes)

        # 发送文件大小
        file_size = os.path.getsize(filename)
        writer.write(struct.pack(">Q", file_size))

        # 发送文件内容，添加总体超时控制
        await writer.drain()
        with open(filename, "rb") as f:
            while True:
                data = f.read(4096)
                if not data:
                    break
                writer.write(data)
                await writer.drain()

        logger.debug(f"文件 {file_name} 发送成功")

        # 接收接收端发送的文件绝对路径
        try:
            file_abs_path_len_data = await recv_all(reader, 4)
            if not file_abs_path_len_data:
                logger.error("无法接收文件绝对路径长度")
                return None
            file_abs_path_len = struct.unpack(">I", file_abs_path_len_data)[0]

            file_abs_path_data = await recv_all(reader, file_abs_path_len)
            if not file_abs_path_data:
                logger.error("无法接收文件绝对路径")
                return None

            file_abs_path = file_abs_path_data.decode("utf-8")
            logger.debug(f"文件在远程服务器保存为: {file_abs_path}")
            return file_abs_path

        except Exception as e:
            logger.error(f"接收远程文件路径失败: {e}")
            return None

    except asyncio.TimeoutError:
        logger.error(f"连接 {host}:{port} 超时")
        return None
    except Exception as e:
        logger.error(f"发送文件失败: {e}")
        return None
    finally:
        if writer:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
        if reader:
            try:
                reader.close()
            except Exception:
                pass


class AvatarManager:
    """头像管理器"""

    def __init__(self, images_dir: Path | None = None):
        self.images_dir = images_dir

    async def get_avatar(self, user_id: str, cache_name: str) -> str | None:
        """
        获取用户头像

        Args:
            user_id: 用户ID
            cache_name: 缓存名称

        Returns:
            base64格式的头像数据
        """
        return await download_qq_avatar(user_id, cache_name, self.images_dir)

    async def cleanup_cache(self):
        """清理头像缓存"""
        if self.images_dir is None:
            self.images_dir = get_plugin_data_dir() / "images"

        cache_dir = self.images_dir / "avatar_cache"
        if cache_dir.exists():
            # 清理超过7天的头像缓存
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(days=7)

            cleaned_count = 0
            for avatar_file in cache_dir.glob("*_avatar.jpg"):
                try:
                    file_mtime = datetime.fromtimestamp(avatar_file.stat().st_mtime)
                    if file_mtime < cutoff_time:
                        avatar_file.unlink()
                        cleaned_count += 1
                        logger.debug(f"已清理过期头像缓存: {avatar_file.name}")
                except Exception as e:
                    logger.warning(f"清理头像缓存 {avatar_file} 时出错: {e}")

            if cleaned_count > 0:
                logger.debug(f"共清理 {cleaned_count} 个过期头像缓存文件")

    async def cleanup_used_avatars(self):
        """清理已使用的头像缓存（别名方法）"""
        await self.cleanup_cache()


# 为了向后兼容，提供一些旧名称的别名
def download_qq_avatar_legacy(user_id: str, cache_name: str) -> str | None:
    """
    下载QQ头像的兼容函数

    Args:
        user_id: QQ用户ID
        cache_name: 缓存文件名

    Returns:
        base64格式的头像数据，失败返回None
    """
    # 使用asyncio运行同步调用
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(download_qq_avatar(user_id, cache_name))
    finally:
        loop.close()
