import logging
import os
import zipfile

from PIL import Image

from .tl_utils import get_plugin_data_dir

logger = logging.getLogger("astrbot")


def split_image(
    image_path: str, rows: int = 6, cols: int = 4, output_dir: str = None
) -> list[str]:
    """
    将图片切分为指定行列的网格

    Args:
        image_path: 源图片路径
        rows: 行数，默认6行
        cols: 列数，默认4列
        output_dir: 输出目录，如果不指定则使用插件数据目录下的 split_output

    Returns:
        List[str]: 切分后的图片文件路径列表，按顺序排列
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size

            # 如果图片是横向的（宽 > 高），交换行列数
            # 默认是纵向切割：6行4列
            if width > height and rows > cols:
                rows, cols = cols, rows
                logger.debug(f"检测到横向图片，自动调整切割网格为 {rows}行 x {cols}列")

            # 计算每个切片的宽高
            piece_width = width // cols
            piece_height = height // rows

            # 如果未指定输出目录，则使用插件的标准数据目录
            if not output_dir:
                data_dir = get_plugin_data_dir()
                output_dir = os.path.join(str(data_dir), "split_output")

            # 获取源文件名（不含扩展名和路径）作为子目录，避免文件混淆
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            # 最终存储目录: .../split_output/base_name/
            final_output_dir = os.path.join(output_dir, base_name)

            if not os.path.exists(final_output_dir):
                os.makedirs(final_output_dir)

            output_files = []

            # 遍历网格进行切分
            # r 从 0 到 rows-1, c 从 0 到 cols-1
            for r in range(rows):
                for c in range(cols):
                    # 计算裁剪区域 (left, upper, right, lower)
                    left = c * piece_width
                    upper = r * piece_height
                    right = left + piece_width
                    lower = upper + piece_height

                    # 裁剪
                    piece = img.crop((left, upper, right, lower))

                    # 生成文件名，格式：{base_name}_{id}.png
                    # id 从 1 开始，按行优先顺序
                    idx = r * cols + c + 1
                    file_name = f"{base_name}_{idx}.png"
                    file_path = os.path.join(final_output_dir, file_name)

                    # 保存
                    piece.save(file_path, "PNG")
                    output_files.append(file_path)

            return output_files

    except Exception as e:
        logger.error(f"Error splitting image: {e}")
        return []


def create_zip(files: list[str], output_filename: str = None) -> str:
    """
    将文件列表打包成zip

    Args:
        files: 文件路径列表
        output_filename: 输出zip文件名（包含路径）。如果不指定，则使用第一个文件的目录 + 目录名.zip

    Returns:
        str: zip文件路径，失败返回None
    """
    if not files:
        return None

    try:
        if not output_filename:
            first_file = files[0]
            dir_path = os.path.dirname(first_file)
            dir_name = os.path.basename(dir_path)
            # 输出到目录的同级，即 .../split_output/base_name.zip
            output_filename = os.path.join(os.path.dirname(dir_path), f"{dir_name}.zip")

        with zipfile.ZipFile(output_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file in files:
                zipf.write(file, os.path.basename(file))

        return output_filename
    except Exception as e:
        logger.error(f"Error creating zip: {e}")
        return None
