"""图像生成提示词模块"""


def enhance_prompt_for_gemini(prompt: str) -> str:
    """优化提示词"""

    return prompt


def get_avatar_prompt(prompt: str) -> str:
    """获取头像模式提示词"""
    base_prompt = "设计一个高质量的社交媒体头像，构图居中，主体清晰，背景简洁大方，适合圆形裁剪，具有极高的辨识度。"
    if prompt:
        return f"""{base_prompt}
用户需求：{prompt}"""
    return base_prompt


def get_poster_prompt(prompt: str) -> str:
    """获取海报模式提示词"""
    base_prompt = "设计一张具有视觉冲击力的电影级宣传海报，排版专业，构图富有张力，色彩搭配具有艺术感，细节丰富。"
    if prompt:
        return f"""{base_prompt}
海报主题与要求：{prompt}"""
    return base_prompt


def get_wallpaper_prompt(prompt: str) -> str:
    """获取壁纸模式提示词"""
    base_prompt = "创作一张 4K 超高清电脑桌面壁纸，画面细腻，视野开阔，光影效果绝佳，令人赏心悦目，无噪点。"
    if prompt:
        return f"""{base_prompt}
画面描述：{prompt}"""
    return base_prompt


def get_card_prompt(prompt: str) -> str:
    """获取卡片模式提示词"""
    base_prompt = "设计一张精致的卡片插画，风格唯美，色彩清新，适合作为明信片或收藏卡，画面具有故事感。"
    if prompt:
        return f"""{base_prompt}
卡片内容：{prompt}"""
    return base_prompt


def get_mobile_prompt(prompt: str) -> str:
    """获取手机壁纸模式提示词"""
    base_prompt = "设计一张竖屏手机壁纸，构图完美适配手机屏幕，主体位置合理不遮挡时间显示，视觉效果惊艳。"
    if prompt:
        return f"""{base_prompt}
画面描述：{prompt}"""
    return base_prompt


def get_sticker_prompt(prompt: str = "") -> str:
    """获取表情包提示词"""
    base_prompt = """为我生成图中角色的绘制 Q 版的，LINE 风格的半身像表情包，注意头饰要正确
彩色手绘风格，使用 6x4 布局，涵盖各种各样的常用聊天语句，或是一些有关的娱乐 meme
其他需求：不要原图复制，高清修复，高质量。所有标注为手写简体中文。
"""

    if prompt:
        return f"""{base_prompt}
附加要求：{prompt}"""
    return base_prompt


def get_figure_prompt(prompt: str, style_type: int = 1) -> str:
    """
    获取手办化提示词

    Args:
        prompt: 用户的额外描述
        style_type: 风格类型 (1: PVC标准版, 2: 树脂GK收藏版)
    """
    if style_type == 2:
        base_prompt = "将画面中的角色重塑为顶级收藏级树脂手办，全身动态姿势，置于角色主题底座，高精度材质，手工涂装，肌肤纹理与服装材质真实分明。戏剧性硬光为主光源，凸显立体感，无过曝；强效补光消除死黑，细节完整可见。背景为窗边景深模糊，侧后方隐约可见产品包装盒。博物馆级摄影质感，全身细节无损，面部结构精准。禁止：任何2D元素或照搬原图、塑料感、面部模糊、五官错位、细节丢失。"
    else:
        base_prompt = f"""Create a high-quality 1/7 scale PVC figure based on the following description.
The figure should be positioned on a circular plastic display base with a collectible box in the background.
The box should feature a large clear window displaying the main artwork, product name, brand logo, barcode, and specifications panel.
Include a small price tag on the corner of the box.
Place a computer monitor in the background showing the ZBrush modeling process of this figure.

Figure requirements:
- Realistic 3D appearance with proper PVC texture detail
- Natural pose with accurate body proportions
- High-quality sculpting with no execution errors
- If original is not full body, extend to full figure version
- Expressions and poses must match the description exactly
- Head should not be oversized, legs should not be too short
- Avoid cartoonish proportions unless specified as chibi style
- For animal figures, reduce fur texture realism to look more like collectible figures
- No outline strokes, must be fully 3D dimensional
- Proper perspective with foreground/background relationship

Subject: {prompt}

Technical specifications:
- Professional photography lighting setup
- High-resolution rendering with detailed textures
- Realistic material properties and reflections
- Collectible figure presentation style"""

    if prompt and style_type == 2:
        return f"""{base_prompt}
{prompt}"""

    # PVC mode already includes prompt in Subject field
    return base_prompt


def get_generation_prompt(prompt: str) -> str:
    """获取生图专用提示词"""
    return f"""图像生成任务：{prompt}

重要要求：
- 根据用户的描述生成全新的原创图像
- 生成图像要完全符合用户的描述要求

重要：这是一项图像生成任务，请根据描述创建全新的图像！"""


def get_modification_prompt(prompt: str) -> str:
    """获取改图专用提示词"""
    return f"""请根据参考图像进行以下修改：{prompt}

重要要求：
- 必须基于提供的参考图像进行修改，不能忽略原图
- 保持图像的整体构图和主要对象
- 严格按照用户要求进行修改，不要返回原图
- 如果修改涉及颜色、风格或背景，必须有明显变化
- 确保修改后的图像与原图有可区分的差异"""


def get_auto_modification_prompt(prompt: str) -> str:
    """获取自动改图提示词（用于快捷模式中的自动判断）"""
    return f"""图像修改任务：{prompt}

请严格按照用户要求修改参考图像，确保：
1. 必须基于提供的参考图像进行修改
2. 保持主要对象和构图，只修改用户要求的部分
3. 修改后的图像要与原图有明显区别
4. 不要返回完全相同的原图
5. 修改要自然、合理，保持图像质量

重要：这是一项图像修改任务，不是生成新图像，必须基于参考图像进行修改！"""


def get_style_change_prompt(style: str, prompt: str = "") -> str:
    """获取风格转换提示词"""
    full_prompt = f"将参考图像改为{style}风格"
    if prompt:
        full_prompt += f"，{prompt}"
    return full_prompt


def enhance_prompt_for_figure(prompt: str) -> str:
    """兼容旧接口"""
    return get_figure_prompt(prompt, style_type=1)
