"""优化的图像生成提示词模块（基于 Gemini 官方文档）"""


def enhance_prompt_for_gemini(prompt: str) -> str:
    """基于 Gemini 官方文档优化提示词"""
    # 直接返回用户的提示词，不添加强制的风格要求
    # 让用户自己控制风格（二次元、写实等）
    return prompt


def enhance_prompt_for_figure(prompt: str) -> str:
    """手办化提示词增强（基于官方文档优化）"""
    figure_keywords = ["手办", "figure", "模型", "手办化", "手办模型"]
    if any(keyword in prompt.lower() for keyword in figure_keywords):
        return f"""Create a high-quality 1/7 scale PVC figure based on the following description.
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

    # 如果不是手办相关，使用通用优化
    return enhance_prompt_for_gemini(prompt)
