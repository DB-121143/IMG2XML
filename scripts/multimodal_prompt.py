import os
import json
import base64
import yaml
from pathlib import Path
from openai import OpenAI
import httpx
import urllib3

# 屏蔽 InsecureRequestWarning 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# -------------------------- 全局配置与常量 --------------------------
# SAM3初始提示词（用于过滤重复，避免返回无效提示词）
SAM3_INITIAL_PROMPTS = [
    "icon", "picture", "rectangle", "section_panel",
    "title_bar", "arrow", "rounded rectangle"
]

# -------------------------- 配置加载函数 --------------------------
def load_multimodal_config():
    """加载multimodal配置（带容错处理）"""
    try:
        CONFIG_PATH = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "config",
            "config.yaml"
        )
        if not Path(CONFIG_PATH).exists():
            raise FileNotFoundError(f"配置文件不存在：{CONFIG_PATH}")
        
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            CONFIG = yaml.safe_load(f)
        
        # 校验multimodal节点是否存在
        if "multimodal" not in CONFIG:
            raise KeyError("配置文件中缺少'multimodal'节点")
        
        return CONFIG["multimodal"]
    
    except Exception as e:
        print(f"配置加载失败：{str(e)}")
        return None

# 提前加载配置（全局单例，避免重复读取文件）
MULTIMODAL_CONFIG = load_multimodal_config()

# -------------------------- 工具函数 --------------------------
def image_to_base64(image_path: str) -> tuple[str, str]:
    """
    优化版：将图片转为base64编码，并返回图片格式（适配png/jpg/jpeg）
    :param image_path: 图片路径
    :return: (img_base64_str, img_format_str)
    """
    img_path = Path(image_path)
    
    # 校验图片是否存在
    if not img_path.exists():
        raise FileNotFoundError(f"图片文件不存在：{image_path}")
    
    # 获取并处理图片格式（兼容jpg/jpeg，统一转为小写）
    img_format = img_path.suffix.lstrip(".").lower()
    img_format = "jpeg" if img_format == "jpg" else img_format
    
    # 读取并编码为base64
    with open(image_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")
    
    return img_base64, img_format

def parse_json_from_response(content: str) -> dict:
    """Helper to parse JSON from VLM response with loose filtering"""
    try:
        content = str(content).strip()
        # Remove code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # Basic cleanup: sometimes models add text before/after
        start = content.find('{')
        end = content.rfind('}')
        
        if start != -1 and end != -1:
            json_str = content[start:end+1]
            return json.loads(json_str)
        return {}
    except Exception as e:
        print(f"JSON Parse Warning: {e}")
        return {}

# -------------------------- 核心函数：获取补充提示词 --------------------------
def get_supplement_prompts(
    mask_vis_path: str, 
    existing_prompts: list = None,
    round_index: int = 1, 
    original_image_path: str = None
) -> dict:
    """
    优化版：分轮次获取补充提示词
    
    :param mask_vis_path: 掩码可视化图路径 (SECOND image)
    :param existing_prompts: 已识别的提示词列表（告诉模型这些不需要了）
    :param round_index: 当前轮次 (2=SingleWord, 3=TwoWords, 4=Phrase)
    :param original_image_path: 原始图片路径 (FIRST image)
    :return: 字典 {"icon_prompts": [], "picture_prompts": [], "has_missing": bool}
    """
    # 前置校验：配置加载失败直接返回空结构
    if not MULTIMODAL_CONFIG:
        print("错误：多模态配置加载失败，无法调用API")
        return {"icon_prompts": [], "picture_prompts": [], "has_missing": False}
    
    # 前置校验：图片路径有效性
    if not mask_vis_path or not Path(mask_vis_path).exists():
        print(f"错误：掩码可视化图路径无效或文件不存在：{mask_vis_path}")
        return {"icon_prompts": [], "picture_prompts": [], "has_missing": False}
        
    if not original_image_path or not Path(original_image_path).exists():
        # 如果没传原图，回退到只发 mask_vis
        print(f"警告：未传入 original_image_path，VLM 仅看 Mask 图")
        original_image_path = mask_vis_path

    # 1. 准备图片数据 
    try:
        # 原始图 (FIRST)
        orig_base64, orig_format = image_to_base64(original_image_path)
        # Mask图 (SECOND)
        mask_base64, mask_format = image_to_base64(mask_vis_path)
    except Exception as e:
        print(f"图片处理失败: {e}")
        return {"icon_prompts": [], "picture_prompts": [], "has_missing": False}

    # 2. 根据轮次构建 Prompt
    base_prompt = """You are given two images:
1. FIRST: Original image
2. SECOND: Same image with colored overlays (GREEN/Color=icons/elements, BLUE=photos)

TASK: Scan blank areas in SECOND image (no colored overlay).
Find any MISSED icon, diagram, graph, or picture in blank areas.

CATEGORY DISTINCTION (based on visual features):
- icon_prompts: for SIMPLE graphics, shapes, flat icons, cartoon-like (e.g. server, user, database)
- picture_prompts: for COMPLEX images, rich textures, screenshots, logos (e.g. photo, logo, complex diagram)

"""
    
    if round_index == 2:
        print(f"[VLM Round 2] Scanning for SINGLE WORD prompts...")
        specific_rules = """RULES:
- Each prompt must be EXACTLY ONE WORD (single noun)
- Provide AT MOST 3 prompts for icon_prompts, AT MOST 3 prompts for picture_prompts
- For picture_prompts: use CONCRETE OBJECT nouns (what object is shown), NOT abstract words like "photo" or "image"
- DO NOT include: arrow, line, connector, text, label (we only want icons and photos)
- If nothing is missing, set has_missing to false
"""
    elif round_index == 3:
        print(f"[VLM Round 3] Scanning for TWO WORD prompts...")
        specific_rules = """RULES:
- Each prompt must be EXACTLY TWO WORDS (adjective + noun or noun + noun)
- Provide AT MOST 2 prompts for icon_prompts, AT MOST 2 prompts for picture_prompts
- For picture_prompts: use CONCRETE OBJECT descriptions
- DO NOT include: arrow, line, connector, text, label
- If nothing is missing, set has_missing to false
"""
    elif round_index >= 4:
        print(f"[VLM Round 4] Scanning for SHORT PHRASES (3-5 words)...")
        specific_rules = """RULES:
- Use short noun phrases (3-5 words per phrase)
- Provide AT MOST 2 prompts for icon_prompts, AT MOST 2 prompts for picture_prompts
- DO NOT include: arrow, line, connector, text, label
- If nothing is missing, set has_missing to false
"""
    else:
        # Default / Fallback
        specific_rules = """RULES:
- Provide specific object names.
- DO NOT include: arrow, line, connector, text, label.
"""

    prompt = base_prompt + specific_rules + """
Output JSON Format:
{"has_missing": true/false, "icon_prompts": ["word1", "word2"], "picture_prompts": ["word1"]}
"""

    messages = [
        {
            "role": "system",
            "content": "You are an expert in analyzing flowchart and diagram masks."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/{orig_format};base64,{orig_base64}"}
                },
                {
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/{mask_format};base64,{mask_base64}"}
                }
            ]
        }
    ]

    # 3. 调用API
    try:
        mode = MULTIMODAL_CONFIG.get("mode", "api")
        if mode == "local":
            api_key = MULTIMODAL_CONFIG.get("local_api_key", "ollama")
            base_url = MULTIMODAL_CONFIG.get("local_base_url", "http://localhost:11434/v1")
            model_name = MULTIMODAL_CONFIG.get("local_model")
        else:
            api_key = MULTIMODAL_CONFIG.get('api_key')
            base_url = MULTIMODAL_CONFIG.get('base_url')
            model_name = MULTIMODAL_CONFIG.get("model")

        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=1,
            http_client=httpx.Client(verify=False)
        )
        
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=MULTIMODAL_CONFIG.get("max_tokens", 4000),
            timeout=float(MULTIMODAL_CONFIG.get("timeout", 60)),
            temperature=0.1
        )
        
        if not response.choices:
            print("错误：API返回无有效choices内容")
            return {"icon_prompts": [], "picture_prompts": [], "has_missing": False}
            
        content = response.choices[0].message.content
        # print(f"Raw VLM Response: {content}") # Debug

        # 4. JSON 解析与清理
        cleaned_json = parse_json_from_response(content)
        
        # 确保 keys 存在
        if "icon_prompts" not in cleaned_json: cleaned_json["icon_prompts"] = []
        if "picture_prompts" not in cleaned_json: cleaned_json["picture_prompts"] = []
        if "has_missing" not in cleaned_json: cleaned_json["has_missing"] = False
        
        # 显式标记成功
        cleaned_json["error"] = False
        
        return cleaned_json

    except Exception as e:
        print(f"API调用失败: {e}")
        # 显式标记失败
        return {"icon_prompts": [], "picture_prompts": [], "has_missing": False, "error": True}

# -------------------------- 独立测试入口 --------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", required=True)
    parser.add_argument("--orig", required=True)
    parser.add_argument("--round", type=int, default=2)
    args = parser.parse_args()
    
    res = get_supplement_prompts(args.mask, round_index=args.round, original_image_path=args.orig)
    print(res)

