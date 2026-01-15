import os
import json
import base64
import yaml
from pathlib import Path
from openai import OpenAI
import httpx

# -------------------------- 全局配置与常量 --------------------------
# SAM3初始提示词（用于过滤重复，避免返回无效提示词）
SAM3_INITIAL_PROMPTS = [
    "icon", "picture", "rectangle", "section_panel",
    "text_bubble", "title_bar", "arrow", "rounded rectangle"
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

# -------------------------- 核心函数：获取补充提示词 --------------------------
def get_supplement_prompts(mask_vis_path: str, existing_prompts: list = None) -> list:
    """
    优化版：统一使用OpenAI SDK调用大模型（支持Remote API和Local Ollama）
    :param mask_vis_path: 掩码可视化图路径
    :param existing_prompts: 已识别的提示词列表（告诉模型这些不需要了）
    :return: 补充提示词列表（如["diamond", "ellipse"]）
    """
    # 前置校验：配置加载失败直接返回空列表
    if not MULTIMODAL_CONFIG:
        print("错误：多模态配置加载失败，无法调用API")
        return []
    
    # 前置校验：图片路径有效性
    if not mask_vis_path or not Path(mask_vis_path).exists():
        print(f"错误：掩码可视化图路径无效或文件不存在：{mask_vis_path}")
        return []
    
    # 构造已有元素描述
    existing_str = ""
    if existing_prompts:
        unique_existing = list(set(existing_prompts))
        existing_str = f"\n   (已知已识别的元素：{', '.join(unique_existing)}，请忽略这些类别)"

    # 1. 准备图片数据 (无论Local还是Remote都尽量走OpenAI Vision格式)
    try:
        img_base64, img_format = image_to_base64(mask_vis_path)
    except Exception as e:
        print(f"图片处理失败: {e}")
        return []

    # 2. 构建提示词 (Optimized from icon/vlm_client.py)
    prompt = f"""
You are an expert in analyzing flowchart and diagram masks.
INPUT: An image of a flowchart/diagram.
- Masked areas (colored/dark): ALREADY IDENTIFIED elements.{existing_str}
- Blank areas (white/bright): UNIDENTIFIED elements.

TASK: Scan the BLANK areas (unidentified) and list the names of missed elements.

CATEGORY DISTINCTION:
- icon_prompts: for SIMPLE graphics, shapes, flat icons (e.g., diamond, cylinder, cloud, user, server).
- picture_prompts: for COMPLEX images, screenshots, logos (e.g., photo, logo, complex diagram).

RULES:
- Provide EXACTLY ONE WORD (single noun) or specific DrawIO shape names.
- Prefer standard DrawIO shapes: diamond, cylinder, cloud, actor, hexagon, triangle, parallelogram.
- Avoid abstract words like "image", "shape", "thing". Use concrete names like "server", "database", "user".
- DO NOT include: arrow, line, connector, text, label.
- If nothing is missed, return empty lists.

Output JSON Format:
{{
  "icon_prompts": ["word1", "word2"],
  "picture_prompts": ["word1", "word2"]
}}
    """.strip()

    try:
        # 3. 确定配置 (Local vs Remote)
        mode = MULTIMODAL_CONFIG.get("mode", "api")
        
        if mode == "local":
            print(f"Using LOCAL Ollama: {MULTIMODAL_CONFIG.get('local_model')}")
            api_key = MULTIMODAL_CONFIG.get("local_api_key", "ollama")
            base_url = MULTIMODAL_CONFIG.get("local_base_url", "http://localhost:11434/v1")
            model_name = MULTIMODAL_CONFIG.get("local_model")
        else:
            print(f"Using REMOTE API: {MULTIMODAL_CONFIG.get('model')}")
            api_key = MULTIMODAL_CONFIG['api_key']
            base_url = MULTIMODAL_CONFIG['base_url']
            model_name = MULTIMODAL_CONFIG["model"]

        # 4. 初始化客户端
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=1,
            http_client=httpx.Client(verify=False)
        )
        
        # 5. 调用API
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{img_format};base64,{img_base64}",
                            }
                        }
                    ]
                }
            ],
            max_tokens=MULTIMODAL_CONFIG["max_tokens"],
            temperature=0.1  # 降低随机性
        )
        
        # check response
        if not response.choices:
            print("错误：API返回无有效choices内容")
            return []
            
        content = response.choices[0].message.content.strip()

        # ----------------- 通用处理逻辑 -----------------
        print(f"多模态模型返回内容：{content}")
        
        # 尝试解析JSON
        try:
            cleaned_content = content.replace("```json", "").replace("```", "").strip()
            
            # 1. 尝试解析对象格式 {"icon_prompts": [], ...}
            start_obj = cleaned_content.find('{')
            end_obj = cleaned_content.rfind('}')
            
            prompts_list = []
            
            if start_obj != -1 and end_obj != -1:
                try:
                    json_str = cleaned_content[start_obj:end_obj+1]
                    data = json.loads(json_str)
                    if isinstance(data, dict):
                        prompts_list.extend(data.get("icon_prompts", []))
                        prompts_list.extend(data.get("picture_prompts", []))
                        # 兼容直接返回 dict 但 key 不一样的情况 (rare fallback)
                        if not prompts_list and "prompts" in data:
                            prompts_list.extend(data["prompts"])
                except:
                    pass # Continue to try array parsing

            # 2. 尝试解析纯列表格式 ["a", "b"] (兼容旧模型输出)
            if not prompts_list:
                start_arr = cleaned_content.find('[')
                end_arr = cleaned_content.rfind(']')
                if start_arr != -1 and end_arr != -1:
                    try:
                        json_str = cleaned_content[start_arr:end_arr+1]
                        data = json.loads(json_str)
                        if isinstance(data, list):
                            prompts_list = data
                    except:
                        pass
            
            # 3. 最终清理与过滤
            if prompts_list:
                # 过滤非字符串、空串、去重
                final_list = list(set([str(p).strip().lower() for p in prompts_list if p and isinstance(p, (str, int))]))
                # 再次过滤黑名单 (箭头/连线/文字)
                blacklist = {"arrow", "line", "connector", "text", "label", "word", "link"}
                final_list = [p for p in final_list if p not in blacklist]
                return final_list
            
            print(f"警告：无法解析有效JSON，原始内容：{content}")
            return []
            
        except json.JSONDecodeError:
            print(f"JSON解析异常，原始内容：{content}")
            return []

    except Exception as e:
        print(f"调用多模态大模型失败：{str(e)}")
        return []

# -------------------------- 独立测试入口 --------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="调用Qwen3多模态API获取SAM3补充提示词")
    parser.add_argument("--mask", "-m", required=True, help="掩码可视化图路径（png/jpg/jpeg）")
    args = parser.parse_args()
    
    final_prompts = get_supplement_prompts(args.mask)
    print(f"\n测试完成 | 最终返回补充提示词列表：{final_prompts}")
