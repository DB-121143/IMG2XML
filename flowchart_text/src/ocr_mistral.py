"""
Mistral OCR 模块
调用 Mistral AI API 获取 LaTeX/Markdown 格式的文字识别结果
主要用于校对 Azure OCR 结果，特别是处理数学公式
"""

import base64
import re
import io
import time
import tempfile
import json
from pathlib import Path
from dataclasses import dataclass
from PIL import Image

from mistralai import Mistral

# Import types for compatibility
try:
    from src.ocr_azure import OCRResult, TextBlock
except ImportError:
    pass # Will handle inside method if needed or fail later


import sys
# Add project root to path to import scripts
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(Path(__file__).parent.parent))

from config import MISTRAL_API_KEY
from openai import OpenAI
import httpx

# Import local model configuration
try:
    from scripts.multimodal_prompt import MULTIMODAL_CONFIG
except ImportError:
    print("Warning: Could not import local model config from scripts.multimodal_prompt")
    MULTIMODAL_CONFIG = None

@dataclass
class MistralOCRResult:
    """Mistral OCR 识别结果"""
    raw_text: str  # 原始识别文本
    latex_blocks: list[str]  # 提取的 LaTeX 公式块
    markdown_content: str  # Markdown 格式内容


class MistralOCR:
    """Mistral AI OCR 客户端 (支持本地 Qwen 模型)"""
    
    def __init__(self):
        self.mode = "api" # default
        
        # Check global multimodal config for local mode preference
        if MULTIMODAL_CONFIG and MULTIMODAL_CONFIG.get("mode") == "local":
            self.mode = "local"
            print("OCR initialized in LOCAL mode (using Ollama/Qwen)")
            # No preload needed for API/Ollama
        else:
            if not MISTRAL_API_KEY:
                # If local mode is not enabled, we must have API key
                raise ValueError("Mistral API Key 未配置，且未启用本地模式")
            
            self.client = Mistral(api_key=MISTRAL_API_KEY)
            # 使用支持视觉的模型
            self.model = "pixtral-12b-2409"  # Mistral 的多模态模型
    
    def analyze_image(self, image_path: str) -> MistralOCRResult:
        """
        使用 Mistral/Local 视觉模型分析图像
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
            
        if self.mode == "local":
            return self._analyze_image_local(str(image_path))
        else:
            return self._analyze_image_api(str(image_path))

    def _analyze_image_api(self, image_path: str) -> MistralOCRResult:
        """ 原 Mistral API 调用逻辑 """
        # 读取图像并编码为 base64
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        
        # 确定图像 MIME 类型
        suffix = Path(image_path).suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp"
        }
        mime_type = mime_types.get(suffix, "image/png")
        
        # 构建提示词
        prompt = """请仔细识别这张图片中的所有文字内容。

要求：
1. 保留所有文字的原始内容，不要进行翻译或解释
2. 如果发现数学公式或特殊符号（如 $、∑、∫、√ 等），请使用 LaTeX 格式输出
3. 数学公式请用 $...$ 或 $$...$$ 包裹
4. 按照图片中的位置顺序输出文字
5. 保持原有的换行和段落结构

请直接输出识别到的文字内容："""
        
        # 调用 Mistral 视觉 API
        response = self.client.chat.complete(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:{mime_type};base64,{image_base64}"
                        }
                    ]
                }
            ]
        )
        
        raw_text = response.choices[0].message.content
        
        # 提取 LaTeX 公式块
        latex_blocks = self._extract_latex(raw_text)
        
        return MistralOCRResult(
            raw_text=raw_text,
            latex_blocks=latex_blocks,
            markdown_content=raw_text
        )

    def _analyze_image_local(self, image_path: str) -> MistralOCRResult:
        """ 本地 Qwen 模型调用逻辑 (Use Ollama) """
        try:
            # 读取图像并编码为 base64
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            
            # 确定图像 MIME 类型
            suffix = Path(image_path).suffix.lower()
            # 简单处理，默认为 jpeg 如果不确定
            mime_type = "image/jpeg"
            if suffix == ".png": mime_type = "image/png"
            elif suffix == ".webp": mime_type = "image/webp"

            prompt = (
                "Please identify the text content in the image.\n"
                "If the image contains a mathematical formula, please output it in standard LaTeX format (wrapped in $).\n"
                "If the image contains standard text, output it directly.\n"
            )
            
            # 初始化 OpenAI 兼容客户端
            client = OpenAI(
                base_url=MULTIMODAL_CONFIG.get("local_base_url", "http://localhost:11434/v1"),
                api_key=MULTIMODAL_CONFIG.get("local_api_key", "ollama"),
                http_client=httpx.Client(verify=False)
            )

            response = client.chat.completions.create(
                model=MULTIMODAL_CONFIG.get("local_model"),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_base64}",
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1024,
                temperature=0.1
            )
            
            raw_text = response.choices[0].message.content.strip()
            
            latex_blocks = self._extract_latex(raw_text)
            return MistralOCRResult(
                raw_text=raw_text,
                latex_blocks=latex_blocks,
                markdown_content=raw_text
            )
            
        except Exception as e:
            print(f"Local inference failed: {e}")
            return MistralOCRResult(raw_text="", latex_blocks=[], markdown_content="")

    def recognize_crops(self, crops: list) -> list[dict]:
        """
        批量识别裁剪后的图片片段 (Crop API)
        Supports both Mistral SDK and Generic OpenAI-compatible API
        Args:
            crops: List of (id, base64) OR (id, base64, hint_text)
        """
        # Determine mode from self.mode (initialized in __init__)
        # If 'local' OR if we are in 'api' mode but using the generic multimodal config (Qwen/Dashscope)
        if self.mode == "local" or (self.mode == "api" and MULTIMODAL_CONFIG):
             return self._recognize_crops_generic(crops)
             
        # Only fallback to Mistral SDK if strict Mistral API usage is intended (no MULTIMODAL_CONFIG)
        if self.mode == "local":
            return self._recognize_crops_local(crops)
            
        # API Mode logic follows...
        # 分批处理，避免单次请求过大
        BATCH_SIZE = 4 # 每次处理4张图，防止Token超限或混乱

        results = []
        
        for i in range(0, len(crops), BATCH_SIZE):
            batch = crops[i:i+BATCH_SIZE]
            
            # 构建 Prompt
            content_list = [
                {
                    "type": "text", 
                    "text": "Please identify the text content in the following images in order.\n"
                            "If the image contains a mathematical formula, please output it in standard LaTeX format (wrapped in $).\n"
                            "If the image contains standard text, output it directly.\n"
                            "The output format must be strictly as follows (one line per image):\n"
                            "Image 1: <text>\n"
                            "Image 2: <text>\n"
                            "...\n"
                            "Do not output any other nonsense."
                }
            ]
            
            for idx, (img_id, b64) in enumerate(batch):
                content_list.append({
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{b64}"
                })
            
            try:
                response = self.client.chat.complete(
                    model=self.model,
                    messages=[{"role": "user", "content": content_list}]
                )
                
                response_text = response.choices[0].message.content
                # 解析结果
                lines = response_text.strip().split('\n')
                # 建立简单的映射
                # 假设模型严格遵循顺序
                
                # 简单解析器
                batch_result_map = {}
                for line in lines:
                    match = re.match(r"Image\s+(\d+)[:：]\s*(.*)", line)
                    if match:
                        seq_idx = int(match.group(1)) - 1 # 0-indexed
                        text_content = match.group(2).strip()
                        if 0 <= seq_idx < len(batch):
                            real_id = batch[seq_idx][0]
                            # 简单的公式判断
                            # 如果 text 包含特殊 LaTeX 符号，或者以 $ 开头
                            is_formula = bool(re.search(r"[\\]", text_content)) or ("$" in text_content)
                            batch_result_map[real_id] = {
                                "id": real_id,
                                "text": text_content,
                                "is_formula": is_formula
                            }
                
                # 填充缺失项
                for img_id, _ in batch:
                    if img_id in batch_result_map:
                        results.append(batch_result_map[img_id])
                    else:
                        print(f"Warning: Mistral missed image {img_id}")
                        results.append({"id": img_id, "text": "", "is_formula": False})
                        
            except Exception as e:
                print(f"Mistral Batch Error: {e}")
                for img_id, _ in batch:
                    results.append({"id": img_id, "text": "", "is_formula": False})
                    
        return results
    
    def _recognize_crops_generic(self, crops: list) -> list[dict]:
        """ Generic/Local VLM Batch Recognition (Batched for Efficiency) """
        results = []
        BATCH_SIZE = 5
        
        # Load Config
        if self.mode == "api":
            base_url = MULTIMODAL_CONFIG.get("base_url")
            api_key = MULTIMODAL_CONFIG.get("api_key")
            model_name = MULTIMODAL_CONFIG.get("model")
            print(f"   Using Generic API: {model_name} (Batch Size: {BATCH_SIZE})")
            time.sleep(1.0) # Pre-emptive wait
        else:
            base_url = MULTIMODAL_CONFIG.get("local_base_url", "http://localhost:11434/v1")
            api_key = MULTIMODAL_CONFIG.get("local_api_key", "ollama")
            model_name = MULTIMODAL_CONFIG.get("local_model")
            print(f"   Using Local API: {model_name} (Batch Size: {BATCH_SIZE})")

        # Initialize client
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=httpx.Client(verify=False)
        )

        for i in range(0, len(crops), BATCH_SIZE):
            batch = crops[i:i+BATCH_SIZE]
            
            # Rate limit pacing
            if self.mode == "api" and i > 0:
                print(f"   Sleeping 3s to cool down TPM...")
                time.sleep(3.0) 

            # Construct Batch Prompt
            content_list = []
            sys_msg = (
                "You are an OCR engine. Identify text in the images below.\n"
                "Return a strict JSON list: [{\"id\": \"<id>\", \"text\": \"<content>\"}, ...]\n"
                "If formula, use LaTeX ($...$)."
            )
            content_list.append({"type": "text", "text": sys_msg})
            
            for item in batch:
                if len(item) == 3: img_id, b64, hint = item
                else: img_id, b64 = item; hint = ""
                content_list.append({"type": "text", "text": f"\nImage ID: {img_id} (Hint: {hint})"})
                content_list.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
            content_list.append({"type": "text", "text": "\nJSON List:"})

            # Retry Loop
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": content_list}],
                        max_tokens=1024,
                        temperature=0.01,
                        response_format={"type": "json_object"}
                    )
                    content = response.choices[0].message.content.strip()
                    # Cleanup markdown
                    if content.startswith("```json"): content = content[7:]
                    if content.endswith("```"): content = content[:-3]
                    
                    data = json.loads(content)
                    if isinstance(data, dict) and "result" in data: data = data["result"] # common wrapper
                    if isinstance(data, dict): data = list(data.values())[0] if data else [] # heuristic wrapper
                    
                    if not isinstance(data, list): raise ValueError("Format error")
                    
                    # Map results
                    res_map = {str(x.get("id")): x.get("text","") for x in data}
                    for item in batch:
                         rid = str(item[0])
                         txt = res_map.get(rid, "")
                         is_f = "$" in txt or "\\" in txt
                         results.append({"id": rid, "text": txt, "is_formula": is_f})
                    break # Success

                except Exception as e:
                    if "429" in str(e) or "limit" in str(e).lower():
                        wait = (2 ** attempt) * 4.0
                        print(f"      Batch 429. Retry in {wait}s...")
                        time.sleep(wait)
                        if attempt == max_retries -1:
                            # Fail batch
                            for item in batch: results.append({"id": item[0], "text": "", "is_formula": False})
                    else:
                        print(f"      Batch Error: {e}")
                        for item in batch: results.append({"id": item[0], "text": "", "is_formula": False})
                        break
        
        return results

    def _recognize_crops_local(self, crops: list[tuple[str, str]]) -> list[dict]:
        """ 本地 Qwen 模型批量识别 (逐个处理以保证准确性) - 使用 Ollama 接口 """
        results = []
        
        # 初始化 client
        client = OpenAI(
            base_url=MULTIMODAL_CONFIG.get("local_base_url", "http://localhost:11434/v1"),
            api_key=MULTIMODAL_CONFIG.get("local_api_key", "ollama"),
            http_client=httpx.Client(verify=False)
        )
        model_name = MULTIMODAL_CONFIG.get("local_model")

        prompt = (
            "Please identify the short text content in the image.\n"
            "If it is a formula, use LaTeX format.\n"
            "Output ONLY the text detected."
        )
        
        for img_id, b64 in crops:
            try:
                # 调用 Ollama API
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    # 默认使用 png header, Ollama 通常能处理
                                    "image_url": {
                                        "url": f"data:image/png;base64,{b64}",
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=256,
                    temperature=0.1
                )
                
                output_text = response.choices[0].message.content.strip()
                
                text_content = output_text
                is_formula = bool(re.search(r"[\\]", text_content)) or ("$" in text_content)
                
                results.append({
                    "id": img_id,
                    "text": text_content,
                    "is_formula": is_formula
                })
                
            except Exception as e:
                print(f"Local crop inference failed for {img_id}: {e}")
                results.append({"id": img_id, "text": "", "is_formula": False})
                
        return results
    
    def _extract_latex(self, text: str) -> list[str]:
        """
        从文本中提取 LaTeX 公式块
        
        Args:
            text: 包含 LaTeX 的文本
            
        Returns:
            list[str]: 提取的 LaTeX 公式列表
        """
        latex_blocks = []
        
        # 匹配 $$...$$ (块级公式)
        block_pattern = r'\$\$(.*?)\$\$'
        block_matches = re.findall(block_pattern, text, re.DOTALL)
        latex_blocks.extend(block_matches)
        
        # 匹配 $...$ (行内公式)，排除 $$
        inline_pattern = r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)'
        inline_matches = re.findall(inline_pattern, text)
        latex_blocks.extend(inline_matches)
        
        return latex_blocks
    
    def analyze_image_end_to_end(self, image_path: str):
        """
        使用 VLM 直接进行端到端的文字检测和识别 (替换 Azure)
        返回 OCRResult 对象
        """
        # Ensure imports
        try:
            from src.ocr_azure import OCRResult, TextBlock
        except ImportError:
            try:
                from ocr_azure import OCRResult, TextBlock
            except ImportError as e:
                print(f"Error importing types: {e}")
                return None

        image_path = Path(image_path)
        with Image.open(image_path) as img:
            width, height = img.size
            
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        
        suffix = image_path.suffix.lower()
        mime_type = "image/jpeg"
        if suffix == ".png": mime_type = "image/png"
        elif suffix == ".webp": mime_type = "image/webp"

        prompt = (
            "Detect all text in this image. "
            "Return a JSON object with a key 'text_blocks'. "
            "Each item must have 'text' and 'bbox' [ymin, xmin, ymax, xmax] (normalized 0-1000). "
            "Do not use markdown code blocks."
        )

        # Determine configuration based on mode
        mode = MULTIMODAL_CONFIG.get("mode", "local")
        if mode == "api":
            base_url = MULTIMODAL_CONFIG.get("base_url")
            api_key = MULTIMODAL_CONFIG.get("api_key")
            model_name = MULTIMODAL_CONFIG.get("model")
            print(f"Sending End-to-End VLM OCR request (API Mode, Model: {model_name})...")
        else:
            base_url = MULTIMODAL_CONFIG.get("local_base_url", "http://localhost:11434/v1")
            api_key = MULTIMODAL_CONFIG.get("local_api_key", "ollama")
            model_name = MULTIMODAL_CONFIG.get("local_model")
            print(f"Sending End-to-End VLM OCR request (Local Mode, Model: {model_name})...")
        
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=httpx.Client(verify=False)
        )
        
        try:
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
                                    "url": f"data:{mime_type};base64,{image_base64}",
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2048,
                temperature=0.01,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            content = content.replace("```json", "").replace("```", "").strip()
            
            data = json.loads(content)
            blocks_data = data.get("text_blocks", [])
            
            final_blocks = []
            for item in blocks_data:
                txt = item.get("text", "")
                bbox = item.get("bbox", [0,0,0,0])
                if len(bbox) == 4:
                    ymin, xmin, ymax, xmax = bbox
                    # Clamp
                    ymin = max(0, min(1000, ymin))
                    xmin = max(0, min(1000, xmin))
                    ymax = max(0, min(1000, ymax))
                    xmax = max(0, min(1000, xmax))
                    
                    x1 = (xmin / 1000.0) * width
                    y1 = (ymin / 1000.0) * height
                    x2 = (xmax / 1000.0) * width
                    y2 = (ymax / 1000.0) * height
                    
                    # 8-point polygon: TL, TR, BR, BL
                    polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                    
                    # Calculate font size from height
                    font_height = abs(y2 - y1)
                    if font_height < 1: font_height = 10.0 # fallback
                    
                    final_blocks.append(TextBlock(
                        text=txt, polygon=polygon, confidence=1.0, 
                        font_size_px=font_height, font_style=None
                    ))
            
            print(f"VLM Detected {len(final_blocks)} text blocks.")
            return OCRResult(image_width=width, image_height=height, text_blocks=final_blocks)

        except Exception as e:
            print(f"End-to-End VLM OCR failed: {e}")
            return OCRResult(image_width=width, image_height=height, text_blocks=[])

    def verify_text(self, azure_text: str, image_path: str) -> tuple[str, bool]:
        """
        使用 Mistral 校验 Azure OCR 的识别结果
        
        Args:
            azure_text: Azure OCR 识别的文本
            image_path: 原图路径
            
        Returns:
            tuple[str, bool]: (校正后的文本, 是否包含LaTeX)
        """
        result = self.analyze_image(image_path)
        
        has_latex = len(result.latex_blocks) > 0 or '$' in result.raw_text
        
        # 如果 Azure 文本中没有正确识别数学符号，使用 Mistral 结果
        if has_latex and '$' not in azure_text:
            return result.raw_text, True
        
        return azure_text, has_latex


def analyze_image_with_mistral(image_path: str) -> MistralOCRResult:
    """
    便捷函数：使用 Mistral OCR 分析图像
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        MistralOCRResult: OCR 识别结果
    """
    ocr = MistralOCR()
    return ocr.analyze_image(image_path)


if __name__ == "__main__":
    # 测试代码
    import sys
    if len(sys.argv) > 1:
        result = analyze_image_with_mistral(sys.argv[1])
        print("识别结果:")
        print(result.raw_text)
        print(f"\n提取到 {len(result.latex_blocks)} 个 LaTeX 公式块:")
        for i, latex in enumerate(result.latex_blocks[:5]):
            print(f"  {i+1}. {latex}")

