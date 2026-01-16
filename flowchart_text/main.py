#!/usr/bin/env python3
"""
OCR 矢量还原系统 - 主入口
将位图格式的学术/技术插图还原为 draw.io (mxGraph XML) 可编辑的矢量格式

使用方法:
    python main.py <input_image> [output_file]
    
示例:
    python main.py input.png
    python main.py input.png output.drawio.xml
"""

import argparse
import sys
import os
import io
import base64
from pathlib import Path
from PIL import Image

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config import validate_config
from src.ocr_azure import AzureOCR, OCRResult
from src.ocr_mistral import MistralOCR, MistralOCRResult
from src.content_aligner import align_ocr_results, AlignedTextBlock, calculate_bbox
from src.coord_processor import CoordProcessor
from src.font_calculator import FontCalculator
from src.xml_generator import MxGraphXMLGenerator, TextCellData


class OCRVectorRestorer:
    """
    OCR 矢量还原器
    整合所有模块完成从图像到 draw.io 文件的转换
    """
    
    def __init__(self, use_mistral: bool = True):
        """
        初始化还原器
        
        Args:
            use_mistral: 是否使用 Mistral OCR 进行校对
        """
        self.use_vlm_only = os.environ.get("USE_VLM_OCR", "false").lower() == "true"
        # In VLM-only mode, disable separate refinement step
        self.use_mistral = use_mistral if not self.use_vlm_only else False
        
        self.azure_ocr = None
        self.mistral_ocr = None
        
    def _init_ocr_clients(self):
        """延迟初始化 OCR 客户端"""
        if self.use_vlm_only:
            if self.mistral_ocr is None:
                try:
                    print("🔧 Mode: VLM-Only (Bypassing Azure)")
                    self.mistral_ocr = MistralOCR()
                except Exception as e:
                    print(f"警告: 无法初始化 VLM OCR: {e}")
                    raise
            return

        if self.azure_ocr is None:
            try:
                self.azure_ocr = AzureOCR()
            except ValueError as e:
                print(f"警告: 无法初始化 Azure OCR: {e}")
                raise
        
        if self.use_mistral and self.mistral_ocr is None:
            try:
                self.mistral_ocr = MistralOCR()
            except ValueError as e:
                print(f"警告: 无法初始化 Mistral OCR: {e}")
                self.use_mistral = False
    
    def process_image(self, image_path: str, output_path: str = None) -> str:
        """
        处理图像并生成 draw.io 文件
        
        Args:
            image_path: 输入图像路径
            output_path: 输出文件路径（可选）
            
        Returns:
            str: 输出文件路径
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        # 修复后的输出路径逻辑
        if output_path is None:
            output_path = image_path.with_suffix(".drawio.xml")
        else:
            output_path = Path(output_path)
            # 正确判断是否以 .drawio.xml 结尾
            if not output_path.name.lower().endswith(".drawio.xml"):
                # 先清空现有后缀，再添加目标后缀，避免重复
                output_path = output_path.with_suffix("").with_suffix(".drawio.xml")
        
        print(f"📄 输入文件: {image_path}")
        print(f"📝 输出文件: {output_path}")
        print()
        
        # 获取图像尺寸
        with Image.open(image_path) as img:
            image_width, image_height = img.size
        print(f"📐 图像尺寸: {image_width} x {image_height} 像素")
        
        # 初始化 OCR 客户端
        print("\n🔧 初始化 OCR 服务...")
        self._init_ocr_clients()
        
        # 步骤 1: Azure OCR 或 VLM End-to-End
        if self.use_vlm_only:
            print("\n📖 步骤 1/5: 使用 VLM (End-to-End) 识别文字...")
            try:
                azure_result = self.mistral_ocr.analyze_image_end_to_end(str(image_path))
            except AttributeError:
                # Fallback if method missing (shouldn't happen with my edits)
                raise RuntimeError("MistralOCR missing analyze_image_end_to_end method")
        else:
            print("\n📖 步骤 1/5: 使用 Azure OCR 识别文字...")
            azure_result = self.azure_ocr.analyze_image(str(image_path))
            
        print(f"   识别到 {len(azure_result.text_blocks)} 个文字块")
        
        # 步骤 2: Mistral OCR (Crop Strategy - 升级版)
        aligned_blocks = []
        if self.use_mistral:
            print("\n🔍 步骤 2/5: 使用 Mistral OCR (Crop Mode) 进行精准识别...")
            try:
                # 准备 Crops
                pil_img = Image.open(image_path)
                crop_data = [] # (id, b64)
                block_map = {} # id -> AzureBlock
                
                print(f"   正在裁剪 {len(azure_result.text_blocks)} 个文本区域...")
                for i, block in enumerate(azure_result.text_blocks):
                    # 获取 bbox
                    poly = block.polygon
                    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                    
                    # Padding (重要：防止切掉边缘)
                    pad = 5
                    x1 = max(0, int(x1 - pad))
                    y1 = max(0, int(y1 - pad))
                    x2 = min(pil_img.width, int(x2 + pad))
                    y2 = min(pil_img.height, int(y2 + pad))
                    
                    crop = pil_img.crop((x1, y1, x2, y2))
                    
                    # Base64
                    buf = io.BytesIO()
                    crop.save(buf, format="PNG")
                    b64 = base64.b64encode(buf.getvalue()).decode()
                    
                    cid = str(i)
                    # Pass hint text to help VLM focus
                    crop_data.append((cid, b64, block.text))
                    block_map[cid] = block

                # 批量识别
                print(f"   发送 Mistral 视觉识别请求 (分批处理)...")
                if crop_data:
                    mistral_crop_results = self.mistral_ocr.recognize_crops(crop_data)
                    
                    # 合并结果
                    for res in mistral_crop_results:
                        cid = res["id"]
                        m_text = res["text"]
                        is_formula = res["is_formula"]
                        orig_block = block_map[cid]
                        
                        # 决策逻辑：如果 Mistral 返回空，回退 Azure
                        final_text = m_text if m_text and m_text.strip() else orig_block.text
                        
                        # 保留 Azure 坐标
                        aligned = AlignedTextBlock(
                            text=final_text,
                            polygon=orig_block.polygon,
                            confidence=orig_block.confidence, 
                            font_size_px=orig_block.font_size_px,
                            is_latex=is_formula,
                            original_azure_text=orig_block.text,
                            latex_source="mistral_crop" if m_text and m_text.strip() else None
                        )
                        aligned_blocks.append(aligned)
                else:
                    print("   没有需要识别的文本块。")

            except Exception as e:
                print(f"   ⚠️  Mistral Crop OCR 失败: {e}")
                import traceback
                traceback.print_exc()
                self.use_mistral = False # 标记失败，触发后续 Fallback
        
        if not self.use_mistral or not aligned_blocks:
             if not aligned_blocks and azure_result.text_blocks:
                 print("\n⏭️  步骤 2/5 (Fallback): 仅使用 Azure OCR")
                 for block in azure_result.text_blocks:
                     aligned_blocks.append(AlignedTextBlock(
                        text=block.text,
                        polygon=block.polygon,
                        confidence=block.confidence,
                        font_size_px=block.font_size_px,
                        is_latex=False,
                        original_azure_text=block.text,
                        latex_source=None
                     ))
        
        # 步骤 3: 内容对齐 (此版本Crop模式已完成对齐，这里主要用于日志或跳过)
        print("\n🔗 步骤 3/5: 对齐逻辑已集成在 Crop 流程中")
        # aligned_blocks = align_ocr_results(azure_result, mistral_result) # DEPRECATED in this mode
        
        # --- 新增：文本块合并（段落/行合并）---
        from src.content_aligner import merge_text_blocks
        print("🧩 执行文本块/段落合并...")
        # 默认阈值10-15px，可视情况调整
        aligned_blocks = merge_text_blocks(aligned_blocks, line_threshold=12.0)
        # ------------------------------------
        
        latex_count = sum(1 for b in aligned_blocks if b.is_latex)
        print(f"   对齐完成，其中 {latex_count} 个为 LaTeX 公式")
        
        # 步骤 4: 坐标和字号处理
        print("\n📐 步骤 4/5: 处理坐标和字号...")
        # 不进行坐标归一化，直接使用原图坐标
        # 这样可以保持原图中文字的相对位置和大小
        coord_processor = CoordProcessor(
            source_width=image_width,
            source_height=image_height,
            canvas_width=None,  # 使用原图宽度
            canvas_height=None  # 使用原图高度
        )
        # 初始化字号计算器（使用 Cap-Height 算法）
        font_calculator = FontCalculator(
            canvas_scale=coord_processor.uniform_scale
        )
        from config import RENDER_RATIO, CAP_HEIGHT_RATIO
        print(f"   使用 Cap-Height 算法 (CAP_HEIGHT_RATIO={CAP_HEIGHT_RATIO}, RENDER_RATIO={RENDER_RATIO})")
        
        # 处理每个文本块
        processed_cells = []
        for block in aligned_blocks:
            # 归一化坐标
            geometry = coord_processor.polygon_to_geometry(block.polygon)
            
            # 计算字号（传递边界框信息以支持竖排文字处理）
            font_result = font_calculator.calculate_font_size(
                text=block.text, 
                polygon_height_px=block.font_size_px,
                bbox_width=geometry.get("width"),
                bbox_height=geometry.get("height"),
                rotation=geometry.get("rotation", 0)
            )
            
            processed_cells.append({
                "text": block.text,
                "geometry": geometry,
                "font_size": font_result.estimated_pt,
                "is_latex": block.is_latex
            })
        
        print(f"   处理了 {len(processed_cells)} 个文本单元格")
        
        # 步骤 5: 生成 XML
        print("\n📄 步骤 5/5: 生成 draw.io XML...")
        # 使用原图尺寸作为页面大小
        generator = MxGraphXMLGenerator(
            diagram_name=image_path.stem,
            page_width=image_width,
            page_height=image_height
        )
        
        text_cells = []
        for cell_data in processed_cells:
            geo = cell_data["geometry"]
            cell = generator.create_text_cell(
                text=cell_data["text"],
                x=geo["x"],
                y=geo["y"],
                width=max(geo["width"], 20),  # 最小宽度
                height=max(geo["height"], 10),  # 最小高度
                font_size=cell_data["font_size"],
                is_latex=cell_data["is_latex"],
                rotation=geo.get("rotation", 0)
            )
            text_cells.append(cell)
        
        # 保存文件
        generator.save_to_file(text_cells, str(output_path))
        
        print(f"\n✅ 完成！已生成 {len(text_cells)} 个文本单元格")
        print(f"   输出文件: {output_path}")
        
        return str(output_path)
    
    def preview_ocr(self, image_path: str) -> None:
        """
        预览 OCR 结果（不生成文件）
        
        Args:
            image_path: 图像路径
        """
        self._init_ocr_clients()
        
        print(f"预览 OCR 结果: {image_path}\n")
        
        # Azure OCR
        print("=== Azure OCR 结果 ===")
        azure_result = self.azure_ocr.analyze_image(image_path)
        for i, block in enumerate(azure_result.text_blocks[:10]):
            print(f"{i+1}. '{block.text}' (置信度: {block.confidence:.2f})")
        if len(azure_result.text_blocks) > 10:
            print(f"... 还有 {len(azure_result.text_blocks) - 10} 个文字块")
        
        # Mistral OCR
        if self.use_mistral:
            print("\n=== Mistral OCR 结果 ===")
            mistral_result = self.mistral_ocr.analyze_image(image_path)
            print(mistral_result.raw_text[:500])
            if len(mistral_result.raw_text) > 500:
                print("...")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="OCR 矢量还原系统 - 将图像转换为 draw.io 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python main.py input.png
    python main.py input.png output.drawio.xml
    python main.py input.png --preview
    python main.py input.png --no-mistral
        """
    )
    
    parser.add_argument(
        "input",
        help="输入图像文件路径 (PNG, JPG, BMP, PDF)"
    )
    
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="输出 .drawio.xml 文件路径（可选，默认与输入同名）"
    )
    
    parser.add_argument(
        "--preview",
        action="store_true",
        help="仅预览 OCR 结果，不生成文件"
    )
    
    parser.add_argument(
        "--no-mistral",
        action="store_true",
        help="不使用 Mistral OCR 校对"
    )
    
    args = parser.parse_args()
    
    # 验证配置
    print("🔐 验证 API 配置...")
    if not validate_config():
        print("\n❌ 配置验证失败，请先设置 API 密钥")
        print("   复制 .env.example 为 .env 并填写密钥")
        sys.exit(1)
    print("   配置验证通过\n")
    
    # 创建还原器
    restorer = OCRVectorRestorer(use_mistral=not args.no_mistral)
    
    try:
        if args.preview:
            restorer.preview_ocr(args.input)
        else:
            restorer.process_image(args.input, args.output)
    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

