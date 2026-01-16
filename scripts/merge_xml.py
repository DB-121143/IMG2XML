import os
import sys
import xml.etree.ElementTree as ET
import yaml
import argparse
from pathlib import Path
from PIL import Image
import subprocess

# 加载配置
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.yaml")

# 检查配置文件是否存在
if not os.path.exists(CONFIG_PATH):
    print(f"Warning: Config file not found at {CONFIG_PATH}")
    CONFIG = {"paths": {}, "xml_merge": {"layer_rules": {}}}
else:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        CONFIG = yaml.safe_load(f)

FLOWCHART_CONFIG = CONFIG.get("paths", {})
LAYER_CONFIG = CONFIG.get("xml_merge", {}).get("layer_rules", {})

def get_image_size(image_path: str) -> tuple[float, float]:
    """获取图片尺寸"""
    try:
        with Image.open(image_path) as img:
            return float(img.width), float(img.height)
    except Exception as e:
        print(f"获取图片尺寸失败: {e}，将尝试从XML推断或使用默认值")
        return 0, 0

def run_text_extraction(image_path: str) -> str:
    """
    运行文字识别（调用flowchart_text模块）
    返回生成的文字XML路径
    """
    # 确定脚本路径和输出目录
    script_path = FLOWCHART_CONFIG.get("flowchart_text_script")
    
    # 核心修正：强制将OCR文字输出到图片所在目录（任务目录），避免多任务并发时写到同一个全局目录造成覆盖
    output_dir = os.path.dirname(image_path)
    
    # 路径容错处理
    if not script_path or not os.path.exists(script_path):
        # 尝试默认相对路径
        base_dir = os.path.dirname(os.path.dirname(__file__))
        script_path = os.path.join(base_dir, "flowchart_text", "main.py")
    
    # 构建输出文件名
    
    # 构建输出文件名 (flowchart_text通常生成 {stem}_text.drawio.xml)
    stem = Path(image_path).stem
    expected_xml_path = os.path.join(output_dir, f"{stem}_text.drawio.xml")
    
    print(f"正在运行文字提取: {script_path} -> {image_path}")
    
    try:
        # 调用子进程运行 flowchart_text/main.py
        # flowchart_text/main.py 参数: <input_image> [output_file]
        # 注意：flowchart_text 可能会写入到默认路径或其内部逻辑决定的路径
        # 这里为了稳妥，我们指定输出路径（如果脚本支持）
        # 查看flowchart_text/main.py源码（之前已查看）：支持 python main.py <input> [output]
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 使用绝对路径
        abs_image_path = os.path.abspath(image_path)
        abs_output_path = os.path.abspath(expected_xml_path)
        
        # 设置工作目录为 flowchart_text 所在目录，避免import错误
        cwd = os.path.dirname(script_path)
        
        # 传递 VLM OCR 配置
        env = os.environ.copy()
        if CONFIG.get("multimodal", {}).get("force_vlm_ocr", False):
            env["USE_VLM_OCR"] = "true"
        
        subprocess.run(
            [sys.executable, os.path.basename(script_path), abs_image_path, abs_output_path],
            cwd=cwd,
            env=env,
            check=True
        )
        
        if os.path.exists(expected_xml_path):
            return expected_xml_path
        else:
            print(f"Error: 文字提取脚本运行成功但未生成文件: {expected_xml_path}")
            return ""
            
    except subprocess.CalledProcessError as e:
        print(f"文字提取脚本运行失败: {e}")
        return ""
    except Exception as e:
        print(f"调用文字提取脚本时发生错误: {e}")
        return ""

def merge_xml(sam3_xml_path: str, text_xml_path: str, image_path: str, output_path: str):
    """
    合并SAM3提取的图形XML和OCR提取的文字XML
    """
    print(f"开始合并XML: {sam3_xml_path} + {text_xml_path} -> {output_path}")
    
    # 1. 解析SAM3 XML（图形）
    try:
        sam3_tree = ET.parse(sam3_xml_path)
        sam3_root = sam3_tree.getroot()
    except Exception as e:
        print(f"解析SAM3 XML失败: {e}")
        return

    # 2. 解析文字 XML
    text_root = None
    if text_xml_path and os.path.exists(text_xml_path):
        try:
            text_tree = ET.parse(text_xml_path)
            text_root = text_tree.getroot()
        except Exception as e:
            print(f"解析文字 XML失败: {e}，将只保留图形")
    
    # 3. 初始化合并后的 XML 结构
    # 使用 sam3_root 作为基础，确保画板设置（尺寸、网格等）保留
    # 但需要清空 root 下的 mxCell，重新按顺序添加
    
    # 查找 root 节点 (mxGraphModel -> root)
    model = sam3_root.find(".//mxGraphModel")
    if model is None:
        print("Error: Invalid DrawIO XML (no mxGraphModel)")
        return
        
    root_elem = model.find("root")
    if root_elem is None:
        print("Error: Invalid DrawIO XML (no root)")
        return

    # 提取所有 cells
    sam3_cells = []
    text_cells = []
    
    # 提取 layer 0 和 1 (画板基础 cell)
    base_cells = [] 
    
    # 收集 SAM3 cells
    for cell in list(root_elem):
        cell_id = cell.get("id")
        if cell_id in ["0", "1"]:
            base_cells.append(cell)
            continue
        sam3_cells.append(cell)
        
    # 收集文字 cells
    if text_root:
        text_model = text_root.find(".//mxGraphModel")
        if text_model:
            text_root_elem = text_model.find("root")
            if text_root_elem:
                for cell in text_root_elem:
                    cell_id = cell.get("id")
                    if cell_id in ["0", "1"]:
                        continue # 忽略基础 layer cell
                    text_cells.append(cell)

    # 4. 清空当前 root
    for cell in list(root_elem):
        root_elem.remove(cell)
        
    # 5. 重建 root 内容
    # 添加基础 cells (id=0, id=1)
    for cell in base_cells:
        root_elem.append(cell)
    # 如果基础 cells 缺失，补全
    if not base_cells:
        ET.SubElement(root_elem, "mxCell", {"id": "0"})
        ET.SubElement(root_elem, "mxCell", {"id": "1", "parent": "0"})

    # 6. 层级排序与添加
    # 逻辑：文字(最上) > Base64 > 基础图形(按面积降序)
    # 实际上 sam3_cells 内部已经是 [基础图形..., Base64..., 箭头...] 的某种顺序
    # 更好的方式是根据 cell 的 style 属性判断类型
    
    # 分类 SAM3 cells
    basic_shapes = []
    base64_shapes = []
    arrows = []
    others = []
    
    for cell in sam3_cells:
        style = cell.get("style", "")
        if "image=data:image" in style:
            base64_shapes.append(cell)
        elif "endArrow" in style or "edge" in cell.attrib:
            arrows.append(cell)
        else:
            # 假设是基础图形
            basic_shapes.append(cell)
            
    # SAM3 提取时已经对 basic_shapes 做了面积排序，这里保留原顺序即可，
    # 或者如果需要在合并时重新保证 "面积越大越在下"，可以再次排序，
    # 但由于无法简单从 xml element 获取面积，这里信任 sam3_extractor 的排序。
    
    # 层级堆叠：
    # 底层: 基础图形 (basic_shapes)
    # 中层: Base64 图片 (base64_shapes)
    # 上层: 箭头 (arrows) - 箭头通常在物体上方
    # 顶层: 文字 (text_cells)
    
    # 重新分配 ID 以免冲突
    # 找到最大 ID 并递增
    current_id = 2
    
    def add_cells(cells_list):
        nonlocal current_id
        for cell in cells_list:
            # 更新 ID
            old_id = cell.get("id")
            cell.set("id", str(current_id))
            
            # 更新 parent (通常都是 1)
            if cell.get("parent") != "0": # 0 是 layer 容器
                cell.set("parent", "1")
                
            root_elem.append(cell)
            current_id += 1
            
    add_cells(basic_shapes)
    add_cells(base64_shapes)
    add_cells(arrows)
    add_cells(text_cells)
    
    # 7. 保存合并后的文件
    try:
        tree = ET.ElementTree(sam3_root)
        # 保持缩进
        ET.indent(tree, space="  ", level=0)
        tree.write(output_path, encoding="utf-8", xml_declaration=False)
        print(f"合并完成: {output_path}")
    except Exception as e:
        print(f"保存合并XML失败: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合并SAM3 XML和文字XML")
    parser.add_argument("--sam3_xml", required=True, help="SAM3生成的XML路径")
    parser.add_argument("--text_xml", required=False, help="文字识别生成的XML路径") # 可选
    parser.add_argument("--image", "-i", required=True, help="原始图片路径")
    parser.add_argument("--output", "-o", required=True, help="输出XML路径")
    
    args = parser.parse_args()
    
    # 如果作为脚本单独运行，text_xml 需要传入，或者在这里不执行自动提取逻辑
    merge_xml(args.sam3_xml, args.text_xml, args.image, args.output)
