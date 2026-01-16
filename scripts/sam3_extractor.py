import os
import sys
import base64
import json
import argparse
import cv2
import numpy as np
import torch
import yaml
import onnxruntime as ort
from PIL import Image, ImageColor
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from collections import OrderedDict
import threading

# 导入SAM3模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 指向sam3_workflow
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from scripts.multimodal_prompt import get_supplement_prompts  # 导入多模态提示词生成函数

# 加载配置
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "config.yaml")

# 检查配置文件是否存在
if not os.path.exists(CONFIG_PATH):
    print(f"Warning: Config file not found at {CONFIG_PATH}")
    # Fallback or exit
    sys.exit(1)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

# 全局配置（从yaml读取）
SAM3_CONFIG = CONFIG["sam3"]
OUTPUT_CONFIG = CONFIG["paths"]

# RMBG-2.0 模型路径（自动推导相对路径）
RMBG_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "rmbg", "model.onnx")

# 掩码可视化配色
MASK_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

# DrawIO样式配置（扩展更多基本形状）
DRAWIO_STYLES = {
    "icon": "shape=image;verticalLabelPosition=bottom;labelBackgroundColor=default;verticalAlign=top;aspect=fixed;imageAspect=0;",
    "picture": "shape=image;verticalLabelPosition=bottom;labelBackgroundColor=default;verticalAlign=top;aspect=fixed;imageAspect=0;",
    "rectangle": "rounded=0;whiteSpace=wrap;html=1;",
    "rounded rectangle": "rounded=1;whiteSpace=wrap;html=1;",
    #"chat_bubble_rect": "shape=comment;whiteSpace=wrap;html=1;perimeter=commentPerimeter;", # 备用：方形气泡
    "title_bar": "rounded=0;whiteSpace=wrap;html=1;fillColor=#E6E6E6;", # 默认灰底
    "section_panel": "rounded=0;whiteSpace=wrap;html=1;dashed=1;dashPattern=1 1;", # 默认虚线
    "arrow": "endArrow=classic;strokeWidth=2;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;entryX=0.5;entryY=0.5;exitX=0.5;exitY=0.5;", # 增加连接点
    # --- 新增形状 ---
    "diamond": "rhombus;whiteSpace=wrap;html=1;",
    "ellipse": "ellipse;whiteSpace=wrap;html=1;", # 圆形/椭圆
    "cylinder": "shape=cylinder3;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;size=15;", # 数据库
    "cloud": "ellipse;shape=cloud;whiteSpace=wrap;html=1;",
    "actor": "shape=umlActor;verticalLabelPosition=bottom;verticalAlign=top;html=1;outlineConnect=0;",
    "hexagon": "shape=hexagon;perimeter=hexagonPerimeter2;whiteSpace=wrap;html=1;fixedSize=1;",
    "triangle": "triangle;whiteSpace=wrap;html=1;",
    "parallelogram": "shape=parallelogram;perimeter=parallelogramPerimeter;whiteSpace=wrap;html=1;fixedSize=1;"
}

# 矢量化支持的Prompt列表 (不在列表中的将被视为图片处理)
VECTOR_SUPPORTED_PROMPTS = [
    "rectangle", "rounded rectangle", "title_bar", "section_panel", 
    "diamond", "ellipse", "cylinder", "cloud", "actor", "hexagon", "triangle", "parallelogram"
]

# -------------------------- RMBG-2.0 抠图核心类 --------------------------
class RMBGInference:
    def __init__(self, model_path):
        """初始化RMBG-2.0 ONNX模型"""
        self.model_path = model_path
        assert os.path.exists(model_path), f"RMBG模型文件不存在：{model_path}"
        
        # 初始化ONNX Runtime（优先CUDA，无则CPU）
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        # 过滤不可用的provider
        available_providers = ort.get_available_providers()
        self.providers = [p for p in providers if p in available_providers]
        
        # ---------- 新增：配置ONNX Runtime选项，屏蔽警告 ----------
        session_options = ort.SessionOptions()
        # 设置日志级别为ERROR（仅显示严重错误，屏蔽WARNING/INFO）
        session_options.log_severity_level = 3  # 0=VERBOSE, 1=INFO, 2=WARNING, 3=ERROR, 4=FATAL
        # 可选：关闭额外的性能优化日志（进一步减少冗余输出）
        session_options.enable_profiling = False
        
        # ---------- 传入session_options ----------
        self.session = ort.InferenceSession(
            model_path,
            providers=self.providers,
            sess_options=session_options  # 新增这一行
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_size = (1024, 1024)  # RMBG-2.0固定输入尺寸

    def preprocess(self, img):
        """图片预处理：缩放、归一化、转CHW格式"""
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RMBG要求BGR格式
        h, w = img.shape[:2]
        # 缩放到模型输入尺寸
        img_resized = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)
        # 归一化到[0,1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        # 转CHW格式 (HWC -> CHW)
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        # 增加batch维度 (3,1024,1024) -> (1,3,1024,1024)
        img_batch = np.expand_dims(img_transposed, axis=0)
        return img_batch, (h, w)

    def postprocess(self, pred, original_size):
        """后处理：提取alpha通道并还原到原图尺寸"""
        # 移除batch维度，提取alpha通道 (1,1,1024,1024) -> (1024,1024)
        alpha = pred[0, 0, :, :]
        # 缩放回原图尺寸
        alpha_resized = cv2.resize(alpha, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
        # 归一化到[0,255]并转8位
        alpha_resized = (alpha_resized * 255).astype(np.uint8)
        return alpha_resized

    def remove_background(self, img_rgb):
        """核心抠图接口：输入RGB格式PIL/OpenCV图片，返回抠图后带透明通道的PIL图片"""
        # 统一转换为OpenCV格式（HWC, RGB）
        if isinstance(img_rgb, Image.Image):
            img = np.array(img_rgb)
        else:
            img = img_rgb.copy()
        
        # 预处理
        img_input, original_size = self.preprocess(img)
        # 模型推理
        pred = self.session.run([self.output_name], {self.input_name: img_input})[0]
        # 后处理得到alpha通道
        alpha = self.postprocess(pred, original_size)
        # 合并alpha通道到原图（生成RGBA图片）
        img_rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        img_rgba[:, :, 3] = alpha
        # 转换为PIL图片
        pil_rgba = Image.fromarray(img_rgba)
        return pil_rgba

# -------------------------- 工具函数（优化版取色+核心修改箭头中点计算） --------------------------
def extract_style_colors(image: np.ndarray, bbox: list) -> tuple:
    """
    精细化取色逻辑：区分 边框区域(Stroke) 和 内部区域(Fill)
    逻辑：
    1. Fill: 收缩边界框20%，取内部区域的中值颜色（鲁棒性强于均值）。
    2. Stroke: 提取边界框外围5%~10%区域。由于通常是白底黑框，我们提取该区域中最暗的25%像素的均值作为边框色。
    
    :param image: BGR格式的OpenCV图像
    :param bbox: [x1, y1, x2, y2]
    :return: (fill_color_hex, stroke_color_hex)
    """
    x1, y1, x2, y2 = map(int, bbox)
    h_box, w_box = y2 - y1, x2 - x1
    
    # 截取ROI
    roi = image[y1:y2, x1:x2]
    if roi.size == 0: 
        return "#ffffff", "#000000"
        
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    # --- 1. 提取填充色 (Fill Color) ---
    # 收缩比例: 20% 或 至少5像素
    margin_x = int(min(w_box * 0.2, max(5, w_box * 0.1)))
    margin_y = int(min(h_box * 0.2, max(5, h_box * 0.1)))
    
    # 提取内部区域
    inner_roi = roi_rgb[margin_y:h_box-margin_y, margin_x:w_box-margin_x]
    
    # 兜底：如果收缩没了，就用整体
    if inner_roi.size == 0:
        inner_roi = roi_rgb
        
    # 计算中值颜色 (Median)，抗噪性好
    inner_pixels = inner_roi.reshape(-1, 3)
    fill_rgb = np.median(inner_pixels, axis=0).astype(int)
    
    # --- 2. 提取描边色 (Stroke Color) ---
    # 提取四周边缘
    border_width = max(2, int(min(w_box, h_box) * 0.1)) # 10% 宽度作为边缘带
    
    top = roi_rgb[:border_width, :]
    bottom = roi_rgb[h_box-border_width:, :]
    left = roi_rgb[:, :border_width]
    right = roi_rgb[:, w_box-border_width:]
    
    # 拼接所有边缘像素
    border_pixels = np.concatenate([
        top.reshape(-1, 3), 
        bottom.reshape(-1, 3),
        left.reshape(-1, 3),
        right.reshape(-1, 3)
    ], axis=0)
    
    # 计算亮度 (Luminance)
    # L = 0.299*R + 0.587*G + 0.114*B
    luminance = np.dot(border_pixels, [0.299, 0.587, 0.114])
    
    # 提取最暗的 25% 像素 (假设边框比背景深)
    dark_threshold = np.percentile(luminance, 25)
    darker_pixels = border_pixels[luminance <= dark_threshold]
    
    if len(darker_pixels) > 0:
        stroke_rgb = np.mean(darker_pixels, axis=0).astype(int)
    else:
        stroke_rgb = np.mean(border_pixels, axis=0).astype(int)

    # 辅助函数：RGB -> Hex
    def rgb2hex(rgb):
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    return rgb2hex(fill_rgb), rgb2hex(stroke_rgb)

def image_to_base64(image: Image.Image) -> str:
    import io
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def prettify_xml(elem: ET.Element) -> str:
    """
    格式化XML（移除版本声明行 + 过滤空行）
    """
    rough_string = ET.tostring(elem, "utf-8")
    reparsed = minidom.parseString(rough_string)
    # 关键修改：1. 排除<?xml开头的版本行 2. 过滤空行
    return '\n'.join([
        line for line in reparsed.toprettyxml(indent="  ").split('\n')
        if line.strip() and not line.strip().startswith("<?xml")
    ])

def calculate_arrow_midpoints(bbox: list) -> tuple:
    """
    核心修改：箭头起点/终点取边界框较短边的中点（近似）
    逻辑：
    1. 计算边界框的宽度（w=x2-x1）和高度（h=y2-y1）
    2. 判定较短边：宽<高（竖长框）→ 短边为左右边；高<宽（横长框）→ 短边为上下边
    3. 短边中点作为箭头的起点和终点，箭头沿长边延伸
    """
    x1, y1, x2, y2 = map(int, bbox)
    bbox_width = x2 - x1  # 水平方向长度（宽）
    bbox_height = y2 - y1  # 垂直方向长度（高）
    
    # 情况1：宽 < 高（竖长框，较短边为水平方向，箭头左右延伸）
    if bbox_width < bbox_height:
        # 左边中点（起点）：x1，垂直中点
        start_point = (x1, (y1 + y2) // 2)
        # 右边中点（终点）：x2，垂直中点
        end_point = (x2, (y1 + y2) // 2)
    # 情况2：高 <= 宽（横长框/正方形，较短边为垂直方向，箭头上下列延伸）
    else:
        # 上边中点（起点）：水平中点，y1
        start_point = ((x1 + x2) // 2, y1)
        # 下边中点（终点）：水平中点，y2
        end_point = ((x1 + x2) // 2, y2)
    
    return start_point, end_point

def calculate_element_area(bbox: list) -> float:
    """计算元素面积（用于后续层级排序）"""
    x1, y1, x2, y2 = map(int, bbox)
    return (x2 - x1) * (y2 - y1)

def build_drawio_xml(canvas_width: int, canvas_height: int, elements_data: dict) -> ET.Element:
    mxfile = ET.Element("mxfile", {"host": "app.diagrams.net", "type": "device"})
    diagram = ET.SubElement(mxfile, "diagram", {"id": "ERDiagram", "name": "Page-1"})
    mx_graph_model = ET.SubElement(diagram, "mxGraphModel", {
        "dx": str(canvas_width),
        "dy": str(canvas_height),
        "grid": "1",
        "gridSize": "10",
        "guides": "1",
        "tooltips": "1",
        "connect": "1",
        "arrows": "1",
        "fold": "1",
        "page": "1",
        "pageScale": "1",
        "pageWidth": str(canvas_width),
        "pageHeight": str(canvas_height),
        "math": "0",
        "shadow": "0",
        "background": "#ffffff"
    })
    root = ET.SubElement(mx_graph_model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})
    cell_id = 2

    # 优化版：元素层级排序逻辑 (Z-Index Optimization)
    # 原则1: 面积大的元素在底层 (Background panels)
    # 原则2: 面积小的元素在顶层 (Text bubbles, specific shapes)
    # 原则3: 图片/图标在更上层
    # 原则4: 连线(Arrows)最后绘制，确保在所有形状上方，方便连接

    # 1. 收集所有 【普通形状】
    shape_elements = []
    normal_types = ["rectangle", "section_panel", "title_bar", "rounded rectangle"]
    for t in normal_types:
        if t in elements_data:
            for item in elements_data[t]:
                item["_type"] = t # 临时标记类型
                shape_elements.append(item)
    
    # 按面积 降序 排序 (最大的在最前 -> 也就最先写入XML -> 位于最底层)
    # Draw.io XML顺序：先写的在下面(被覆盖)，后写的在上面。
    shape_elements.sort(key=lambda x: calculate_element_area(x["bbox"]), reverse=True)

    # 2. 写入 【图片/图标】与【其他基本形状】 (通常显示在面板之上)
    image_elements = []
    
    # 动态获取所有需要在第二层处理的类型
    # 排除已处理的 (normal_types 已在上面处理? 不，之前代码只处理了 normal_types)
    # 现在的逻辑是：
    # Layer 1: shape_elements (所有基本形状，按面积排序)
    # Layer 2: image_elements (icon/picture) -> 这里需要扩展吗？
    # 用户需求：希望统一处理。
    # 让我重构这部分逻辑：
    
    # 获取所有的 Keys
    all_keys = set(DRAWIO_STYLES.keys())
    edge_keys = {"arrow"}
    pic_keys = {"icon", "picture"}
    # 扩展：VECTOR_SUPPORTED_PROMPTS 里的都算 shape
    VECTOR_SUPPORTED_PROMPTS = {
        "rectangle", "rounded rectangle", "section_panel", "title_bar",
        "diamond", "ellipse", "cylinder", "cloud", "hexagon", "triangle", "parallelogram"
    }
    
    # 动态逻辑：
    # Layer 1 (Vector): 所有在 VECTOR_SUPPORTED_PROMPTS 里的类型
    # Layer 2 (Image): 所有不在 Layer 1 且不是 arrow 的类型 (包含 icon, picture 以及 MLLM 返回的未知类型)
    
    # --- 1. 写入所有基本形状 (Layer 1) ---
    shape_elements_list = []
    # 遍历所有存在的 key
    for t in elements_data.keys():
        if t in VECTOR_SUPPORTED_PROMPTS:
            for item in elements_data[t]:
                item["_type"] = t
                shape_elements_list.append(item)
    
    # 按面积 降序
    shape_elements_list.sort(key=lambda x: calculate_element_area(x["bbox"]), reverse=True)

    for item in shape_elements_list:
        elem_type = item["_type"]
        x1, y1, x2, y2 = item["bbox"]
        width = x2 - x1
        height = y2 - y1
        cell_base_attrs = {"id": str(cell_id), "parent": "1", "vertex": "1", "value": ""}
        
        fill_color = item.get("fill_color", "#ffffff")
        stroke_color = item.get("stroke_color", "#000000")
        stroke_width = item.get("stroke_width", 1)
        
        base_style = DRAWIO_STYLES.get(elem_type, "rounded=0;whiteSpace=wrap;html=1;")
        style = f"{base_style}fillColor={fill_color};strokeColor={stroke_color};strokeWidth={stroke_width};"
        
        cell = ET.SubElement(root, "mxCell", {**cell_base_attrs, "style": style})
        ET.SubElement(cell, "mxGeometry", {"x": str(x1), "y": str(y1), "width": str(width), "height": str(height), "as": "geometry"})
        cell_id += 1

    # --- 2. 写入图片类型的元素 (Layer 2) ---
    # 所有没在 Layer 1 处理的非 Vector 元素 (包括 arrow, icon, picture)
    pic_elements_list = []
    for t in elements_data.keys():
         if t not in VECTOR_SUPPORTED_PROMPTS:
            for item in elements_data[t]:
                item["_type"] = t
                pic_elements_list.append(item)
    
    pic_elements_list.sort(key=lambda x: calculate_element_area(x["bbox"]), reverse=True)
    
    for item in pic_elements_list:
        elem_type = item["_type"]
        x1, y1, x2, y2 = item["bbox"]
        width = x2 - x1
        height = y2 - y1
        
        style = "shape=image;verticalLabelPosition=bottom;verticalAlign=top;imageAspect=0;aspect=fixed;"
        # Base64处理: 如果太长可能影响XML性能，但Draw.io支持
        # 注意：这里需要添加 image=data:image/png;base64,.....
        b64_str = item.get("base64", "")
        if b64_str:
            # 用户请求去除 ";base64"
            style += f"image=data:image/png,{b64_str};"
            
        cell = ET.SubElement(root, "mxCell", {
            "id": str(cell_id),
            "parent": "1",
            "vertex": "1",
            "value": "",
            "style": style
        })
        ET.SubElement(cell, "mxGeometry", {
            "x": str(x1),
            "y": str(y1),
            "width": str(width),
            "height": str(height),
            "as": "geometry"
        })
        cell_id += 1
        
    # 3. (已移除) 连线 Arrow 不再作为 Edge 处理，而是作为普通图片处理
    # 这样可以避免朝向和路由错误的问题


    return mxfile


# -------------------------- SAM3推理类（集成抠图） --------------------------
def calculate_iou(box1, box2):
    """
    计算两个矩形框的 Intersection over Union (IoU)
    box: [x1, y1, x2, y2]
    """
    # 确定交集矩形
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area

    if union_area <= 0:
        return 0.0
    
    return intersection_area / union_area

def deduplicate_elements(elements_data: dict, iou_threshold: float = 0.85) -> dict:
    """
    对所有提取出的元素进行全局去重 (NMS-like strategy)
    优先保留可信度高的元素，或者保留面积大的（或特定的层级优先）
    这里策略：优先保留 VECTOR_SUPPORTED_PROMPTS 里的特定类型，其次保留 score 高的
    """
    all_items = []
    # 1. 扁平化所有元素，并带上原始 key 信息
    for key, items in elements_data.items():
        for item in items:
            item['_temp_key'] = key # 暂存 key
            # 给不同类型赋予权重，用于冲突解决
            # Vector 形状 > Icon > Picture > Arrow
            priority = 1
            if key in VECTOR_SUPPORTED_PROMPTS:
                priority = 3
            elif key == "icon":
                priority = 2
            
            item['_priority'] = priority
            all_items.append(item)

    # 2. 按优先级降序，分数降序 排序
    all_items.sort(key=lambda x: (x['_priority'], x['score']), reverse=True)
    
    keep_indices = []
    dropped_indices = set()
    
    for i in range(len(all_items)):
        if i in dropped_indices:
            continue
            
        keep_indices.append(i)
        
        for j in range(i + 1, len(all_items)):
            if j in dropped_indices:
                continue
                
            # 计算 IoU
            iou = calculate_iou(all_items[i]['bbox'], all_items[j]['bbox'])
            if iou > iou_threshold:
                # 认为是重复的，由于 all_items 已按优先级排序，丢弃 j 即可
                dropped_indices.add(j)
    
    # 3. 重建字典
    new_elements_data = {}
    for i in keep_indices:
        item = all_items[i]
        key = item['_temp_key']
        
        # 清理临时字段
        del item['_temp_key']
        del item['_priority']
        
        if key not in new_elements_data:
            new_elements_data[key] = []
        new_elements_data[key].append(item)
        
    return new_elements_data

class Sam3ElementExtractor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_path = SAM3_CONFIG["checkpoint_path"]
        self.bpe_path = SAM3_CONFIG["bpe_path"]
        print(f"加载SAM3模型（设备：{self.device}）...")
        self.model = build_sam3_image_model(
            bpe_path=self.bpe_path,
            checkpoint_path=self.checkpoint_path,
            load_from_HF=False,
            device=self.device
        )
        self.processor = Sam3Processor(self.model)
        print("SAM3模型加载完成！")
        
        # 初始化RMBG抠图模型（仅针对icon）
        print(f"加载RMBG-2.0抠图模型（路径：{RMBG_MODEL_PATH}）...")
        self.rmbg_infer = RMBGInference(RMBG_MODEL_PATH)
        print("RMBG-2.0模型加载完成！")

        # 初始化状态缓存 (LRU Cache)
        self.state_cache = OrderedDict()
        self.max_cache_size = 3  # 最多缓存3张图片的Embedding
        self.cache_lock = threading.Lock() # 保护缓存读写
        
        # 记录应当被视为"图片"(保留背景)的prompt集合
        self.known_picture_prompts = {"picture"}

    def _get_image_state(self, image_path: str):
        """获取或创建图像状态 (线程安全 + LRU缓存)"""
        with self.cache_lock:
            # 1. 命中缓存
            if image_path in self.state_cache:
                self.state_cache.move_to_end(image_path)
                return self.state_cache[image_path]

        # 2. 未命中，加载新图像 (释放锁，允许其他线程读缓存，但此处我们还是串行化比较安全，或者只锁这个计算过程)
        # 注意：这里如果多个线程请求同一个新图片，可能会重复计算。为了简单，我们接受这个开销，或者使用更细粒度的锁。
        # 考虑到显存有限，我们还是在外部加锁比较好。但为了独立性，这里处理。
        
        print(f"SAM3(Interactive): Loading & Embedding image -> {image_path}")
        pil_image = Image.open(image_path).convert("RGB")
        cv2_image = cv2.imread(image_path)
        canvas_size = pil_image.size
        
        # 编码 (耗时操作)
        image_state = self.processor.set_image(pil_image)
        
        cache_item = {
            "image_state": image_state,
            "pil_image": pil_image,
            "cv2_image": cv2_image,
            "canvas_size": canvas_size
        }

        with self.cache_lock:
            # 再次检查（防止并发插入）
            if image_path in self.state_cache:
                self.state_cache.move_to_end(image_path)
                return self.state_cache[image_path]
            
            # 插入新项
            self.state_cache[image_path] = cache_item
            
            # 淘汰旧项
            if len(self.state_cache) > self.max_cache_size:
                removed_path, _ = self.state_cache.popitem(last=False)
                print(f"Evicted cache: {removed_path}")
        
        return cache_item


    def extract_with_new_prompts(self, image_path: str, new_prompts: list, existing_result: dict = None) -> dict:
        """
        优化版：增量提取接口 (支持LRU缓存)
        1. 重用已编码的图像Embedding (avoid expensive image encoding)
        2. 仅对新提示词进行Mask Decoding
        3. 将新结果合并到现有结果中
        """
        # 1. 获取图像状态 (自动处理缓存)
        cache_item = self._get_image_state(image_path)
        
        state = cache_item["image_state"]
        pil_image = cache_item["pil_image"]
        canvas_size = cache_item["canvas_size"]
        cv2_image = cache_item["cv2_image"]

        # 2. 如果是第一次调用(existing_result is None)
        if existing_result is None:
            elements_data = {}
            full_metadata = {
                "image_path": image_path,
                "image_size": {"width": canvas_size[0], "height": canvas_size[1]},
                "elements": {},
                "total_elements": 0,
                "total_area": 0
            }
        else:
            elements_data = existing_result["elements"]
            full_metadata = existing_result["full_metadata"]

        # 3. 仅对新提示词进行推理 (注意：state 复用需要 Reset)
        # 警告：由于 state 是对象引用，我们在修改 prompt 时要注意是否会影响并发
        # 但目前我们假设单图操作是串行的 (由 GLOBAL_LOCK 或 业务逻辑保证)
        
        for prompt in new_prompts:
            if prompt in elements_data: 
                continue # 已存在的提示词跳过

            elements_data[prompt] = []
            
            self.processor.reset_all_prompts(state)
            result_state = self.processor.set_text_prompt(prompt=prompt, state=state)
            masks = result_state.get("masks", [])
            boxes = result_state.get("boxes", [])
            scores = result_state.get("scores", [])

            num_masks = masks.shape[0] if (isinstance(masks, torch.Tensor) and masks.dim() > 0) else len(masks)
            if num_masks == 0:
                continue

            for i in range(num_masks):
                score = scores[i]
                score_val = score.item() if hasattr(score, 'item') else float(score)
                if score_val < SAM3_CONFIG["score_threshold"]:
                    continue

                box = boxes[i]
                bbox = box.cpu().numpy().tolist() if isinstance(box, torch.Tensor) else box
                bbox = [int(coord) for coord in bbox]
                x1, y1, x2, y2 = bbox

                mask = masks[i]
                binary_mask = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else np.array(mask)
                if binary_mask.ndim > 2:
                    binary_mask = binary_mask.squeeze()
                binary_mask = (binary_mask > 0.5).astype(np.uint8) * 255

                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                polygon = []
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < SAM3_CONFIG["min_area"]:
                        continue
                    epsilon = SAM3_CONFIG["epsilon_factor"] * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    polygon = approx.reshape(-1, 2).tolist()
                    break

                if not polygon:
                    continue

                elem_item = {
                    "id": full_metadata["total_elements"], # 使用全局ID
                    "score": score_val,
                    "bbox": bbox,
                    "polygon": polygon,
                    "area": calculate_element_area(bbox)
                }
                
                # 更新元数据
                full_metadata["total_area"] += elem_item["area"]
                full_metadata["total_elements"] += 1

                # 元素后处理 (RMBG / Color / Arrow)
                if prompt == "icon":
                    cropped = pil_image.crop((x1, y1, x2, y2))
                    cropped_rmbg = self.rmbg_infer.remove_background(cropped)
                    elem_item["base64"] = image_to_base64(cropped_rmbg)
                elif prompt == "picture":
                    cropped = pil_image.crop((x1, y1, x2, y2))
                    elem_item["base64"] = image_to_base64(cropped)
                elif prompt in VECTOR_SUPPORTED_PROMPTS:
                    # 使用新的双重提取逻辑
                    fill_color, stroke_color = extract_style_colors(cv2_image, bbox)
                    elem_item["fill_color"] = fill_color
                    elem_item["stroke_color"] = stroke_color
                elif prompt == "arrow":
                    # 箭头转为图片处理 (抠图)
                    # 优化策略：Padding + Mask过滤 + RMBG
                    # 1. 扩大裁剪范围 (Padding)
                    pad = 15
                    img_w, img_h = pil_image.size
                    p_x1 = max(0, x1 - pad)
                    p_y1 = max(0, y1 - pad)
                    p_x2 = min(img_w, x2 + pad)
                    p_y2 = min(img_h, y2 + pad)
                    
                    # 裁剪原始 RGB 图像
                    cropped_pil = pil_image.crop((p_x1, p_y1, p_x2, p_y2))
                    cropped_np = np.array(cropped_pil)

                    # 2. 利用 SAM3 预测的 Mask 进行粗略过滤 (Masking)
                    # 目的：去除 bbox 内但不属于箭头的邻居（如附近的文字）
                    if binary_mask is not None:
                        # 裁剪对应的 Mask 区域
                        mask_crop = binary_mask[p_y1:p_y2, p_x1:p_x2]
                        
                        # 3. 膨胀 Mask (Dilation)
                        # 目的：防止 SAM3 Mask 过于紧致导致切掉箭头边缘，同时保留一定边缘上下文给 RMBG
                        kernel = np.ones((10, 10), np.uint8)
                        dilated_mask = cv2.dilate(mask_crop, kernel, iterations=1)
                        
                        # 4. 将 Mask 外的区域涂白 (White-out Background)
                        # 这样 RMBG 就不会被背景里的杂物干扰，只会看到白底上的箭头
                        mask_3ch = cv2.cvtColor(dilated_mask, cv2.COLOR_GRAY2RGB)
                        white_bg = np.full_like(cropped_np, 255)
                        
                        # 保留 Mask 区域内的原始像素，其他变白
                        masked_img_np = np.where(mask_3ch > 0, cropped_np, white_bg)
                        input_pil = Image.fromarray(masked_img_np)
                    else:
                        input_pil = cropped_pil

                    # 5. 送入 RMBG 做最终精细抠图
                    cropped_rmbg = self.rmbg_infer.remove_background(input_pil)
                    
                    elem_item["base64"] = image_to_base64(cropped_rmbg)
                    # 更新 bbox 为带 padding 的坐标，确保 xml 中位置正确
                    elem_item["bbox"] = [p_x1, p_y1, p_x2, p_y2]
                    # 更新 area (虽然后续排序用了，但不影响核心逻辑)
                    elem_item["area"] = (p_x2 - p_x1) * (p_y2 - p_y1)

                elements_data[prompt].append(elem_item)

        # 4. 更新Elements (应用去重)
        # NMS 去重：因为多次增量提取或不同 Prompt 可能指向同一位置
        if full_metadata["total_elements"] > 0:
            print(f"执行NMS去重，去重前元素总数: {full_metadata['total_elements']}")
            deduped_elements = deduplicate_elements(elements_data, iou_threshold=0.85)
            elements_data = deduped_elements
            
            # 重新计算 total_elements 和 total_area
            new_total = 0
            new_area = 0
            for k, items in elements_data.items():
                new_total += len(items)
                for item in items:
                    new_area += item['area']
            
            full_metadata["total_elements"] = new_total
            full_metadata["total_area"] = new_area
            print(f"去重后元素总数: {full_metadata['total_elements']}")

        full_metadata["elements"] = elements_data
        
        return {
            "canvas_size": canvas_size,
            "elements": elements_data,
            "full_metadata": full_metadata,
            "pil_image": pil_image,
            "cv2_image": cv2_image
        }

    def extract_at_point(self, image_path: str, point: tuple) -> dict:
        """
        单点交互式提取 (支持多用户并发/切换)
        :param point: (x, y) 像素坐标
        """
        # 1. 获取状态 (自动处理缓存)
        cache_item = self._get_image_state(image_path)
        
        state = cache_item["image_state"]
        pil_image = cache_item["pil_image"]
        w, h = cache_item["canvas_size"]
        
        # 2. 归一化坐标 & 重置Prompt
        # 注意：processor.reset_all_prompts(state) 是原地修改 state 字典
        # 如果还要支持同一张图并发点击，我们需要 深度拷贝 state，或者加锁串行化 processor 操作。
        # 鉴于 SAM3 state 数据量大，深拷贝昂贵。我们假设对同一张图的操作需要串行。
        # 不同的图(不同 cache_item) 理论上 state 独立，但 processor 可能是共享权重的。
        
        norm_point = [point[0] / w, point[1] / h]
        self.processor.reset_all_prompts(state)
        
        # 3. 推理 (Label=1: Positive click)
        # 此处调用模型推理，如果 weight 共享且无中间状态，则安全。
        # 但通常建议外部加锁。
        result_state = self.processor.add_point_prompt(norm_point, 1, state)
        
        # 4. 处理结果 (取Top1)
        scores = result_state.get("scores", [])
        masks = result_state.get("masks", [])
        boxes = result_state.get("boxes", [])
        
        if len(scores) == 0:
            return None
            
        # 找分数最高的
        if isinstance(scores, torch.Tensor):
            best_idx = torch.argmax(scores).item()
            score = scores[best_idx].item()
            mask = masks[best_idx].cpu().numpy()
            box = boxes[best_idx].cpu().numpy()
        else:
            best_idx = np.argmax(scores)
            score = scores[best_idx]
            mask = masks[best_idx]
            box = boxes[best_idx]
        
        bbox = [int(c) for c in box]
        binary_mask = (mask > 0.5).astype(np.uint8) * 255
        
        # 必须确保 bbox 合法
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1: return None
        
        # Padding (类似箭头，给予一定余量)
        padding = 15
        x1_p = max(0, x1 - padding)
        y1_p = max(0, y1 - padding)
        x2_p = min(w, x2 + padding)
        y2_p = min(h, y2 + padding)
        
        crop_box = (x1_p, y1_p, x2_p, y2_p)
        crop_img = pil_image.crop(crop_box)
        
        # Mask处理
        full_mask = Image.fromarray(binary_mask)
        crop_mask = full_mask.crop(crop_box)
        
        # 膨胀Mask并白化背景
        mask_np = np.array(crop_mask)
        # kernel大小根据图像尺寸动态调整可能更好，这里固定5x5
        kernel = np.ones((5, 5), np.uint8) 
        mask_dilated = cv2.dilate(mask_np, kernel, iterations=1)
        
        crop_np = np.array(crop_img)
        white_bg = np.ones_like(crop_np) * 255
        
        # 扩展mask维度
        if len(mask_dilated.shape) == 2:
            mask_3ch = np.stack([mask_dilated]*3, axis=2)
        else:
            mask_3ch = mask_dilated
            
        # 融合
        processed_crop = np.where(mask_3ch > 127, crop_np, white_bg)
        processed_pil = Image.fromarray(processed_crop)
        
        # RMBG
        final_rgba = self.rmbg_infer.remove_background(processed_pil)
        base64_str = image_to_base64(final_rgba)
        
        return {
            "bbox": bbox,
            "crop_bbox": [x1_p, y1_p, x2_p, y2_p],
            "base64": base64_str,
            "score": score
        }

    def extract_with_prompts(self, image_path: str, prompts: list) -> dict:
        """兼容旧接口，内部调用增量提取"""
        # 兼容性包装，不直接操作 current_image_path
        return self.extract_with_new_prompts(image_path, prompts, existing_result=None)

    def generate_mask_visualization(self, cv2_image: np.ndarray, elements_data: dict, output_path: str):
        """生成掩码可视化图（用于传给多模态大模型）"""
        image = cv2_image.copy()
        overlay = cv2_image.copy()
        global_id = 0
        for elem_type, items in elements_data.items():
            for item in items:
                color = MASK_COLORS[global_id % len(MASK_COLORS)]
                points = np.array(item["polygon"], dtype=np.int32)
                cv2.fillPoly(overlay, [points], color)
                x1, y1, x2, y2 = item["bbox"]
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, f"#{global_id} ({elem_type})", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if elem_type == "arrow":
                    source_mid, target_mid = calculate_arrow_midpoints(item["bbox"])
                    cv2.arrowedLine(image, source_mid, target_mid, color, 2, tipLength=0.1)
                    cv2.circle(image, source_mid, 4, (0, 0, 255), -1)
                    cv2.circle(image, target_mid, 4, (0, 255, 0), -1)
                global_id += 1
        result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        cv2.imwrite(output_path, result)
        return output_path

    def iterative_extract(self, image_path: str, specific_output_dir: str = None) -> dict:
        """
        循环迭代提取优化版 (Fixed 4-Round Strategy):
        Round 1: Initial generic prompts.
        Round 2: Single-word specific prompts.
        Round 3: Two-word specific prompts.
        Round 4: Short phrases.
        :param specific_output_dir: 指定输出目录（用于Server隔离任务），如果不传则使用默认的 output/temp/<filename>
        """
        if specific_output_dir:
            temp_dir = specific_output_dir
        else:
            temp_dir = os.path.join(OUTPUT_CONFIG["temp_dir"], Path(image_path).stem)
            
        os.makedirs(temp_dir, exist_ok=True)
        self.vis_dir = Path(temp_dir) 

        # Round 1: Initial Extraction
        print(f"--- [Round 1] Initial Extraction ---")
        current_prompts = list(SAM3_CONFIG.get("initial_prompts", ["rectangle", "icon", "arrow"]))
        known_prompts = set(current_prompts)
        
        # Reset known picture prompts for new image
        self.known_picture_prompts = {"picture"}
        
        current_result = self.extract_with_new_prompts(image_path, current_prompts, existing_result=None)
        
        # Save initial visualization
        vis_path = str(self.vis_dir / "mask_vis_round_1.jpg")
        self.generate_mask_visualization(current_result["cv2_image"], current_result["elements"], vis_path)

        # Iteration Rounds 2, 3, 4
        for round_idx in range(2, 5):
            print(f"--- [Round {round_idx}] VLM Scanning ---")
            
            # 1. Update Visualization from previous round
            vis_path = str(self.vis_dir / f"mask_vis_round_{round_idx-1}.jpg")
            # Ensure the visualization exists (it should, from end of previous block)
            if not os.path.exists(vis_path):
                self.generate_mask_visualization(current_result["cv2_image"], current_result["elements"], vis_path)
            
            # 2. Call VLM
            # Note: We pass original image path for VLM comparison
            vlm_response = get_supplement_prompts(
                mask_vis_path=vis_path, 
                existing_prompts=list(known_prompts),
                round_index=round_idx,
                original_image_path=image_path
            )
            
            # --- 错误检测逻辑 ---
            if vlm_response.get("error", False):
                print(f"⚠️ Warning: Round {round_idx} VLM call failed (see logs above). Stopping iterative scan.")
                break
            # --------------------

            icon_prompts = vlm_response.get("icon_prompts", [])
            picture_prompts = vlm_response.get("picture_prompts", [])
            has_missing = vlm_response.get("has_missing", False)
            
            print(f"VLM Response: icons={icon_prompts}, pictures={picture_prompts}, missing={has_missing}")
            
            # 3. Update Picture Prompts Registry
            for p in picture_prompts:
                self.known_picture_prompts.add(p)
                
            # 4. Combine and Filter
            new_candidates = list(set(icon_prompts + picture_prompts))
            valid_new_prompts = [p for p in new_candidates if p not in known_prompts]
            
            if not valid_new_prompts:
                print(f"Round {round_idx}: No new valid prompts found.")
                if not has_missing:
                    print("VLM indicates no missing elements. Stopping early.")
                    break
                else:
                    print("VLM thinks elements are missing but provided no new unique prompts. Continue to next round.")
                    continue

            # 5. Incremental Extraction
            print(f"Extracting new prompts: {valid_new_prompts}")
            current_result = self.extract_with_new_prompts(image_path, valid_new_prompts, existing_result=current_result)
            
            known_prompts.update(valid_new_prompts)
            
            # Generate visualization for next round
            next_vis_path = str(self.vis_dir / f"mask_vis_round_{round_idx}.jpg")
            self.generate_mask_visualization(current_result["cv2_image"], current_result["elements"], next_vis_path)

        print(f"Final Total Elements: {current_result['full_metadata']['total_elements']}")
        
        # Save results
        self.save_temp_files(current_result, temp_dir)
        
        return current_result


    def save_temp_files(self, extract_result: dict, output_dir: str):
        """保存裁切元素、掩码图、元数据"""
        # 裁切元素（含抠图后的icon）
        icons_dir = os.path.join(output_dir, "icons")
        os.makedirs(icons_dir, exist_ok=True)
        
        pil_image = extract_result["pil_image"]
        cv2_image = extract_result["cv2_image"]

        for elem_type, items in extract_result["elements"].items():
            for item in items:
                x1, y1, x2, y2 = item["bbox"]
                padding = 5
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(cv2_image.shape[1], x2 + padding)
                y2 = min(cv2_image.shape[0], y2 + padding)
                
                # 判定处理策略：不在矢量列表中的，包含icon/picture或未知的全部视为图片
                is_vector_type = (elem_type in VECTOR_SUPPORTED_PROMPTS) or (elem_type == "arrow")
                
                # 保存原图/抠图后的图
                if not is_vector_type:
                    # 图片模式：裁切 -> 抠图(RMBG) -> 保存 -> Base64
                    # 重新从原始PIL图裁切以获得高质量输入
                    cropped = pil_image.crop((x1, y1, x2, y2))
                    
                    if elem_type in self.known_picture_prompts:
                         # 特殊情况：picture保留背景
                         final_img = cropped
                         save_suffix = ".png"
                    else:
                         # icon 或 任意未知形状 (如 server, user 等)：执行抠图
                         final_img = self.rmbg_infer.remove_background(cropped)
                         save_suffix = "_rmbg.png"
                    
                    # 保存PNG
                    save_name = f"{elem_type}_{item['id']:04d}{save_suffix}"
                    final_img.save(os.path.join(icons_dir, save_name))
                    
                    # 如果元素中没有base64（比如新发现的prompt），这里补充
                    # 注意：extract_with_new_prompts 可能已经生成过，但这里统一重新生成保证一致性
                    item["base64"] = image_to_base64(final_img)
                else:
                    # 矢量模式：保存原图截图 (可选，用于调试)
                    cropped = extract_result["cv2_image"][y1:y2, x1:x2]
                    cv2.imwrite(os.path.join(icons_dir, f"{elem_type}_{item['id']:04d}.png"), cropped)

        # 保存元数据
        # 净化倒排索引：移除";base64"字段，减小日志体积 (结合以后的代码我希望倒排索引去除";base64")
        metadata_clean = {
            "image_path": extract_result["full_metadata"]["image_path"],
            "image_size": extract_result["full_metadata"]["image_size"],
            "total_elements": extract_result["full_metadata"]["total_elements"],
            "total_area": extract_result["full_metadata"]["total_area"],
            "elements": {}
        }
        for k, v in extract_result["full_metadata"]["elements"].items():
            metadata_clean["elements"][k] = []
            for item in v:
                item_copy = item.copy()
                # 兼容移除 "base64" 和 ";base64" 键
                if "base64" in item_copy:
                    del item_copy["base64"] 
                if ";base64" in item_copy:
                    del item_copy[";base64"]
                metadata_clean["elements"][k].append(item_copy)

        with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata_clean, f, indent=2, ensure_ascii=False)

        # 生成最终XML（含修改后箭头逻辑）
        xml_root = build_drawio_xml(
            extract_result["canvas_size"][0],
            extract_result["canvas_size"][1],
            extract_result["elements"]
        )
        xml_path = os.path.join(output_dir, "sam3_output.drawio.xml")
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(prettify_xml(xml_root))

        print(f"临时文件保存到：{output_dir}")

# -------------------------- 主函数 --------------------------
def main(image_path: str):
    """主函数：处理单张图片"""
    if not os.path.exists(image_path):
        print(f"错误：图片不存在 → {image_path}")
        sys.exit(1)

    # 初始化提取器（自动加载SAM3+RMBG模型）
    extractor = Sam3ElementExtractor()
    # 迭代提取元素（icon自动抠图）
    extract_result = extractor.iterative_extract(image_path)
    # 返回最终XML路径
    temp_dir = os.path.join(OUTPUT_CONFIG["temp_dir"], Path(image_path).stem)
    xml_path = os.path.join(temp_dir, "sam3_output.drawio.xml")
    print(f"处理完成！最终XML文件：{xml_path}")
    return xml_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM3迭代提取元素（icon自动抠图）并生成DrawIO XML")
    parser.add_argument("--image", "-i", required=True, help="输入图片路径")
    args = parser.parse_args()
    main(args.image)
