import os
import subprocess

def get_freest_gpu():
    """Run nvidia-smi to find the GPU with low utilization and sufficient memory."""
    try:
        # Query GPU index, utilization.gpu, and memory.free
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.free', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode != 0:
            print(f"⚠️ Warning: nvidia-smi failed, defaulting to GPU 0.")
            return "0"

        gpu_candidates = []
        for line in result.stdout.strip().split('\n'):
            parts = line.split(',')
            if len(parts) == 3:
                index = parts[0].strip()
                utilization = int(parts[1].strip())
                memory_free = int(parts[2].strip())
                gpu_candidates.append({
                    "index": index, 
                    "util": utilization, 
                    "mem": memory_free
                })

        if not gpu_candidates:
            return "0"

        # Sort by utilization (ascending) then memory (descending)
        # We prefer a GPU with 0% utilization over one with 90%, even if the 90% one has more RAM.
        best_gpu = sorted(gpu_candidates, key=lambda x: (x["util"], -x["mem"]))[0]
        
        print(f"✅ Auto-selected GPU {best_gpu['index']} (Util: {best_gpu['util']}%, Free: {best_gpu['mem']} MiB)")
        return best_gpu["index"]

    except Exception as e:
        print(f"⚠️ GPU auto-selection failed: {e}. Defaulting to GPU 0.")
        return "0"

# CRITICAL: Set CUDA device BEFORE importing torch or any library that imports torch
# If CUDA_VISIBLE_DEVICES is already set in shell (e.g. via 'CUDA_VISIBLE_DEVICES=1 python server.py'), use it.
# Otherwise, dynamically find the freest GPU.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = get_freest_gpu()
else:
    print(f"ℹ️ Using environment provided CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

import shutil
import uuid
import yaml
import hashlib
import json
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 导入你的核心处理逻辑
# 确保你的 scripts 目录在 PYTHONPATH 中，或者我们动态添加
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))
from scripts.sam3_extractor import main as sam3_main, Sam3ElementExtractor
from scripts.merge_xml import run_text_extraction, merge_xml

app = FastAPI(title="SAM3 Image to DrawIO API")

@app.get("/")
def root():
    return {"message": "SAM3 Backend is running correctly. Please access via Frontend (http://localhost:5173)."}

# 允许跨域，方便前端开发
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------- 全局配置与路径管理 --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

# 加载配置
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

# Web 任务的存储目录
WEB_TASKS_DIR = os.path.join(BASE_DIR, "output", "web_tasks")
os.makedirs(WEB_TASKS_DIR, exist_ok=True)

# 任务缓存文件（用于持久化和重用）
TASKS_CACHE_FILE = os.path.join(WEB_TASKS_DIR, "tasks_history.json")

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str
    cached: bool = False # 新增字段标识是否命中缓存

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: float  # 0.0 - 1.0
    original_image_url: Optional[str] = None
    result_xml_url: Optional[str] = None
    error: Optional[str] = None

# 使用内存简单的任务状态存储
# 结构: { task_id: { status, progress, image_path, ... } }
TASKS_DB = {}
# 图片哈希映射缓存
# 结构: { file_hash: task_id }
IMAGE_HASH_MAP = {}

import threading

# 全局 SAM3 实例 (Lazy Loaded)
GLOBAL_EXTRACTOR = None
GLOBAL_LOCK = threading.Lock() # 保护 GLOBAL_EXTRACTOR 的使用

def get_extractor():
    global GLOBAL_EXTRACTOR
    if GLOBAL_EXTRACTOR is None:
        print("Initializing Global SAM3 Extractor...")
        GLOBAL_EXTRACTOR = Sam3ElementExtractor()
    return GLOBAL_EXTRACTOR

def load_tasks_history():
    """从磁盘加载任务历史记录"""
    global TASKS_DB, IMAGE_HASH_MAP
    if os.path.exists(TASKS_CACHE_FILE):
        try:
            with open(TASKS_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                TASKS_DB = data.get("tasks", {})
                IMAGE_HASH_MAP = data.get("hashes", {})
            print(f"Loaded {len(TASKS_DB)} tasks from history.")
        except Exception as e:
            print(f"Error loading tasks history: {e}")

def save_tasks_history():
    """保存任务历史记录到磁盘"""
    try:
        data = {
            "tasks": TASKS_DB,
            "hashes": IMAGE_HASH_MAP
        }
        with open(TASKS_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving tasks history: {e}")

# 初始化时加载历史
load_tasks_history()

# -------------------------- 核心处理逻辑 --------------------------
def process_image_background(task_id: str, image_path: str, output_dir: str, file_hash: str = None):
    """后台任务：执行完整的 SAM3 + OCR + Merge 流程"""
    try:
        TASKS_DB[task_id]["status"] = "processing"
        TASKS_DB[task_id]["progress"] = 0.1
        # 实时保存状态
        save_tasks_history()
        
        img_stem = Path(image_path).stem
        
        # 1. & 2. 并行执行 SAM3 提取 和 OCR 提取
        print(f"[{task_id}] Starting parallel processing: SAM3 + OCR...")
        
        import concurrent.futures

        def run_sam3_with_global_model(img_path):
            """Helper to run SAM3 using the shared loaded model protected by a lock."""
            extractor = get_extractor()
            # 保护 GPU 推理过程，防止多任务并发导致 OOM
            with GLOBAL_LOCK:
                print(f"[{task_id}] Acquired Global SAM3 Lock. Starting inference...")
                # 关键修改：传入 task_dir 作为输出目录，避免 output/temp/original 冲突
                extractor.iterative_extract(img_path, specific_output_dir=output_dir)
                print(f"[{task_id}] Global SAM3 Inference finished. Lock released.")
            
            # 结果现在直接位于 task_dir (即 output_dir)
            return os.path.join(output_dir, "sam3_output.drawio.xml")

        # 使用线程池并行运行
        # 注意：如果显存有限，同时运行两个大模型可能会导致 OOM。
        # 如果遇到显存不足，请将 max_workers 改为 1 或回退到串行模式。
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # 提交任务 - 使用全局模型而不是运行脚本 main (避免重复加载)
            future_sam3 = executor.submit(run_sam3_with_global_model, image_path)
            future_text = executor.submit(run_text_extraction, image_path)
            
            # 获取结果 (会阻塞直到完成)
            sam3_xml_path = future_sam3.result()
            print(f"[{task_id}] Step 1: SAM3 Extraction completed.")
            
            text_xml_path = future_text.result()
            print(f"[{task_id}] Step 2: Text Extraction completed.")

        TASKS_DB[task_id]["progress"] = 0.8
        
        # 3. 合并 XML
        print(f"[{task_id}] Step 3: Merging XML...")
        final_xml_path = os.path.join(output_dir, "result.drawio.xml")
        merge_xml(sam3_xml_path, text_xml_path, image_path, final_xml_path)
        
        TASKS_DB[task_id]["progress"] = 1.0
        TASKS_DB[task_id]["status"] = "completed"
        TASKS_DB[task_id]["result_xml_path"] = final_xml_path
        
        # 任务成功完成后，记录哈希映射
        if file_hash:
            IMAGE_HASH_MAP[file_hash] = task_id
            
        save_tasks_history()
        
    except Exception as e:
        print(f"[{task_id}] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        TASKS_DB[task_id]["status"] = "failed"
        TASKS_DB[task_id]["error"] = str(e)
        save_tasks_history()

# -------------------------- API 接口 --------------------------

@app.post("/upload", response_model=TaskResponse)
async def upload_image(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...), 
    force: bool = False
):
    """
    上传图片并开始处理任务
    :param force: 是否强制重新处理（即使有缓存）
    """
    print(f"Receive upload request: {file.filename} (Force={force})") # Debug Log
    
    # 读取所有内容计算哈希
    content = await file.read()
    file_hash = hashlib.sha256(content).hexdigest()
    
    # 检查缓存 (如果不是强制刷新)
    if not force and file_hash in IMAGE_HASH_MAP:
        existing_task_id = IMAGE_HASH_MAP[file_hash]
        if existing_task_id in TASKS_DB:
            existing_task = TASKS_DB[existing_task_id]
            # 确保任务确实是成功的，并且文件还在
            if existing_task["status"] == "completed" and os.path.exists(existing_task.get("result_xml_path", "")):
                print(f"Cache Hit! HASH={file_hash} -> TaskID={existing_task_id}")
                return {
                    "task_id": existing_task_id,
                    "status": "completed",
                    "message": "Image processed loaded from cache.",
                    "cached": True
                }
            else:
                # 缓存无效（文件丢失或之前失败），清除
                del IMAGE_HASH_MAP[file_hash]
    
    # 如果没有缓存或缓存无效，创建新任务
    # 指针重置回开头（虽然 read() 后 file.file 可能需要 seek(0)，但我们已经读到 content 了）
    # 由于已经读到 content，可以直接写入文件，不再需要 copyfileobj
    
    task_id = str(uuid.uuid4())
    task_dir = os.path.join(WEB_TASKS_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)
    
    file_ext = Path(file.filename).suffix
    if not file_ext:
        file_ext = ".jpg" 
        
    image_path = os.path.join(task_dir, f"original{file_ext}")
    
    with open(image_path, "wb") as f:
        f.write(content)
        
    # 初始化任务状态
    TASKS_DB[task_id] = {
        "status": "pending",
        "progress": 0.0,
        "image_path": image_path,
        "original_filename": file.filename,
        "hash": file_hash
    }
    save_tasks_history()
    
    # 启动后台处理任务
    background_tasks.add_task(process_image_background, task_id, image_path, task_dir, file_hash)
    
    return {
        "task_id": task_id,
        "status": "pending",
        "message": "Image uploaded and processing started.",
        "cached": False
    }

@app.get("/task/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """查询任务状态"""
    if task_id not in TASKS_DB:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = TASKS_DB[task_id]
    
    response = {
        "task_id": task_id,
        "status": task["status"],
        "progress": task.get("progress", 0.0),
        "error": task.get("error")
    }
    
    # 构建资源 URL
    if task["status"] in ["processing", "completed"]:
         response["original_image_url"] = f"/files/{task_id}/original"
         
    if task["status"] == "completed":
        response["result_xml_url"] = f"/files/{task_id}/xml"
        
    return response

@app.get("/files/{task_id}/original")
async def get_original_image(task_id: str):
    """获取原图"""
    if task_id not in TASKS_DB:
        raise HTTPException(status_code=404, detail="Task not found")
    
    image_path = TASKS_DB[task_id]["image_path"]
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    return FileResponse(image_path)

@app.get("/files/{task_id}/xml")
async def get_result_xml(task_id: str):
    """获取生成的 XML"""
    if task_id not in TASKS_DB:
        raise HTTPException(status_code=404, detail="Task not found")
        
    if TASKS_DB[task_id]["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed yet")
    
    xml_path = TASKS_DB[task_id].get("result_xml_path")
    if not xml_path or not os.path.exists(xml_path):
        raise HTTPException(status_code=404, detail="Result file not found")
        
    # 读取内容并返回（或者直接下载）
    # 为了让 Draw.io 嵌入能够读取，直接返回文本内容也不错，或者文件流
    return FileResponse(
        xml_path, 
        media_type="application/xml", 
        filename=f"drawio_export_{task_id}.xml"
    )

class SegmentationRequest(BaseModel):
    task_id: str
    x: int
    y: int

@app.post("/interactive/segment")
async def segment_at_point(req: SegmentationRequest):
    """交互式单点分割"""
    if req.task_id not in TASKS_DB:
        raise HTTPException(status_code=404, detail="Task not found")
        
    task_info = TASKS_DB[req.task_id]
    image_path = task_info["image_path"]
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image file not found")
        
    try:
        extractor = get_extractor()
        
        # 加锁以确保单个 GPU 模型实例不会并发调用导致错误
        # 即使我们在 extractor 内部使用 state 字典隔离了数据，PyTorch 模型推理本身在同一 CUDA Stream 下是串行的，
        # 但为了避免 Python 侧的状态混乱，加上 Lock 是最安全的做法。
        # 注意：由于我们在 extractor 内部实现了 cache，这里 switching users 不会触发重新 embedding
        with GLOBAL_LOCK:
            result = extractor.extract_at_point(image_path, (req.x, req.y))
        
        if result is None:
             raise HTTPException(status_code=500, detail="Failed to segment object at location")
             
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # 服务器启动入口
    uvicorn.run(app, host="0.0.0.0", port=8001)
