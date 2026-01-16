import os
import sys
import uuid
import shutil
from pathlib import Path
from typing import Dict, Optional

import yaml
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Ensure project root on path
ROOT = Path(__file__).resolve().parent
SAM3_SERVICE_DIR = ROOT / "sam3_service"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SAM3_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SAM3_SERVICE_DIR))

from sam3_service.client import Sam3ServicePool
from sam3_service.rmbg_client import RMBGServicePool
from sam3_service.service_extractor import Sam3ServiceElementExtractor
from scripts.merge_xml import run_text_extraction, merge_xml

# ----------------------- Config -----------------------
CONFIG_PATH = ROOT / "config" / "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

SAM3_CONFIG = CONFIG.get("sam3", {})
OUTPUT_CONFIG = CONFIG.get("paths", {})
INPUT_DIR = Path(OUTPUT_CONFIG.get("input_dir", ROOT / "input"))
TEMP_DIR = Path(OUTPUT_CONFIG.get("temp_dir", ROOT / "output" / "temp"))
FINAL_DIR = Path(OUTPUT_CONFIG.get("final_dir", ROOT / "output" / "final"))

SAM3_ENDPOINTS_CFG = CONFIG.get("services", {}).get("sam3_endpoints", [])
RMBG_ENDPOINTS_CFG = CONFIG.get("services", {}).get("rmbg_endpoints", [])

SAM3_ENDPOINTS_ENV = os.environ.get("SAM3_ENDPOINTS", "")
RMBG_ENDPOINTS_ENV = os.environ.get("RMBG_ENDPOINTS", "")


def _pick_endpoints(arg_val: str, env_val: str, cfg_val):
    arg_list = [e.strip() for e in arg_val.split(",") if e.strip()] if arg_val else []
    env_list = [e.strip() for e in env_val.split(",") if e.strip()] if env_val else []
    return arg_list or env_list or cfg_val


# ----------------------- Pools & extractor -----------------------
sam3_pool = Sam3ServicePool(_pick_endpoints("", SAM3_ENDPOINTS_ENV, SAM3_ENDPOINTS_CFG))
rmbg_pool = RMBGServicePool(_pick_endpoints("", RMBG_ENDPOINTS_ENV, RMBG_ENDPOINTS_CFG))
extractor = Sam3ServiceElementExtractor(sam3_pool, rmbg_pool, SAM3_CONFIG, OUTPUT_CONFIG)

# ----------------------- FastAPI app -----------------------
app = FastAPI(title="SAM3 Service Pipeline API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Task store (in-memory)
TASKS: Dict[str, Dict] = {}


class TaskStatus(BaseModel):
    task_id: str
    status: str
    progress: float
    result_xml_url: Optional[str] = None
    error: Optional[str] = None


@app.get("/")
def root():
    return {"message": "SAM3 pipeline API is running"}


@app.post("/upload")
async def upload_image(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    task_id = str(uuid.uuid4())
    task_dir = TEMP_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(file.filename).suffix or ".jpg"
    img_path = task_dir / f"input{ext}"
    with open(img_path, "wb") as f:
        f.write(content)

    TASKS[task_id] = {"status": "pending", "progress": 0.0, "img_path": str(img_path)}
    background_tasks.add_task(_run_pipeline, task_id)
    return {"task_id": task_id, "status": "pending"}


def _run_pipeline(task_id: str):
    try:
        task = TASKS.get(task_id)
        if not task:
            return
        task["status"] = "processing"
        task["progress"] = 0.1

        img_path = task["img_path"]
        img_stem = Path(img_path).stem
        out_dir = Path(img_path).parent

        sam3_xml = extractor.iterative_extract(img_path, specific_output_dir=str(out_dir))
        task["progress"] = 0.5

        text_xml = run_text_extraction(img_path)
        task["progress"] = 0.7

        merged_path = out_dir / f"{img_stem}_merged.drawio.xml"
        merge_xml(sam3_xml, text_xml, img_path, merged_path)

        FINAL_DIR.mkdir(parents=True, exist_ok=True)
        final_path = FINAL_DIR / f"{img_stem}.drawio.xml"
        shutil.copy(str(merged_path), str(final_path))

        task["status"] = "completed"
        task["progress"] = 1.0
        task["result_xml"] = str(final_path)
    except Exception as exc:  # pragma: no cover
        task = TASKS.get(task_id, {})
        task["status"] = "failed"
        task["error"] = str(exc)
        task["progress"] = 1.0
        TASKS[task_id] = task


@app.get("/task/{task_id}", response_model=TaskStatus)
def get_task(task_id: str):
    task = TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskStatus(
        task_id=task_id,
        status=task.get("status", "unknown"),
        progress=task.get("progress", 0.0),
        result_xml_url=f"/files/{task_id}/xml" if task.get("status") == "completed" else None,
        error=task.get("error"),
    )


@app.get("/files/{task_id}/xml")
def download_xml(task_id: str):
    task = TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Task not completed")
    xml_path = task.get("result_xml")
    if not xml_path or not os.path.exists(xml_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(xml_path, media_type="application/xml", filename=Path(xml_path).name)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
