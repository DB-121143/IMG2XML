import os
import sys
import yaml
import shutil
import argparse
import concurrent.futures
from pathlib import Path
from typing import List

# Ensure project root on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_service.client import Sam3ServicePool
from sam3_service.rmbg_client import RMBGServicePool
from sam3_service.service_extractor import Sam3ServiceElementExtractor
from scripts.merge_xml import run_text_extraction, merge_xml

# Config paths
CONFIG_PATH = ROOT / "config" / "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

SAM3_CONFIG = CONFIG["sam3"]
OUTPUT_CONFIG = CONFIG["paths"]
INPUT_DIR = Path(OUTPUT_CONFIG["input_dir"])
FINAL_DIR = Path(OUTPUT_CONFIG["final_dir"])
TEMP_DIR = Path(OUTPUT_CONFIG["temp_dir"])

# Endpoints for the service pool (argument > env > config)
SAM3_ENDPOINTS_ENV = os.environ.get("SAM3_ENDPOINTS", "")
RMBG_ENDPOINTS_ENV = os.environ.get("RMBG_ENDPOINTS", "")
SAM3_ENDPOINTS_CFG = CONFIG.get("services", {}).get("sam3_endpoints", [])
RMBG_ENDPOINTS_CFG = CONFIG.get("services", {}).get("rmbg_endpoints", [])


def process_single_image(image_path: Path, extractor: Sam3ServiceElementExtractor):
    print(f"\n========== 开始处理：{image_path} ==========")
    img_stem = image_path.stem

    sam3_xml_path = extractor.iterative_extract(str(image_path), specific_output_dir=str(TEMP_DIR / img_stem))
    if not os.path.exists(sam3_xml_path) or os.path.getsize(sam3_xml_path) == 0:
        raise RuntimeError(f"SAM3生成的XML为空或不存在：{sam3_xml_path}")

    print("步骤2：文字识别...")
    text_xml_path = run_text_extraction(str(image_path))
    if not os.path.exists(text_xml_path) or os.path.getsize(text_xml_path) == 0:
        raise RuntimeError(f"文字识别生成的XML为空或不存在：{text_xml_path}")

    print("步骤3：合并XML...")
    merged_xml_path = TEMP_DIR / f"{img_stem}_merged_temp.drawio.xml"
    merge_xml(str(sam3_xml_path), text_xml_path, str(image_path), str(merged_xml_path))

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    final_output_path = FINAL_DIR / f"{img_stem}.drawio.xml"
    shutil.copy(str(merged_xml_path), str(final_output_path))
    print(f"✅ 处理完成，结果已保存: {final_output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pipeline using SAM3 HTTP service")
    parser.add_argument(
        "--endpoints",
        default="",
        help="逗号分隔的 SAM3 服务端地址列表，优先级：参数 > 环境变量 SAM3_ENDPOINTS > config.services.sam3_endpoints",
    )
    parser.add_argument(
        "--rmbg-endpoints",
        default="",
        help="逗号分隔的 RMBG 服务端地址列表，优先级：参数 > 环境变量 RMBG_ENDPOINTS > config.services.rmbg_endpoints",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并发处理图片的线程数（建议不要超过服务端总进程数）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    def _pick_endpoints(arg_val: str, env_val: str, cfg_val: List[str]):
        arg_list = [e.strip() for e in arg_val.split(",") if e.strip()]
        env_list = [e.strip() for e in env_val.split(",") if e.strip()]
        return arg_list or env_list or cfg_val

    endpoints = _pick_endpoints(args.endpoints, SAM3_ENDPOINTS_ENV, SAM3_ENDPOINTS_CFG)
    if not endpoints:
        print("错误：未提供有效的 SAM3 端点，使用 --endpoints、环境变量 SAM3_ENDPOINTS 或 config.services.sam3_endpoints 配置")
        sys.exit(1)

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    sam3_pool = Sam3ServicePool(endpoints)
    rmbg_endpoints = _pick_endpoints(args.rmbg_endpoints, RMBG_ENDPOINTS_ENV, RMBG_ENDPOINTS_CFG)
    if not rmbg_endpoints:
        print("错误：未提供 RMBG 端点，使用 --rmbg-endpoints、环境变量 RMBG_ENDPOINTS 或 config.services.rmbg_endpoints 配置")
        sys.exit(1)
    rmbg_pool = RMBGServicePool(rmbg_endpoints)
    extractor = Sam3ServiceElementExtractor(sam3_pool, rmbg_pool, SAM3_CONFIG, OUTPUT_CONFIG)

    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    images = [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in supported]
    if not images:
        print(f"错误：输入目录{INPUT_DIR}中未找到支持的图片文件")
        sys.exit(1)

    print(f"发现 {len(images)} 张图片，将使用 {args.workers} 线程处理，SAM3 端点：{endpoints}，RMBG 端点：{rmbg_endpoints}")

    def _task(img: Path):
        try:
            process_single_image(img, extractor)
            return True, str(img)
        except Exception as exc:
            print(f"处理 {img} 失败：{exc}")
            return False, str(img)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(_task, images))

    failed = [img for ok, img in results if not ok]
    if failed:
        print(f"\n有 {len(failed)} 张图片处理失败：{failed}")
    else:
        print(f"\n所有图片处理完成！最终文件保存在：{FINAL_DIR}")


if __name__ == "__main__":
    main()
