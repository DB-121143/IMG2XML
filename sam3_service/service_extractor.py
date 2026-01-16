import os
import io
import base64
import time
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from scripts.sam3_extractor import (
    extract_style_colors,
    image_to_base64,
    deduplicate_elements,
    build_drawio_xml,
    calculate_element_area,
    calculate_arrow_midpoints,
    prettify_xml,
    VECTOR_SUPPORTED_PROMPTS,
)
from scripts.multimodal_prompt import get_supplement_prompts


class Sam3ServiceElementExtractor:
    """HTTP 服务版的 SAM3 迭代提取器，复刻本地 Sam3ElementExtractor 的流程。"""

    def __init__(self, sam3_pool, rmbg_pool, sam3_config: Dict, output_config: Dict):
        self.sam3_pool = sam3_pool
        self.rmbg_pool = rmbg_pool
        self.sam3_config = sam3_config
        self.output_config = output_config
        self.known_picture_prompts = {"picture"}

    # -------------- 内部工具 --------------
    @staticmethod
    def _decode_mask(det: Dict) -> Optional[np.ndarray]:
        mask_b64 = det.get("mask") or ""
        if not mask_b64:
            return None
        try:
            mask_bytes = base64.b64decode(mask_b64)
            mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
            return np.array(mask_img)
        except Exception:
            return None

    @staticmethod
    def _polygon_from_mask(mask: np.ndarray) -> List[List[int]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1:
                continue
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            return approx.reshape(-1, 2).tolist()
        return []

    def _process_detections(
        self,
        image_path: Path,
        detections: List[Dict],
        pil_image: Image.Image,
        cv2_image: np.ndarray,
        existing_result: Optional[Dict],
    ) -> Dict:
        if existing_result is None:
            elements_data: Dict[str, List[Dict]] = {}
            full_metadata = {
                "image_path": str(image_path),
                "image_size": {"width": pil_image.size[0], "height": pil_image.size[1]},
                "elements": {},
                "total_elements": 0,
                "total_area": 0,
            }
        else:
            elements_data = existing_result["elements"]
            full_metadata = existing_result["full_metadata"]

        rmbg_jobs = []  # (elem_ref, b64_input, log_label)

        for det in detections:
            prompt = det.get("prompt", "")
            score = float(det.get("score", 0))
            if score < self.sam3_config.get("score_threshold", 0.5):
                continue
            bbox = [int(v) for v in det.get("bbox", [0, 0, 0, 0])]
            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1:
                continue

            area = calculate_element_area(bbox)
            if area < self.sam3_config.get("min_area", 0):
                continue

            polygon = det.get("polygon", []) or []
            mask_np = self._decode_mask(det)
            if not polygon and mask_np is not None:
                polygon = self._polygon_from_mask(mask_np)

            elem = {
                "id": full_metadata["total_elements"],
                "score": score,
                "bbox": bbox,
                "polygon": polygon,
                "area": area,
            }

            # 属性生成
            if prompt == "icon":
                crop = pil_image.crop((x1, y1, x2, y2))
                buf = io.BytesIO()
                crop.save(buf, format="PNG")
                buf.seek(0)
                crop_b64 = base64.b64encode(buf.read()).decode("ascii")
                rmbg_jobs.append((elem, crop_b64, f"icon bbox={bbox}"))
            elif prompt == "picture":
                crop = pil_image.crop((x1, y1, x2, y2))
                elem["base64"] = image_to_base64(crop)
            elif prompt in VECTOR_SUPPORTED_PROMPTS:
                # 矢量类型取色
                fill_color, stroke_color = extract_style_colors(cv2_image, bbox)
                elem["fill_color"] = fill_color
                elem["stroke_color"] = stroke_color
            elif prompt == "arrow":
                # 参考本地实现：使用 mask 定位并 RMBG
                pad = 15
                img_w, img_h = pil_image.size
                p_x1 = max(0, x1 - pad)
                p_y1 = max(0, y1 - pad)
                p_x2 = min(img_w, x2 + pad)
                p_y2 = min(img_h, y2 + pad)
                cropped_pil = pil_image.crop((p_x1, p_y1, p_x2, p_y2))
                cropped_np = np.array(cropped_pil)
                if mask_np is not None:
                    mask_crop = mask_np[p_y1:p_y2, p_x1:p_x2]
                    kernel = np.ones((10, 10), np.uint8)
                    dilated = cv2.dilate(mask_crop, kernel, iterations=1)
                    mask_3ch = cv2.cvtColor(dilated, cv2.COLOR_GRAY2RGB)
                    white_bg = np.full_like(cropped_np, 255)
                    masked_np = np.where(mask_3ch > 0, cropped_np, white_bg)
                    input_pil = Image.fromarray(masked_np)
                else:
                    input_pil = cropped_pil
                rmbg_jobs.append(
                    (elem, image_to_base64(input_pil), f"arrow bbox={bbox} padded={[p_x1,p_y1,p_x2,p_y2]}")
                )
                elem["bbox"] = [p_x1, p_y1, p_x2, p_y2]
                elem["area"] = (p_x2 - p_x1) * (p_y2 - p_y1)

            if prompt not in elements_data:
                elements_data[prompt] = []
            elements_data[prompt].append(elem)
            full_metadata["total_elements"] += 1
            full_metadata["total_area"] += elem["area"]

        # 并行执行 RMBG 调用（限制线程数不超过端点数，避免单一端口排队过长）
        if rmbg_jobs:
            max_per_image = max(1, len(getattr(self.rmbg_pool, "clients", [])))
            max_workers = min(max_per_image, len(rmbg_jobs))

            def _run_rmbg(b64_input: str, submitted_at: float):
                start = time.time()
                out = self.rmbg_pool.remove(b64_input)
                end = time.time()
                queue_elapsed = start - submitted_at
                exec_elapsed = end - start
                return out, queue_elapsed, exec_elapsed

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                future_map = {}
                for elem_ref, b64_input, log_label in rmbg_jobs:
                    submitted_at = time.time()
                    fut = ex.submit(_run_rmbg, b64_input, submitted_at)
                    future_map[fut] = (elem_ref, log_label, submitted_at)

                for fut in concurrent.futures.as_completed(future_map):
                    elem_ref, log_label, submitted_at = future_map[fut]
                    rgba_b64, queue_elapsed, exec_elapsed = fut.result()
                    total_elapsed = time.time() - submitted_at
                    elem_ref["base64"] = rgba_b64
                    print(
                        f"[RMBG] {log_label} queue={queue_elapsed:.3f}s exec={exec_elapsed:.3f}s total={total_elapsed:.3f}s"
                    )

        # 去重
        if full_metadata["total_elements"] > 0:
            elements_data = deduplicate_elements(elements_data, iou_threshold=0.85)
            # 重新统计
            total = 0
            area_sum = 0
            for _, items in elements_data.items():
                total += len(items)
                for item in items:
                    area_sum += item["area"]
            full_metadata["total_elements"] = total
            full_metadata["total_area"] = area_sum

        full_metadata["elements"] = elements_data
        return {
            "canvas_size": (pil_image.size[0], pil_image.size[1]),
            "elements": elements_data,
            "full_metadata": full_metadata,
            "pil_image": pil_image,
            "cv2_image": cv2_image,
        }

    # -------------- 可视化 & 保存 --------------
    @staticmethod
    def generate_mask_visualization(cv2_image: np.ndarray, elements_data: dict, output_path: str):
        image = cv2_image.copy()
        overlay = cv2_image.copy()
        global_id = 0
        for elem_type, items in elements_data.items():
            for item in items:
                color = (0, 255, 0) if global_id % 2 == 0 else (255, 0, 0)
                points = item.get("polygon") or []
                if points:
                    pts = np.array(points, dtype=np.int32)
                    cv2.fillPoly(overlay, [pts], color)
                    cv2.rectangle(image, (item["bbox"][0], item["bbox"][1]), (item["bbox"][2], item["bbox"][3]), color, 2)
                else:
                    x1, y1, x2, y2 = item["bbox"]
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                if elem_type == "arrow":
                    src_mid, tgt_mid = calculate_arrow_midpoints(item["bbox"])
                    cv2.arrowedLine(image, src_mid, tgt_mid, color, 2, tipLength=0.1)
                global_id += 1
        result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        cv2.imwrite(output_path, result)
        return output_path

    def save_xml(self, extract_result: dict, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        xml_root = build_drawio_xml(
            extract_result["canvas_size"][0],
            extract_result["canvas_size"][1],
            extract_result["elements"],
        )
        xml_path = os.path.join(output_dir, "sam3_output.drawio.xml")
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(prettify_xml(xml_root))
        return xml_path

    # -------------- 迭代主流程 --------------
    def iterative_extract(self, image_path: str, specific_output_dir: Optional[str] = None) -> str:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片不存在: {image_path}")

        if specific_output_dir:
            temp_dir = specific_output_dir
        else:
            temp_dir = os.path.join(self.output_config["temp_dir"], Path(image_path).stem)
        os.makedirs(temp_dir, exist_ok=True)
        vis_dir = Path(temp_dir)

        pil_image = Image.open(image_path).convert("RGB")
        cv2_image = cv2.imread(image_path)

        # Round 1
        current_prompts = list(self.sam3_config.get("initial_prompts", ["rectangle", "icon", "arrow"]))
        known_prompts = set(current_prompts)
        t0 = time.time()
        resp = self.sam3_pool.predict(
            image_path=str(image_path),
            prompts=current_prompts,
            return_masks=True,
            mask_format="png",
        )
        print(f"[SAM3] Round1 prompts={current_prompts} time={time.time()-t0:.3f}s")
        current_result = self._process_detections(image_path, resp.get("results", []), pil_image, cv2_image, existing_result=None)
        vis_path = str(vis_dir / "mask_vis_round_1.jpg")
        self.generate_mask_visualization(current_result["cv2_image"], current_result["elements"], vis_path)

        # Rounds 2-4
        for round_idx in range(2, 5):
            vis_prev = str(vis_dir / f"mask_vis_round_{round_idx-1}.jpg")
            if not os.path.exists(vis_prev):
                self.generate_mask_visualization(current_result["cv2_image"], current_result["elements"], vis_prev)

            vlm_resp = get_supplement_prompts(
                mask_vis_path=vis_prev,
                existing_prompts=list(known_prompts),
                round_index=round_idx,
                original_image_path=image_path,
            )
            if vlm_resp.get("error", False):
                break

            icon_prompts = vlm_resp.get("icon_prompts", [])
            picture_prompts = vlm_resp.get("picture_prompts", [])
            has_missing = vlm_resp.get("has_missing", False)

            for p in picture_prompts:
                self.known_picture_prompts.add(p)

            new_candidates = list(set(icon_prompts + picture_prompts))
            valid_new = [p for p in new_candidates if p not in known_prompts]
            if not valid_new:
                if not has_missing:
                    break
                else:
                    continue

            t1 = time.time()
            resp = self.sam3_pool.predict(
                image_path=str(image_path),
                prompts=valid_new,
                return_masks=True,
                mask_format="png",
            )
            print(f"[SAM3] Round{round_idx} prompts={valid_new} time={time.time()-t1:.3f}s")
            current_result = self._process_detections(
                image_path,
                resp.get("results", []),
                pil_image,
                cv2_image,
                existing_result=current_result,
            )
            known_prompts.update(valid_new)
            vis_next = str(vis_dir / f"mask_vis_round_{round_idx}.jpg")
            self.generate_mask_visualization(current_result["cv2_image"], current_result["elements"], vis_next)

        xml_path = self.save_xml(current_result, temp_dir)
        return xml_path
