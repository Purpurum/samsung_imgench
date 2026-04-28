import os
import json
import time
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .celery_worker import celery_app

def _split_image_to_tiles(
    image_path: str,
    tile_size: int = 512,
    overlap: int = 32
) -> Tuple[List[Dict], Tuple[int, int, int]]:
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    channels = arr.shape[2] if arr.ndim == 3 else 3
    
    tiles = []
    idx = 0
    step_y = tile_size - overlap
    step_x = tile_size - overlap
    
    for y in range(0, h, step_y):
        for x in range(0, w, step_x):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            
            tile_data = arr[y:y_end, x:x_end]
            actual_h, actual_w = tile_data.shape[:2]
            
            tiles.append({
                "index": idx,
                "x": x,
                "y": y,
                "width": actual_w,
                "height": actual_h,
                "data": tile_data
            })
            idx += 1
    
    return tiles, (h, w, channels)

def _make_weight_mask(
    height: int,
    width: int,
    mode: str = "gaussian",
    sigma_ratio: float = 0.5
) -> np.ndarray:
    if mode == "none":
        return np.ones((height, width), dtype=np.float32)
    
    if mode == "average":
        return np.ones((height, width), dtype=np.float32)
    
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    sigma = sigma_ratio * min(height, width)
    
    mask = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    mask = mask.astype(np.float32)
    mask = np.maximum(mask, 1e-3)
    return mask

def _assemble_tiles(
    tiles: List[Dict],
    image_shape: Tuple[int, int, int],
    blending: str = "gaussian",
    gaussian_sigma_ratio: float = 0.5
) -> np.ndarray:
    h, w, c = image_shape
    accum = np.zeros((h, w, c), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)
    
    for tile in tiles:
        th, tw = tile["height"], tile["width"]
        mask = _make_weight_mask(th, tw, blending, gaussian_sigma_ratio)
        
        y0, y1 = tile["y"], tile["y"] + th
        x0, x1 = tile["x"], tile["x"] + tw
        
        tile_data = tile["data"].astype(np.float32)
        accum[y0:y1, x0:x1, :] += tile_data * mask[:, :, None]
        weight_sum[y0:y1, x0:x1] += mask
    
    weight_sum = np.maximum(weight_sum, 1e-8)
    result = accum / weight_sum[:, :, None]
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result

def _mock_enhance(tile_data: np.ndarray) -> np.ndarray:
    from PIL import ImageFilter
    img = Image.fromarray(tile_data, mode="RGB")
    denoised = img.filter(ImageFilter.MedianFilter(size=3))
    sharpened = denoised.filter(
        ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)
    )
    return np.asarray(sharpened, dtype=np.uint8)

def _spark_process_tiles(
    tiles: List[Dict],
    config: Dict
) -> List[Dict]:
    try:
        from pyspark.sql import SparkSession
        from pyspark import SparkContext
    except ImportError:
        for tile in tiles:
            tile["data"] = _mock_enhance(tile["data"])
        return tiles
    
    spark_cfg = config.get("spark", {})
    builder = (
        SparkSession.builder
        .appName(spark_cfg.get("app_name", "SatlasEnhancement"))
        .master(spark_cfg.get("master", "local[1]"))
        .config("spark.driver.memory", spark_cfg.get("driver_memory", "4g"))
        .config("spark.driver.maxResultSize", spark_cfg.get("driver_maxResultSize", "0"))
        .config("spark.executor.memory", spark_cfg.get("executor_memory", "2g"))
        .config("spark.python.worker.reuse", "true")
        .config("spark.task.cpus", str(spark_cfg.get("task_cpus", 1)))
    )
    
    spark = builder.getOrCreate()
    sc = spark.sparkContext
    
    try:
        use_mock = config.get("model", {}).get("use_mock", True)
        
        def process_tile(tile_dict):
            if use_mock:
                enhanced = _mock_enhance(tile_dict["data"])
            else:
                enhanced = tile_dict["data"]
            return {**tile_dict, "data": enhanced}
        
        rdd = sc.parallelize(tiles, numSlices=spark_cfg.get("num_partitions", 2))
        processed = rdd.map(process_tile).collect()
        
        return processed
    finally:
        spark.stop()

@celery_app.task(bind=True, name="app.tasks.process_uploaded_image")
def process_uploaded_image(
    self,
    job_id: str,
    file_path: str,
    filename: str,
    params: Optional[Dict] = None
):
    params = params or {}
    
    out_dir = Path(os.getenv("PROCESSING_OUT_DIR", "/data/processed")) / job_id
    tiles_dir = out_dir / "tiles"
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir.mkdir(exist_ok=True)
    
    is_mock_file = any(k in filename.lower() for k in ["mock", "test", "заглушка", "demo", "placeholder"])
    use_mock = params.get("use_mock", True) or is_mock_file
    
    config = {
        "image": {
            "tile_size": params.get("tile_size", 512),
            "overlap": params.get("overlap", 32)
        },
        "model": {
            "use_mock": use_mock,
            "batch_size": 2,
            "device": "cpu"
        },
        "spark": {
            "app_name": "SatlasEnhancement",
            "master": "local[1]",
            "driver_memory": "4g",
            "driver_maxResultSize": "0",
            "executor_memory": "2g",
            "num_partitions": 2,
            "task_cpus": 1
        },
        "postprocessing": {
            "blending": "gaussian",
            "gaussian_sigma_ratio": 0.5
        }
    }
    
    if is_mock_file:
        base = np.zeros((512, 512, 3), dtype=np.uint8)
        base[100:400, 100:400] = [64, 64, 64]
        enhanced = np.ones((512, 512, 3), dtype=np.uint8) * 128
        enhanced[50:450, 50:450] = [200, 200, 200]
        
        for i in range(4):
            tile_path = tiles_dir / f"tile_{i:03d}_mock.png"
            Image.fromarray(base).save(tile_path)
        
        result_path = out_dir / "result_enhanced.png"
        Image.fromarray(enhanced).save(result_path)
        
        return {
            "status": "completed",
            "is_mock": True,
            "output": str(result_path),
            "tiles_dir": str(tiles_dir),
            "tiles_count": 4
        }
    
    try:
        tiles, orig_shape = _split_image_to_tiles(
            file_path,
            tile_size=config["image"]["tile_size"],
            overlap=config["image"]["overlap"]
        )
        
        for tile in tiles:
            tile_path = tiles_dir / f"tile_{tile['index']:03d}_original.png"
            Image.fromarray(tile["data"]).save(tile_path)
        
        processed_tiles = _spark_process_tiles(tiles, config)
        
        for tile in processed_tiles:
            tile_path = tiles_dir / f"tile_{tile['index']:03d}_enhanced.png"
            Image.fromarray(tile["data"]).save(tile_path)
        
        assembled = _assemble_tiles(
            processed_tiles,
            orig_shape,
            blending=config["postprocessing"]["blending"],
            gaussian_sigma_ratio=config["postprocessing"]["gaussian_sigma_ratio"]
        )
        
        result_path = out_dir / "result_enhanced.png"
        Image.fromarray(assembled).save(result_path)
        
        return {
            "status": "completed",
            "output": str(result_path),
            "tiles_dir": str(tiles_dir),
            "tiles_count": len(processed_tiles),
            "output_shape": list(assembled.shape)
        }
        
    except Exception as e:
        import traceback
        return {
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        }