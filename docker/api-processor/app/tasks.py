import os
import json
import time
import logging
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .celery_worker import celery_app

log = logging.getLogger(__name__)

def _get_hdfs_client():
    """Создаёт HDFS клиент с конфигурацией из settings.yaml.

    Возвращает None, если HDFS недоступен (pyarrow + CLI).
    """
    try:
        from src.utils.config import load_config
        from src.storage.hdfs_client import HDFSClient, HDFSConfig

        config_path = Path("/app/config/settings.yaml")
        if not config_path.exists():
            config_path = Path("/workspace/config/settings.yaml")

        if not config_path.exists():
            log.warning("Settings file not found, HDFS unavailable")
            return None

        cfg = load_config(config_path)
        storage_cfg = cfg.get("storage", {})
        hdfs_cfg = HDFSConfig(
            root=storage_cfg.get("hdfs_root", "hdfs://namenode:9000"),
            input_dir=storage_cfg.get("input_dir", "/data/input"),
            output_dir=storage_cfg.get("output_dir", "/data/output"),
            metadata_dir=storage_cfg.get("metadata_dir", "/data/metadata"),
            replication=storage_cfg.get("replication", 1)
        )
        client = HDFSClient(hdfs_cfg)

        # Проверяем работоспособность клиента
        fs = client._get_fs()
        if fs is None:
            # pyarrow не работает, пробуем CLI
            import subprocess
            r = subprocess.run(["hdfs", "dfs", "-test", "-e", "/"], capture_output=True)
            if r.returncode != 0:
                log.warning("HDFS unavailable: pyarrow failed and CLI not accessible")
                return None

        log.info("HDFS client initialized successfully")
        return client
    except Exception as e:
        log.warning(f"HDFS client unavailable: {e}")
    return None

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
    config: Dict,
    job_id: str
) -> List[Dict]:
    """Обрабатывает тайлы через Spark.

    Возвращает обработанные тайлы. Если Spark недоступен — fallback на локальную обработку.
    """
    try:
        from pyspark.sql import SparkSession
        from pyspark import SparkContext
        from src.utils.config import get_spark_config, load_config
    except ImportError:
        log.warning("PySpark not available, using local processing")
        for tile in tiles:
            tile["data"] = _mock_enhance(tile["data"])
        return tiles

    # Загружаем полную конфигурацию из settings.yaml
    try:
        config_path = Path("/app/config/settings.yaml")
        if not config_path.exists():
            config_path = Path("/workspace/config/settings.yaml")
        full_config = load_config(config_path) if config_path.exists() else {}
    except Exception as e:
        log.warning(f"Could not load settings.yaml: {e}")
        full_config = {}

    spark_cfg = full_config.get("spark", config.get("spark", {}))

    builder = (
        SparkSession.builder
        .appName(spark_cfg.get("app_name", "SatlasEnhancement"))
        .master(spark_cfg.get("master", "spark://spark-master:7077"))
        .config("spark.driver.memory", spark_cfg.get("driver_memory", "4g"))
        .config("spark.driver.maxResultSize", spark_cfg.get("driver_maxResultSize", "0"))
        .config("spark.executor.memory", spark_cfg.get("executor_memory", "2g"))
        .config("spark.python.worker.reuse", str(spark_cfg.get("python_worker_reuse", True)).lower())
        .config("spark.task.cpus", str(spark_cfg.get("task_cpus", 1)))
    )

    # Добавляем конфиги из get_spark_config
    for key, value in get_spark_config(full_config).items():
        builder = builder.config(key, value)

    try:
        spark = builder.getOrCreate()
        log.info(f"Spark session created: {spark.sparkContext.appName}")
    except Exception as e:
        log.error(f"Failed to create Spark session: {e}, falling back to local")
        for tile in tiles:
            tile["data"] = _mock_enhance(tile["data"])
        return tiles

    sc = spark.sparkContext

    try:
        use_mock = config.get("model", {}).get("use_mock", True)

        def process_tile(tile_dict):
            """Функция обработки одного тайла на воркере"""
            tile_data = tile_dict["data"]

            if use_mock:
                enhanced = _mock_enhance(tile_data)
            else:
                # Здесь должна быть реальная модель
                enhanced = tile_data

            return {**tile_dict, "data": enhanced}

        num_partitions = spark_cfg.get("num_partitions", 2)
        rdd = sc.parallelize(tiles, numSlices=num_partitions)
        processed = rdd.map(process_tile).collect()

        log.info(f"Spark processed {len(processed)} tiles")
        return processed

    except Exception as e:
        log.error(f"Spark processing failed: {e}, falling back to local")
        for tile in tiles:
            tile["data"] = _mock_enhance(tile["data"])
        return tiles
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
    """Celery задача для обработки изображения с использованием Spark и HDFS"""
    params = params or {}

    out_dir = Path(os.getenv("PROCESSING_OUT_DIR", "/data/processed")) / job_id
    tiles_dir = out_dir / "tiles"
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir.mkdir(exist_ok=True)

    # Определяем режим обработки: mock только если явно указано в параметрах
    use_mock = params.get("use_mock", False)

    # Загружаем конфигурацию из settings.yaml
    try:
        from src.utils.config import load_config
        config_path = Path("/app/config/settings.yaml")
        if not config_path.exists():
            config_path = Path("/workspace/config/settings.yaml")

        if config_path.exists():
            full_config = load_config(config_path)
        else:
            full_config = {}
    except Exception as e:
        log.warning(f"Could not load settings.yaml: {e}")
        full_config = {}

    config = {
        "image": {
            "tile_size": params.get("tile_size", full_config.get("image", {}).get("tile_size", 512)),
            "overlap": params.get("overlap", full_config.get("image", {}).get("overlap", 32))
        },
        "model": {
            "use_mock": use_mock,
            "batch_size": full_config.get("model", {}).get("batch_size", 2),
            "device": full_config.get("model", {}).get("device", "cpu")
        },
        "spark": {
            "app_name": full_config.get("spark", {}).get("app_name", "SatlasEnhancement"),
            "master": full_config.get("spark", {}).get("master", "spark://spark-master:7077"),
            "driver_memory": full_config.get("spark", {}).get("driver_memory", "4g"),
            "driver_maxResultSize": full_config.get("spark", {}).get("driver_maxResultSize", "0"),
            "executor_memory": full_config.get("spark", {}).get("executor_memory", "2g"),
            "num_partitions": full_config.get("spark", {}).get("num_partitions", 2),
            "task_cpus": full_config.get("spark", {}).get("task_cpus", 1),
            "python_worker_reuse": full_config.get("spark", {}).get("python_worker_reuse", True)
        },
        "postprocessing": {
            "blending": params.get("blending", full_config.get("postprocessing", {}).get("blending", "gaussian")),
            "gaussian_sigma_ratio": full_config.get("postprocessing", {}).get("gaussian_sigma_ratio", 0.5)
        }
    }

    # Инициализируем HDFS клиент
    hdfs_client = _get_hdfs_client()
    hdfs_output_path = f"/data/output/{job_id}"
    log.info(f"HDFS client initialized: {hdfs_client is not None}")
    log.info(f"HDFS output path: {hdfs_output_path}")
    log.info(f"Processing mode: use_mock={use_mock}")

    if use_mock:
        log.info(f"Using mock processing for job {job_id}")
        base = np.zeros((512, 512, 3), dtype=np.uint8)
        base[100:400, 100:400] = [64, 64, 64]
        enhanced = np.ones((512, 512, 3), dtype=np.uint8) * 128
        enhanced[50:450, 50:450] = [200, 200, 200]

        for i in range(4):
            tile_path = tiles_dir / f"tile_{i:03d}_mock.png"
            Image.fromarray(base).save(tile_path)
            log.info(f"Saved mock tile to {tile_path}")

            # Сохраняем тайлы в HDFS
            if hdfs_client:
                try:
                    hdfs_tile_path = f"{hdfs_output_path}/tiles/tile_{i:03d}_mock.png"
                    hdfs_client.put_image(base, hdfs_tile_path)
                    log.info(f"Saved mock tile to HDFS: {hdfs_tile_path}")
                except Exception as e:
                    log.warning(f"Failed to save tile to HDFS: {e}")

        result_path = out_dir / "result_enhanced.png"
        Image.fromarray(enhanced).save(result_path)

        # Сохраняем результат в HDFS
        if hdfs_client:
            try:
                hdfs_client.put_image(enhanced, f"{hdfs_output_path}/result_enhanced.png")
            except Exception as e:
                log.warning(f"Failed to save result to HDFS: {e}")

        return {
            "status": "completed",
            "is_mock": True,
            "output": str(result_path),
            "hdfs_output": f"{hdfs_output_path}/result_enhanced.png" if hdfs_client else None,
            "tiles_dir": str(tiles_dir),
            "hdfs_tiles_dir": f"{hdfs_output_path}/tiles" if hdfs_client else None,
            "tiles_count": 4
        }

    try:
        tiles, orig_shape = _split_image_to_tiles(
            file_path,
            tile_size=config["image"]["tile_size"],
            overlap=config["image"]["overlap"]
        )

        # Сохраняем оригинальные тайлы локально и в HDFS
        for tile in tiles:
            tile_path = tiles_dir / f"tile_{tile['index']:03d}_original.png"
            Image.fromarray(tile["data"]).save(tile_path)

            if hdfs_client:
                try:
                    hdfs_client.put_image(tile["data"], f"{hdfs_output_path}/tiles/tile_{tile['index']:03d}_original.png")
                except Exception as e:
                    log.warning(f"Failed to save original tile to HDFS: {e}")

        # Обрабатываем тайлы через Spark
        processed_tiles = _spark_process_tiles(tiles, config, job_id)

        # Сохраняем обработанные тайлы локально и в HDFS
        for tile in processed_tiles:
            tile_path = tiles_dir / f"tile_{tile['index']:03d}_enhanced.png"
            Image.fromarray(tile["data"]).save(tile_path)

            if hdfs_client:
                try:
                    hdfs_client.put_image(tile["data"], f"{hdfs_output_path}/tiles/tile_{tile['index']:03d}_enhanced.png")
                except Exception as e:
                    log.warning(f"Failed to save enhanced tile to HDFS: {e}")

        # Собираем изображение
        assembled = _assemble_tiles(
            processed_tiles,
            orig_shape,
            blending=config["postprocessing"]["blending"],
            gaussian_sigma_ratio=config["postprocessing"]["gaussian_sigma_ratio"]
        )

        result_path = out_dir / "result_enhanced.png"
        Image.fromarray(assembled).save(result_path)

        # Сохраняем результат в HDFS
        if hdfs_client:
            try:
                hdfs_client.put_image(assembled, f"{hdfs_output_path}/result_enhanced.png")

                # Сохраняем метаданные
                metadata = {
                    "job_id": job_id,
                    "filename": filename,
                    "original_shape": list(orig_shape),
                    "output_shape": list(assembled.shape),
                    "tiles_count": len(processed_tiles),
                    "config": config
                }
                hdfs_client.put_json(metadata, f"{hdfs_output_path}/metadata.json")
            except Exception as e:
                log.warning(f"Failed to save result to HDFS: {e}")

        return {
            "status": "completed",
            "output": str(result_path),
            "hdfs_output": f"{hdfs_output_path}/result_enhanced.png" if hdfs_client else None,
            "tiles_dir": str(tiles_dir),
            "hdfs_tiles_dir": f"{hdfs_output_path}/tiles" if hdfs_client else None,
            "tiles_count": len(processed_tiles),
            "output_shape": list(assembled.shape)
        }

    except Exception as e:
        import traceback
        log.error(f"Processing failed: {e}\n{traceback.format_exc()}")
        return {
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        }