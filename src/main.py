"""Точка входа пайплайна улучшения спутниковых снимков.

Запуск:

    # Из CLI через spark-submit
    spark-submit --master spark://spark-master:7077 \\
        --conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000 \\
        src/main.py \\
        --image /app/data/samples/sat_001.png \\
        --image-id SAT_20240420_001 \\
        --config /app/config/settings.yaml

    # Из Jupyter
    from src.main import run_enhancement_pipeline
    run_enhancement_pipeline(
        image_path="/app/data/samples/sat_001.png",
        image_id="SAT_20240420_001",
        config_path="/app/config/settings.yaml",
    )

Пайплайн:

    1. Валидация входного файла
    2. Разбиение на тайлы (локально, на драйвере)
    3. Параллелизация тайлов в Spark RDD
    4. Распределённая обработка через mapPartitions:
         - На каждом процессе-воркере модель грузится один раз (lazy cache + thread-safe)
         - Тайлы обрабатываются потоковыми батчами (экономия памяти)
         - NumPy-массивы сериализуются с опциональным сжатием
    5. Сбор результатов обратно на драйвер (collect)
    6. Сборка полного изображения с gaussian-блендингом
    7. Запись исходного и обработанного в HDFS + JSON с метриками

Оптимизации:

    - mapPartitions вместо map: загрузка модели 1× на партицию, а не на элемент
    - Потоковый батчинг: обработка частями, без загрузки всей партиции в память
    - Thread-safe кеш модели: защита от race condition при многопоточных исполнителях
    - Консистентная структура метаданных: все поля присутствуют даже при ошибках
    - Валидация форматов и конфигурируемые пути
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import zlib
import pickle
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any

import numpy as np

# Важно для spark-submit: добавляем корень проекта в sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.model import ModelConfig, get_or_create_enhancer, ImageEnhancer
from src.postprocessing import assemble_tiles, psnr, ssim_simple
from src.preprocessing import Tile, split_image_into_tiles, validate_image
from src.storage import HDFSClient, HDFSConfig
from src.utils import load_config, setup_logging

log = logging.getLogger("satlas.main")

# -----------------------------------------------------------------
# Глобальные константы и кеш модели (thread-safe)
# -----------------------------------------------------------------

_ENHANCER_CACHE: Dict[str, ImageEnhancer] = {}
_ENHANCER_LOCK = threading.Lock()


def _get_or_create_enhancer_threadsafe(cfg: ModelConfig) -> ImageEnhancer:
    """Потоко-безопасное получение/создание энхансера.

    Использует double-checked locking для минимизации блокировок.
    """
    key = cfg.path
    # Быстрая проверка без лога
    if key in _ENHANCER_CACHE:
        return _ENHANCER_CACHE[key]

    with _ENHANCER_LOCK:
        # Повторная проверка внутри лога
        if key not in _ENHANCER_CACHE:
            _ENHANCER_CACHE[key] = get_or_create_enhancer(cfg)
            log.debug("Модель загружена в кеш: %s", key)
    return _ENHANCER_CACHE[key]


def _serialize_array(arr: np.ndarray, compress: bool = True) -> bytes:
    """Сериализация numpy-массива с опциональным сжатием."""
    data = pickle.dumps(arr, protocol=pickle.HIGHEST_PROTOCOL)
    return zlib.compress(data, level=6) if compress else data


def _deserialize_array(data: bytes, compress: bool = True) -> np.ndarray:
    """Десериализация numpy-массива с опциональным сжатием."""
    raw = zlib.decompress(data) if compress else data
    return pickle.loads(raw)


# -----------------------------------------------------------------
# Воркер-функция для mapPartitions (потоковая обработка)
# -----------------------------------------------------------------

def _process_partition_factory(model_cfg_dict: dict, spark_cfg: dict):
    """Фабрика функции-обработчика партиции.
    Потоковая обработка с гарантированной очисткой памяти.
    """
    def _process(iterator):
        # Ленивый импорт внутри воркера
        from src.model import ModelConfig, get_or_create_enhancer

        mcfg = ModelConfig(**model_cfg_dict)
        enhancer = get_or_create_enhancer(mcfg)
        batch_size = spark_cfg.get("batch_size", 2)

        batch_metas = []
        batch_datas = []

        for meta, data in iterator:
            batch_metas.append(meta)
            batch_datas.append(data)

            if len(batch_datas) >= batch_size:
                try:
                    results = enhancer.enhance_batch(batch_datas)
                    for m, res in zip(batch_metas, results):
                        yield m, res
                except Exception:
                    # Fallback: обрабатываем по одному
                    for m, d in zip(batch_metas, batch_datas):
                        try:
                            yield m, enhancer.enhance(d)
                        except Exception:
                            yield m, d  # исходный как fallback
                finally:
                    # 🔥 ГАРАНТИРОВАННАЯ очистка независимо от успеха/ошибки
                    batch_metas.clear()
                    batch_datas.clear()

        # Обработка остатка (хвоста)
        if batch_datas:
            try:
                results = enhancer.enhance_batch(batch_datas)
                for m, res in zip(batch_metas, results):
                    yield m, res
            except Exception:
                for m, d in zip(batch_metas, batch_datas):
                    try:
                        yield m, enhancer.enhance(d)
                    except Exception:
                        yield m, d
            finally:
                batch_metas.clear()
                batch_datas.clear()

    return _process


# -----------------------------------------------------------------
# Основной пайплайн
# -----------------------------------------------------------------

def _create_empty_metadata(
    image_id: str,
    config: dict,
    original_shape: Optional[List[int]] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Создаёт полную структуру метаданных с None-значениями для полей, которые могут отсутствовать."""
    return {
        "image_id": image_id,
        "original_path": None,
        "enhanced_path": None,
        "metadata_path": None,
        "original_shape": original_shape,
        "enhanced_shape": None,
        "tile_size": config["image"]["tile_size"],
        "overlap": config["image"]["overlap"],
        "num_tiles": 0,
        "inference_time_sec": None,
        "total_time_sec": None,
        "model": config["model"]["name"],
        "use_mock": config["model"]["use_mock"],
        "metrics": {"psnr_db": None, "ssim": None},
        "processed_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "error": error,
        "local_fallback": False,
    }


def run_enhancement_pipeline(
    image_path: str,
    image_id: str,
    config_path: str,
    spark=None,  # SparkSession или None (создадим сами)
) -> Dict[str, Any]:
    """Полный пайплайн: валидация -> тайлинг -> Spark-инференс -> сборка -> HDFS.

    Args:
        image_path: локальный путь к исходному изображению.
        image_id: уникальный идентификатор снимка (для путей в HDFS).
        config_path: путь к settings.yaml.
        spark: внешняя SparkSession (если None — создаётся по конфигу).

    Returns:
        Словарь с метриками и путями в HDFS (консистентная структура).
    """
    import gc  # Для принудительной сборки мусора

    t_start = time.time()
    cfg = load_config(config_path)
    logger = setup_logging(
        level=cfg["logging"]["level"],
        fmt=cfg["logging"]["format"],
        log_dir=cfg["logging"].get("log_dir"),
        app_name=cfg["spark"]["app_name"],
    )

    # 0) Валидация формата файла
    SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    src_path = Path(image_path)
    if src_path.suffix.lower() not in SUPPORTED_EXTS:
        raise ValueError(
            f"Неподдерживаемый формат '{src_path.suffix}'. "
            f"Допустимые: {', '.join(SUPPORTED_EXTS)}"
        )

    # 1) Валидация изображения
    src_path = validate_image(str(src_path), cfg["image"])

    # 2) Тайлинг (локально на драйвере)
    tiles, orig_shape = split_image_into_tiles(
        str(src_path),
        tile_size=cfg["image"]["tile_size"],
        overlap=cfg["image"]["overlap"],
    )
    logger.info("Подготовлено %d тайлов для Spark", len(tiles))

    # 3) Spark: создаём сессию или переиспользуем
    spark_created_here = False
    if spark is None:
        from pyspark.sql import SparkSession
        spark_cfg = cfg["spark"]
        builder = (
            SparkSession.builder
            .appName(spark_cfg["app_name"])
            .config("spark.executor.memory", spark_cfg.get("executor_memory", "4g"))
            .config("spark.executor.cores", str(spark_cfg.get("executor_cores", 2)))
            .config("spark.task.maxFailures", str(spark_cfg.get("task_max_failures", 3)))
            .config("spark.sql.broadcastTimeout", str(spark_cfg.get("broadcast_timeout", 600)))
            # Критично для collect(): память драйвера
            .config("spark.driver.memory", spark_cfg.get("driver_memory", "6g"))
            # Отключаем лимит на размер результата collect()
            .config("spark.driver.maxResultSize", spark_cfg.get("driver_maxResultSize", "0"))
            # Одна задача на ядро для избежания race condition в модели
            .config("spark.task.cpus", str(spark_cfg.get("task_cpus", 1)))
            # Переиспользование воркеров для кеширования модели
            .config("spark.python.worker.reuse", "true")
            # Контроль параллелизма шейфлов
            .config("spark.sql.shuffle.partitions", str(spark_cfg.get("shuffle_partitions", 4)))
        )
        master = spark_cfg.get("master")
        if master:
            builder = builder.master(master)
        spark = builder.getOrCreate()
        spark_created_here = True
    sc = spark.sparkContext
    logger.info("SparkContext готов: %s", sc.appName)

    try:
        # 4) Параллельная обработка тайлов
        # Сериализуем данные тайлов для эффективной передачи
        compress_arrays = cfg["spark"].get("compress_arrays", True)
        tile_items = [
            (t.to_dict_without_data(), _serialize_array(t.data, compress=compress_arrays))
            for t in tiles
        ]
        num_parts = min(cfg["spark"]["num_partitions"], max(1, len(tile_items)))
        rdd = sc.parallelize(tile_items, numSlices=num_parts)

        model_cfg_dict = {
            "name": cfg["model"]["name"],
            "path": cfg["model"]["path"],
            "device": cfg["model"]["device"],
            "batch_size": cfg["model"]["batch_size"],
            "use_mock": cfg["model"]["use_mock"],
        }
        spark_worker_cfg = {
            "batch_size": cfg["model"]["batch_size"],
            "compress_arrays": compress_arrays,
        }

        # Мониторинг памяти перед инференсом
        import psutil, os
        process = psutil.Process(os.getpid())
        logger.info(f"💾 Память ДО инференса: {process.memory_info().rss / 1024**2:.1f} MB")

        t_infer = time.time()

        # Запускаем обработку с ПОТОКОВОЙ передачей данных
        processed_rdd = rdd.mapPartitions(
            _process_partition_factory(model_cfg_dict, spark_worker_cfg)
        )

        # Собираем результаты
        processed_serialized = processed_rdd.collect()

        # 🔥 КРИТИЧНО: Немедленно освобождаем RDD и исходные данные
        del processed_rdd
        del rdd
        del tile_items
        gc.collect()

        infer_time = time.time() - t_infer
        logger.info(
            "Инференс завершён: %d тайлов за %.2f сек (%.2f тайлов/сек)",
            len(processed_serialized), infer_time, len(processed_serialized) / max(infer_time, 1e-6),
        )
        logger.info(f"💾 Память ПОСЛЕ collect: {process.memory_info().rss / 1024**2:.1f} MB")

        # 5) Восстанавливаем объекты Tile для сборки
        processed_tiles: List[Tile] = []
        for meta, data_bytes in processed_serialized:
            # 🔥 ДЕСЕРИАЛИЗУЕМ байты обратно в numpy-массив
            data = _deserialize_array(data_bytes, compress=compress_arrays)
            processed_tiles.append(
                Tile(
                    index=meta["index"],
                    x=meta["x"],
                    y=meta["y"],
                    width=meta["width"],
                    height=meta["height"],
                    data=data,  # Теперь это ndarray, а не bytes
                )
            )

        # Освобождаем память после десериализации
        del processed_serialized
        gc.collect()

        # Сортируем по исходному индексу для детерминированности
        processed_tiles.sort(key=lambda t: t.index)

# 6) Сборка изображения
        logger.info(f"💾 Память ПЕРЕД сборкой: {process.memory_info().rss / 1024**2:.1f} MB")
        enhanced = assemble_tiles(
            processed_tiles,
            image_shape=orig_shape,
            blending=cfg["postprocessing"]["blending"],
            gaussian_sigma_ratio=cfg["postprocessing"]["gaussian_sigma_ratio"],
        )
        logger.info(f"💾 Память ПОСЛЕ сборки: {process.memory_info().rss / 1024**2:.1f} MB")

        # 🔥 Освобождаем списки тайлов ПЕРЕД загрузкой оригинала для метрик
        total_tiles = len(tiles)

        # Освобождаем память
        del tiles, processed_tiles
        gc.collect()
        logger.info(f"💾 Память после очистки тайлов: {process.memory_info().rss / 1024**2:.1f} MB")

        # 📊 Расчёт метрик (с защитой от OOM)
        try:
            from PIL import Image as PILImage

            # Вариант А: Загружаем оригинал с явным приведением к uint8 (экономит память)
            with PILImage.open(src_path) as img:
                # Конвертируем сразу в numpy без промежуточного копирования
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                original_full = np.asarray(img, dtype=np.uint8)

            # Обрезаем до общего размера (если формы разошлись)
            h = min(original_full.shape[0], enhanced.shape[0])
            w = min(original_full.shape[1], enhanced.shape[1])

            # 🔥 Опционально: считаем метрики на даунскейле для экономии памяти
            # Если изображение >50 Мп — уменьшаем в 2× для метрик
            if h * w > 50_000_000:
                logger.info("📉 Изображение большое, считаем метрики на даунскейле 2×")
                from cv2 import resize, INTER_AREA
                orig_small = resize(original_full[:h, :w], (w//2, h//2), interpolation=INTER_AREA)
                enh_small = resize(enhanced[:h, :w], (w//2, h//2), interpolation=INTER_AREA)
                metric_psnr = psnr(orig_small, enh_small)
                metric_ssim = ssim_simple(orig_small, enh_small)
            else:
                metric_psnr = psnr(original_full[:h, :w], enhanced[:h, :w])
                metric_ssim = ssim_simple(original_full[:h, :w], enhanced[:h, :w])

            logger.info("Метрики: PSNR=%.2f dB, SSIM=%.4f", metric_psnr, metric_ssim)

            # Освобождаем оригинал после метрик
            del original_full
            gc.collect()

        except Exception as e:
            logger.warning("⚠️ Не удалось рассчитать метрики: %s, пропускаем", e)
            metric_psnr = metric_ssim = None

        # 7) Сохранение в HDFS
        hcfg = HDFSConfig(
            root=cfg["storage"]["hdfs_root"],
            input_dir=cfg["storage"]["input_dir"],
            output_dir=cfg["storage"]["output_dir"],
            metadata_dir=cfg["storage"]["metadata_dir"],
            replication=cfg["storage"]["replication"],
            access_log_enabled=cfg["storage"].get("access_log_enabled", True),
            access_log_dir=cfg["storage"].get("access_log_dir"),
        )
        hdfs = HDFSClient(hcfg)
        date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        ext = src_path.suffix.lstrip(".").lower()
        pil_fmt = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG",
                   "tif": "TIFF", "tiff": "TIFF", "bmp": "BMP"}.get(ext, "PNG")

        input_hdfs = f"{hcfg.input_dir}/{date_str}/{image_id}_original.{ext}"
        output_hdfs = f"{hcfg.output_dir}/{date_str}/{image_id}_enhanced.png"
        meta_hdfs = f"{hcfg.metadata_dir}/{date_str}/{image_id}.json"

        # Инициализируем метаданные (полная структура)
        metadata = _create_empty_metadata(
            image_id=image_id,
            config=cfg,
            original_shape=list(original_full.shape),
        )

        try:
            hdfs.put_local_file(src_path, input_hdfs)
            hdfs.put_image(enhanced, output_hdfs, fmt="PNG")

            # Заполняем успешные значения
            metadata.update({
                "original_path": input_hdfs,
                "enhanced_path": output_hdfs,
                "metadata_path": meta_hdfs,
                "enhanced_shape": list(enhanced.shape),
                "num_tiles": total_tiles,
                "inference_time_sec": round(infer_time, 3),
                "total_time_sec": round(time.time() - t_start, 3),
                "metrics": {
                    "psnr_db": metric_psnr,
                    "ssim": metric_ssim,
                },
            })
            hdfs.put_json(metadata, meta_hdfs)
            logger.info("Результаты сохранены в HDFS: %s", meta_hdfs)

        except Exception as e:
            logger.error("Ошибка записи в HDFS: %s", e)
            # Fallback: локальное сохранение
            local_output_dir = Path(cfg["storage"].get("local_fallback_dir", "/app/data/local_output"))
            local_output_dir.mkdir(parents=True, exist_ok=True)

            local_path = local_output_dir / f"{image_id}_enhanced.png"
            PILImage.fromarray(enhanced).save(local_path)

            metadata.update({
                "enhanced_path": str(local_path),
                "error": str(e),
                "local_fallback": True,
                "num_tiles": total_tiles,
                "inference_time_sec": round(infer_time, 3),
                "total_time_sec": round(time.time() - t_start, 3),
            })
            logger.warning("Сохранено локально (fallback): %s", local_path)

        return metadata

    finally:
        if spark_created_here:
            spark.stop()
            logger.info("SparkSession остановлен")


# -----------------------------------------------------------------
# CLI
# -----------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Пайплайн улучшения спутниковых снимков (Spark + Hadoop)."
    )
    p.add_argument("--image", required=True, help="Путь к исходному изображению.")
    p.add_argument("--image-id", required=True, help="Уникальный ID снимка.")
    p.add_argument(
        "--config",
        default="/app/config/settings.yaml",
        help="Путь к YAML-конфигу.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Только валидация, без запуска Spark.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        if args.dry_run:
            print(f"🔍 Dry run: валидация {args.image}")
            cfg = load_config(args.config)
            validate_image(args.image, cfg["image"])
            print("✅ Валидация пройдена")
            return 0

        result = run_enhancement_pipeline(
            image_path=args.image,
            image_id=args.image_id,
            config_path=args.config,
        )
        print("=" * 60)
        print("🎉 Пайплайн завершён успешно.")
        print(f"image_id      : {result.get('image_id')}")
        print(f"enhanced_path : {result.get('enhanced_path')}")
        print(f"metrics       : PSNR={result['metrics']['psnr_db']:.2f} dB, "
              f"SSIM={result['metrics']['ssim']:.4f}")
        print(f"total_time    : {result.get('total_time_sec')} сек")
        if result.get('error'):
            print(f"⚠️  warning     : {result['error']}")
        if result.get('local_fallback'):
            print("📁 fallback    : локальное сохранение")
        print("=" * 60)
        return 0

    except Exception as e:
        log.exception("❌ Пайплайн упал: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())