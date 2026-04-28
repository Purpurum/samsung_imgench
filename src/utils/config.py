"""Утилиты для конфигурации и логирования."""
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Загружает YAML-конфигурацию из файла.

    Args:
        config_path: путь к файлу settings.yaml

    Returns:
        Словарь с параметрами конфигурации.

    Raises:
        FileNotFoundError: если файл отсутствует.
        yaml.YAMLError: если YAML некорректен.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл конфигурации не найден: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Конфиг должен быть YAML-объектом (dict)")
    return cfg
    
def get_spark_config(config: Dict[str, Any]) -> Dict[str, str]:
    spark_cfg = config.get("spark", {})
    return {
        "spark.app.name": spark_cfg.get("app_name", "SatlasEnhancement"),
        "spark.master": spark_cfg.get("master", "local[1]"),
        "spark.driver.memory": spark_cfg.get("driver_memory", "4g"),
        "spark.driver.maxResultSize": spark_cfg.get("driver_maxResultSize", "0"),
        "spark.executor.memory": spark_cfg.get("executor_memory", "2g"),
        "spark.executor.cores": str(spark_cfg.get("executor_cores", 2)),
        "spark.task.maxFailures": str(spark_cfg.get("task_max_failures", 3)),
        "spark.sql.broadcastTimeout": str(spark_cfg.get("broadcast_timeout", 600)),
        "spark.task.cpus": str(spark_cfg.get("task_cpus", 1)),
        "spark.python.worker.reuse": str(spark_cfg.get("python_worker_reuse", "true")).lower(),
        "spark.sql.shuffle.partitions": str(spark_cfg.get("spark_sql_shuffle_partitions", 4)),
        "spark.default.parallelism": str(spark_cfg.get("spark_default_parallelism", 4)),
    }


def setup_logging(
    level: str = "INFO",
    fmt: str | None = None,
    log_dir: str | Path | None = None,
    app_name: str = "satlas",
) -> logging.Logger:
    """Настраивает корневой логгер: вывод в stdout + опционально в файл.

    Args:
        level: уровень логирования (DEBUG/INFO/WARNING/ERROR).
        fmt: строка формата сообщений.
        log_dir: директория для файла лога (если None - только stdout).
        app_name: имя приложения (используется в имени файла и логгера).

    Returns:
        Настроенный логгер.
    """
    fmt = fmt or "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    formatter = logging.Formatter(fmt)

    root = logging.getLogger()
    root.setLevel(level.upper())
    # Очищаем старые хэндлеры (важно при повторных запусках в Jupyter)
    for h in list(root.handlers):
        root.removeHandler(h)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    root.addHandler(sh)

    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path / f"{app_name}.log", encoding="utf-8")
        fh.setFormatter(formatter)
        root.addHandler(fh)

    # Приглушаем шумные сторонние логи
    logging.getLogger("py4j").setLevel(logging.WARNING)
    logging.getLogger("pyspark").setLevel(logging.WARNING)

    return logging.getLogger(app_name)
