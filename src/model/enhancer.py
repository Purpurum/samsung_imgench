"""Обёртка над моделью улучшения качества снимков.

Поддерживает два режима:

1. `use_mock: true` — применяет лёгкий CPU-алгоритм (unsharp mask + denoise),
   чтобы пайплайн был полностью воспроизводим на учебной машине без GPU
   и без скачивания весов (~сотни МБ). На вход/выход — те же размеры, что и
   у реальной модели, поэтому остальные модули не различают режимы.

2. `use_mock: false` — загружает настоящую модель из
   `satlaspretrain_models` (https://github.com/allenai/satlaspretrain_models).
   Требует установленного пакета и скачанных весов. Реализация специально
   сделана lazy: импорт torch и satlaspretrain_models происходит только
   в момент первого обращения, чтобы Spark-драйвер не тянул тяжёлые
   зависимости без необходимости.

На каждый Spark-воркер модель грузится один раз — через lazy-инициализацию
внутри `mapPartitions` (см. src/main.py).
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter

log = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Параметры инициализации модели."""
    name: str
    path: str
    device: str
    batch_size: int
    use_mock: bool


class ImageEnhancer:
    """Универсальный интерфейс для инференса одного тайла или батча.

    Инстанс создаётся один раз на процесс-воркер. Под капотом либо
    применяется мок-алгоритм, либо загружается настоящая модель Satlas.
    """

    # Блокировка ленивой инициализации (на случай многопоточности)
    _lock = threading.Lock()

    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self._model = None  # реальная torch-модель, грузится лениво
        self._torch = None

    # ----------------------- public API -----------------------

    def enhance(self, tile: np.ndarray) -> np.ndarray:
        """Применяет модель к одному тайлу (H, W, C) uint8 и возвращает результат.

        Args:
            tile: исходный тайл shape=(H, W, 3) dtype=uint8.

        Returns:
            Улучшенный тайл той же формы и dtype.
        """
        if tile.dtype != np.uint8:
            raise ValueError(f"Ожидается uint8, получен {tile.dtype}")
        if self.cfg.use_mock:
            return self._mock_enhance(tile)
        return self._torch_enhance(tile)

    def enhance_batch(self, tiles: list[np.ndarray]) -> list[np.ndarray]:
        """Батч-инференс: эффективнее, когда доступен GPU.

        В mock-режиме сводится к последовательным вызовам.
        """
        if self.cfg.use_mock:
            return [self._mock_enhance(t) for t in tiles]
        return self._torch_enhance_batch(tiles)

    # ----------------------- mock-алгоритм -----------------------

    @staticmethod
    def _mock_enhance(tile: np.ndarray) -> np.ndarray:
        """Легковесное улучшение: unsharp-mask + лёгкий denoise.

        Даёт заметный визуальный эффект (повышение резкости + сглаживание
        шума) и позволяет проверить PSNR/SSIM — достаточно для учебной
        демонстрации работы пайплайна.
        """
        img = Image.fromarray(tile, mode="RGB")
        # Snoothing (median) ослабляет соль-перец шум
        denoised = img.filter(ImageFilter.MedianFilter(size=3))
        # Unsharp mask: усиление высоких частот
        sharpened = denoised.filter(
            ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)
        )
        return np.asarray(sharpened, dtype=np.uint8)

    # ----------------------- реальная модель -----------------------

    def _ensure_loaded(self) -> None:
        """Ленивая загрузка torch-модели (один раз на процесс)."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            log.info("Инициализация модели Satlas (device=%s)...", self.cfg.device)
            import torch  # локальный импорт, чтобы не грузить на драйвере
            try:
                import satlaspretrain_models  # noqa: F401
            except ImportError as e:
                raise RuntimeError(
                    "Пакет satlaspretrain_models не установлен. "
                    "Установите: pip install satlaspretrain_models, "
                    "или включите use_mock: true в конфиге."
                ) from e

            from satlaspretrain_models import Weights

            self._torch = torch
            weights_manager = Weights()
            # Имя модели из конфига резолвится в каталоге satlaspretrain
            self._model = weights_manager.get_pretrained_model(
                model_identifier=self.cfg.name,
                fpn=True,
            )
            self._model.eval()
            self._model.to(self.cfg.device)
            log.info("Модель %s загружена на %s", self.cfg.name, self.cfg.device)

    def _torch_enhance(self, tile: np.ndarray) -> np.ndarray:
        """Инференс одного тайла через torch-модель."""
        self._ensure_loaded()
        import torch
        # (H, W, C) -> (1, C, H, W), float32 [0,1]
        t = torch.from_numpy(tile).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        t = t.to(self.cfg.device)
        with torch.no_grad():
            out = self._model(t)
        # Обратно к uint8 [0,255]
        out = out.squeeze(0).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
        return (out * 255).astype(np.uint8)

    def _torch_enhance_batch(self, tiles: list[np.ndarray]) -> list[np.ndarray]:
        """Батч-инференс через torch-модель."""
        self._ensure_loaded()
        import torch
        bs = self.cfg.batch_size
        results: list[np.ndarray] = []
        for i in range(0, len(tiles), bs):
            chunk = tiles[i : i + bs]
            batch = torch.stack(
                [torch.from_numpy(t).permute(2, 0, 1).float() / 255.0 for t in chunk]
            ).to(self.cfg.device)
            with torch.no_grad():
                out = self._model(batch)
            out = out.clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
            results.extend([(x * 255).astype(np.uint8) for x in out])
        return results


# Кэш инстансов модели на процесс-воркер.
# Ключ = tuple из полей cfg, чтобы переиспользовать при одинаковых настройках.
_enhancer_cache: dict[tuple, ImageEnhancer] = {}


def get_or_create_enhancer(cfg: ModelConfig) -> ImageEnhancer:
    """Возвращает единственный на процесс инстанс ImageEnhancer.

    Используется в Spark-воркерах внутри mapPartitions, чтобы избежать
    перезагрузки модели на каждый тайл.
    """
    key = (cfg.name, cfg.path, cfg.device, cfg.batch_size, cfg.use_mock)
    inst = _enhancer_cache.get(key)
    if inst is None:
        inst = ImageEnhancer(cfg)
        _enhancer_cache[key] = inst
        log.info("Создан новый ImageEnhancer (use_mock=%s)", cfg.use_mock)
    return inst
