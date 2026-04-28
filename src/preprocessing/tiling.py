"""Разбиение изображения на тайлы с перекрытием.

Формирует сетку тайлов фиксированного размера с заданным overlap так,
чтобы при последующей сборке можно было точно восстановить исходное
изображение. Последний ряд/столбец тайлов при необходимости «прижимается»
к правому/нижнему краю, чтобы избежать паддинга.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


@dataclass
class Tile:
    """Один тайл с метаданными о его позиции в исходном изображении.

    Attributes:
        index: порядковый номер тайла в изображении.
        x: координата левого верхнего угла по X (столбец).
        y: координата левого верхнего угла по Y (строка).
        width: ширина тайла в пикселях.
        height: высота тайла в пикселях.
        data: numpy-массив формы (H, W, C) с dtype=uint8.
    """
    index: int
    x: int
    y: int
    width: int
    height: int
    data: np.ndarray

    def to_dict_without_data(self) -> dict:
        """Возвращает метаданные тайла без payload (для логов/метрик)."""
        d = asdict(self)
        d.pop("data", None)
        return d


def validate_image(path: str | Path, cfg_image: dict) -> Path:
    """Проверяет существование и базовые свойства изображения.

    Args:
        path: путь к файлу.
        cfg_image: секция image конфига (formats, min_size, max_size).

    Returns:
        Нормализованный Path.

    Raises:
        FileNotFoundError: если файл отсутствует.
        ValueError: если формат не поддерживается или размер вне допустимого.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Изображение не найдено: {p}")
    if p.suffix.lower() not in [ext.lower() for ext in cfg_image["formats"]]:
        raise ValueError(
            f"Формат {p.suffix} не поддерживается. "
            f"Ожидается один из: {cfg_image['formats']}"
        )
    with Image.open(p) as img:
        w, h = img.size
    min_s, max_s = cfg_image["min_size"], cfg_image["max_size"]
    if not (min_s <= min(w, h) and max(w, h) <= max_s):
        raise ValueError(
            f"Размер {w}x{h} вне допустимого диапазона [{min_s}, {max_s}]"
        )
    log.info("Валидация пройдена: %s (%dx%d)", p.name, w, h)
    return p


def _compute_starts(total: int, tile: int, overlap: int) -> List[int]:
    """Вычисляет список стартовых координат тайлов вдоль одной оси.

    Последний тайл всегда прижат к краю (total - tile), даже если
    перекрытие с предыдущим оказывается больше заданного.

    Args:
        total: общий размер по оси (ширина или высота изображения).
        tile: размер тайла.
        overlap: желаемое перекрытие.

    Returns:
        Список стартовых координат.
    """
    if tile >= total:
        return [0]
    step = tile - overlap
    if step <= 0:
        raise ValueError(f"overlap ({overlap}) должен быть меньше tile_size ({tile})")
    starts = list(range(0, total - tile + 1, step))
    # Гарантируем, что последний тайл покрывает правый/нижний край
    if starts[-1] + tile < total:
        starts.append(total - tile)
    return starts


def split_image_into_tiles(
    image_path: str | Path,
    tile_size: int,
    overlap: int,
) -> Tuple[List[Tile], Tuple[int, int, int]]:
    """Разбивает изображение на тайлы.

    Args:
        image_path: путь к изображению.
        tile_size: размер тайла в пикселях.
        overlap: перекрытие между соседними тайлами.

    Returns:
        Кортеж (список тайлов, (H, W, C)) — тайлы и форма исходного
        изображения для последующей сборки.
    """
    p = Path(image_path)
    with Image.open(p) as img:
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")
        elif img.mode == "RGBA":
            img = img.convert("RGB")
        elif img.mode == "L":
            img = img.convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    h, w, c = arr.shape
    log.info("Изображение загружено: shape=%s dtype=%s", arr.shape, arr.dtype)

    # Если изображение меньше тайла - возвращаем его как единственный тайл
    effective_tile = min(tile_size, h, w)
    xs = _compute_starts(w, effective_tile, overlap)
    ys = _compute_starts(h, effective_tile, overlap)

    tiles: List[Tile] = []
    idx = 0
    for y in ys:
        for x in xs:
            patch = arr[y : y + effective_tile, x : x + effective_tile, :].copy()
            tiles.append(
                Tile(
                    index=idx,
                    x=x,
                    y=y,
                    width=patch.shape[1],
                    height=patch.shape[0],
                    data=patch,
                )
            )
            idx += 1
    log.info(
        "Сформировано %d тайлов (grid=%dx%d, tile=%d, overlap=%d)",
        len(tiles), len(ys), len(xs), effective_tile, overlap,
    )
    return tiles, (h, w, c)
