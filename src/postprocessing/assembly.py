"""Сборка обработанных тайлов обратно в полное изображение.

Ключевой нюанс — перекрытие между тайлами. На стыках один и тот же
пиксель оказывается обработан сразу несколькими тайлами. Если
просто перезаписывать пиксели последним тайлом, на границах возникает
видимый «шов». Поэтому мы используем взвешенное усреднение: каждому
пикселю тайла сопоставляется вес (1 в центре, 0 к краю), а финальное
значение пикселя изображения = sum(weight_i * pixel_i) / sum(weight_i).

Для маски весов применяется гауссова функция, что даёт самый плавный
переход. Дополнительно поддерживается режим "average" (равные веса)
и "none" (последний тайл перезаписывает).
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

from src.preprocessing.tiling import Tile

log = logging.getLogger(__name__)


def _make_weight_mask(height: int, width: int, mode: str, sigma_ratio: float) -> np.ndarray:
    """Формирует маску весов для блендинга.

    Args:
        height: высота тайла.
        width: ширина тайла.
        mode: тип маски ("gaussian" | "average" | "none").
        sigma_ratio: доля от меньшей стороны для sigma гауссианы.

    Returns:
        Массив float32 формы (H, W) со значениями в [0, 1].
    """
    if mode == "none":
        return np.ones((height, width), dtype=np.float32)

    if mode == "average":
        # Константный вес, но ненулевой — чтобы знаменатель при усреднении
        # не обнулился на пикселях, покрытых лишь одним тайлом.
        return np.ones((height, width), dtype=np.float32)

    # Гауссова маска: максимум в центре, плавный спад к краям.
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    sigma = sigma_ratio * min(height, width)
    # eps, чтобы вес по краям не был ровно 0 (иначе останутся пустые пиксели
    # там, где тайлы вообще не перекрываются — например, по краям изображения).
    mask = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    mask = mask.astype(np.float32)
    mask = np.maximum(mask, 1e-3)
    return mask


def assemble_tiles(
    tiles: List[Tile],
    image_shape: Tuple[int, int, int],
    blending: str = "gaussian",
    gaussian_sigma_ratio: float = 0.5,
) -> np.ndarray:
    """Собирает тайлы в полное изображение с учётом overlap.

    Args:
        tiles: список обработанных тайлов (с оригинальными x/y координатами).
        image_shape: форма исходного изображения (H, W, C).
        blending: режим блендинга ("gaussian" | "average" | "none").
        gaussian_sigma_ratio: sigma гауссианы как доля от размера тайла.

    Returns:
        Собранное изображение shape=(H, W, C) dtype=uint8.
    """
    h, w, c = image_shape
    # Накапливаем взвешенную сумму и общий вес в float32, затем делим.
    accum = np.zeros((h, w, c), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)

    for tile in tiles:
        th, tw = tile.height, tile.width
        if tile.data.shape[:2] != (th, tw):
            raise ValueError(
                f"Tile #{tile.index}: shape {tile.data.shape[:2]} "
                f"не совпадает с ожидаемой ({th}, {tw})"
            )
        mask = _make_weight_mask(th, tw, blending, gaussian_sigma_ratio)
        y0, y1 = tile.y, tile.y + th
        x0, x1 = tile.x, tile.x + tw
        # Broadcast (H, W, 1) * (H, W, C) по каналам
        accum[y0:y1, x0:x1, :] += tile.data.astype(np.float32) * mask[:, :, None]
        weight_sum[y0:y1, x0:x1] += mask

    # Защита от деления на 0 (на практике не должно случаться при валидном tiling)
    weight_sum = np.maximum(weight_sum, 1e-8)
    result = accum / weight_sum[:, :, None]
    result = np.clip(result, 0, 255).astype(np.uint8)

    log.info(
        "Собрано изображение %dx%d из %d тайлов (blending=%s)",
        w, h, len(tiles), blending,
    )
    return result
