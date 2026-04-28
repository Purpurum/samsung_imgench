#!/usr/bin/env python3
"""Генерирует синтетический «спутниковый» снимок для тестирования пайплайна.

Создаёт изображение с имитацией географических структур (поля, реки,
облака, шум), чтобы mock-модель могла продемонстрировать эффект
улучшения (шарпен + денойз).

Использование:
    python scripts/generate_sample.py --output data/samples/sat_001.png --size 2048
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def generate_satellite_like(size: int, seed: int = 42) -> np.ndarray:
    """Создаёт RGB-массив size×size, имитирующий спутниковый снимок."""
    rng = np.random.default_rng(seed)

    # Базовая поверхность - «поля» разного цвета, крупные блоки
    block = size // 16
    base = rng.integers(60, 180, size=(size // block, size // block, 3), dtype=np.uint8)
    base = np.kron(base, np.ones((block, block, 1), dtype=np.uint8))
    base = base[:size, :size]

    img = Image.fromarray(base)
    # Сглаживаем границы блоков, чтобы это выглядело менее «пиксельно»
    img = img.filter(ImageFilter.GaussianBlur(radius=8))
    arr = np.asarray(img, dtype=np.int32)

    # «Река» - извилистая тёмно-синяя полоса
    yy, xx = np.mgrid[0:size, 0:size]
    river = np.abs((yy - size // 2) - 40 * np.sin(xx / (size / 8))) < 20
    arr[river] = np.array([30, 60, 120])

    # «Облака» - крупные светлые пятна
    cloud_mask = np.zeros((size, size), dtype=np.float32)
    for _ in range(5):
        cy, cx = rng.integers(0, size, size=2)
        r = rng.integers(size // 15, size // 8)
        dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
        cloud_mask = np.maximum(cloud_mask, np.exp(-dist2 / (2 * r ** 2)))
    arr = arr + (cloud_mask[..., None] * 100).astype(np.int32)

    # Шум (имитация низкого качества исходника - mock-модель его снизит)
    noise = rng.normal(0, 15, size=arr.shape)
    arr = arr + noise

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, help="Путь для сохранения PNG.")
    p.add_argument("--size", type=int, default=1024, help="Сторона изображения в пикселях.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    arr = generate_satellite_like(args.size, args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(out)
    print(f"Сохранено: {out} ({args.size}x{args.size})")


if __name__ == "__main__":
    main()
