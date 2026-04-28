"""Метрики качества улучшения изображений.

Реализовано без scikit-image, чтобы минимизировать зависимости:
— PSNR: пиковое отношение сигнал/шум.
— SSIM: структурное сходство, упрощённая реализация (без окна Гаусса,
  по всей картинке), достаточная для сравнительной оценки в учебном
  проекте. Для production-метрик рекомендуется использовать
  `skimage.metrics.structural_similarity`.
"""
from __future__ import annotations

import numpy as np


def psnr(original: np.ndarray, processed: np.ndarray, max_val: float = 255.0) -> float:
    """Peak Signal-to-Noise Ratio (дБ). Выше — лучше.

    Сравнивает исходное и обработанное изображение попиксельно.
    Для идентичных изображений возвращает +inf.
    """
    if original.shape != processed.shape:
        raise ValueError("Формы изображений должны совпадать")
    mse = np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * float(np.log10(max_val / np.sqrt(mse)))


def ssim_simple(original: np.ndarray, processed: np.ndarray) -> float:
    """Упрощённый SSIM по всей картинке (без скользящего окна).

    Значение в [-1, 1], обычно в [0, 1] для осмысленных изображений.
    1.0 — идентичны, ниже — тем сильнее различаются.
    """
    if original.shape != processed.shape:
        raise ValueError("Формы изображений должны совпадать")
    x = original.astype(np.float64)
    y = processed.astype(np.float64)
    mu_x, mu_y = x.mean(), y.mean()
    var_x, var_y = x.var(), y.var()
    cov = ((x - mu_x) * (y - mu_y)).mean()

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    num = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    return float(num / den)
