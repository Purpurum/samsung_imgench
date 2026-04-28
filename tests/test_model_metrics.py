"""Тесты mock-модели и метрик качества."""
import numpy as np
import pytest

from src.model import ModelConfig, get_or_create_enhancer
from src.postprocessing import psnr, ssim_simple


@pytest.fixture
def mock_cfg():
    return ModelConfig(
        name="mock", path="", device="cpu",
        batch_size=4, use_mock=True,
    )


def test_mock_enhancer_preserves_shape(mock_cfg):
    enhancer = get_or_create_enhancer(mock_cfg)
    tile = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    out = enhancer.enhance(tile)
    assert out.shape == tile.shape
    assert out.dtype == np.uint8


def test_mock_enhancer_batch_consistency(mock_cfg):
    enhancer = get_or_create_enhancer(mock_cfg)
    tiles = [np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(5)]
    batch_out = enhancer.enhance_batch(tiles)
    single_out = [enhancer.enhance(t) for t in tiles]
    assert len(batch_out) == len(single_out) == 5
    for b, s in zip(batch_out, single_out):
        # В mock-режиме batch = последовательный вызов, должны совпадать
        assert np.array_equal(b, s)


def test_mock_enhancer_changes_image(mock_cfg):
    """Убеждаемся, что mock-алгоритм реально что-то делает."""
    enhancer = get_or_create_enhancer(mock_cfg)
    tile = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    out = enhancer.enhance(tile)
    assert not np.array_equal(tile, out), "Mock не должен возвращать identity"


def test_psnr_identical_is_infinity():
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    assert psnr(img, img) == float("inf")


def test_psnr_different_is_finite():
    # Умеренная разница - PSNR конечный и в разумных пределах
    a = np.full((64, 64, 3), 100, dtype=np.uint8)
    b = np.full((64, 64, 3), 120, dtype=np.uint8)  # MSE = 400
    val = psnr(a, b)
    # 10*log10(255^2/400) ≈ 22.1 dB
    assert 15 < val < 30


def test_psnr_max_difference_is_zero_db():
    # Граничный случай: MSE = 255^2, PSNR = 0 dB
    a = np.zeros((64, 64, 3), dtype=np.uint8)
    b = np.full((64, 64, 3), 255, dtype=np.uint8)
    assert psnr(a, b) == pytest.approx(0.0, abs=1e-6)


def test_ssim_identical_is_one():
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    assert ssim_simple(img, img) == pytest.approx(1.0, abs=1e-6)


def test_ssim_shape_mismatch_raises():
    with pytest.raises(ValueError):
        ssim_simple(np.zeros((10, 10, 3)), np.zeros((20, 20, 3)))
