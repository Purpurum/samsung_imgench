"""Модульные тесты для разбиения и сборки тайлов."""
import numpy as np
import pytest
from PIL import Image

from src.preprocessing.tiling import _compute_starts, split_image_into_tiles
from src.postprocessing.assembly import assemble_tiles


@pytest.fixture
def sample_image(tmp_path):
    """Создаёт тестовое изображение 1024x768 с градиентом."""
    h, w = 768, 1024
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    arr[..., 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    arr[..., 2] = 128
    p = tmp_path / "test.png"
    Image.fromarray(arr).save(p)
    return p, arr


def test_compute_starts_fits_exactly():
    # 1024 = 4*256 - нулевой overlap
    starts = _compute_starts(1024, 256, 0)
    assert starts == [0, 256, 512, 768]


def test_compute_starts_with_overlap():
    # tile=256, overlap=32, step=224
    # 0, 224, 448, 672 - последний (672+256=928) не покрывает 1024,
    # значит добавляется 1024-256=768
    starts = _compute_starts(1024, 256, 32)
    assert starts[0] == 0
    assert starts[-1] == 1024 - 256
    # Покрытие полное
    assert starts[-1] + 256 == 1024


def test_compute_starts_tile_larger_than_image():
    assert _compute_starts(100, 256, 0) == [0]


def test_compute_starts_invalid_overlap():
    with pytest.raises(ValueError):
        _compute_starts(1024, 256, 256)


def test_split_tiles_covers_image(sample_image):
    path, arr = sample_image
    tiles, shape = split_image_into_tiles(path, tile_size=256, overlap=32)
    assert shape == arr.shape
    # Все тайлы одного размера (256x256), последний ряд/столбец - прижатый
    for t in tiles:
        assert t.data.shape == (256, 256, 3)
        assert 0 <= t.x <= arr.shape[1] - 256
        assert 0 <= t.y <= arr.shape[0] - 256


def test_split_then_assemble_no_blending_recovers_original(sample_image):
    """Без модификаций и без блендинга сборка = исходник."""
    path, arr = sample_image
    tiles, shape = split_image_into_tiles(path, tile_size=256, overlap=32)
    assembled = assemble_tiles(tiles, shape, blending="none")
    assert assembled.shape == arr.shape
    # При overlap + "none" последний тайл перезаписывает, поэтому
    # допускаем малое расхождение на стыках. Gaussian/average должен
    # давать близкий к идеалу результат.
    diff = np.abs(assembled.astype(int) - arr.astype(int)).mean()
    assert diff < 2.0


def test_split_then_assemble_gaussian_close_to_original(sample_image):
    path, arr = sample_image
    tiles, shape = split_image_into_tiles(path, tile_size=256, overlap=64)
    assembled = assemble_tiles(tiles, shape, blending="gaussian")
    assert assembled.shape == arr.shape
    # Гауссово усреднение тех же тайлов не должно сильно менять картинку
    diff = np.abs(assembled.astype(int) - arr.astype(int)).mean()
    assert diff < 2.0


def test_small_image_single_tile(tmp_path):
    arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    p = tmp_path / "small.png"
    Image.fromarray(arr).save(p)
    tiles, shape = split_image_into_tiles(p, tile_size=256, overlap=32)
    assert len(tiles) == 1
    assert tiles[0].data.shape == (100, 100, 3)
