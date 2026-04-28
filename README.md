# Satlas Image Enhancement

Распределённая система обработки спутниковых снимков на базе **Apache Spark** и **Hadoop (HDFS)** с интеграцией нейросетевой модели улучшения качества изображений. Реализация учебного ТЗ с полным production-стеком: REST API, асинхронная обработка через Celery, Web UI на Streamlit и мониторинг задач.

---

## Содержание

1. [Что делает проект](#что-делает-проект)
2. [Архитектура](#архитектура)
3. [Структура репозитория](#структура-репозитория)
4. [Быстрый старт](#быстрый-старт)
5. [Как это работает внутри](#как-это-работает-внутри)
6. [Конфигурация](#конфигурация)
7. [WebUI кластера](#webui-кластера)
8. [Запуск пайплайна](#запуск-пайплайна)
9. [Тестирование](#тестирование)
10. [Переход на реальную модель Satlas](#переход-на-реальную-модель-satlas)
11. [Диагностика проблем](#диагностика-проблем)
12. [Соответствие ТЗ](#соответствие-тз)

---

## Что делает проект

Принимает спутниковый снимок (PNG / JPEG / TIFF), распределённо обрабатывает его через Spark-воркеры нейросетевой моделью улучшения качества, и сохраняет исходную и улучшенную версии в HDFS вместе с JSON-метаданными и метриками PSNR / SSIM.

Основные возможности:

- Разбиение снимка на тайлы с настраиваемым overlap.
- Распределённый инференс через `mapPartitions` — модель грузится **один раз на воркер-процесс** и переиспользуется.
- Gaussian-блендинг на стыках тайлов — без видимых швов.
- Два режима работы модели: `mock` (лёгкий CPU-алгоритм, готов «из коробки») и `production` (реальная модель `satlaspretrain_models`).
- Полная Docker-инфраструктура: NameNode, DataNode, Spark Master, два Spark Worker, Jupyter, FastAPI, Celery, Redis, Streamlit, Flower.
- REST API для интеграции с внешними системами.
- Асинхронная обработка задач через очередь Celery.
- Веб-интерфейс для пользователей (Streamlit).
- Мониторинг очереди задач через Flower.
- Модульные тесты (16 штук, покрывают тайлинг, сборку, метрики, mock-модель).

---

## Архитектура

```
                ┌──────────────────────┐
                │   Streamlit Frontend │   ← пользовательский UI (8501)
                │      (8501)          │
                └──────────┬───────────┘
                           │ HTTP API
                           ▼
                ┌──────────────────────┐
                │   FastAPI Processor  │   ← REST API (8000)
                │      (8000)          │
                └──────────┬───────────┘
                           │ задачи
                           ▼
                ┌──────────────────────┐
                │    Celery Worker     │   ← асинхронная обработка
                │   (Spark + HDFS)     │
                └──────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   Spark Master (7077)  │
              └──────┬────────┬────────┘
                     │        │
             ┌───────▼──┐  ┌──▼───────┐
             │ Worker 1 │  │ Worker 2 │   ← параллельный инференс тайлов
             │ 2 ядра   │  │ 2 ядра   │
             └──────────┘  └──────────┘
                     │        │
                     ▼        ▼
             ┌────────────────────────┐
             │   HDFS NameNode (9870) │ ← метаданные
             └───────────┬────────────┘
                         │
             ┌───────────▼────────────┐
             │    HDFS DataNode       │ ← блоки данных
             │  /data/input/          │
             │  /data/output/         │
             │  /data/metadata/       │
             └────────────────────────┘

                ┌──────────────────────┐
                │        Redis         │   ← брокер Celery
                │       (6379)         │
                └──────────────────────┘

                ┌──────────────────────┐
                │       Flower         │   ← мониторинг Celery (5555)
                │       (5555)         │
                └──────────────────────┘
```

**Альтернативный вход:** Jupyter Lab для интерактивной разработки и тестирования:

```
                ┌──────────────────────┐
                │      Jupyter Lab     │   ← интерактивный режим (8888)
                │  (Spark-драйвер)     │
                └──────────┬───────────┘
                           │  submit
                           ▼
              ┌────────────────────────┐
              │   Spark Master (7077)  │
              └──────┬────────┬────────┘
                     │        │
             ┌───────▼──┐  ┌──▼───────┐
             │ Worker 1 │  │ Worker 2 │
             └──────────┘  └──────────┘
```

**Поток данных для одного снимка:**

```
Image (N×M) ──► валидация ──► split на тайлы ──► RDD ──► mapPartitions
                                                              │
                    (на каждом воркере: модель в памяти,      │
                     батч-инференс тайлов партиции)           │
                                                              ▼
HDFS ◄── gaussian blending ◄── collect() ◄── обработанные тайлы
```

---

## Структура репозитория

```
satlas-project/
├── config/
│   └── settings.yaml                   # Все параметры пайплайна
├── data/
│   └── samples/                        # Тестовые снимки (генерируются скриптом)
├── docker/
│   ├── api-processor/
│   │   ├── Dockerfile                  # FastAPI + Celery образ
│   │   ├── Dockerfile.celery           # Celery worker образ
│   │   └── app/                        # FastAPI приложение
│   ├── frontend/
│   │   ├── Dockerfile                  # Streamlit образ
│   │   └── app.py                      # Streamlit UI
│   ├── Dockerfile.jupyter              # Образ для Jupyter + драйвера
│   ├── Dockerfile.spark                # Образ для Spark Master/Workers
│   └── docker-compose.yml              # Оркестрация всего стека
├── notebooks/
│   └── demo_pipeline.ipynb             # Интерактивное демо
├── scripts/
│   ├── init_hdfs.sh                    # Создание директорий в HDFS
│   ├── generate_sample.py              # Генератор тестовых снимков
│   └── run_pipeline.sh                 # Обёртка над spark-submit
├── src/
│   ├── main.py                         # Точка входа пайплайна
│   ├── preprocessing/
│   │   └── tiling.py                   # Валидация + разбиение на тайлы
│   ├── model/
│   │   └── enhancer.py                 # Mock + реальная Satlas-модель
│   ├── postprocessing/
│   │   ├── assembly.py                 # Сборка тайлов с блендингом
│   │   └── metrics.py                  # PSNR, SSIM
│   ├── storage/
│   │   └── hdfs_client.py              # PyArrow + CLI fallback
│   └── utils/
│       └── config.py                   # Загрузка YAML + логи
├── tests/
│   ├── test_tiling.py                  # 8 тестов разбиения/сборки
│   └── test_model_metrics.py           # 8 тестов mock-модели и метрик
├── Makefile                            # Удобные команды
├── requirements.txt
└── README.md
```

---

## Быстрый старт

**Требования:** Docker 20+, Docker Compose v2, 6+ ГБ свободной RAM, ~4 ГБ на диске под образы.

```bash
# 1. Клонировать репозиторий и зайти в него
cd satlas-project

# 2. Собрать образы (первый раз ~5-10 минут)
make build

# 3. Поднять весь кластер (NameNode, DataNode, Spark×3, Jupyter)
make up

# 4. Сгенерировать тестовый снимок и прогнать пайплайн
make run
```

После `make up` доступны:

| URL | Сервис |
|---|---|
| http://localhost:8501 | Streamlit Frontend (основной UI) |
| http://localhost:8000 | FastAPI Processor (REST API) |
| http://localhost:8888 | Jupyter Lab (для разработки) |
| http://localhost:8080 | Spark Master UI |
| http://localhost:9870 | HDFS NameNode UI |
| http://localhost:4040 | Spark Application UI (во время задачи) |
| http://localhost:5555 | Flower (мониторинг Celery) |

В браузере откройте http://localhost:8501 для работы с основным интерфейсом. Для интерактивной разработки используйте Jupyter по адресу http://localhost:8888 и ноутбук `notebooks/demo_pipeline.ipynb`.

Остановить всё: `make down`. Удалить и данные HDFS: `make clean`.

---

## Как это работает внутри

### 1. Валидация снимка (`src/preprocessing/tiling.py::validate_image`)

Проверяет: файл существует, расширение в белом списке (`.png`, `.tif`, `.jpg`, …), размер в диапазоне `[min_size, max_size]` из конфига. Падает с человекочитаемой ошибкой, если что-то не так.

### 2. Разбиение на тайлы (`split_image_into_tiles`)

Формирует сетку тайлов размера `tile_size × tile_size` с перекрытием `overlap`. Ключевая деталь — **последний тайл в ряду/столбце всегда прижат к правому/нижнему краю** (`start = total - tile_size`), даже если из-за этого его overlap с предыдущим больше заявленного. Это гарантирует, что при сборке мы получим ровно исходный размер без паддинга.

Возвращаемый `Tile` содержит `(index, x, y, width, height, data)` — координаты нужны для последующей сборки.

### 3. Распределённый инференс (`src/main.py::run_enhancement_pipeline` + `mapPartitions`)

Тайлы упаковываются в RDD и перепартицируются по количеству воркеров:

```python
tile_items = [(t.to_dict_without_data(), t.data) for t in tiles]
rdd = sc.parallelize(tile_items, numSlices=num_partitions)
processed = rdd.mapPartitions(_process_partition_factory(model_cfg_dict)).collect()
```

Внутри `_process_partition_factory` — замыкание, которое на каждом воркер-процессе:

1. Импортирует `ImageEnhancer` (импорт ленивый, чтобы драйвер не тянул torch).
2. Получает инстанс через `get_or_create_enhancer` — он **кешируется в модуле**, так что для всех последующих партиций на этом процессе модель уже в памяти.
3. Собирает всю партицию в память и вызывает `enhance_batch` — батчи эффективнее на GPU, а на CPU сводятся к последовательным вызовам без штрафа.

**Почему `mapPartitions`, а не `map`?** С `map` модель бы загружалась на каждый тайл — это катастрофически медленно при реальной нейросети в сотни мегабайт. С `mapPartitions` накладные расходы оплачиваются один раз на партицию (а фактически — один раз на процесс благодаря кешу).

### 4. Сборка тайлов (`src/postprocessing/assembly.py::assemble_tiles`)

Наивный подход «последний тайл перезаписывает» даёт видимый шов на стыках. Правильное решение — **взвешенное усреднение**:

```
result[y, x] = sum(mask_i[y, x] * tile_i[y, x]) / sum(mask_i[y, x])
```

Маска `mask_i` для каждого тайла — гауссиана с максимумом в центре и плавным спадом к краям. В перекрытии с соседом вклад «центральных» пикселей тайла всегда больше, поэтому переходы сглаженные. Эпсилон `1e-3` в маске гарантирует, что пиксели, покрытые одним тайлом, не обнулятся.

Поддерживаются режимы `gaussian` (по умолчанию), `average` (равные веса) и `none` (без усреднения — для проверки).

### 5. Метрики (`src/postprocessing/metrics.py`)

**PSNR** (Peak Signal-to-Noise Ratio) — стандартная метрика в дБ. Для идентичных изображений `+inf`, при сильных различиях — 0 и ниже.

**SSIM** (Structural Similarity) — упрощённая реализация по всей картинке без скользящего окна. Для учебных целей её достаточно; для production-валидации стоит подключить `skimage.metrics.structural_similarity`.

### 6. Сохранение в HDFS (`src/storage/hdfs_client.py`)

Основной путь — `pyarrow.fs.HadoopFileSystem`: нативный доступ через `libhdfs` из дистрибутива Hadoop (`HADOOP_HOME`, `CLASSPATH`, `ARROW_LIBHDFS_DIR` настраиваются в Dockerfile). Если нативная библиотека недоступна — автоматический fallback на CLI `hdfs dfs -put`.

Структура результатов:

```
hdfs:///data/input/2024-04-20/SAT_DEMO_001_original.png
hdfs:///data/output/2024-04-20/SAT_DEMO_001_enhanced.png
hdfs:///data/metadata/2024-04-20/SAT_DEMO_001.json
```

JSON-метаданные содержат: пути, формы, параметры тайлинга, количество тайлов, время инференса, метрики, модель, timestamp.

---

## Конфигурация

Все параметры — в `config/settings.yaml`. Самые важные секции:

| Параметр | Назначение | По умолчанию |
|---|---|---|
| `image.tile_size` | Размер тайла в пикселях | 512 |
| `image.overlap` | Перекрытие тайлов | 32 |
| `model.use_mock` | Использовать заглушку вместо реальной модели | `true` |
| `model.device` | `cpu` или `cuda` | `cpu` |
| `model.batch_size` | Размер батча для инференса | 4 |
| `spark.num_partitions` | На сколько частей дробить RDD | 4 |
| `postprocessing.blending` | `gaussian` / `average` / `none` | `gaussian` |

Изменения в `settings.yaml` применяются сразу — файл монтируется в контейнер через volume, перезапуск не нужен.

---

## WebUI кластера

- **Streamlit Frontend (8501)** — основной пользовательский интерфейс для загрузки и обработки снимков.
- **FastAPI Processor (8000)** — REST API для программного взаимодействия (`/docs` доступен для Swagger).
- **Spark Master (8080)** — статус воркеров, список приложений, ресурсы.
- **HDFS NameNode (9870)** — браузер файловой системы (вкладка *Utilities → Browse the file system*), живой просмотр `/data/input/…`, `/data/output/…`. Можно скачать файлы в один клик.
- **Spark App UI (4040)** — появляется на время работы задачи, показывает DAG, стадии, распределение тайлов по воркерам, время каждой таски.
- **Flower (5555)** — мониторинг задач Celery: очередь, выполненные/активные задачи, воркеры.
- **Jupyter Lab (8888)** — интерактивная разработка и отладка пайплайна.

---

## Запуск пайплайна

### Вариант 1: через Web UI (Streamlit)

Откройте http://localhost:8501 в браузере, загрузите изображение через интерфейс и нажмите кнопку обработки. Статус задачи отображается в реальном времени.

### Вариант 2: через REST API (FastAPI)

```bash
curl -X POST http://localhost:8000/process \
  -F "image=@/path/to/image.png" \
  -F "image_id=SAT_20240420_001"
```

Swagger-документация доступна по адресу http://localhost:8000/docs.

### Вариант 3: из Jupyter

Открыть `notebooks/demo_pipeline.ipynb` и выполнить ячейки — самый наглядный способ, с графиками сравнения.

### Вариант 4: программно из Python

```python
from src.main import run_enhancement_pipeline

result = run_enhancement_pipeline(
    image_path="/app/data/samples/sat_001.png",
    image_id="SAT_20240420_001",
    config_path="/app/config/settings.yaml",
)
print(result["metrics"])  # {'psnr_db': ..., 'ssim': ...}
```

### Вариант 5: через spark-submit

```bash
bash scripts/run_pipeline.sh /app/data/samples/sat_001.png SAT_20240420_001
```

или вручную из контейнера `satlas-jupyter`:

```bash
spark-submit \
    --master spark://spark-master:7077 \
    --conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000 \
    /app/src/main.py \
        --image /app/data/samples/sat_001.png \
        --image-id SAT_20240420_001 \
        --config /app/config/settings.yaml
```

### Вариант 6: массовая загрузка

```bash
# Положить файлы в HDFS напрямую
docker exec satlas-namenode hdfs dfs -put *.png /data/input/raw/
# Затем обработать их циклом из вашего скрипта
```

---

## Тестирование

```bash
make test
```

Запускает 16 unit-тестов внутри контейнера:

- **8 тестов `test_tiling.py`** — разбиение/сборка: корректность стартовых координат, покрытие всего изображения, round-trip без потерь, обработка маленьких изображений, валидация невалидных параметров.
- **8 тестов `test_model_metrics.py`** — mock-модель сохраняет форму и действительно меняет изображение; батч совпадает с последовательным вызовом; PSNR/SSIM дают ожидаемые граничные значения.

Все 16 проходят на чистой установке.

Для покрытия: `pytest tests/ --cov=src --cov-report=html`.

---

## Переход на реальную модель Satlas

По умолчанию `use_mock: true` — пайплайн работает без внешних зависимостей и без GPU. Для интеграции реальной модели [`satlaspretrain_models`](https://github.com/allenai/satlaspretrain_models):

1. В `requirements.txt` раскомментировать блок с `torch`, `torchvision`, `satlaspretrain_models`.
2. Пересобрать образы: `make build`.
3. В `config/settings.yaml`:
   ```yaml
   model:
     use_mock: false
     name: "Sentinel2_SwinB_SI_RGB"   # см. каталог satlaspretrain
     device: "cuda"                   # если доступен GPU
   ```
4. При первом запуске веса скачаются в `/models` (volume общий для Jupyter и Spark-воркеров — веса скачаются один раз).

Интерфейс `ImageEnhancer` одинаков для обоих режимов — остальные модули не меняются.

---

## Диагностика проблем

**`Celery worker не подключается к Redis`**

Проверьте, что сервис `redis` запущен: `docker ps | grep satlas-redis`. Логи Celery: `docker logs satlas-celery`. Перезапуск: `docker restart satlas-celery satlas-redis`.

**`API Processor не стартует / 502 Bad Gateway`**

Сервису может не хватать памяти. Проверьте логи: `docker logs satlas-api`. Увеличьте лимит в `docker/docker-compose.yml` (секция `deploy.resources.limits`).

**`NameNode не стартует` / `Connection refused hdfs://namenode:9000`**
Дайте кластеру 30–60 секунд после `make up` — в `docker-compose.yml` прописан `healthcheck`, но на медленных машинах первый старт длиннее. Логи: `make logs`.

**`py4j.protocol.Py4JJavaError ... java.io.IOException: No FileSystem for scheme: hdfs`**
Не настроен `CLASSPATH` для pyarrow. В ноутбуке выполните:
```python
import subprocess, os
os.environ["CLASSPATH"] = subprocess.check_output(
    ["/opt/hadoop/bin/hadoop", "classpath", "--glob"]
).decode().strip()
```
Для `spark-submit` это делает `scripts/run_pipeline.sh`.

**`WorkerLostException` во время инференса**
Воркеру не хватает памяти (по умолчанию 2 ГБ). В `docker/docker-compose.yml` увеличьте `SPARK_WORKER_MEMORY`, в `settings.yaml` — `spark.executor.memory`.

**`image_search: use_mock=true, а PSNR = inf`**
Вы обрабатываете изображение, на котором median-фильтр и unsharp-mask не дают заметного эффекта (например, полностью монотонное). Попробуйте сгенерированный `scripts/generate_sample.py` — там специально добавлен шум и структуры, чтобы эффект был виден.

**Тесты падают локально, а в контейнере проходят**
Скорее всего, отсутствуют `pyyaml` / `pillow` / `numpy` на хосте. Установите: `pip install -r requirements.txt`.

---

## Соответствие ТЗ

| Требование ТЗ | Где реализовано |
|---|---|
| Этап 1: приём и валидация | `src/preprocessing/tiling.py::validate_image` |
| Этап 2: разбиение на тайлы с overlap | `split_image_into_tiles` |
| Этап 3: распределённая обработка через Spark | `src/main.py` + `mapPartitions` |
| Загрузка модели один раз на воркер | `get_or_create_enhancer` (модульный кеш) |
| Batch-инференс | `ImageEnhancer.enhance_batch` |
| Этап 4: сборка с blending | `assemble_tiles` + gaussian-маска |
| Этап 5: сохранение в HDFS | `HDFSClient` + структура `/data/{input,output,metadata}` |
| Метаданные обработки (JSON) | `HDFSClient.put_json` в `run_enhancement_pipeline` |
| Spark-конфигурация (память, ядра, retries) | `config/settings.yaml` + `docker-compose.yml` |
| Модульные тесты | `tests/test_tiling.py`, `tests/test_model_metrics.py` |
| Интеграционный тест full-pipeline | `notebooks/demo_pipeline.ipynb` |
| Метрики PSNR/SSIM | `src/postprocessing/metrics.py` |
| Логирование в файл и stdout | `src/utils/config.py::setup_logging` |
| Расширяемость (замена модели) | Интерфейс `ImageEnhancer` + флаг `use_mock` |
| REST API для обработки | `docker/api-processor/app/main.py` (FastAPI) |
| Асинхронная обработка задач | Celery worker (`docker/api-processor/Dockerfile.celery`) |
| Web UI для пользователей | Streamlit frontend (`docker/frontend/app.py`) |
| Мониторинг очереди задач | Flower (порт 5555) |
| Брокер сообщений | Redis (порт 6379) |

---

## Дальнейшие улучшения (за рамками ТЗ)

- Поддержка **GeoTIFF** с сохранением геопривязки через `rasterio` (сейчас при сохранении в PNG метаданные теряются).
- Параллельная обработка **пачки снимков** через `spark.read.format("binaryFile")` — чтобы Spark сам распределял файлы.
- **Persistent model cache** — при `use_mock: false` скачивать веса один раз в volume `/models`, а не на каждый старт воркера.
- Метрики **MS-SSIM** и **LPIPS** для более точной оценки перцептивного качества.
- **Kerberos** для защищённого HDFS в продакшене.

---

## Лицензия

Учебный проект, публикуется под MIT License.