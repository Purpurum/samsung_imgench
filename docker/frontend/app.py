import streamlit as st
import requests
import json
import os
import glob
from pathlib import Path
import base64

API_URL = os.getenv("API_URL", "http://api-processor:8000")
DATA_DIR = os.getenv("FRONTEND_DATA_DIR", "/app/data/processed")

st.set_page_config(page_title="Улучшение спутниковых снимков", layout="wide")
st.title("Улучшитель качества спутниковых снимков")

tab1, tab2, tab3 = st.tabs(["Загрузка и обработка", "Прямой доступ к API", "Результаты и тайлы"])

def upload_file(file_obj, params_dict=None):
    """Отправка файла на обработку через API"""
    files = {"file": (file_obj.name, file_obj.getvalue())}
    data = {"params": json.dumps(params_dict or {})}
    try:
        response = requests.post(f"{API_URL}/api/process/upload", files=files, data=data, timeout=30)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {e}")

def check_status(job_id):
    """Проверка статуса задачи"""
    try:
        response = requests.get(f"{API_URL}/api/jobs/{job_id}", timeout=10)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        raise Exception(f"Status check failed: {e}")

def get_tiles(job_id):
    """Получение списка тайлов для задачи"""
    try:
        response = requests.get(f"{API_URL}/api/jobs/{job_id}/tiles", timeout=10)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        raise Exception(f"Get tiles failed: {e}")

def get_image_base64(image_path):
    """Converts image to base64 for display"""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None

with tab1:
    st.header("Загрузка изображения для улучшения")
    uploaded = st.file_uploader("Выберите файл", type=["tif", "tiff", "SAFE", "png", "jpg"])

    if uploaded:
        st.info(f"Выбрано: {uploaded.name}")

        col1, col2 = st.columns(2)
        with col1:
            tile_size = st.selectbox("Размер тайла", [256, 512, 1024], index=1)
            overlap = st.selectbox("Перекрытие", [16, 32, 64], index=1)
        with col2:
            use_mock = st.checkbox("Использовать тестовую модель (быстрее)", value=False)  # По умолчанию реальная обработка
            blending = st.selectbox("Смешивание", ["gaussian", "average", "none"], index=0)

        if st.button("Начать обработку", type="primary"):
            with st.spinner("Загрузка и постановка задачи в очередь..."):
                params = {
                    "tile_size": tile_size,
                    "overlap": overlap,
                    "use_mock": use_mock,
                    "blending": blending
                }
                try:
                    resp = upload_file(uploaded, params)
                    if resp.status_code == 200:
                        res = resp.json()
                        st.success(f"Задача {res['job_id']} добавлена в очередь")
                        st.session_state["last_job_id"] = res["job_id"]
                    else:
                        st.error(f"Ошибка API: {resp.text}")
                except Exception as e:
                    st.error(f"Ошибка сети: {e}")

with tab2:
    st.header("Прямой доступ к API через HTTP")
    st.markdown("""
    Эта вкладка позволяет напрямую взаимодействовать с API обработки.
    Загрузите изображение и отправьте его на обработку в бэкенд.
    """)

    st.code(f"""# Пример команды curl
curl -X POST "{API_URL}/api/process/upload" \\
  -F "file=@your_image.png" \\
  -F 'params={{"tile_size":512,"overlap":32,"use_mock":true}}'

# Ответ будет содержать job_id для проверки статуса
""", language="bash")

    col1, col2 = st.columns([2, 1])
    with col1:
        api_file = st.file_uploader("Загрузить файл для тестирования API", type=["tif", "tiff", "SAFE", "png", "jpg"], key="api_uploader")
    with col2:
        api_tile_size = st.selectbox("Размер тайла", [256, 512, 1024], index=1, key="api_tile")
        api_overlap = st.selectbox("Перекрытие", [16, 32, 64], index=1, key="api_overlap")
        api_use_mock = st.checkbox("Использовать тестовую модель", value=False, key="api_mock")  # По умолчанию реальная обработка

    if api_file and st.button("Отправить через API", type="primary", key="api_send"):
        with st.spinner("Запрос..."):
            try:
                params = {
                    "tile_size": api_tile_size,
                    "overlap": api_overlap,
                    "use_mock": api_use_mock
                }
                resp = upload_file(api_file, params)

                if resp.status_code == 200:
                    result = resp.json()
                    st.success("Запрос успешно выполнен!")
                    st.json(result)
                    st.session_state["last_job_id"] = result["job_id"]

                    # Показать следующие шаги
                    st.info(f"""
                    **Следующие шаги:**
                    1. Job ID: `{result['job_id']}`
                    2. Перейдите на вкладку 'Результаты и тайлы'
                    3. Вставьте Job ID и нажмите 'Обновить статус'
                    """)
                else:
                    st.error(f"Ошибка API ({resp.status_code}): {resp.text}")
            except Exception as e:
                st.error(f"Ошибка: {e}")

    # Документация эндпоинтов API
    st.subheader("Доступные эндпоинты")
    st.markdown(f"""
    | Метод | Эндпоинт | Описание |
    |--------|----------|-------------|
    | POST | `/api/process/upload` | Загрузка изображения для обработки |
    | GET | `/api/jobs/{{job_id}}` | Получить статус задачи |
    | GET | `/api/jobs/{{job_id}}/tiles` | Получить список тайлов |
    | GET | `/api/health` | Проверка работоспособности |
    """)

with tab3:
    st.header("Мониторинг и просмотр результатов")

    job_input = st.text_input("Job ID", value=st.session_state.get("last_job_id", ""))

    if st.button("Обновить статус", type="primary"):
        if not job_input:
            st.warning("Введите Job ID")
        else:
            with st.spinner("Проверка статуса..."):
                try:
                    resp = check_status(job_input)
                    if resp.status_code == 200:
                        data = resp.json()

                        # Отображение статуса
                        status = data.get("status")
                        status_colors = {
                            "completed": "success",
                            "failed": "error",
                            "processing": "info",
                            "pending": "warning",
                            "queued": "warning"
                        }
                        status_color = status_colors.get(status, "info")

                        if status_color == "success":
                            st.success(f"✅ Статус: **{status.upper()}**")
                        elif status_color == "error":
                            st.error(f"❌ Статус: **{status.upper()}**")
                        elif status_color == "info":
                            st.info(f"🔄 Статус: **{status.upper()}**")
                        else:
                            st.warning(f"⏳ Статус: **{status.upper()}**")

                        # Показать полный ответ в раскрывающемся блоке
                        with st.expander("Необработанный ответ", expanded=False):
                            st.json(data)

                        if status == "completed":
                            res_info = data.get("result", {})

                            if res_info.get("is_mock"):
                                st.warning("ℹ️ Отображаются тестовые результаты (обнаружен тестовый файл)")

                            # Отображение путей HDFS
                            hdfs_output = res_info.get("hdfs_output")
                            hdfs_tiles = res_info.get("hdfs_tiles_dir")
                            if hdfs_output:
                                st.info(f"**HDFS Output:** `{hdfs_output}`")
                            if hdfs_tiles:
                                st.info(f"**HDFS Tiles:** `{hdfs_tiles}`")

                            job_dir = Path(DATA_DIR) / job_input
                            tiles_dir = job_dir / "tiles"

                            # Отображение улучшенного результата
                            result_files = list(job_dir.glob("result*.png"))
                            if result_files:
                                st.subheader("📸 Улучшенный результат")
                                st.image(str(result_files[0]), caption="Улучшенное изображение", use_column_width=True)

                                # Кнопка скачивания
                                with open(result_files[0], "rb") as f:
                                    st.download_button(
                                        label="📥 Скачать результат",
                                        data=f.read(),
                                        file_name=result_files[0].name,
                                        mime="image/png"
                                    )

                            # Отображение тайлов в сетке
                            if tiles_dir.exists():
                                original_tiles = sorted(tiles_dir.glob("*_original.png"))
                                enhanced_tiles = sorted(tiles_dir.glob("*_enhanced.png"))
                                mock_tiles = sorted(tiles_dir.glob("*_mock.png"))
                                all_tiles = sorted(tiles_dir.glob("*.png"))

                                if all_tiles:
                                    st.subheader(f"🔲 Тайлы (всего: {len(all_tiles)})")

                                    # Создаем вкладки в зависимости от типов тайлов
                                    tab_names = []
                                    if original_tiles:
                                        tab_names.append("Исходные тайлы")
                                    if enhanced_tiles:
                                        tab_names.append("Улучшенные тайлы")
                                    if mock_tiles and not original_tiles and not enhanced_tiles:
                                        tab_names.append("Тестовые тайлы")

                                    if len(tab_names) > 1:
                                        tile_tabs = st.tabs(tab_names)
                                        tab_idx = 0

                                        if original_tiles:
                                            with tile_tabs[tab_idx]:
                                                cols = st.columns(min(4, len(original_tiles)))
                                                for i, t_path in enumerate(original_tiles):
                                                    with cols[i % 4]:
                                                        st.image(str(t_path), caption=t_path.name, use_column_width=True)
                                                tab_idx += 1

                                        if enhanced_tiles:
                                            with tile_tabs[tab_idx]:
                                                cols = st.columns(min(4, len(enhanced_tiles)))
                                                for i, t_path in enumerate(enhanced_tiles):
                                                    with cols[i % 4]:
                                                        st.image(str(t_path), caption=t_path.name, use_column_width=True)
                                                tab_idx += 1

                                        if mock_tiles and not original_tiles and not enhanced_tiles:
                                            with tile_tabs[tab_idx]:
                                                cols = st.columns(min(4, len(mock_tiles)))
                                                for i, t_path in enumerate(mock_tiles):
                                                    with cols[i % 4]:
                                                        st.image(str(t_path), caption=t_path.name, use_column_width=True)
                                    elif len(tab_names) == 1:
                                        # Только один тип тайлов - показываем все сразу
                                        cols = st.columns(min(4, len(all_tiles)))
                                        for i, t_path in enumerate(all_tiles):
                                            with cols[i % 4]:
                                                st.image(str(t_path), caption=t_path.name, use_column_width=True)

                                    # Статистика тайлов
                                    st.subheader("📊 Статистика тайлов")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Всего тайлов", len(all_tiles))
                                    with col2:
                                        st.metric("Исходные тайлы", len(original_tiles))
                                    with col3:
                                        st.metric("Улучшенные тайлы", len(enhanced_tiles))

                            elif status in ["pending", "processing", "STARTED", "PENDING"]:
                                st.info("Обработка выполняется...")
                        elif status == "failed":
                            st.error(f"❌ Ошибка: {data.get('error', 'Неизвестная ошибка')}")
                            if "traceback" in data:
                                with st.expander("Traceback"):
                                    st.code(data["traceback"])
                        else:
                            st.info(f"⏳ Статус: {status}")
                    else:
                        st.error(f"Ошибка запроса: {resp.status_code}")
                except Exception as e:
                    st.error(f"Не удалось получить статус: {e}")

    # Секция последних задач
    if Path(DATA_DIR).exists():
        st.subheader("📁 Последние задачи")
        job_dirs = sorted([d for d in Path(DATA_DIR).iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)[:5]
        if job_dirs:
            for jd in job_dirs:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"`{jd.name}`")
                with col2:
                    if st.button("Просмотреть", key=f"view_{jd.name}"):
                        st.session_state["last_job_id"] = jd.name
                        st.rerun()
        else:
            st.info("Обработанные задачи не найдены")

st.sidebar.header("Информация")
st.sidebar.markdown("""
- **Загрузка**: Постановка задачи обработки изображения через Celery
- **API**: Прямой доступ к эндпоинтам с примерами curl
- **Результаты**: Проверка статуса, просмотр улучшенного изображения и тайлов
""")

try:
    health = requests.get(f"{API_URL}/api/health", timeout=2)
    if health.status_code == 200:
        st.sidebar.success("API: Активен")
    else:
        st.sidebar.error("API: Неожиданный ответ")
except Exception:
    st.sidebar.error("API: Недоступен")