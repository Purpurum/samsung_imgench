import streamlit as st
import requests
import json
import os
import glob
from pathlib import Path
import base64

API_URL = os.getenv("API_URL", "http://api-processor:8000")
DATA_DIR = os.getenv("FRONTEND_DATA_DIR", "/app/data/processed")

st.set_page_config(page_title="Satellite Image Enhancer", layout="wide")
st.title("Satellite Image Quality Enhancer")

tab1, tab2, tab3 = st.tabs(["Upload and Process", "API Direct Access", "Results and Tiles"])

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
    st.header("Upload image for enhancement")
    uploaded = st.file_uploader("Select file", type=["tif", "tiff", "SAFE", "png", "jpg"])

    if uploaded:
        st.info(f"Selected: {uploaded.name}")

        col1, col2 = st.columns(2)
        with col1:
            tile_size = st.selectbox("Tile size", [256, 512, 1024], index=1)
            overlap = st.selectbox("Overlap", [16, 32, 64], index=1)
        with col2:
            use_mock = st.checkbox("Use mock model (faster)", value=False)  # По умолчанию реальная обработка
            blending = st.selectbox("Blending", ["gaussian", "average", "none"], index=0)

        if st.button("Start Processing", type="primary"):
            with st.spinner("Uploading and queuing task..."):
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
                        st.success(f"Task {res['job_id']} queued")
                        st.session_state["last_job_id"] = res["job_id"]
                    else:
                        st.error(f"API error: {resp.text}")
                except Exception as e:
                    st.error(f"Network error: {e}")

with tab2:
    st.header("Direct API access via HTTP")
    st.markdown("""
    This tab allows you to interact with the processing API directly.
    Upload an image and send it to the backend for processing.
    """)

    st.code(f"""# Example curl command
curl -X POST "{API_URL}/api/process/upload" \\
  -F "file=@your_image.png" \\
  -F 'params={{"tile_size":512,"overlap":32,"use_mock":true}}'

# Response will contain job_id for status checking
""", language="bash")

    col1, col2 = st.columns([2, 1])
    with col1:
        api_file = st.file_uploader("Upload file for API test", type=["tif", "tiff", "SAFE", "png", "jpg"], key="api_uploader")
    with col2:
        api_tile_size = st.selectbox("Tile size", [256, 512, 1024], index=1, key="api_tile")
        api_overlap = st.selectbox("Overlap", [16, 32, 64], index=1, key="api_overlap")
        api_use_mock = st.checkbox("Use mock", value=False, key="api_mock")  # По умолчанию реальная обработка

    if api_file and st.button("Send via API", type="primary", key="api_send"):
        with st.spinner("Requesting..."):
            try:
                params = {
                    "tile_size": api_tile_size,
                    "overlap": api_overlap,
                    "use_mock": api_use_mock
                }
                resp = upload_file(api_file, params)

                if resp.status_code == 200:
                    result = resp.json()
                    st.success("Request successful!")
                    st.json(result)
                    st.session_state["last_job_id"] = result["job_id"]

                    # Show next steps
                    st.info(f"""
                    **Next steps:**
                    1. Job ID: `{result['job_id']}`
                    2. Go to 'Results and Tiles' tab
                    3. Paste the Job ID and click 'Refresh Status'
                    """)
                else:
                    st.error(f"API error ({resp.status_code}): {resp.text}")
            except Exception as e:
                st.error(f"Error: {e}")

    # API endpoints documentation
    st.subheader("Available Endpoints")
    st.markdown(f"""
    | Method | Endpoint | Description |
    |--------|----------|-------------|
    | POST | `/api/process/upload` | Upload image for processing |
    | GET | `/api/jobs/{{job_id}}` | Get job status |
    | GET | `/api/jobs/{{job_id}}/tiles` | Get tiles list |
    | GET | `/api/health` | Health check |
    """)

with tab3:
    st.header("Monitor and view results")

    job_input = st.text_input("Job ID", value=st.session_state.get("last_job_id", ""))

    if st.button("Refresh Status", type="primary"):
        if not job_input:
            st.warning("Enter Job ID")
        else:
            with st.spinner("Checking status..."):
                try:
                    resp = check_status(job_input)
                    if resp.status_code == 200:
                        data = resp.json()

                        # Display status prominently
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
                            st.success(f"✅ Status: **{status.upper()}**")
                        elif status_color == "error":
                            st.error(f"❌ Status: **{status.upper()}**")
                        elif status_color == "info":
                            st.info(f"🔄 Status: **{status.upper()}**")
                        else:
                            st.warning(f"⏳ Status: **{status.upper()}**")

                        # Show full response in expander
                        with st.expander("Raw Response", expanded=False):
                            st.json(data)

                        if status == "completed":
                            res_info = data.get("result", {})

                            if res_info.get("is_mock"):
                                st.warning("ℹ️ Displaying mock results (test file detected)")

                            # Display HDFS paths if available
                            hdfs_output = res_info.get("hdfs_output")
                            hdfs_tiles = res_info.get("hdfs_tiles_dir")
                            if hdfs_output:
                                st.info(f"**HDFS Output:** `{hdfs_output}`")
                            if hdfs_tiles:
                                st.info(f"**HDFS Tiles:** `{hdfs_tiles}`")

                            job_dir = Path(DATA_DIR) / job_input
                            tiles_dir = job_dir / "tiles"

                            # Display enhanced result
                            result_files = list(job_dir.glob("result*.png"))
                            if result_files:
                                st.subheader("📸 Enhanced Result")
                                st.image(str(result_files[0]), caption="Enhanced image", use_column_width=True)

                                # Download button
                                with open(result_files[0], "rb") as f:
                                    st.download_button(
                                        label="📥 Download Result",
                                        data=f.read(),
                                        file_name=result_files[0].name,
                                        mime="image/png"
                                    )

                            # Display tiles in grid - проверяем все PNG файлы, не только с суффиксами
                            if tiles_dir.exists():
                                original_tiles = sorted(tiles_dir.glob("*_original.png"))
                                enhanced_tiles = sorted(tiles_dir.glob("*_enhanced.png"))
                                mock_tiles = sorted(tiles_dir.glob("*_mock.png"))
                                all_tiles = sorted(tiles_dir.glob("*.png"))

                                if all_tiles:
                                    st.subheader(f"🔲 Tiles ({len(all_tiles)} total)")

                                    # Создаем вкладки в зависимости от типов тайлов
                                    tab_names = []
                                    if original_tiles:
                                        tab_names.append("Original Tiles")
                                    if enhanced_tiles:
                                        tab_names.append("Enhanced Tiles")
                                    if mock_tiles and not original_tiles and not enhanced_tiles:
                                        tab_names.append("Mock Tiles")

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

                                    # Tile statistics
                                    st.subheader("📊 Tile Statistics")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Total Tiles", len(all_tiles))
                                    with col2:
                                        st.metric("Original Tiles", len(original_tiles))
                                    with col3:
                                        st.metric("Enhanced Tiles", len(enhanced_tiles))

                            elif status in ["pending", "processing", "STARTED", "PENDING"]:
                                st.info("Processing in progress...")
                        elif status == "failed":
                            st.error(f"❌ Error: {data.get('error', 'Unknown error')}")
                            if "traceback" in data:
                                with st.expander("Traceback"):
                                    st.code(data["traceback"])
                        else:
                            st.info(f"⏳ Status: {status}")
                    else:
                        st.error(f"Request error: {resp.status_code}")
                except Exception as e:
                    st.error(f"Failed to get status: {e}")

    # Recent jobs section
    if Path(DATA_DIR).exists():
        st.subheader("📁 Recent Jobs")
        job_dirs = sorted([d for d in Path(DATA_DIR).iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)[:5]
        if job_dirs:
            for jd in job_dirs:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"`{jd.name}`")
                with col2:
                    if st.button("View", key=f"view_{jd.name}"):
                        st.session_state["last_job_id"] = jd.name
                        st.rerun()
        else:
            st.info("No processed jobs found")

st.sidebar.header("Info")
st.sidebar.markdown("""
- **Upload**: Queues image processing task via Celery
- **API**: Direct endpoint access with curl examples
- **Results**: Status check, enhanced image and tiles preview
- **Mock files**: Upload files with 'mock', 'test', 'demo' in name for placeholder results
- **HDFS**: Results are also saved to HDFS for distributed storage
""")

try:
    health = requests.get(f"{API_URL}/api/health", timeout=2)
    if health.status_code == 200:
        st.sidebar.success("API: Active")
    else:
        st.sidebar.error("API: Unexpected response")
except Exception:
    st.sidebar.error("API: Unreachable")