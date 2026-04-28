import streamlit as st
import requests
import json
import os
import glob
from pathlib import Path

API_URL = os.getenv("API_URL", "http://api-processor:8000")
DATA_DIR = os.getenv("FRONTEND_DATA_DIR", "/app/data/processed")

st.set_page_config(page_title="Satellite Image Enhancer", layout="wide")
st.title("Satellite Image Quality Enhancer")

tab1, tab2, tab3 = st.tabs(["Upload and Process", "API Direct Access", "Results and Tiles"])

def upload_file(file_obj, params_dict=None):
    files = {"file": (file_obj.name, file_obj.getvalue())}
    data = {"params": json.dumps(params_dict or {})}
    return requests.post(f"{API_URL}/api/process/upload", files=files, data=data)

def check_status(job_id):
    return requests.get(f"{API_URL}/api/jobs/{job_id}")

def get_tiles(job_id):
    return requests.get(f"{API_URL}/api/jobs/{job_id}/tiles")

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
            use_mock = st.checkbox("Use mock model (faster)", value=True)
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
    st.header("Direct API access")
    st.code(f"""curl -X POST "{API_URL}/api/process/upload" \\
  -F "file=@your_image.png" \\
  -F 'params={{"tile_size":512,"overlap":32,"use_mock":true}}'""", language="bash")
    
    api_file = st.file_uploader("File for API test", type=["tif", "tiff", "SAFE", "png", "jpg"], key="api_uploader")
    
    if api_file and st.button("Send via API", type="primary", key="api_send"):
        with st.spinner("Requesting..."):
            try:
                params = {"tile_size": 512, "overlap": 32, "use_mock": True}
                resp = upload_file(api_file, params)
                st.json(resp.json())
                if resp.status_code == 200:
                    st.session_state["last_job_id"] = resp.json()["job_id"]
            except Exception as e:
                st.error(f"Error: {e}")

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
                        st.json(data)
                        status = data.get("status")
                        
                        if status == "completed":
                            st.success("Processing complete")
                            res_info = data.get("result", {})
                            
                            if res_info.get("is_mock"):
                                st.warning("Displaying mock results (test file detected)")
                            
                            job_dir = Path(DATA_DIR) / job_input
                            tiles_dir = job_dir / "tiles"
                            
                            result_files = list(job_dir.glob("result*.png"))
                            if result_files:
                                st.image(str(result_files[0]), caption="Enhanced result", use_container_width=True)
                            
                            if tiles_dir.exists():
                                tiles = sorted(tiles_dir.glob("*.png"))
                                if tiles:
                                    st.subheader(f"Tiles ({len(tiles)})")
                                    cols = st.columns(min(4, len(tiles)))
                                    for i, t_path in enumerate(tiles):
                                        with cols[i % 4]:
                                            st.image(str(t_path), caption=t_path.name, use_container_width=True)
                            elif status in ["pending", "processing", "STARTED", "PENDING"]:
                                st.info("Processing in progress...")
                        elif status == "failed":
                            st.error(f"Error: {data.get('error')}")
                        else:
                            st.info(f"Status: {status}")
                    else:
                        st.error(f"Request error: {resp.status_code}")
                except Exception as e:
                    st.error(f"Failed to get status: {e}")

st.sidebar.header("Info")
st.sidebar.markdown("""
- Upload: Queues image processing task via Celery
- API: Direct endpoint access with curl examples
- Results: Status check, enhanced image and tiles preview
- Mock files: Upload files with 'mock', 'test', 'demo' in name for placeholder results
""")

try:
    health = requests.get(f"{API_URL}/api/health", timeout=2)
    if health.status_code == 200:
        st.sidebar.success("API: Active")
    else:
        st.sidebar.error("API: Unexpected response")
except Exception:
    st.sidebar.error("API: Unreachable")