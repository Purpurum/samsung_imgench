from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import uuid
import os
import json
import tempfile
from pathlib import Path
from .tasks import process_uploaded_image
from .celery_worker import celery_app

app = FastAPI(title="Satellite Image Processor API")

class ProcessingParams(BaseModel):
    product: str = "GRD"
    mode: str = "EW"
    pol: str = "DH"
    band_name: str = "Amplitude_HH"
    max_scenes: int = 100
    tile_size: int = 512
    overlap: int = 32
    use_mock: bool = False  # По умолчанию используем реальную обработку

class JobResponse(BaseModel):
    job_id: str
    status: str
    task_id: Optional[str] = None

@app.post("/api/process/upload", response_model=JobResponse)
async def process_upload(
    file: UploadFile = File(...),
    params: str = Form("{}")
):
    supported = ('.tif', '.tiff', '.SAFE', '.png', '.jpg', '.jpeg')
    if not any(file.filename.lower().endswith(ext) for ext in supported):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    job_id = str(uuid.uuid4())
    upload_dir = Path(os.getenv("PROCESSING_OUT_DIR", "/data/processed")) / "uploads" / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    params_dict = json.loads(params) if params else {}

    task = process_uploaded_image.delay(
        job_id=job_id,
        file_path=str(file_path),
        filename=file.filename,
        params=params_dict
    )

    return JobResponse(job_id=job_id, status="queued", task_id=task.id)

@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    result = celery_app.AsyncResult(job_id)

    if result.state == "PENDING":
        return {"job_id": job_id, "status": "pending"}
    elif result.state == "STARTED":
        return {"job_id": job_id, "status": "processing"}
    elif result.state == "SUCCESS":
        return {"job_id": job_id, "status": "completed", "result": result.result}
    elif result.state == "FAILURE":
        return {"job_id": job_id, "status": "failed", "error": str(result.info)}
    return {"job_id": job_id, "status": result.state}

@app.get("/api/jobs/{job_id}/tiles")
async def get_job_tiles(job_id: str):
    tiles_dir = Path(os.getenv("PROCESSING_OUT_DIR", "/data/processed")) / job_id / "tiles"
    if not tiles_dir.exists():
        return {"job_id": job_id, "tiles": []}

    tiles = []
    for ext in ["*.png", "*.tif", "*.jpg"]:
        tiles.extend([str(p.relative_to(tiles_dir)) for p in tiles_dir.glob(ext)])

    return {"job_id": job_id, "tiles": sorted(tiles), "tiles_dir": str(tiles_dir)}

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "api-processor"}