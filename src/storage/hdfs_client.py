import io
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

@dataclass
class HDFSConfig:
    root: str
    input_dir: str
    output_dir: str
    metadata_dir: str
    replication: int = 1

class HDFSClient:
    def __init__(self, cfg: HDFSConfig) -> None:
        self.cfg = cfg
        self._fs = None
        self._host, self._port = self._parse_root(cfg.root)

    @staticmethod
    def _parse_root(root: str) -> tuple[str, int]:
        without_scheme = root.replace("hdfs://", "")
        host, _, port = without_scheme.partition(":")
        return host, int(port) if port else 9000

    def _get_fs(self):
        if self._fs is not None:
            return self._fs
        try:
            from pyarrow import fs as pafs
            self._fs = pafs.HadoopFileSystem(
                host=self._host,
                port=self._port,
                user=os.getenv("HADOOP_USER_NAME", "root")
            )
            log.info("Connected to HDFS: %s:%d via pyarrow", self._host, self._port)
        except Exception as e:
            log.warning("pyarrow HDFS unavailable (%s), falling back to CLI", e)
            self._fs = None
        return self._fs

    def mkdirs(self, hdfs_path: str) -> None:
        fs = self._get_fs()
        if fs is not None:
            fs.create_dir(hdfs_path, recursive=True)
            return
        self._cli(["hdfs", "dfs", "-mkdir", "-p", self._full_uri(hdfs_path)])

    def exists(self, hdfs_path: str) -> bool:
        fs = self._get_fs()
        if fs is not None:
            from pyarrow.fs import FileType
            info = fs.get_file_info(hdfs_path)
            return info.type != FileType.NotFound
        r = subprocess.run(
            ["hdfs", "dfs", "-test", "-e", self._full_uri(hdfs_path)],
            capture_output=True,
        )
        return r.returncode == 0

    def put_image(self, image: np.ndarray, hdfs_path: str, fmt: str = "PNG") -> None:
        buf = io.BytesIO()
        Image.fromarray(image).save(buf, format=fmt)
        self.put_bytes(buf.getvalue(), hdfs_path)

    def put_bytes(self, data: bytes, hdfs_path: str) -> None:
        fs = self._get_fs()
        self.mkdirs(str(Path(hdfs_path).parent))
        if fs is not None:
            with fs.open_output_stream(hdfs_path) as f:
                f.write(data)
            log.info("Written to HDFS: %s (%d bytes)", hdfs_path, len(data))
            return
        tmp = Path(f"/tmp/hdfs_upload_{os.getpid()}_{Path(hdfs_path).name}")
        tmp.write_bytes(data)
        try:
            self._cli(["hdfs", "dfs", "-put", "-f", str(tmp), self._full_uri(hdfs_path)])
        finally:
            tmp.unlink(missing_ok=True)

    def put_local_file(self, local_path: Union[str, Path], hdfs_path: str) -> None:
        local_path = Path(local_path)
        fs = self._get_fs()
        self.mkdirs(str(Path(hdfs_path).parent))
        if fs is not None:
            with local_path.open("rb") as src, fs.open_output_stream(hdfs_path) as dst:
                dst.write(src.read())
            log.info("Uploaded to HDFS: %s -> %s", local_path, hdfs_path)
            return
        self._cli(["hdfs", "dfs", "-put", "-f", str(local_path), self._full_uri(hdfs_path)])

    def put_json(self, payload: dict, hdfs_path: str) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.put_bytes(data, hdfs_path)

    def get_bytes(self, hdfs_path: str) -> bytes:
        fs = self._get_fs()
        if fs is not None:
            with fs.open_input_stream(hdfs_path) as f:
                return f.read()
        r = subprocess.run(
            ["hdfs", "dfs", "-cat", self._full_uri(hdfs_path)],
            capture_output=True,
            check=True,
        )
        return r.stdout

    def _full_uri(self, hdfs_path: str) -> str:
        if hdfs_path.startswith("hdfs://"):
            return hdfs_path
        return f"{self.cfg.root}{hdfs_path}"

    @staticmethod
    def _cli(args: list[str]) -> None:
        log.debug("HDFS CLI: %s", " ".join(args))
        r = subprocess.run(args, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(
                f"HDFS CLI failed (rc={r.returncode}): {r.stderr or r.stdout}"
            )