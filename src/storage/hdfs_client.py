import io
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Tuple
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

@dataclass
class HDFSAccessLog:
    """Лог обращения к HDFS."""
    timestamp: str
    operation: str  # read, write, mkdir, exists, delete
    path: str
    size_bytes: Optional[int] = None
    duration_ms: Optional[float] = None
    success: bool = True
    error: Optional[str] = None
    user: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "operation": self.operation,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error,
            "user": self.user or os.getenv("HADOOP_USER_NAME", "unknown")
        }

@dataclass
class HDFSConfig:
    root: str
    input_dir: str
    output_dir: str
    metadata_dir: str
    replication: int = 1
    access_log_enabled: bool = True
    access_log_dir: Optional[str] = None

class HDFSClient:
    def __init__(self, cfg: HDFSConfig) -> None:
        self.cfg = cfg
        self._fs = None
        self._host, self._port = self._parse_root(cfg.root)
        self._access_logs: List[HDFSAccessLog] = []
        self._user = os.getenv("HADOOP_USER_NAME", "root")

    def _log_access(self, operation: str, path: str, size_bytes: Optional[int] = None,
                    duration_ms: Optional[float] = None, success: bool = True,
                    error: Optional[str] = None) -> None:
        """Записывает лог обращения к HDFS."""
        if not self.cfg.access_log_enabled:
            return

        log_entry = HDFSAccessLog(
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            operation=operation,
            path=path,
            size_bytes=size_bytes,
            duration_ms=duration_ms,
            success=success,
            error=error,
            user=self._user
        )
        self._access_logs.append(log_entry)
        log.debug("HDFS access: %s %s (%d ms, %d bytes)",
                  operation, path, duration_ms or 0, size_bytes or 0)

        # Сохраняем логи в файл если настроено
        if self.cfg.access_log_dir and len(self._access_logs) % 10 == 0:
            self._flush_access_logs()

    def _flush_access_logs(self) -> None:
        """Сохраняет накопленные логи доступа в HDFS."""
        if not self._access_logs or not self.cfg.access_log_dir:
            return

        log_path = f"{self.cfg.access_log_dir}/hdfs_access_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        lines = [json.dumps(log.to_dict()) for log in self._access_logs]
        data = "\n".join(lines).encode("utf-8")

        try:
            self.mkdirs(str(Path(log_path).parent))
            fs = self._get_fs()
            if fs is not None:
                with fs.open_output_stream(log_path, append=True) as f:
                    f.write(data + b"\n")
            else:
                tmp = Path(f"/tmp/hdfs_access_log_{os.getpid()}")
                tmp.write_bytes(data + b"\n")
                try:
                    self._cli(["hdfs", "dfs", "-put", "-f", str(tmp), self._full_uri(log_path)])
                finally:
                    tmp.unlink(missing_ok=True)
            log.info("Flushed %d access logs to %s", len(self._access_logs), log_path)
            self._access_logs.clear()
        except Exception as e:
            log.warning("Failed to flush access logs: %s", e)

    def get_access_logs(self) -> List[Dict[str, Any]]:
        """Возвращает список всех логов доступа."""
        return [log.to_dict() for log in self._access_logs]

    def clear_access_logs(self) -> None:
        """Очищает накопленные логи доступа."""
        self._access_logs.clear()

    @staticmethod
    def _parse_root(root: str) -> Tuple[str, int]:
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
        t0 = time.time()
        try:
            fs = self._get_fs()
            if fs is not None:
                fs.create_dir(hdfs_path, recursive=True)
            else:
                self._cli(["hdfs", "dfs", "-mkdir", "-p", self._full_uri(hdfs_path)])
            self._log_access("mkdir", hdfs_path, duration_ms=(time.time() - t0) * 1000, success=True)
        except Exception as e:
            self._log_access("mkdir", hdfs_path, duration_ms=(time.time() - t0) * 1000, success=False, error=str(e))
            raise

    def exists(self, hdfs_path: str) -> bool:
        t0 = time.time()
        try:
            fs = self._get_fs()
            if fs is not None:
                from pyarrow.fs import FileType
                info = fs.get_file_info(hdfs_path)
                result = info.type != FileType.NotFound
            else:
                r = subprocess.run(
                    ["hdfs", "dfs", "-test", "-e", self._full_uri(hdfs_path)],
                    capture_output=True,
                )
                result = r.returncode == 0
            self._log_access("exists", hdfs_path, duration_ms=(time.time() - t0) * 1000, success=True)
            return result
        except Exception as e:
            self._log_access("exists", hdfs_path, duration_ms=(time.time() - t0) * 1000, success=False, error=str(e))
            return False

    def put_image(self, image: np.ndarray, hdfs_path: str, fmt: str = "PNG") -> None:
        buf = io.BytesIO()
        Image.fromarray(image).save(buf, format=fmt)
        self.put_bytes(buf.getvalue(), hdfs_path)

    def put_bytes(self, data: bytes, hdfs_path: str) -> None:  # ← переименовали параметр в 'data'
        t0 = time.time()
        try:
            fs = self._get_fs()
            self.mkdirs(str(Path(hdfs_path).parent))
            if fs is not None:
                with fs.open_output_stream(hdfs_path) as f:
                    f.write(data)  # ← теперь 'data' определена
                log.info("Written to HDFS: %s (%d bytes)", hdfs_path, len(data))
            else:
                tmp = Path(f"/tmp/hdfs_upload_{os.getpid()}_{Path(hdfs_path).name}")
                tmp.write_bytes(data)  # ← теперь 'data' определена
                try:
                    self._cli(["hdfs", "dfs", "-put", "-f", str(tmp), self._full_uri(hdfs_path)])
                finally:
                    tmp.unlink(missing_ok=True)
            self._log_access("write", hdfs_path, size_bytes=len(data), duration_ms=(time.time() - t0) * 1000, success=True)
        except Exception as e:
            self._log_access("write", hdfs_path, size_bytes=len(data),  # ← упростили, 'data' всегда определена
                           duration_ms=(time.time() - t0) * 1000, success=False, error=str(e))
            raise

    def put_local_file(self, local_path: Union[str, Path], hdfs_path: str) -> None:
        t0 = time.time()
        local_path = Path(local_path)
        size_bytes = local_path.stat().st_size if local_path.exists() else None
        try:
            fs = self._get_fs()
            self.mkdirs(str(Path(hdfs_path).parent))
            if fs is not None:
                with local_path.open("rb") as src, fs.open_output_stream(hdfs_path) as dst:
                    dst.write(src.read())
                log.info("Uploaded to HDFS: %s -> %s", local_path, hdfs_path)
            else:
                self._cli(["hdfs", "dfs", "-put", "-f", str(local_path), self._full_uri(hdfs_path)])
            self._log_access("write", hdfs_path, size_bytes=size_bytes, duration_ms=(time.time() - t0) * 1000, success=True)
        except Exception as e:
            self._log_access("write", hdfs_path, size_bytes=size_bytes,
                           duration_ms=(time.time() - t0) * 1000, success=False, error=str(e))
            raise

    def put_json(self, payload: Dict, hdfs_path: str) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.put_bytes(data, hdfs_path)

    def get_bytes(self, hdfs_path: str) -> bytes:
        t0 = time.time()
        try:
            fs = self._get_fs()
            if fs is not None:
                with fs.open_input_stream(hdfs_path) as f:
                    data = f.read()
            else:
                r = subprocess.run(
                    ["hdfs", "dfs", "-cat", self._full_uri(hdfs_path)],
                    capture_output=True,
                    check=True,
                )
                data = r.stdout
            self._log_access("read", hdfs_path, size_bytes=len(data), duration_ms=(time.time() - t0) * 1000, success=True)
            return data
        except Exception as e:
            self._log_access("read", hdfs_path, duration_ms=(time.time() - t0) * 1000, success=False, error=str(e))
            raise

    def _full_uri(self, hdfs_path: str) -> str:
        if hdfs_path.startswith("hdfs://"):
            return hdfs_path
        return f"{self.cfg.root}{hdfs_path}"

    @staticmethod
    def _cli(args: List[str]) -> None:
        log.debug("HDFS CLI: %s", " ".join(args))
        r = subprocess.run(args, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(
                f"HDFS CLI failed (rc={r.returncode}): {r.stderr or r.stdout}"
            )