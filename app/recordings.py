from __future__ import annotations

import asyncio
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request

from app.config import Settings


_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_EXTENSIONS_BY_TYPE = {
    "video/webm": "webm",
    "video/mp4": "mp4",
    "video/quicktime": "mov",
    "application/octet-stream": "webm",
}
_MAX_DURATION_MS = 24 * 60 * 60 * 1000


@dataclass(frozen=True)
class RecordingRecord:
    recording_id: str
    filename: str
    path: Path
    content_type: str
    byte_length: int
    duration_ms: int | None
    started_at: str | None
    created_at: float

    def public_payload(self) -> dict[str, Any]:
        return {
            "id": self.recording_id,
            "filename": self.filename,
            "content_type": self.content_type,
            "byte_length": self.byte_length,
            "duration_ms": self.duration_ms,
            "started_at": self.started_at,
            "created_at": self.created_at,
            "created_at_iso": datetime.fromtimestamp(
                self.created_at, timezone.utc
            ).isoformat(),
            "download_url": f"/api/recordings/{self.recording_id}",
        }


class RecordingStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.recordings_saved = 0
        self.bytes_saved = 0
        self.last_recording_id: str | None = None
        self.last_saved_at: float | None = None
        self.last_error: str | None = None
        self._lock = asyncio.Lock()

    async def save_request(self, request: Request) -> RecordingRecord:
        if not self.settings.recording_enabled:
            raise HTTPException(status_code=403, detail="Recording is disabled")

        content_type = _clean_content_type(request.headers.get("content-type"))
        extension = _extension_for_content_type(content_type)
        duration_ms = _duration_header(request.headers.get("x-yolo-elf-duration-ms"))
        started_at = _short_header(request.headers.get("x-yolo-elf-started-at"))
        expected_length = _content_length(request.headers.get("content-length"))
        if expected_length is not None and expected_length > self.settings.recording_max_bytes:
            await self._remember_error("Recording is too large")
            raise HTTPException(status_code=413, detail="Recording is too large")

        created_at = time.time()
        recording_id = _recording_id(created_at)
        filename = f"{recording_id}.{extension}"
        storage_dir = self.settings.recording_storage_dir
        storage_dir.mkdir(parents=True, exist_ok=True)
        path = storage_dir / filename
        temp_path = storage_dir / f"{filename}.part"

        total_bytes = 0
        try:
            with temp_path.open("wb") as output:
                async for chunk in request.stream():
                    if not chunk:
                        continue
                    total_bytes += len(chunk)
                    if total_bytes > self.settings.recording_max_bytes:
                        raise HTTPException(status_code=413, detail="Recording is too large")
                    output.write(chunk)
            if total_bytes <= 0:
                raise HTTPException(status_code=400, detail="Recording body is empty")
            temp_path.replace(path)
        except Exception as exc:
            temp_path.unlink(missing_ok=True)
            await self._remember_error(str(exc))
            raise

        record = RecordingRecord(
            recording_id=recording_id,
            filename=filename,
            path=path,
            content_type=content_type,
            byte_length=total_bytes,
            duration_ms=duration_ms,
            started_at=started_at,
            created_at=created_at,
        )
        async with self._lock:
            self.recordings_saved += 1
            self.bytes_saved += total_bytes
            self.last_recording_id = recording_id
            self.last_saved_at = created_at
            self.last_error = None
        return record

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            return {
                "enabled": self.settings.recording_enabled,
                "storage_dir": str(self.settings.recording_storage_dir),
                "max_bytes": self.settings.recording_max_bytes,
                "recordings_saved": self.recordings_saved,
                "bytes_saved": self.bytes_saved,
                "last_recording_id": self.last_recording_id,
                "last_saved_at": self.last_saved_at,
                "last_error": self.last_error,
            }

    def resolve(self, recording_id: str) -> tuple[Path, str]:
        if not _SAFE_ID.fullmatch(recording_id):
            raise HTTPException(status_code=404, detail="Recording not found")

        storage_dir = self.settings.recording_storage_dir.resolve()
        if not storage_dir.exists():
            raise HTTPException(status_code=404, detail="Recording not found")

        for candidate in storage_dir.glob(f"{recording_id}.*"):
            path = candidate.resolve()
            if path.parent != storage_dir or not path.is_file():
                continue
            content_type = _content_type_for_extension(path.suffix.lower().lstrip("."))
            if content_type:
                return path, content_type

        raise HTTPException(status_code=404, detail="Recording not found")

    async def _remember_error(self, message: str) -> None:
        async with self._lock:
            self.last_error = message


def _recording_id(created_at: float) -> str:
    stamp = datetime.fromtimestamp(created_at, timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"rec-{stamp}-{uuid.uuid4().hex[:8]}"


def _clean_content_type(raw: str | None) -> str:
    if not raw:
        return "application/octet-stream"
    content_type = raw.split(";", 1)[0].strip().lower()
    return content_type or "application/octet-stream"


def _extension_for_content_type(content_type: str) -> str:
    return _EXTENSIONS_BY_TYPE.get(content_type, "webm")


def _content_type_for_extension(extension: str) -> str | None:
    for content_type, mapped_extension in _EXTENSIONS_BY_TYPE.items():
        if mapped_extension == extension and content_type != "application/octet-stream":
            return content_type
    return None


def _duration_header(raw: str | None) -> int | None:
    if raw is None or raw.strip() == "":
        return None
    try:
        duration_ms = int(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Recording duration is invalid") from exc
    if duration_ms < 0 or duration_ms > _MAX_DURATION_MS:
        raise HTTPException(status_code=400, detail="Recording duration is out of range")
    return duration_ms


def _content_length(raw: str | None) -> int | None:
    if raw is None or raw.strip() == "":
        return None
    try:
        length = int(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Content-Length is invalid") from exc
    if length < 0:
        raise HTTPException(status_code=400, detail="Content-Length is invalid")
    return length


def _short_header(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    return value[:80]
