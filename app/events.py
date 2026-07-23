from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.config import Settings

# Hard cap on how many rows a single /api/events query can return, regardless of
# the requested limit, so a huge history can't be pulled in one response.
QUERY_MAX = 500


@dataclass
class _ActiveSighting:
    track_id: int
    label: str
    first_seen: float
    last_seen: float
    frames: int
    max_confidence: float
    zones: set[str] = field(default_factory=set)
    camera_id: str | None = None


class SightingAggregator:
    """Folds per-frame tracked boxes into one row per object.

    Keyed by ``(camera_id, track_id)``: tracker ids restart per stream, so two
    cameras would otherwise collide and merge unrelated objects into one row.

    Pure in-memory bookkeeping — no I/O. ``observe`` is called on the hot path
    for every frame; ``take_expired`` hands finished sightings to the store once
    a track has gone quiet, so writes are decoupled from the frame rate.
    """

    def __init__(self) -> None:
        self._active: dict[tuple[str | None, int], _ActiveSighting] = {}

    @property
    def active_count(self) -> int:
        return len(self._active)

    def observe(
        self, detection: dict[str, Any], now: float, camera_id: str | None = None
    ) -> None:
        if detection.get("error"):
            return
        for box in detection.get("boxes") or []:
            track_id = box.get("track_id")
            if track_id is None:
                continue
            track_id = int(track_id)
            confidence = float(box.get("confidence") or 0.0)
            label = str(box.get("label", ""))
            zones = {str(zone) for zone in (box.get("zones") or [])}

            key = (camera_id, track_id)
            current = self._active.get(key)
            if current is None:
                self._active[key] = _ActiveSighting(
                    track_id=track_id,
                    label=label,
                    first_seen=now,
                    last_seen=now,
                    frames=1,
                    max_confidence=confidence,
                    zones=zones,
                    camera_id=camera_id,
                )
                continue

            current.last_seen = now
            current.frames += 1
            current.zones |= zones
            # Keep the label from the most confident observation — a track's
            # class can flicker frame to frame.
            if confidence > current.max_confidence:
                current.max_confidence = confidence
                current.label = label

    def take_expired(self, now: float, expiry_sec: float) -> list[dict[str, Any]]:
        expired_keys = [
            key
            for key, sighting in self._active.items()
            if now - sighting.last_seen > expiry_sec
        ]
        return [self._finalize(self._active.pop(key)) for key in expired_keys]

    def take_all(self) -> list[dict[str, Any]]:
        finalized = [self._finalize(sighting) for sighting in self._active.values()]
        self._active.clear()
        return finalized

    @staticmethod
    def _finalize(sighting: _ActiveSighting) -> dict[str, Any]:
        return {
            "camera_id": sighting.camera_id,
            "track_id": sighting.track_id,
            "label": sighting.label,
            "first_seen": sighting.first_seen,
            "last_seen": sighting.last_seen,
            "frames": sighting.frames,
            "max_confidence": round(sighting.max_confidence, 4),
            "zones": sorted(sighting.zones),
        }


class EventStore:
    """SQLite-backed sighting history with a simple filtered query.

    A fresh connection is opened per operation (all run off the event loop via
    ``asyncio.to_thread``), which sidesteps SQLite's thread-affinity without a
    shared connection or lock. Volume is one write per finished sighting, so the
    per-call connect cost is irrelevant.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.enabled = settings.event_log_enabled
        self.path = settings.event_db_path
        self.default_camera_id = settings.default_camera_id
        self._initialized = False
        self.sightings_written = 0

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_schema(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sightings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER,
                label TEXT,
                first_seen REAL,
                last_seen REAL,
                frames INTEGER,
                max_confidence REAL,
                zones TEXT,
                camera_id TEXT
            )
            """
        )
        # Databases written before multi-camera predate the column. Add it in
        # place rather than rebuilding the table; existing rows keep camera_id
        # NULL and are read back as the default camera.
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(sightings)").fetchall()
        }
        if "camera_id" not in columns:
            connection.execute("ALTER TABLE sightings ADD COLUMN camera_id TEXT")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_sightings_last_seen ON sightings(last_seen)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_sightings_camera ON sightings(camera_id)"
        )

    async def write(self, rows: list[dict[str, Any]]) -> None:
        if not self.enabled or not rows:
            return
        await asyncio.to_thread(self._write_sync, rows)

    def _write_sync(self, rows: list[dict[str, Any]]) -> None:
        if self.path.parent and not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            if not self._initialized:
                self._ensure_schema(connection)
                self._initialized = True
            connection.executemany(
                """
                INSERT INTO sightings
                    (track_id, label, first_seen, last_seen, frames, max_confidence,
                     zones, camera_id)
                VALUES (:track_id, :label, :first_seen, :last_seen, :frames, :max_confidence,
                        :zones, :camera_id)
                """,
                [
                    {
                        "camera_id": row.get("camera_id") or self.default_camera_id,
                        "track_id": row["track_id"],
                        "label": row["label"],
                        "first_seen": row["first_seen"],
                        "last_seen": row["last_seen"],
                        "frames": row["frames"],
                        "max_confidence": row["max_confidence"],
                        "zones": json.dumps(row.get("zones") or []),
                    }
                    for row in rows
                ],
            )
        self.sightings_written += len(rows)

    async def query(
        self,
        limit: int = 100,
        since: float | None = None,
        until: float | None = None,
        label: str | None = None,
        zone: str | None = None,
        camera_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        return await asyncio.to_thread(
            self._query_sync, limit, since, until, label, zone, camera_id
        )

    def _query_sync(
        self,
        limit: int,
        since: float | None,
        until: float | None,
        label: str | None,
        zone: str | None,
        camera_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        clauses: list[str] = []
        params: list[Any] = []
        if camera_id:
            if camera_id == self.default_camera_id:
                # Rows written before the column existed are NULL and belong to
                # the (then only) default camera.
                clauses.append("(camera_id = ? OR camera_id IS NULL)")
            else:
                clauses.append("camera_id = ?")
            params.append(camera_id)
        if since is not None:
            clauses.append("last_seen >= ?")
            params.append(since)
        if until is not None:
            clauses.append("first_seen <= ?")
            params.append(until)
        if label:
            clauses.append("label = ?")
            params.append(label)
        if zone:
            # zones is a JSON array of names; match the exact quoted token so
            # "door" doesn't also match "door-2".
            clauses.append("zones LIKE ?")
            params.append(f'%"{zone}"%')

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        bounded_limit = max(1, min(QUERY_MAX, int(limit)))
        params.append(bounded_limit)
        sql = f"SELECT * FROM sightings{where} ORDER BY first_seen DESC LIMIT ?"

        with self._connect() as connection:
            # A DB written by an older build has no camera_id column, which would
            # make the filter above fail; migrate on read too so /history works
            # against an existing file before the first new write lands.
            if not self._initialized:
                self._ensure_schema(connection)
                self._initialized = True
            rows = connection.execute(sql, params).fetchall()
        return [self._row_to_public(row) for row in rows]

    def _row_to_public(self, row: sqlite3.Row) -> dict[str, Any]:
        first_seen = row["first_seen"]
        last_seen = row["last_seen"]
        try:
            zones = json.loads(row["zones"]) if row["zones"] else []
        except (ValueError, TypeError):
            zones = []
        return {
            "camera_id": row["camera_id"] or self.default_camera_id,
            "track_id": row["track_id"],
            "label": row["label"],
            "first_seen": first_seen,
            "first_seen_iso": datetime.fromtimestamp(first_seen, timezone.utc).isoformat(),
            "last_seen": last_seen,
            "dwell_sec": round(max(0.0, last_seen - first_seen), 2),
            "frames": row["frames"],
            "max_confidence": row["max_confidence"],
            "zones": zones,
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "sightings_written": self.sightings_written,
        }
