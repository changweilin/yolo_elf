from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.config import Settings

ANCHORS = ("center", "bottom")


def point_in_polygon(x: float, y: float, points: tuple[tuple[float, float], ...]) -> bool:
    """Ray-casting point-in-polygon test on normalized coordinates.

    The ``(yi > y) != (yj > y)`` guard guarantees ``yj != yi`` before the divide,
    so horizontal edges never raise ZeroDivisionError.
    """
    inside = False
    count = len(points)
    j = count - 1
    for i in range(count):
        xi, yi = points[i]
        xj, yj = points[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


@dataclass(frozen=True)
class Zone:
    name: str
    points: tuple[tuple[float, float], ...]
    anchor: str

    def anchor_point(self, xyxy: list[float]) -> tuple[float, float]:
        """Return the pixel point tested for membership: box centre or foot."""
        x1, y1, x2, y2 = xyxy
        cx = (x1 + x2) / 2.0
        if self.anchor == "bottom":
            return cx, y2
        return cx, (y1 + y2) / 2.0

    def public(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "points": [list(point) for point in self.points],
            "anchor": self.anchor,
        }


def _parse_zone(item: Any, index: int) -> Zone:
    if not isinstance(item, dict):
        raise ValueError(f"zone #{index} must be an object")

    name = str(item.get("name") or "").strip()
    if not name:
        raise ValueError(f"zone #{index} requires a non-empty name")

    raw_points = item.get("points")
    if not isinstance(raw_points, (list, tuple)) or len(raw_points) < 3:
        raise ValueError(f"zone {name!r}: points must be a list of at least 3 [x, y] pairs")

    points: list[tuple[float, float]] = []
    for pair in raw_points:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"zone {name!r}: each point must be an [x, y] pair")
        try:
            x = float(pair[0])
            y = float(pair[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"zone {name!r}: point coordinates must be numbers") from exc
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"zone {name!r}: point coordinates must be normalized to 0..1")
        points.append((x, y))

    anchor = str(item.get("anchor") or "center").strip().lower()
    if anchor not in ANCHORS:
        raise ValueError(f"zone {name!r}: anchor must be one of {', '.join(ANCHORS)}")

    return Zone(name=name, points=tuple(points), anchor=anchor)


def build_zones(data: Any) -> tuple[Zone, ...]:
    """Validate a decoded zones list into ``Zone`` objects (fail loud)."""
    if not isinstance(data, list):
        raise ValueError("zones must be a JSON array of zone objects")
    zones = tuple(_parse_zone(item, index) for index, item in enumerate(data))
    names = [zone.name for zone in zones]
    if len(names) != len(set(names)):
        raise ValueError("zone names must be unique")
    return zones


def parse_zones(raw: str) -> tuple[Zone, ...]:
    raw = (raw or "").strip()
    if not raw:
        return ()
    try:
        data = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"ZONES must be valid JSON: {exc}") from exc
    return build_zones(data)


def build_camera_zones(data: Any, default_camera_id: str) -> dict[str, tuple[Zone, ...]]:
    """Validate a decoded ``ZONES`` value into per-camera zone lists.

    Two shapes are accepted: a bare array (all zones belong to the default
    camera, unless an entry carries its own ``camera_id``), or an object mapping
    ``camera_id`` to an array. The bare array is the pre-multi-camera format, so
    an existing ``ZONES`` value keeps meaning exactly what it meant before.
    """
    if isinstance(data, dict):
        return {
            str(camera_id): build_zones(items) for camera_id, items in data.items()
        }
    if not isinstance(data, list):
        raise ValueError("zones must be a JSON array of zone objects, or a camera map")

    grouped: dict[str, list[Any]] = {}
    for item in data:
        camera_id = default_camera_id
        if isinstance(item, dict) and item.get("camera_id"):
            camera_id = str(item["camera_id"]).strip() or default_camera_id
        grouped.setdefault(camera_id, []).append(item)
    return {camera_id: build_zones(items) for camera_id, items in grouped.items()}


def parse_camera_zones(raw: str, default_camera_id: str) -> dict[str, tuple[Zone, ...]]:
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"ZONES must be valid JSON: {exc}") from exc
    return build_camera_zones(data, default_camera_id)


class ZoneEngine:
    """Tags detection boxes with the ROI polygons they fall inside.

    Membership is tested in normalized frame coordinates so a zone follows the
    scene regardless of capture resolution. Zones are per-camera: two recorders
    point at different physical places, so a shared polygon list would be
    meaningless. Pure CPU work with no I/O — the detection worker calls
    ``annotate`` inline; ``set_zones`` swaps a tuple reference atomically, so no
    lock is needed on the single event loop.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.default_camera_id = settings.default_camera_id
        self._zones_by_camera = parse_camera_zones(
            settings.zones_json, self.default_camera_id
        )

    @property
    def enabled(self) -> bool:
        return any(self._zones_by_camera.values())

    def zones_for(self, camera_id: str | None = None) -> tuple[Zone, ...]:
        key = camera_id or self.default_camera_id
        return self._zones_by_camera.get(key, ())

    def annotate(
        self, detection: dict[str, Any], camera_id: str | None = None
    ) -> dict[str, Any]:
        """In place: add ``box['zones']`` and ``detection['zone_counts']``.

        No-op (zero added fields) when the camera has no zones or the frame
        errored, so the feature has no cost when unused.
        """
        zones = self.zones_for(camera_id)
        if not zones or detection.get("error"):
            return detection

        width = detection.get("width") or 0
        height = detection.get("height") or 0
        counts = {zone.name: 0 for zone in zones}
        for box in detection.get("boxes") or []:
            names: list[str] = []
            xyxy = box.get("xyxy")
            if xyxy and width and height:
                for zone in zones:
                    px, py = zone.anchor_point(xyxy)
                    if point_in_polygon(px / width, py / height, zone.points):
                        names.append(zone.name)
                        counts[zone.name] += 1
            box["zones"] = names
        detection["zone_counts"] = counts
        return detection

    def set_zones(self, data: Any, camera_id: str | None = None) -> None:
        """Replace one camera's zone list (fail loud on an invalid polygon)."""
        key = camera_id or self.default_camera_id
        self._zones_by_camera = {**self._zones_by_camera, key: build_zones(data)}

    def snapshot(self, camera_id: str | None = None) -> dict[str, Any]:
        # `zones` stays the flat single-camera view (the requested camera, or the
        # default) so existing readers — /api/status, /metrics, viewer.js — keep
        # working; `cameras` carries the full per-camera picture.
        return {
            "enabled": self.enabled,
            "camera_id": camera_id or self.default_camera_id,
            "zones": [zone.public() for zone in self.zones_for(camera_id)],
            "cameras": {
                key: [zone.public() for zone in zones]
                for key, zones in self._zones_by_camera.items()
            },
        }
