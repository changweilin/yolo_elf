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


class ZoneEngine:
    """Tags detection boxes with the ROI polygons they fall inside.

    Membership is tested in normalized frame coordinates so a zone follows the
    scene regardless of capture resolution. Pure CPU work with no I/O — the
    detection worker calls ``annotate`` inline; ``set_zones`` swaps the tuple
    reference atomically, so no lock is needed on the single event loop.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._zones = parse_zones(settings.zones_json)

    @property
    def enabled(self) -> bool:
        return bool(self._zones)

    def annotate(self, detection: dict[str, Any]) -> dict[str, Any]:
        """In place: add ``box['zones']`` and ``detection['zone_counts']``.

        No-op (zero added fields) when no zones are configured or the frame
        errored, so the feature has no cost when unused.
        """
        zones = self._zones
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

    def set_zones(self, data: Any) -> None:
        self._zones = build_zones(data)

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "zones": [zone.public() for zone in self._zones],
        }
