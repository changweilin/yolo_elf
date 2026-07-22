import json

import pytest

from app.config import get_settings
from app.zones import ZoneEngine, build_zones, parse_zones, point_in_polygon


# A unit square covering the top-left quadrant of a normalized frame.
SQUARE = ({"name": "tl", "points": [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]},)


def _engine(monkeypatch, zones=None):
    monkeypatch.delenv("ZONES", raising=False)
    if zones is not None:
        monkeypatch.setenv("ZONES", json.dumps(zones))
    return ZoneEngine(get_settings())


def _detection(boxes, width=100, height=100, error=None):
    detection = {"frame_id": 1, "width": width, "height": height, "boxes": boxes}
    if error is not None:
        detection["error"] = error
    return detection


# --- geometry -----------------------------------------------------------------


def test_point_in_polygon_inside_and_outside():
    square = ((0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5))
    assert point_in_polygon(0.25, 0.25, square) is True
    assert point_in_polygon(0.75, 0.25, square) is False


def test_point_in_polygon_handles_horizontal_edges():
    # A triangle with a flat bottom edge must not raise on the divide.
    triangle = ((0.0, 0.0), (1.0, 0.0), (0.5, 1.0))
    assert point_in_polygon(0.5, 0.3, triangle) is True
    assert point_in_polygon(0.9, 0.9, triangle) is False


# --- parsing / validation -----------------------------------------------------


def test_parse_zones_empty_is_disabled():
    assert parse_zones("") == ()


def test_parse_zones_reads_points_and_anchor():
    zones = parse_zones(
        json.dumps([{"name": "door", "anchor": "bottom", "points": [[0, 0], [1, 0], [1, 1]]}])
    )
    assert zones[0].name == "door"
    assert zones[0].anchor == "bottom"
    assert zones[0].points == ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))


@pytest.mark.parametrize(
    "raw",
    [
        "{not json",
        json.dumps({"name": "x"}),  # not an array
        json.dumps([{"points": [[0, 0], [1, 0], [1, 1]]}]),  # missing name
        json.dumps([{"name": "x", "points": [[0, 0], [1, 0]]}]),  # < 3 points
        json.dumps([{"name": "x", "points": [[0, 0], [1, 0], [2, 2]]}]),  # out of range
        json.dumps([{"name": "x", "points": [[0, 0], [1, 0], [1, 1]], "anchor": "middle"}]),
        json.dumps(
            [
                {"name": "dup", "points": [[0, 0], [1, 0], [1, 1]]},
                {"name": "dup", "points": [[0, 0], [1, 0], [1, 1]]},
            ]
        ),
    ],
)
def test_build_zones_rejects_invalid(raw):
    with pytest.raises(ValueError):
        parse_zones(raw)


# --- annotation ---------------------------------------------------------------


def test_annotate_tags_boxes_inside_zone(monkeypatch):
    engine = _engine(monkeypatch, list(SQUARE))
    # Centre at (25, 25) → normalized (0.25, 0.25), inside the top-left square.
    inside = {"xyxy": [10.0, 10.0, 40.0, 40.0], "label": "person"}
    # Centre at (75, 75) → outside.
    outside = {"xyxy": [60.0, 60.0, 90.0, 90.0], "label": "person"}
    detection = _detection([inside, outside])

    engine.annotate(detection)

    assert inside["zones"] == ["tl"]
    assert outside["zones"] == []
    assert detection["zone_counts"] == {"tl": 1}


def test_annotate_bottom_anchor_uses_foot_point(monkeypatch):
    # Box centre is inside the top-left square but its bottom edge is outside it.
    engine = _engine(
        monkeypatch,
        [{"name": "tl", "anchor": "bottom", "points": [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]}],
    )
    box = {"xyxy": [10.0, 10.0, 40.0, 80.0], "label": "person"}  # foot at y=80 → 0.8
    engine.annotate(_detection([box]))
    assert box["zones"] == []


def test_annotate_is_noop_without_zones(monkeypatch):
    engine = _engine(monkeypatch, None)
    box = {"xyxy": [10.0, 10.0, 40.0, 40.0], "label": "person"}
    detection = _detection([box])
    engine.annotate(detection)
    assert "zones" not in box
    assert "zone_counts" not in detection


def test_annotate_skips_error_frames(monkeypatch):
    engine = _engine(monkeypatch, list(SQUARE))
    detection = _detection([], error="bad frame")
    engine.annotate(detection)
    assert "zone_counts" not in detection


def test_set_zones_swaps_at_runtime(monkeypatch):
    engine = _engine(monkeypatch, None)
    assert engine.enabled is False
    engine.set_zones([{"name": "tl", "points": [[0, 0], [0.5, 0], [0.5, 0.5]]}])
    assert engine.enabled is True
    snapshot = engine.snapshot()
    assert snapshot["zones"][0]["name"] == "tl"
    assert snapshot["zones"][0]["anchor"] == "center"


def test_set_zones_rejects_invalid(monkeypatch):
    engine = _engine(monkeypatch, None)
    with pytest.raises(ValueError):
        engine.set_zones([{"points": [[0, 0], [1, 0], [1, 1]]}])
