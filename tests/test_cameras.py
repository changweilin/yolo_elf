"""Multi-camera unit coverage: config parsing, registry isolation, and the
camera_id dimension added to zones / alerts / events.

The single-camera path is covered by the pre-existing per-module tests; those
must stay green untouched, which is what proves multi-camera is an added
dimension rather than changed behaviour.
"""

import asyncio
import json
import sqlite3

import pytest

from app.alerts import AlertEngine
from app.config import DEFAULT_CAMERA_ID, parse_cameras
from app.events import EventStore, SightingAggregator
from app.stream_state import StreamRegistry
from app.zones import ZoneEngine
from helpers import FakeViewer as _FakeViewer
from helpers import box as _box
from helpers import detection as _detection
from helpers import settings_from as _settings


# --- config -------------------------------------------------------------------


def test_empty_cameras_yields_the_implicit_default(monkeypatch):
    settings = _settings(monkeypatch)

    assert settings.cameras == ((DEFAULT_CAMERA_ID, "Camera"),)
    assert settings.default_camera_id == DEFAULT_CAMERA_ID
    assert settings.multi_camera is False
    assert settings.max_cameras == 4


def test_cameras_parse_ids_and_display_names(monkeypatch):
    settings = _settings(monkeypatch, CAMERAS="front:前門, back:後院 , side")

    assert settings.cameras == (("front", "前門"), ("back", "後院"), ("side", "side"))
    assert settings.camera_ids == ("front", "back", "side")
    assert settings.default_camera_id == "front"
    assert settings.multi_camera is True


@pytest.mark.parametrize(
    "value",
    ["front,front", "front,ba d", "front,-lead", "front," + "x" * 33],
)
def test_cameras_reject_invalid_lists(value):
    with pytest.raises(ValueError, match="CAMERAS"):
        parse_cameras(value, 4)


def test_cameras_reject_more_than_max(monkeypatch):
    with pytest.raises(ValueError, match="MAX_CAMERAS"):
        _settings(monkeypatch, CAMERAS="a,b,c", MAX_CAMERAS="2")


# --- registry -----------------------------------------------------------------


def test_channels_keep_independent_counters(monkeypatch):
    registry = StreamRegistry(_settings(monkeypatch, CAMERAS="front,back"))

    async def run():
        await registry.submit_frame(b"a", "front")
        await registry.submit_frame(b"b", "front")  # replaces the queued frame
        await registry.submit_frame(b"c", "back")
        return registry.channels

    channels = asyncio.run(run())
    assert channels["front"].frames_received == 2
    assert channels["front"].frames_dropped == 1
    assert channels["back"].frames_received == 1
    assert channels["back"].frames_dropped == 0
    # Frame ids restart per camera, so an id is only unique within its stream.
    assert channels["back"].frame_queue.qsize() == 1


def test_ready_queue_serves_cameras_round_robin(monkeypatch):
    registry = StreamRegistry(_settings(monkeypatch, CAMERAS="front,back"))

    async def run():
        await registry.submit_frame(b"f1", "front")
        await registry.submit_frame(b"b1", "back")
        await registry.submit_frame(b"f2", "front")
        first = await registry.next_frame()
        second = await registry.next_frame()
        return first, second

    (first_id, first_frame), (second_id, second_frame) = asyncio.run(run())
    assert [first_id, second_id] == ["front", "back"]
    # A busy camera never queues twice: the newest frame wins, the older is gone.
    assert first_frame.jpeg == b"f2"
    assert second_frame.jpeg == b"b1"


def test_unknown_camera_id_is_rejected(monkeypatch):
    registry = StreamRegistry(_settings(monkeypatch, CAMERAS="front,back"))

    assert registry.resolve_camera_id(None) == "front"
    assert registry.resolve_camera_id("back") == "back"
    with pytest.raises(ValueError, match="Unknown camera_id"):
        registry.resolve_camera_id("garage")


def test_tracker_key_reuses_the_primary_model_for_the_first_camera(monkeypatch):
    registry = StreamRegistry(_settings(monkeypatch, CAMERAS="front,back"))

    assert registry.tracker_key("front") is None
    assert registry.tracker_key("back") == "back"


def test_viewer_only_receives_its_subscribed_camera(monkeypatch):
    registry = StreamRegistry(_settings(monkeypatch, CAMERAS="front,back"))
    watcher = _FakeViewer()
    everyone = _FakeViewer()

    async def run():
        await registry.add_viewer(watcher, "back")
        await registry.add_viewer(everyone)
        await registry.submit_frame(b"f1", "front")
        camera_id, frame = await registry.next_frame()
        await registry.publish_detection(frame, _detection([]), None, camera_id)

    asyncio.run(run())
    assert watcher.json == []
    assert [message["camera_id"] for message in everyone.json] == ["front"]
    assert everyone.binary == [b"f1"]


def test_alert_broadcast_respects_the_viewer_subscription(monkeypatch):
    registry = StreamRegistry(_settings(monkeypatch, CAMERAS="front,back"))
    watcher = _FakeViewer()

    async def run():
        await registry.add_viewer(watcher, "back")
        await registry.broadcast_alert([{"type": "alert", "camera_id": "front"}])
        await registry.broadcast_alert([{"type": "alert", "camera_id": "back"}])

    asyncio.run(run())
    assert [event["camera_id"] for event in watcher.json] == ["back"]


def test_snapshot_carries_per_camera_and_aggregate_numbers(monkeypatch):
    registry = StreamRegistry(_settings(monkeypatch, CAMERAS="front,back"))

    async def run():
        await registry.submit_frame(b"f1", "front")
        camera_id, frame = await registry.next_frame()
        await registry.publish_detection(frame, _detection([]), None, camera_id)
        await registry.submit_frame(b"b1", "back")
        return await registry.snapshot()

    status = asyncio.run(run())
    assert status["camera_ids"] == ["front", "back"]
    assert status["multi_camera"] is True
    assert status["frames_received"] == 2
    assert status["frames_processed"] == 1
    assert status["cameras"]["front"]["frames_processed"] == 1
    assert status["cameras"]["back"]["frames_processed"] == 0
    assert status["cameras"]["back"]["queue_depth"] == 1
    assert status["cameras"]["front"]["display_name"] == "front"


# --- zones --------------------------------------------------------------------


def test_zones_are_scoped_per_camera(monkeypatch):
    engine = ZoneEngine(
        _settings(
            monkeypatch,
            CAMERAS="front,back",
            ZONES=json.dumps(
                {
                    "front": [
                        {"name": "door", "points": [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]]}
                    ],
                    "back": [
                        {"name": "yard", "points": [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]]}
                    ],
                }
            ),
        )
    )

    front = engine.annotate(_detection([_box(track_id=1)]), "front")
    back = engine.annotate(_detection([_box(track_id=1)]), "back")

    assert front["boxes"][0]["zones"] == ["door"]
    assert back["boxes"][0]["zones"] == ["yard"]
    assert engine.snapshot("back")["zones"][0]["name"] == "yard"
    assert set(engine.snapshot()["cameras"]) == {"front", "back"}


def test_zones_entries_can_name_their_camera(monkeypatch):
    engine = ZoneEngine(
        _settings(
            monkeypatch,
            CAMERAS="front,back",
            ZONES=json.dumps(
                [
                    {"name": "door", "points": [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]]},
                    {
                        "camera_id": "back",
                        "name": "yard",
                        "points": [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]],
                    },
                ]
            ),
        )
    )

    # No camera_id = the default (first) camera, i.e. the legacy ZONES meaning.
    assert [zone.name for zone in engine.zones_for("front")] == ["door"]
    assert [zone.name for zone in engine.zones_for("back")] == ["yard"]


def test_set_zones_only_replaces_the_named_camera(monkeypatch):
    engine = ZoneEngine(_settings(monkeypatch, CAMERAS="front,back"))

    engine.set_zones([{"name": "door", "points": [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]]}], "front")
    engine.set_zones([{"name": "yard", "points": [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]]}], "back")
    engine.set_zones([], "back")

    assert [zone.name for zone in engine.zones_for("front")] == ["door"]
    assert engine.zones_for("back") == ()
    assert engine.enabled is True


# --- alerts -------------------------------------------------------------------


def test_alert_cooldown_is_per_camera(monkeypatch):
    engine = AlertEngine(
        _settings(
            monkeypatch,
            CAMERAS="front,back",
            ALERT_RULES=json.dumps([{"name": "person", "classes": ["person"]}]),
            ALERT_COOLDOWN_SEC="3600",
        )
    )
    detection = _detection([_box(track_id=1)])

    async def run():
        front = await engine.process(None, detection, "front")
        back = await engine.process(None, detection, "back")
        front_again = await engine.process(None, detection, "front")
        return front, back, front_again

    front, back, front_again = asyncio.run(run())
    # A fire on one camera must not silence the same rule on another.
    assert [event["camera_id"] for event in front] == ["front"]
    assert [event["camera_id"] for event in back] == ["back"]
    assert front_again == []


def test_alert_rule_can_be_pinned_to_one_camera(monkeypatch):
    engine = AlertEngine(
        _settings(
            monkeypatch,
            CAMERAS="front,back",
            ALERT_RULES=json.dumps(
                [{"name": "yard-only", "classes": ["person"], "camera_id": "back"}]
            ),
        )
    )
    detection = _detection([_box(track_id=1)])

    async def run():
        return (
            await engine.process(None, detection, "front"),
            await engine.process(None, detection, "back"),
        )

    front, back = asyncio.run(run())
    assert front == []
    assert [event["rule"] for event in back] == ["yard-only"]


# --- events -------------------------------------------------------------------


def test_sightings_do_not_collide_across_cameras():
    aggregator = SightingAggregator()

    aggregator.observe(_detection([_box("person", track_id=1)]), 100.0, "front")
    aggregator.observe(_detection([_box("dog", track_id=1)]), 100.0, "back")

    assert aggregator.active_count == 2
    rows = sorted(aggregator.take_all(), key=lambda row: row["camera_id"])
    assert [(row["camera_id"], row["label"]) for row in rows] == [
        ("back", "dog"),
        ("front", "person"),
    ]


def test_event_store_filters_by_camera(monkeypatch, tmp_path):
    monkeypatch.setenv("EVENT_LOG_ENABLED", "1")
    monkeypatch.setenv("EVENT_DB_PATH", str(tmp_path / "events.db"))
    store = EventStore(_settings(monkeypatch, CAMERAS="front,back"))

    async def run():
        await store.write(
            [
                {"camera_id": "front", "track_id": 1, "label": "person", "first_seen": 1.0,
                 "last_seen": 2.0, "frames": 3, "max_confidence": 0.9, "zones": []},
                {"camera_id": "back", "track_id": 1, "label": "dog", "first_seen": 3.0,
                 "last_seen": 4.0, "frames": 3, "max_confidence": 0.8, "zones": []},
            ]
        )
        return await store.query(camera_id="back"), await store.query()

    back, everything = asyncio.run(run())
    assert [row["label"] for row in back] == ["dog"]
    assert {row["camera_id"] for row in everything} == {"front", "back"}


def test_event_store_migrates_a_pre_camera_database(monkeypatch, tmp_path):
    """A DB written before multi-camera gains the column and reads as default."""
    db_path = tmp_path / "events.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE sightings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER, label TEXT, first_seen REAL, last_seen REAL,
                frames INTEGER, max_confidence REAL, zones TEXT
            )
            """
        )
        connection.execute(
            "INSERT INTO sightings (track_id, label, first_seen, last_seen, frames,"
            " max_confidence, zones) VALUES (7, 'person', 1.0, 2.0, 5, 0.9, '[]')"
        )

    monkeypatch.setenv("EVENT_LOG_ENABLED", "1")
    monkeypatch.setenv("EVENT_DB_PATH", str(db_path))
    store = EventStore(_settings(monkeypatch, CAMERAS="front,back"))

    rows = asyncio.run(store.query(camera_id="front"))
    assert [row["track_id"] for row in rows] == [7]
    # The legacy row had no camera; it belongs to the (then only) default camera.
    assert rows[0]["camera_id"] == "front"
    with sqlite3.connect(db_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(sightings)")}
    assert "camera_id" in columns


def test_metrics_count_zones_across_every_camera(monkeypatch):
    from app.metrics import render_metrics

    engine = ZoneEngine(_settings(monkeypatch, CAMERAS="front,back"))
    triangle = [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]]
    engine.set_zones([{"name": "door", "points": triangle}], "front")
    engine.set_zones(
        [{"name": "yard", "points": triangle}, {"name": "drive", "points": triangle}],
        "back",
    )

    body = render_metrics({"zones": engine.snapshot()})
    assert "yolo_elf_zones 3" in body
