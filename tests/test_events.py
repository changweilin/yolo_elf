import asyncio

from app.events import EventStore, SightingAggregator
from helpers import box as _box
from helpers import detection as _detection
from helpers import settings_from


def _store(monkeypatch, tmp_path, enabled=True):
    return EventStore(
        settings_from(
            monkeypatch,
            EVENT_LOG_ENABLED="1" if enabled else "0",
            EVENT_DB_PATH=tmp_path / "events.db",
        )
    )


# --- aggregation --------------------------------------------------------------


def test_observe_starts_and_updates_a_sighting():
    agg = SightingAggregator()
    agg.observe(_detection([_box("person", confidence=0.6, track_id=7)]), 1000.0)
    agg.observe(_detection([_box("person", confidence=0.8, track_id=7, zones=["door"])]), 1001.0)

    assert agg.active_count == 1
    finalized = agg.take_all()
    assert len(finalized) == 1
    sighting = finalized[0]
    assert sighting["track_id"] == 7
    assert sighting["first_seen"] == 1000.0
    assert sighting["last_seen"] == 1001.0
    assert sighting["frames"] == 2
    assert sighting["max_confidence"] == 0.8
    assert sighting["zones"] == ["door"]


def test_label_follows_highest_confidence_observation():
    agg = SightingAggregator()
    agg.observe(_detection([_box("cat", confidence=0.55, track_id=3)]), 1.0)
    agg.observe(_detection([_box("dog", confidence=0.91, track_id=3)]), 2.0)
    agg.observe(_detection([_box("fox", confidence=0.40, track_id=3)]), 3.0)

    assert agg.take_all()[0]["label"] == "dog"


def test_observe_skips_error_and_untracked_boxes():
    agg = SightingAggregator()
    agg.observe(_detection([_box("person", track_id=1)], error="bad frame"), 1.0)
    agg.observe(_detection([_box("person", track_id=None)]), 1.0)
    assert agg.active_count == 0


def test_take_expired_finalizes_only_idle_tracks():
    agg = SightingAggregator()
    agg.observe(_detection([_box("person", track_id=1)]), 1000.0)
    agg.observe(_detection([_box("person", track_id=2)]), 1050.0)

    expired = agg.take_expired(now=1055.0, expiry_sec=5.0)

    assert [s["track_id"] for s in expired] == [1]
    assert agg.active_count == 1  # track 2 is still fresh


# --- SQLite store -------------------------------------------------------------


def test_store_round_trip(monkeypatch, tmp_path):
    store = _store(monkeypatch, tmp_path)

    async def run():
        await store.write(
            [
                {
                    "track_id": 5,
                    "label": "person",
                    "first_seen": 1000.0,
                    "last_seen": 1004.0,
                    "frames": 12,
                    "max_confidence": 0.9,
                    "zones": ["door"],
                }
            ]
        )
        return await store.query(limit=10)

    events = asyncio.run(run())
    assert len(events) == 1
    assert events[0]["track_id"] == 5
    assert events[0]["dwell_sec"] == 4.0
    assert events[0]["zones"] == ["door"]
    assert events[0]["first_seen_iso"].startswith("1970-01-01")


def test_store_filters(monkeypatch, tmp_path):
    store = _store(monkeypatch, tmp_path)
    rows = [
        {"track_id": 1, "label": "person", "first_seen": 100.0, "last_seen": 105.0,
         "frames": 3, "max_confidence": 0.9, "zones": ["door"]},
        {"track_id": 2, "label": "dog", "first_seen": 200.0, "last_seen": 205.0,
         "frames": 3, "max_confidence": 0.8, "zones": ["yard"]},
        {"track_id": 3, "label": "person", "first_seen": 300.0, "last_seen": 305.0,
         "frames": 3, "max_confidence": 0.7, "zones": []},
    ]

    async def run(**filters):
        await store.write(rows)
        return await store.query(**filters)

    by_label = asyncio.run(run(label="person"))
    assert {e["track_id"] for e in by_label} == {1, 3}

    by_zone = asyncio.run(run(zone="door"))
    assert {e["track_id"] for e in by_zone} == {1}

    recent = asyncio.run(run(since=250.0))
    assert {e["track_id"] for e in recent} == {3}


def test_disabled_store_is_inert(monkeypatch, tmp_path):
    store = _store(monkeypatch, tmp_path, enabled=False)

    async def run():
        await store.write([{"track_id": 1, "label": "x", "first_seen": 1.0, "last_seen": 2.0,
                            "frames": 1, "max_confidence": 0.5, "zones": []}])
        return await store.query()

    assert asyncio.run(run()) == []
    assert not (tmp_path / "events.db").exists()
