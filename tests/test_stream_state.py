import asyncio

from app.config import get_settings
from app.stream_state import StreamRegistry


class _FakeViewer:
    """Minimal viewer websocket: records JSON sends, optionally fails."""

    def __init__(self, fail=False):
        self.sent = []
        self._fail = fail

    async def send_json(self, payload):
        if self._fail:
            raise RuntimeError("socket closed")
        self.sent.append(payload)


def test_broadcast_alert_sends_events_to_viewers():
    hub = StreamRegistry(get_settings())
    viewer = _FakeViewer()
    event = {"type": "alert", "rule": "person", "count": 2}

    async def run():
        await hub.add_viewer(viewer)
        await hub.broadcast_alert([event])

    asyncio.run(run())
    assert viewer.sent == [event]


def test_broadcast_alert_drops_dead_viewer():
    hub = StreamRegistry(get_settings())
    viewer = _FakeViewer(fail=True)

    async def run():
        await hub.add_viewer(viewer)
        await hub.broadcast_alert([{"type": "alert"}])
        return len(hub.viewer_clients)

    assert asyncio.run(run()) == 0


def test_broadcast_alert_noop_without_events():
    hub = StreamRegistry(get_settings())
    viewer = _FakeViewer()

    async def run():
        await hub.add_viewer(viewer)
        await hub.broadcast_alert([])

    asyncio.run(run())
    assert viewer.sent == []


def test_broadcast_vlm_sends_and_retains_latest():
    hub = StreamRegistry(get_settings())
    viewer = _FakeViewer()
    payload = {"type": "vlm", "camera_id": "default", "boxes": []}

    async def run():
        await hub.add_viewer(viewer)
        await hub.broadcast_vlm("default", payload)

    asyncio.run(run())
    assert viewer.sent == [payload]
    # Retained so a viewer joining later can be primed.
    assert hub.channels["default"].latest_vlm == payload


def test_broadcast_vlm_unknown_camera_is_noop():
    hub = StreamRegistry(get_settings())
    viewer = _FakeViewer()

    async def run():
        await hub.add_viewer(viewer)
        await hub.broadcast_vlm("ghost", {"type": "vlm"})

    asyncio.run(run())
    assert viewer.sent == []


def test_send_latest_primes_new_viewer_with_vlm():
    hub = StreamRegistry(get_settings())
    viewer = _FakeViewer()
    payload = {"type": "vlm", "camera_id": "default", "boxes": []}

    async def run():
        hub.channels["default"].latest_vlm = payload
        await hub.add_viewer(viewer)
        await hub.send_latest_to_viewer(viewer)

    asyncio.run(run())
    # No frames yet, so the only thing primed is the retained VLM payload.
    assert viewer.sent == [payload]
