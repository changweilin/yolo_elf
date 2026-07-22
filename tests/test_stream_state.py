import asyncio

from app.config import get_settings
from app.stream_state import StreamHub


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
    hub = StreamHub(get_settings())
    viewer = _FakeViewer()
    event = {"type": "alert", "rule": "person", "count": 2}

    async def run():
        await hub.add_viewer(viewer)
        await hub.broadcast_alert([event])

    asyncio.run(run())
    assert viewer.sent == [event]


def test_broadcast_alert_drops_dead_viewer():
    hub = StreamHub(get_settings())
    viewer = _FakeViewer(fail=True)

    async def run():
        await hub.add_viewer(viewer)
        await hub.broadcast_alert([{"type": "alert"}])
        return len(hub.viewer_clients)

    assert asyncio.run(run()) == 0


def test_broadcast_alert_noop_without_events():
    hub = StreamHub(get_settings())
    viewer = _FakeViewer()

    async def run():
        await hub.add_viewer(viewer)
        await hub.broadcast_alert([])

    asyncio.run(run())
    assert viewer.sent == []
