"""Fixtures-free builders shared by the per-feature test modules.

These were copy-pasted into test_alerts / test_cameras / test_events /
test_zones with small divergences; the divergences were accidental, not
meaningful, so they live here once.
"""

from app.config import get_settings


def settings_from(monkeypatch, **env):
    """`get_settings()` with only the given variables set.

    Everything else is already cleared by the autouse fixture in conftest.
    """
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    return get_settings()


def detection(boxes, *, frame_id=1, width=100, height=100, error=None, **extra):
    """A detection payload as the worker broadcasts it."""
    payload = {
        "frame_id": frame_id,
        "width": width,
        "height": height,
        "boxes": boxes,
        **extra,
    }
    if error is not None:
        payload["error"] = error
    return payload


def box(label="person", confidence=0.9, track_id=None, xyxy=None, zones=None):
    """One detection box. Defaults are a valid, in-frame, tracked person."""
    return {
        "label": label,
        "confidence": confidence,
        "track_id": track_id,
        "xyxy": list(xyxy) if xyxy is not None else [10.0, 10.0, 30.0, 30.0],
        "zones": list(zones) if zones is not None else [],
    }


class FakeViewer:
    """Minimal viewer websocket: records JSON payloads and binary frames."""

    def __init__(self):
        self.json = []
        self.binary = []

    async def send_json(self, payload):
        self.json.append(payload)

    async def send_bytes(self, payload):
        self.binary.append(payload)
