import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from app.main import create_app


def test_health_and_pages_load():
    app = create_app()
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/phone").status_code == 200
        assert client.get("/viewer").status_code == 200


def test_status_includes_stream_metrics():
    app = create_app()
    with TestClient(app) as client:
        status = client.get("/api/status").json()

    assert status["queue_depth"] == 0
    assert status["receive_fps"] == 0.0
    assert status["process_fps"] == 0.0
    assert status["last_frame_bytes"] == 0
    assert status["avg_total_latency_ms"] == 0.0


def test_camera_websocket_returns_detection_error_for_invalid_jpeg():
    app = create_app()
    with TestClient(app) as client:
        with client.websocket_connect("/ws/camera") as websocket:
            config = websocket.receive_json()
            assert config["type"] == "config"
            websocket.send_bytes(b"not a jpeg")
            message = websocket.receive_json()

    assert message["type"] == "detection"
    detection = message["detection"]
    assert detection["boxes"] == []
    assert detection["error"]
