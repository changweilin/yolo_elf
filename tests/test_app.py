import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from app.main import create_app


REMOTE_ENV = [
    "REMOTE_STORAGE_URL",
    "REMOTE_STORAGE_TOKEN",
    "REMOTE_STORAGE_INCLUDE_FRAME",
    "REMOTE_STORAGE_QUEUE_SIZE",
    "REMOTE_STORAGE_TIMEOUT",
    "REMOTE_STORAGE_RETRIES",
]


@pytest.fixture(autouse=True)
def clear_remote_env(monkeypatch):
    for name in REMOTE_ENV:
        monkeypatch.delenv(name, raising=False)


def test_health_and_pages_load():
    app = create_app()
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/phone").status_code == 200
        assert client.get("/viewer").status_code == 200


def test_root_redirects_to_phone_page():
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/", follow_redirects=False)

    assert response.status_code in {307, 308}
    assert response.headers["location"] == "/phone"


def test_phone_page_exposes_camera_toggle_action():
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/phone")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.text.count("Start camera") == 1
    assert 'id="cameraToggleButton"' in response.text
    assert "/static/phone.js?v=camera-toggle-2" in response.text
    assert "data-start-camera" not in response.text
    assert 'id="stopButton"' not in response.text


def test_viewer_links_to_phone_camera_page():
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/viewer")

    assert response.status_code == 200
    assert 'href="/phone"' in response.text
    assert "Open phone" in response.text


def test_status_includes_stream_metrics():
    app = create_app()
    with TestClient(app) as client:
        status = client.get("/api/status").json()

    assert status["queue_depth"] == 0
    assert status["receive_fps"] == 0.0
    assert status["process_fps"] == 0.0
    assert status["last_frame_bytes"] == 0
    assert status["avg_total_latency_ms"] == 0.0
    assert status["remote_storage"]["enabled"] is False


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


def test_viewer_receives_binary_frame_after_metadata():
    app = create_app()
    frame = b"not a jpeg"

    with TestClient(app) as client:
        with client.websocket_connect("/ws/viewer") as viewer:
            status = viewer.receive_json()
            assert status["type"] == "status"

            with client.websocket_connect("/ws/camera") as camera:
                config = camera.receive_json()
                assert config["type"] == "config"
                camera.send_bytes(frame)

                camera_message = camera.receive_json()
                viewer_metadata = viewer.receive_json()
                viewer_frame = viewer.receive_bytes()

    assert camera_message["type"] == "detection"
    assert viewer_metadata["type"] == "frame"
    assert viewer_metadata["transport"] == "binary"
    assert viewer_metadata["content_type"] == "image/jpeg"
    assert viewer_metadata["byte_length"] == len(frame)
    assert "jpeg" not in viewer_metadata
    assert viewer_frame == frame
