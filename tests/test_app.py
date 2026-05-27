import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from app.main import create_app


REMOTE_ENV = [
    "RECORDING_ENABLED",
    "RECORDING_STORAGE_DIR",
    "RECORDING_MAX_BYTES",
    "REMOTE_STORAGE_URL",
    "REMOTE_STORAGE_TOKEN",
    "REMOTE_STORAGE_INCLUDE_FRAME",
    "REMOTE_STORAGE_RECORDING_URL",
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
    assert "Start camera" not in response.text
    assert 'id="cameraToggleButton"' in response.text
    assert 'id="recordButton"' in response.text
    assert 'id="storageModeGroup"' in response.text
    assert 'data-storage-mode="local"' in response.text
    assert 'data-storage-mode="remote"' in response.text
    assert 'data-storage-mode="both"' in response.text
    assert 'id="settingsToggleButton"' in response.text
    assert 'id="advancedControls"' in response.text
    assert 'id="statusRow"' in response.text
    assert 'id="recordingStatus"' in response.text
    assert 'id="lensToggleButton"' in response.text
    assert 'id="lensSelect"' not in response.text
    assert 'id="zoomInput"' in response.text
    assert 'id="shutterInput"' in response.text
    assert 'id="isoInput"' in response.text
    assert "/static/phone.js?v=recording-2" in response.text
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
    assert status["recordings"]["enabled"] is True
    assert status["recordings"]["storage_modes"] == ["remote", "local", "both"]
    assert status["recordings"]["recordings_saved"] == 0
    assert status["remote_storage"]["enabled"] is False


def test_camera_websocket_returns_detection_error_for_invalid_jpeg():
    app = create_app()
    with TestClient(app) as client:
        with client.websocket_connect("/ws/camera") as websocket:
            config = websocket.receive_json()
            assert config["type"] == "config"
            assert config["recording"]["enabled"] is True
            assert config["recording"]["storage_modes"] == ["remote", "local", "both"]
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
                assert config["recording"]["max_bytes"] > 0
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


def test_recording_upload_saves_file_and_returns_download(tmp_path, monkeypatch):
    monkeypatch.setenv("RECORDING_STORAGE_DIR", str(tmp_path))
    app = create_app()
    body = b"webm-bytes"

    with TestClient(app) as client:
        response = client.post(
            "/api/recordings",
            content=body,
            headers={
                "content-type": "video/webm",
                "x-yolo-elf-duration-ms": "1234",
                "x-yolo-elf-started-at": "2026-05-27T08:00:00.000Z",
            },
        )
        payload = response.json()
        download = client.get(payload["recording"]["download_url"])
        status = client.get("/api/status").json()

    assert response.status_code == 200
    assert payload["type"] == "recording"
    assert payload["recording"]["content_type"] == "video/webm"
    assert payload["recording"]["byte_length"] == len(body)
    assert payload["recording"]["duration_ms"] == 1234
    assert payload["recording"]["storage_mode"] == "local"
    assert payload["recording"]["local_saved"] is True
    assert payload["remote_storage"]["status"] == "skipped"
    assert download.status_code == 200
    assert download.content == body
    assert status["recordings"]["recordings_saved"] == 1
    assert list(tmp_path.glob("*.webm"))


def test_recording_upload_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("RECORDING_STORAGE_DIR", str(tmp_path))
    monkeypatch.setenv("RECORDING_ENABLED", "0")
    app = create_app()

    with TestClient(app) as client:
        response = client.post(
            "/api/recordings",
            content=b"webm-bytes",
            headers={"content-type": "video/webm"},
        )

    assert response.status_code == 403


def test_remote_recording_mode_requires_remote_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("RECORDING_STORAGE_DIR", str(tmp_path))
    app = create_app()

    with TestClient(app) as client:
        response = client.post(
            "/api/recordings",
            content=b"webm-bytes",
            headers={
                "content-type": "video/webm",
                "x-yolo-elf-storage-mode": "remote",
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Remote recording storage is not configured"


def test_invalid_recording_storage_mode_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("RECORDING_STORAGE_DIR", str(tmp_path))
    app = create_app()

    with TestClient(app) as client:
        response = client.post(
            "/api/recordings",
            content=b"webm-bytes",
            headers={
                "content-type": "video/webm",
                "x-yolo-elf-storage-mode": "cloudish",
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Recording storage mode is invalid"
