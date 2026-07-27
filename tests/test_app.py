import asyncio
import json

import pytest

pytest.importorskip("fastapi")

from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app
from app.recordings import RecordingStore


REMOTE_ENV = [
    "RECORDING_ENABLED",
    "RECORDING_KEEP_LOCAL_COPY",
    "RECORDING_STORAGE_DIR",
    "RECORDING_MAX_BYTES",
    "REMOTE_STORAGE_URL",
    "REMOTE_STORAGE_TOKEN",
    "REMOTE_STORAGE_INCLUDE_FRAME",
    "REMOTE_STORAGE_RECORDING_URL",
    "REMOTE_STORAGE_QUEUE_SIZE",
    "REMOTE_STORAGE_TIMEOUT",
    "REMOTE_STORAGE_RETRIES",
    "ALERT_RULES",
    "ALERT_WEBHOOK_URL",
    "ALERT_WEBHOOK_TOKEN",
    "ALERT_COOLDOWN_SEC",
    "ZONES",
    "EVENT_LOG_ENABLED",
    "EVENT_DB_PATH",
    "EVENT_EXPIRY_SEC",
    "METRICS_ENABLED",
    "AUTH_TOKEN",
    "AUTH_SESSION_TTL",
    "CAMERAS",
    "MAX_CAMERAS",
]


@pytest.fixture(autouse=True)
def clear_remote_env(monkeypatch):
    for name in REMOTE_ENV:
        monkeypatch.delenv(name, raising=False)
    # Event logging defaults on and writes a SQLite file; keep it off unless a
    # test opts in with a tmp path, so the suite never touches the repo root.
    monkeypatch.setenv("EVENT_LOG_ENABLED", "0")


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
    # remote/local are independent highlight toggles now; "both" is the derived
    # state when both are lit, so there is no dedicated "both" button.
    assert 'data-storage-mode="both"' not in response.text
    assert 'id="fileLoadInput"' in response.text
    assert 'id="filePlayer"' in response.text
    assert 'id="settingsToggleButton"' in response.text
    assert 'id="advancedControls"' in response.text
    assert 'id="statusRow"' in response.text
    assert 'id="recordingStatus"' in response.text
    assert 'id="lensToggleButton"' in response.text
    assert 'id="lensSelect"' not in response.text
    assert 'id="zoomInput"' in response.text
    assert 'id="shutterInput"' in response.text
    assert 'id="isoInput"' in response.text
    assert 'id="computeStatus"' in response.text
    assert 'id="modeGroup"' in response.text
    assert 'data-detect-mode="fast"' in response.text
    assert 'data-detect-mode="accurate"' in response.text
    assert "/static/phone.js?v=multicam-1" in response.text
    assert "data-start-camera" not in response.text
    assert 'id="stopButton"' not in response.text


def test_viewer_has_role_switch_and_recorder_link():
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/viewer")

    assert response.status_code == 200
    assert 'class="role-switch"' in response.text
    assert 'href="/recorder"' in response.text
    assert 'href="/viewer" aria-current="page"' in response.text
    assert "Open recorder" in response.text


def test_viewer_exposes_detection_mode_switch():
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/viewer")

    assert response.status_code == 200
    assert 'id="modeGroup"' in response.text
    assert 'data-detect-mode="fast"' in response.text
    assert 'data-detect-mode="accurate"' in response.text


def test_detector_mode_can_be_switched():
    app = create_app()
    with TestClient(app) as client:
        detector = client.get("/api/status").json()["detector"]
        assert detector["available_modes"] == ["fast", "accurate"]
        assert detector["model"] == detector["models"][detector["mode"]]

        response = client.post("/api/detector/mode", json={"mode": "accurate"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["type"] == "detector_mode"
        assert payload["detector"]["mode"] == "accurate"
        assert payload["detector"]["model"] == payload["detector"]["models"]["accurate"]

        back = client.post("/api/detector/mode", json={"mode": "fast"})
        assert back.json()["detector"]["mode"] == "fast"
        assert client.get("/api/status").json()["detector"]["mode"] == "fast"


def test_detector_mode_rejects_invalid_mode():
    app = create_app()
    with TestClient(app) as client:
        response = client.post("/api/detector/mode", json={"mode": "ultra"})

    assert response.status_code == 400
    assert response.json()["detail"] == "Detection mode is invalid"


def test_settings_page_loads():
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/settings")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert 'id="settingsForm"' in response.text
    assert 'class="role-switch"' in response.text


def test_detector_config_round_trip():
    app = create_app()
    with TestClient(app) as client:
        current = client.get("/api/detector/config").json()
        assert current["type"] == "detector_config"

        response = client.post(
            "/api/detector/config",
            json={
                "mode": "accurate",
                "fast_model": "yolo11n.pt",
                "accurate_model": "yolo11x.pt",
                "classes": "person, dog",
                "conf_thresh": 0.4,
                "img_size": 960,
            },
        )
        assert response.status_code == 200
        detector = response.json()["detector"]
        assert detector["mode"] == "accurate"
        assert detector["models"] == {"fast": "yolo11n.pt", "accurate": "yolo11x.pt"}
        assert detector["configured_classes"] == ["person", "dog"]
        assert detector["conf_thresh"] == 0.4
        assert detector["img_size"] == 960

        # Changes persist for later reads of the shared detector.
        assert client.get("/api/status").json()["detector"]["img_size"] == 960


def test_detector_config_rejects_invalid_values():
    app = create_app()
    with TestClient(app) as client:
        response = client.post("/api/detector/config", json={"conf_thresh": 2})

    assert response.status_code == 400
    assert "conf_thresh" in response.json()["detail"]


def test_recorder_route_serves_capture_page():
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/recorder")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert 'id="cameraToggleButton"' in response.text
    assert 'class="role-switch"' in response.text
    assert 'href="/viewer"' in response.text


def test_recorder_page_marks_recorder_tab_current():
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/phone")

    assert 'href="/recorder" aria-current="page"' in response.text


def test_status_includes_stream_metrics():
    app = create_app()
    with TestClient(app) as client:
        status = client.get("/api/status").json()

    assert status["queue_depth"] == 0
    assert status["receive_fps"] == 0.0
    assert status["process_fps"] == 0.0
    assert status["last_frame_bytes"] == 0
    assert status["avg_total_latency_ms"] == 0.0
    assert status["camera_storage_mode"] is None
    assert status["camera_recording"] is False
    assert status["recordings"]["enabled"] is True
    assert status["recordings"]["storage_modes"] == ["remote", "local", "both"]
    assert status["recordings"]["recordings_saved"] == 0
    assert status["remote_storage"]["enabled"] is False
    assert status["alerts"]["enabled"] is False
    assert status["alerts"]["rules"] == []
    assert status["zones"]["enabled"] is False
    assert status["zones"]["zones"] == []
    assert status["events"]["enabled"] is False
    assert status["events"]["active_sightings"] == 0


def test_metrics_endpoint_exposes_prometheus():
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "version=0.0.4" in response.headers["content-type"]
    body = response.text
    assert "# TYPE yolo_elf_frames_processed_total counter" in body
    assert "yolo_elf_viewers " in body
    assert "yolo_elf_detector_info{" in body


def test_metrics_can_be_disabled(monkeypatch):
    monkeypatch.setenv("METRICS_ENABLED", "0")
    app = create_app()
    with TestClient(app) as client:
        assert client.get("/metrics").status_code == 404


def test_auth_disabled_by_default():
    app = create_app()
    with TestClient(app) as client:
        assert client.get("/api/status").status_code == 200
        assert client.get("/viewer").status_code == 200
        # login page is always served (harmless when auth is off)
        assert client.get("/login").status_code == 200


def test_auth_blocks_unauthenticated_requests(monkeypatch):
    monkeypatch.setenv("AUTH_TOKEN", "secret")
    app = create_app()
    with TestClient(app) as client:
        # API clients get a 401 they can act on
        assert client.get("/api/status").status_code == 401
        assert client.get("/metrics").status_code == 401
        # exempt paths stay reachable so the login flow can bootstrap
        assert client.get("/health").status_code == 200
        assert client.get("/login").status_code == 200
        assert client.get("/static/app.css").status_code == 200
        # a browser navigation is redirected to the login page
        page = client.get(
            "/viewer", headers={"accept": "text/html"}, follow_redirects=False
        )
        assert page.status_code in {307, 308}
        assert page.headers["location"].startswith("/login")


def test_auth_bearer_token_is_accepted(monkeypatch):
    monkeypatch.setenv("AUTH_TOKEN", "secret")
    app = create_app()
    with TestClient(app) as client:
        headers = {"authorization": "Bearer secret"}
        assert client.get("/api/status", headers=headers).status_code == 200
        assert client.get("/metrics", headers=headers).status_code == 200


def test_auth_login_sets_cookie_and_unlocks(monkeypatch):
    monkeypatch.setenv("AUTH_TOKEN", "secret")
    app = create_app()
    with TestClient(app) as client:
        assert client.post("/api/login", json={"token": "nope"}).status_code == 401
        assert client.get("/api/status").status_code == 401

        assert client.post("/api/login", json={"token": "secret"}).status_code == 200
        # the session cookie now rides along automatically
        assert client.get("/api/status").status_code == 200

        client.post("/api/logout")
        assert client.get("/api/status").status_code == 401


def test_auth_gates_websocket(monkeypatch):
    monkeypatch.setenv("AUTH_TOKEN", "secret")
    app = create_app()
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws/viewer"):
                pass
        with client.websocket_connect("/ws/viewer?token=secret") as ws:
            message = ws.receive_json()
            assert message["type"] == "status"


def test_alerts_default_to_disabled():
    app = create_app()
    with TestClient(app) as client:
        payload = client.get("/api/alerts").json()

    assert payload["type"] == "alerts"
    assert payload["alerts"]["enabled"] is False
    assert payload["alerts"]["webhook_enabled"] is False
    assert payload["alerts"]["rules"] == []


def test_alerts_rules_round_trip():
    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/api/alerts",
            json={"rules": [{"name": "person", "classes": ["person"], "min_count": 2}]},
        )
        assert response.status_code == 200
        alerts = response.json()["alerts"]
        assert alerts["enabled"] is True
        assert alerts["rules"][0]["name"] == "person"
        assert alerts["rules"][0]["classes"] == ["person"]
        assert alerts["rules"][0]["min_count"] == 2

        # The runtime change is reflected in the shared status snapshot.
        assert client.get("/api/status").json()["alerts"]["enabled"] is True


def test_alerts_rejects_invalid_rules():
    app = create_app()
    with TestClient(app) as client:
        response = client.post("/api/alerts", json={"rules": [{"min_count": 2}]})

    assert response.status_code == 400
    assert "name" in response.json()["detail"]


def test_zones_default_to_disabled():
    app = create_app()
    with TestClient(app) as client:
        payload = client.get("/api/zones").json()

    assert payload["type"] == "zones"
    assert payload["zones"]["enabled"] is False
    assert payload["zones"]["zones"] == []


def test_zones_round_trip():
    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/api/zones",
            json={"zones": [{"name": "door", "points": [[0.1, 0.1], [0.4, 0.1], [0.4, 0.6]]}]},
        )
        assert response.status_code == 200
        zones = response.json()["zones"]
        assert zones["enabled"] is True
        assert zones["zones"][0]["name"] == "door"
        assert zones["zones"][0]["anchor"] == "center"
        assert zones["zones"][0]["points"] == [[0.1, 0.1], [0.4, 0.1], [0.4, 0.6]]

        assert client.get("/api/status").json()["zones"]["enabled"] is True


def test_zones_reject_invalid_polygon():
    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/api/zones", json={"zones": [{"name": "bad", "points": [[0.1, 0.1], [0.4, 0.1]]}]}
        )

    assert response.status_code == 400
    assert "points" in response.json()["detail"]


def test_viewer_exposes_zone_editor():
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/viewer")

    assert 'id="zoneEditToggle"' in response.text
    assert 'id="zoneList"' in response.text
    assert 'id="alertStatus"' in response.text
    assert 'href="/history"' in response.text


def test_history_page_loads():
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/history")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert 'id="historyFilters"' in response.text
    assert 'href="/history" aria-current="page"' in response.text


def test_events_query_is_empty_when_disabled():
    app = create_app()
    with TestClient(app) as client:
        payload = client.get("/api/events").json()

    assert payload["type"] == "events"
    assert payload["events"] == []
    assert payload["count"] == 0


def test_events_round_trip_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("EVENT_LOG_ENABLED", "1")
    monkeypatch.setenv("EVENT_DB_PATH", str(tmp_path / "events.db"))
    app = create_app()
    with TestClient(app) as client:
        # Persist a sighting directly through the shared store, then read it back
        # through the query endpoint (no live camera/model needed).
        store = app.state.event_store

        async def seed():
            await store.write(
                [
                    {
                        "track_id": 9,
                        "label": "person",
                        "first_seen": 1000.0,
                        "last_seen": 1006.0,
                        "frames": 20,
                        "max_confidence": 0.88,
                        "zones": ["door"],
                    }
                ]
            )

        asyncio.run(seed())
        payload = client.get("/api/events?label=person&zone=door").json()

    assert payload["count"] == 1
    assert payload["events"][0]["track_id"] == 9
    assert payload["events"][0]["dwell_sec"] == 6.0
    assert payload["events"][0]["zones"] == ["door"]


def test_camera_client_state_is_reflected_in_status():
    app = create_app()
    with TestClient(app) as client:
        with client.websocket_connect("/ws/camera") as camera:
            assert camera.receive_json()["type"] == "config"
            camera.send_text(
                json.dumps(
                    {"type": "client_state", "storage_mode": "both", "recording": True}
                )
            )
            # Frames are processed in order, so receiving the detection guarantees
            # the preceding client_state text was already applied.
            camera.send_bytes(b"not a jpeg")
            assert camera.receive_json()["type"] == "detection"

            status = client.get("/api/status").json()
            assert status["camera_connected"] is True
            assert status["camera_storage_mode"] == "both"
            assert status["camera_recording"] is True


def test_status_is_single_camera_by_default():
    app = create_app()
    with TestClient(app) as client:
        status = client.get("/api/status").json()

    assert status["multi_camera"] is False
    assert status["camera_ids"] == ["default"]
    assert status["default_camera_id"] == "default"
    assert status["cameras"]["default"]["frames_received"] == 0


def test_cameras_endpoint_lists_the_allowlist(monkeypatch):
    monkeypatch.setenv("CAMERAS", "front:前門,back:後院")
    app = create_app()
    with TestClient(app) as client:
        payload = client.get("/api/cameras").json()

    assert payload["type"] == "cameras"
    assert payload["multi_camera"] is True
    assert payload["default_camera_id"] == "front"
    assert payload["cameras"] == [
        {"camera_id": "front", "display_name": "前門"},
        {"camera_id": "back", "display_name": "後院"},
    ]
    assert set(payload["streams"]) == {"front", "back"}


def test_two_cameras_stream_independently(monkeypatch):
    monkeypatch.setenv("CAMERAS", "front,back")
    app = create_app()
    with TestClient(app) as client:
        with client.websocket_connect("/ws/camera?camera_id=front") as front:
            assert front.receive_json()["camera_id"] == "front"
            with client.websocket_connect("/ws/camera?camera_id=back") as back:
                assert back.receive_json()["camera_id"] == "back"
                front.send_bytes(b"not a jpeg")
                assert front.receive_json()["type"] == "detection"
                back.send_bytes(b"not a jpeg")
                assert back.receive_json()["type"] == "detection"

                status = client.get("/api/status").json()

    assert status["cameras"]["front"]["camera_connected"] is True
    assert status["cameras"]["back"]["camera_connected"] is True
    # Each stream counts its own frames; the top level is the roll-up.
    assert status["cameras"]["front"]["frames_processed"] == 1
    assert status["cameras"]["back"]["frames_processed"] == 1
    assert status["frames_processed"] == 2


def test_camera_websocket_rejects_an_unlisted_camera_id(monkeypatch):
    monkeypatch.setenv("CAMERAS", "front,back")
    app = create_app()
    with TestClient(app) as client:
        with client.websocket_connect("/ws/camera?camera_id=garage") as websocket:
            message = websocket.receive_json()

    assert message["type"] == "error"
    assert message["fatal"] is True
    assert "garage" in message["message"]


def test_viewer_can_subscribe_to_one_camera(monkeypatch):
    monkeypatch.setenv("CAMERAS", "front,back")
    app = create_app()
    with TestClient(app) as client:
        with client.websocket_connect("/ws/viewer?camera_id=back") as viewer:
            assert viewer.receive_json()["type"] == "status"
            with client.websocket_connect("/ws/camera?camera_id=front") as front:
                front.receive_json()
                front.send_bytes(b"not a jpeg")
                assert front.receive_json()["type"] == "detection"
            with client.websocket_connect("/ws/camera?camera_id=back") as back:
                back.receive_json()
                back.send_bytes(b"not a jpeg")
                assert back.receive_json()["type"] == "detection"
                # The front frame was never forwarded, so the first thing this
                # viewer sees is the back camera's frame.
                metadata = viewer.receive_json()
                viewer.receive_bytes()

    assert metadata["type"] == "frame"
    assert metadata["camera_id"] == "back"


def test_viewer_frames_carry_their_camera_id():
    app = create_app()
    with TestClient(app) as client:
        with client.websocket_connect("/ws/viewer") as viewer:
            viewer.receive_json()
            with client.websocket_connect("/ws/camera") as camera:
                camera.receive_json()
                camera.send_bytes(b"not a jpeg")
                camera.receive_json()
                metadata = viewer.receive_json()
                viewer.receive_bytes()

    assert metadata["camera_id"] == "default"
    assert metadata["display_name"] == "Camera"


def test_zones_are_stored_per_camera(monkeypatch):
    monkeypatch.setenv("CAMERAS", "front,back")
    app = create_app()
    triangle = [[0.1, 0.1], [0.4, 0.1], [0.4, 0.6]]
    with TestClient(app) as client:
        client.post("/api/zones", json={"camera_id": "front", "zones": [{"name": "door", "points": triangle}]})
        client.post("/api/zones", json={"camera_id": "back", "zones": [{"name": "yard", "points": triangle}]})

        back = client.get("/api/zones?camera_id=back").json()["zones"]
        front = client.get("/api/zones?camera_id=front").json()["zones"]

    assert [zone["name"] for zone in back["zones"]] == ["yard"]
    assert [zone["name"] for zone in front["zones"]] == ["door"]
    assert set(front["cameras"]) == {"front", "back"}


def test_unknown_camera_id_is_a_bad_request(monkeypatch):
    monkeypatch.setenv("CAMERAS", "front,back")
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/api/zones?camera_id=garage")

    assert response.status_code == 400
    assert "garage" in response.json()["detail"]


def test_events_can_be_filtered_by_camera(tmp_path, monkeypatch):
    monkeypatch.setenv("CAMERAS", "front,back")
    monkeypatch.setenv("EVENT_LOG_ENABLED", "1")
    monkeypatch.setenv("EVENT_DB_PATH", str(tmp_path / "events.db"))
    app = create_app()
    with TestClient(app) as client:
        store = app.state.event_store

        async def seed():
            await store.write(
                [
                    {"camera_id": "front", "track_id": 1, "label": "person",
                     "first_seen": 1000.0, "last_seen": 1006.0, "frames": 20,
                     "max_confidence": 0.88, "zones": []},
                    {"camera_id": "back", "track_id": 1, "label": "dog",
                     "first_seen": 1000.0, "last_seen": 1006.0, "frames": 20,
                     "max_confidence": 0.77, "zones": []},
                ]
            )

        asyncio.run(seed())
        scoped = client.get("/api/events?camera_id=back").json()
        everything = client.get("/api/events").json()

    assert scoped["count"] == 1
    assert scoped["events"][0]["label"] == "dog"
    assert everything["count"] == 2


def test_metrics_expose_per_camera_series(monkeypatch):
    monkeypatch.setenv("CAMERAS", "front,back")
    app = create_app()
    with TestClient(app) as client:
        body = client.get("/metrics").text

    assert "yolo_elf_cameras 2" in body
    assert '# TYPE yolo_elf_camera_frames_received_total counter' in body
    assert 'yolo_elf_camera_frames_received_total{camera="front"} 0' in body
    assert 'yolo_elf_camera_frames_received_total{camera="back"} 0' in body


def test_viewer_page_exposes_camera_grid():
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/viewer")

    assert 'id="viewerStage"' in response.text
    assert 'id="viewerPanePrimary"' in response.text
    assert 'id="cameraTabs"' in response.text


def test_history_page_exposes_camera_filter():
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/history")

    assert 'id="cameraFilter"' in response.text


def test_remote_mode_keeps_local_copy_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("RECORDING_STORAGE_DIR", str(tmp_path))
    store = RecordingStore(get_settings())
    assert store._keeps_local("remote") is True
    assert store._keeps_local("local") is True
    assert store._keeps_local("both") is True


def test_remote_mode_can_opt_out_of_local_copy(tmp_path, monkeypatch):
    monkeypatch.setenv("RECORDING_STORAGE_DIR", str(tmp_path))
    monkeypatch.setenv("RECORDING_KEEP_LOCAL_COPY", "0")
    store = RecordingStore(get_settings())
    assert store._keeps_local("remote") is False
    assert store._keeps_local("local") is True
    assert store._keeps_local("both") is True


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
    recording_id = "rec-test-local"
    body = b"webm-bytes"
    metadata = {
        "schema_version": 1,
        "detections": [
            {
                "frame_id": 7,
                "recording_time_ms": 250,
                "wall_time_iso": "2026-05-27T08:00:00.250Z",
                "width": 1280,
                "height": 720,
                "inference_ms": 18.5,
                "boxes": [
                    {
                        "class_id": 2,
                        "label": "package",
                        "confidence": 0.91,
                        "xywh": [10, 20, 30, 40],
                    }
                ],
            }
        ],
    }

    with TestClient(app) as client:
        metadata_response = client.post(
            f"/api/recordings/{recording_id}/metadata",
            json=metadata,
            headers={"x-yolo-elf-storage-mode": "local"},
        )
        response = client.post(
            "/api/recordings",
            content=body,
            headers={
                "content-type": "video/webm",
                "x-yolo-elf-recording-id": recording_id,
                "x-yolo-elf-duration-ms": "1234",
                "x-yolo-elf-started-at": "2026-05-27T08:00:00.000Z",
            },
        )
        payload = response.json()
        download = client.get(payload["recording"]["download_url"])
        metadata_download = client.get(payload["recording"]["metadata_url"])
        status = client.get("/api/status").json()

    assert metadata_response.status_code == 200
    assert metadata_response.json()["metadata"]["metadata_byte_length"] > 0
    assert response.status_code == 200
    assert payload["type"] == "recording"
    assert payload["recording"]["id"] == recording_id
    assert payload["recording"]["content_type"] == "video/webm"
    assert payload["recording"]["byte_length"] == len(body)
    assert payload["recording"]["duration_ms"] == 1234
    assert payload["recording"]["storage_mode"] == "local"
    assert payload["recording"]["local_saved"] is True
    assert payload["recording"]["metadata_byte_length"] > 0
    assert payload["remote_storage"]["status"] == "skipped"
    assert download.status_code == 200
    assert download.content == body
    assert metadata_download.status_code == 200
    assert metadata_download.json()["detections"][0]["boxes"][0]["xywh"] == [10, 20, 30, 40]
    assert status["recordings"]["recordings_saved"] == 1
    assert list(tmp_path.glob("*.webm"))
    assert list(tmp_path.glob("*.detections.json"))


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
        metadata_response = client.post(
            "/api/recordings/rec-test-remote/metadata",
            json={"detections": []},
            headers={"x-yolo-elf-storage-mode": "remote"},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Remote recording storage is not configured"
    assert metadata_response.status_code == 400
    assert metadata_response.json()["detail"] == "Remote recording storage is not configured"


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
