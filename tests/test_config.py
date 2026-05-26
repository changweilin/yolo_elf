import pytest

from app.config import get_settings


SETTINGS_ENV = [
    "PORT",
    "YOLO_MODEL",
    "YOLO_DEVICE",
    "YOLO_HALF",
    "YOLO_WARMUP",
    "YOLO_WARMUP_RUNS",
    "CONF_THRESH",
    "IMG_SIZE",
    "FRAME_FPS",
    "CAPTURE_WIDTH",
    "CAPTURE_HEIGHT",
    "JPEG_QUALITY",
    "MAX_FRAME_BYTES",
    "REMOTE_STORAGE_URL",
    "REMOTE_STORAGE_TOKEN",
    "REMOTE_STORAGE_INCLUDE_FRAME",
    "REMOTE_STORAGE_QUEUE_SIZE",
    "REMOTE_STORAGE_TIMEOUT",
    "REMOTE_STORAGE_RETRIES",
]


def clear_settings_env(monkeypatch):
    for name in SETTINGS_ENV:
        monkeypatch.delenv(name, raising=False)


def test_get_settings_accepts_valid_overrides(monkeypatch):
    clear_settings_env(monkeypatch)
    monkeypatch.setenv("PORT", "8767")
    monkeypatch.setenv("YOLO_HALF", "true")
    monkeypatch.setenv("YOLO_WARMUP", "1")
    monkeypatch.setenv("YOLO_WARMUP_RUNS", "2")
    monkeypatch.setenv("FRAME_FPS", "30")
    monkeypatch.setenv("JPEG_QUALITY", "0.8")
    monkeypatch.setenv("REMOTE_STORAGE_URL", "https://storage.example/events")
    monkeypatch.setenv("REMOTE_STORAGE_TOKEN", "secret")
    monkeypatch.setenv("REMOTE_STORAGE_INCLUDE_FRAME", "yes")
    monkeypatch.setenv("REMOTE_STORAGE_QUEUE_SIZE", "12")
    monkeypatch.setenv("REMOTE_STORAGE_TIMEOUT", "3.5")
    monkeypatch.setenv("REMOTE_STORAGE_RETRIES", "1")

    settings = get_settings()

    assert settings.port == 8767
    assert settings.yolo_half is True
    assert settings.yolo_warmup is True
    assert settings.yolo_warmup_runs == 2
    assert settings.frame_fps == 30
    assert settings.jpeg_quality == 0.8
    assert settings.remote_storage_url == "https://storage.example/events"
    assert settings.remote_storage_token == "secret"
    assert settings.remote_storage_include_frame is True
    assert settings.remote_storage_queue_size == 12
    assert settings.remote_storage_timeout == 3.5
    assert settings.remote_storage_retries == 1


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("PORT", "70000"),
        ("YOLO_HALF", "maybe"),
        ("YOLO_WARMUP", "warm"),
        ("YOLO_WARMUP_RUNS", "0"),
        ("CONF_THRESH", "1.5"),
        ("IMG_SIZE", "16"),
        ("FRAME_FPS", "0"),
        ("CAPTURE_WIDTH", "32"),
        ("CAPTURE_HEIGHT", "32"),
        ("JPEG_QUALITY", "0.1"),
        ("MAX_FRAME_BYTES", "1024"),
        ("REMOTE_STORAGE_INCLUDE_FRAME", "sometimes"),
        ("REMOTE_STORAGE_QUEUE_SIZE", "0"),
        ("REMOTE_STORAGE_TIMEOUT", "0"),
        ("REMOTE_STORAGE_RETRIES", "6"),
    ],
)
def test_get_settings_rejects_out_of_range_values(monkeypatch, name, value):
    clear_settings_env(monkeypatch)
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=name):
        get_settings()
