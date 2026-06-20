import pytest

from app.config import get_settings


SETTINGS_ENV = [
    "PORT",
    "DETECT_MODE",
    "YOLO_MODEL",
    "YOLO_MODEL_ACCURATE",
    "YOLO_CLASSES",
    "YOLO_DEVICE",
    "YOLO_HALF",
    "YOLO_WARMUP",
    "YOLO_WARMUP_RUNS",
    "CONF_THRESH",
    "IMG_SIZE",
    "CLASSIFIER_MODEL",
    "CLASSIFIER_MIN_CONF",
    "CLASSIFIER_MAX_BOXES",
    "FRAME_FPS",
    "CAPTURE_WIDTH",
    "CAPTURE_HEIGHT",
    "JPEG_QUALITY",
    "MAX_FRAME_BYTES",
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
]


def clear_settings_env(monkeypatch):
    for name in SETTINGS_ENV:
        monkeypatch.delenv(name, raising=False)


def test_default_settings_prioritize_detection_recall(monkeypatch):
    clear_settings_env(monkeypatch)

    settings = get_settings()

    assert settings.detect_mode == "fast"
    assert settings.yolo_model == "yolov8s.pt"
    assert settings.yolo_model_accurate == "yolov8x.pt"
    assert settings.yolo_classes == ()
    assert settings.yolo_half is True
    assert settings.conf_thresh == 0.2
    assert settings.img_size == 1280
    assert settings.classifier_model == ""
    assert settings.classifier_min_conf == 0.0
    assert settings.classifier_max_boxes == 5
    assert settings.capture_width == 1920
    assert settings.capture_height == 1080
    assert settings.jpeg_quality == 0.9
    assert settings.recording_enabled is True
    assert settings.recording_keep_local_copy is True
    assert settings.recording_storage_dir.name == "recordings"
    assert settings.recording_max_bytes == 250 * 1024 * 1024


def test_get_settings_accepts_valid_overrides(monkeypatch):
    clear_settings_env(monkeypatch)
    monkeypatch.setenv("PORT", "8767")
    monkeypatch.setenv("DETECT_MODE", "accurate")
    monkeypatch.setenv("YOLO_MODEL_ACCURATE", "yolo11x.pt")
    monkeypatch.setenv("CLASSIFIER_MODEL", " yolov8x-cls.pt ")
    monkeypatch.setenv("CLASSIFIER_MIN_CONF", "0.4")
    monkeypatch.setenv("CLASSIFIER_MAX_BOXES", "3")
    monkeypatch.setenv("YOLO_HALF", "true")
    monkeypatch.setenv("YOLO_WARMUP", "1")
    monkeypatch.setenv("YOLO_WARMUP_RUNS", "2")
    monkeypatch.setenv("FRAME_FPS", "30")
    monkeypatch.setenv("JPEG_QUALITY", "0.8")
    monkeypatch.setenv("RECORDING_ENABLED", "false")
    monkeypatch.setenv("RECORDING_STORAGE_DIR", "test-recordings")
    monkeypatch.setenv("RECORDING_MAX_BYTES", "1048576")
    monkeypatch.setenv("REMOTE_STORAGE_URL", "https://storage.example/events")
    monkeypatch.setenv("REMOTE_STORAGE_TOKEN", "secret")
    monkeypatch.setenv("REMOTE_STORAGE_INCLUDE_FRAME", "yes")
    monkeypatch.setenv("REMOTE_STORAGE_RECORDING_URL", "https://storage.example/recordings")
    monkeypatch.setenv("REMOTE_STORAGE_QUEUE_SIZE", "12")
    monkeypatch.setenv("REMOTE_STORAGE_TIMEOUT", "3.5")
    monkeypatch.setenv("REMOTE_STORAGE_RETRIES", "1")

    settings = get_settings()

    assert settings.port == 8767
    assert settings.detect_mode == "accurate"
    assert settings.yolo_model_accurate == "yolo11x.pt"
    assert settings.classifier_model == "yolov8x-cls.pt"
    assert settings.classifier_min_conf == 0.4
    assert settings.classifier_max_boxes == 3
    assert settings.yolo_half is True
    assert settings.yolo_warmup is True
    assert settings.yolo_warmup_runs == 2
    assert settings.frame_fps == 30
    assert settings.jpeg_quality == 0.8
    assert settings.recording_enabled is False
    assert settings.recording_storage_dir.name == "test-recordings"
    assert settings.recording_max_bytes == 1048576
    assert settings.remote_storage_url == "https://storage.example/events"
    assert settings.remote_storage_token == "secret"
    assert settings.remote_storage_include_frame is True
    assert settings.remote_storage_recording_url == "https://storage.example/recordings"
    assert settings.remote_storage_queue_size == 12
    assert settings.remote_storage_timeout == 3.5
    assert settings.remote_storage_retries == 1


def test_yolo_classes_parses_comma_separated_prompts(monkeypatch):
    clear_settings_env(monkeypatch)
    monkeypatch.setenv("YOLO_CLASSES", " person, backpack ,, fire extinguisher ,")

    settings = get_settings()

    assert settings.yolo_classes == ("person", "backpack", "fire extinguisher")


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("PORT", "70000"),
        ("DETECT_MODE", "ultra"),
        ("YOLO_HALF", "maybe"),
        ("YOLO_WARMUP", "warm"),
        ("YOLO_WARMUP_RUNS", "0"),
        ("CONF_THRESH", "1.5"),
        ("IMG_SIZE", "16"),
        ("CLASSIFIER_MIN_CONF", "1.5"),
        ("CLASSIFIER_MAX_BOXES", "0"),
        ("FRAME_FPS", "0"),
        ("CAPTURE_WIDTH", "32"),
        ("CAPTURE_HEIGHT", "32"),
        ("JPEG_QUALITY", "0.1"),
        ("MAX_FRAME_BYTES", "1024"),
        ("RECORDING_ENABLED", "sometimes"),
        ("RECORDING_MAX_BYTES", "1024"),
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
