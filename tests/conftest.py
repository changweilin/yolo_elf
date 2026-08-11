"""Suite-wide test isolation.

Every setting reachable from the environment is cleared before each test, so a
developer's shell (or a leftover `monkeypatch.setenv` in an earlier module)
can never change what `get_settings()` returns. Individual modules used to keep
their own partial copies of this list; they drifted, and the gaps were real
leaks — `test_config` never cleared `ZONES` or `ALERT_*`, and anything that
built an `EventStore` without opting out wrote `events.db` into the repo root.

`SETTINGS_ENV` is deliberately spelled out rather than derived, so it stays
greppable. `test_config.test_settings_env_covers_every_setting` fails loudly
when `app/config.py` grows a variable that is missing here.
"""

import pytest

SETTINGS_ENV = frozenset(
    {
        # server
        "HOST",
        "PORT",
        # detector: mode, heads, weights
        "DETECT_MODE",
        "DETECT_TASK",
        "DETECT_TASKS",
        "YOLO_MODEL",
        "YOLO_MODEL_ACCURATE",
        "YOLO_MODEL_SEGMENT",
        "YOLO_MODEL_POSE",
        "YOLO_MODEL_OBB",
        "YOLO_MODEL_OPENVOCAB",
        "YOLO_MODEL_SEMANTIC",
        "YOLO_MODEL_DEPTH",
        "RASTER_MAX_SIZE",
        # detector: inference knobs
        "YOLO_CLASSES",
        "YOLO_DEVICE",
        "YOLO_HALF",
        "YOLO_TRACK",
        "YOLO_TRACKER",
        "YOLO_END2END",
        "YOLO_MAX_DET",
        "YOLO_EXPORT",
        "YOLO_WARMUP",
        "YOLO_WARMUP_RUNS",
        "CONF_THRESH",
        "IMG_SIZE",
        # second-stage classifier
        "CLASSIFIER_MODEL",
        "CLASSIFIER_MIN_CONF",
        "CLASSIFIER_MAX_BOXES",
        # VLM channel
        "VLM_ENABLED",
        "VLM_MODEL",
        "VLM_INTERVAL_SEC",
        "VLM_DETECT_TASK",
        "VLM_CAPTION",
        "VLM_CAPTION_TASK",
        # capture / transport
        "FRAME_FPS",
        "CAPTURE_WIDTH",
        "CAPTURE_HEIGHT",
        "JPEG_QUALITY",
        "MAX_FRAME_BYTES",
        # recordings + remote storage
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
        # alerts
        "ALERT_RULES",
        "ALERT_WEBHOOK_URL",
        "ALERT_WEBHOOK_TOKEN",
        "ALERT_COOLDOWN_SEC",
        "ALERT_WEBHOOK_TIMEOUT",
        "ALERT_WEBHOOK_RETRIES",
        # zones / history / metrics / auth / cameras
        "ZONES",
        "EVENT_LOG_ENABLED",
        "EVENT_DB_PATH",
        "EVENT_EXPIRY_SEC",
        "METRICS_ENABLED",
        "AUTH_TOKEN",
        "AUTH_SESSION_TTL",
        "CAMERAS",
        "MAX_CAMERAS",
    }
)


@pytest.fixture(autouse=True)
def isolate_settings_env(monkeypatch):
    for name in SETTINGS_ENV:
        monkeypatch.delenv(name, raising=False)
    # Event logging ships on and writes SQLite to the repo root. Keep it off
    # unless a test opts back in with a tmp_path database.
    monkeypatch.setenv("EVENT_LOG_ENABLED", "0")
