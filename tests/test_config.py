import ast
import pathlib

import pytest

from app.config import TASK_MODEL_ENV, get_settings, parse_detect_tasks
from conftest import SETTINGS_ENV

# Every helper in app/config.py that takes the env var name as its first
# argument. `os.getenv` / `os.environ.get` land here as `getenv` / `get`.
_ENV_READERS = frozenset(
    {
        "_bool_env",
        "_int_env",
        "_float_env",
        "_bounded_int_env",
        "_bounded_float_env",
        "_choice_env",
        "_list_env",
        "_path_env",
        "getenv",
        "get",
    }
)


def _env_names_read_by_config():
    """Env var names `app/config.py` actually reads, found by parsing it.

    AST rather than regex because several of these calls wrap across lines.
    """
    source = pathlib.Path(__file__).resolve().parents[1] / "app" / "config.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names = {env for env, _default in TASK_MODEL_ENV.values()}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        called = getattr(func, "id", None) or getattr(func, "attr", None)
        first = node.args[0]
        if called in _ENV_READERS and isinstance(first, ast.Constant):
            if isinstance(first.value, str) and first.value.isupper():
                names.add(first.value)
    return names


def test_settings_env_covers_every_setting():
    """The suite's isolation list must not drift behind `app/config.py`.

    Without this, adding a setting silently leaves it readable from the
    developer's real environment inside every test.
    """
    missing = _env_names_read_by_config() - SETTINGS_ENV
    assert not missing, (
        f"add {sorted(missing)} to SETTINGS_ENV in tests/conftest.py so the "
        "autouse fixture clears it"
    )


def test_default_settings_prioritize_detection_recall(monkeypatch):
    settings = get_settings()

    assert settings.detect_mode == "fast"
    assert settings.yolo_model == "yolo26s.pt"
    assert settings.yolo_model_accurate == "yolo26x.pt"
    assert settings.yolo_classes == ()
    assert settings.yolo_half is True
    assert settings.yolo_track is True
    assert settings.yolo_tracker == "bytetrack.yaml"
    # "auto" leaves the checkpoint's own head alone — the only value that is
    # bit-for-bit the pre-YOLO26 behaviour on every model generation.
    assert settings.yolo_end2end == "auto"
    assert settings.yolo_max_det == 300
    assert settings.yolo_export == ""
    # The VLM channel is opt-in and off by default.
    assert settings.vlm_enabled is False
    assert settings.vlm_model == "microsoft/Florence-2-base"
    assert settings.vlm_interval_sec == 3.0
    assert settings.vlm_detect_task == ""
    assert settings.vlm_caption is True
    assert settings.vlm_caption_task == "<MORE_DETAILED_CAPTION>"
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


def test_vlm_env_overrides_are_parsed(monkeypatch):
    monkeypatch.setenv("VLM_ENABLED", "1")
    monkeypatch.setenv("VLM_MODEL", "microsoft/Florence-2-large")
    monkeypatch.setenv("VLM_INTERVAL_SEC", "1.5")
    monkeypatch.setenv("VLM_DETECT_TASK", "<OD>")

    settings = get_settings()

    assert settings.vlm_enabled is True
    assert settings.vlm_model == "microsoft/Florence-2-large"
    assert settings.vlm_interval_sec == 1.5
    assert settings.vlm_detect_task == "<OD>"


def test_vlm_caption_can_be_disabled(monkeypatch):
    monkeypatch.setenv("VLM_CAPTION", "0")
    monkeypatch.setenv("VLM_CAPTION_TASK", "<CAPTION>")

    settings = get_settings()

    assert settings.vlm_caption is False
    assert settings.vlm_caption_task == "<CAPTION>"


def test_vlm_interval_out_of_range_raises(monkeypatch):
    monkeypatch.setenv("VLM_INTERVAL_SEC", "0.1")
    with pytest.raises(ValueError):
        get_settings()


def test_get_settings_accepts_valid_overrides(monkeypatch):
    monkeypatch.setenv("PORT", "8767")
    monkeypatch.setenv("DETECT_MODE", "accurate")
    monkeypatch.setenv("YOLO_MODEL_ACCURATE", "yolo11x.pt")
    monkeypatch.setenv("CLASSIFIER_MODEL", " yolov8x-cls.pt ")
    monkeypatch.setenv("CLASSIFIER_MIN_CONF", "0.4")
    monkeypatch.setenv("CLASSIFIER_MAX_BOXES", "3")
    monkeypatch.setenv("YOLO_HALF", "true")
    monkeypatch.setenv("YOLO_TRACK", "off")
    monkeypatch.setenv("YOLO_TRACKER", "tracktrack.yaml")
    monkeypatch.setenv("YOLO_END2END", "ON")
    monkeypatch.setenv("YOLO_MAX_DET", "50")
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
    assert settings.yolo_track is False
    assert settings.yolo_tracker == "tracktrack.yaml"
    assert settings.yolo_end2end == "on"
    assert settings.yolo_max_det == 50
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
    monkeypatch.setenv("YOLO_CLASSES", " person, backpack ,, fire extinguisher ,")

    settings = get_settings()

    assert settings.yolo_classes == ("person", "backpack", "fire extinguisher")


@pytest.mark.parametrize("value", ["engine", "onnx", "ENGINE"])
def test_yolo_export_accepts_known_formats(monkeypatch, value):
    monkeypatch.setenv("YOLO_EXPORT", value)

    assert get_settings().yolo_export == value.lower()


def test_task_defaults_keep_the_original_detect_pipeline(monkeypatch):
    settings = get_settings()

    # An unset DETECT_TASK must behave exactly like every previous release.
    assert settings.detect_task == "detect"
    assert settings.raster_max_size == 256
    assert settings.task_model_map == {
        "segment": "yolo26s-seg.pt",
        "pose": "yolo26s-pose.pt",
        "obb": "yolo26s-obb.pt",
        "openvocab": "yoloe-26s-seg.pt",
        "semantic": "yolo26s-sem.pt",
        "depth": "yolo26s-depth.pt",
    }


def test_detect_tasks_defaults_to_the_single_task(monkeypatch):
    monkeypatch.setenv("DETECT_TASK", "obb")

    # Unset DETECT_TASKS = one head, i.e. the previous pipeline exactly.
    assert get_settings().detect_tasks == ("obb",)


def test_detect_tasks_parses_a_multi_head_list(monkeypatch):
    monkeypatch.setenv("DETECT_TASKS", " detect , POSE ,detect ")

    # Case and padding normalise; a repeat is folded rather than run twice.
    assert get_settings().detect_tasks == ("detect", "pose")


@pytest.mark.parametrize(
    "raw",
    [
        "detect,sorcery",
        "semantic,depth",  # two full-frame rasters cannot both be drawn
        "detect,segment,pose,obb,openvocab",  # over MAX_ACTIVE_TASKS
    ],
)
def test_detect_tasks_rejects_impossible_sets(monkeypatch, raw):
    monkeypatch.setenv("DETECT_TASKS", raw)

    with pytest.raises(ValueError):
        get_settings()


def test_parse_detect_tasks_accepts_a_sequence(monkeypatch):
    # The runtime API hands in an already-split list rather than a string.
    assert parse_detect_tasks(["Depth", " detect "], "detect") == ("depth", "detect")
    assert parse_detect_tasks([], "pose") == ("pose",)


def test_task_models_are_overridable_per_task(monkeypatch):
    monkeypatch.setenv("DETECT_TASK", "POSE")
    monkeypatch.setenv("YOLO_MODEL_POSE", " yolo26x-pose.pt ")
    monkeypatch.setenv("YOLO_MODEL_DEPTH", "")  # blank falls back to the default

    settings = get_settings()

    assert settings.detect_task == "pose"
    assert settings.task_model_map["pose"] == "yolo26x-pose.pt"
    assert settings.task_model_map["depth"] == "yolo26s-depth.pt"


def test_task_models_do_not_alias_across_replaced_copies(monkeypatch):
    # `replace()` is used by the benchmark and the tests; a shared mutable dict
    # here would make one copy's edit leak into every other.
    from dataclasses import replace

    original = get_settings()
    copy = replace(original, task_models=(("pose", "other.pt"),))

    assert original.task_model_map["pose"] == "yolo26s-pose.pt"
    assert copy.task_model_map == {"pose": "other.pt"}


@pytest.mark.parametrize("value", ["auto", "on", "off", " OFF "])
def test_yolo_end2end_accepts_known_modes(monkeypatch, value):
    monkeypatch.setenv("YOLO_END2END", value)

    assert get_settings().yolo_end2end == value.strip().lower()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("PORT", "70000"),
        ("DETECT_MODE", "ultra"),
        ("YOLO_HALF", "maybe"),
        ("YOLO_TRACK", "maybe"),
        ("YOLO_EXPORT", "trt"),
        ("DETECT_TASK", "sorcery"),
        ("RASTER_MAX_SIZE", "16"),
        ("RASTER_MAX_SIZE", "2048"),
        ("YOLO_END2END", "yes"),
        ("YOLO_MAX_DET", "0"),
        ("YOLO_MAX_DET", "1001"),
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
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=name):
        get_settings()
