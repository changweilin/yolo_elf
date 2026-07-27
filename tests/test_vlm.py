import io
from contextlib import nullcontext

from app.config import get_settings
from app.vlm import VLM_BOX_CONFIDENCE, VlmEngine, florence_od_to_boxes


def _settings(monkeypatch, **env):
    for name in (
        "VLM_ENABLED", "VLM_MODEL", "VLM_INTERVAL_SEC", "VLM_DETECT_TASK",
        "VLM_CAPTION", "VLM_CAPTION_TASK", "YOLO_CLASSES", "YOLO_DEVICE", "YOLO_HALF",
    ):
        monkeypatch.delenv(name, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return get_settings()


# --- florence_od_to_boxes (pure) ---------------------------------------------


def test_od_result_maps_labels_and_stamps_sentinels():
    parsed = {"bboxes": [[10, 20, 100, 200], [0, 0, 50, 50]], "labels": ["cat", "dog"]}
    boxes = florence_od_to_boxes(parsed, 640, 480)
    assert [box["label"] for box in boxes] == ["cat", "dog"]
    assert boxes[0]["xyxy"] == [10.0, 20.0, 100.0, 200.0]
    # Florence emits neither confidence nor track ids, so every box carries the
    # fixed sentinel and a null track id.
    assert all(box["confidence"] == VLM_BOX_CONFIDENCE for box in boxes)
    assert all(box["track_id"] is None for box in boxes)
    assert all(box["class_id"] == -1 for box in boxes)


def test_open_vocab_result_uses_bboxes_labels_key():
    parsed = {"bboxes": [[5, 5, 15, 15]], "bboxes_labels": ["fire extinguisher"]}
    boxes = florence_od_to_boxes(parsed, 640, 480)
    assert boxes[0]["label"] == "fire extinguisher"


def test_boxes_are_clamped_to_frame():
    parsed = {"bboxes": [[700, -5, 900, 500]], "labels": ["x"]}
    boxes = florence_od_to_boxes(parsed, 640, 480)
    assert boxes[0]["xyxy"] == [640.0, 0.0, 640.0, 480.0]


def test_missing_label_falls_back_to_index():
    parsed = {"bboxes": [[0, 0, 1, 1], [2, 2, 3, 3]], "labels": ["only"]}
    boxes = florence_od_to_boxes(parsed, 640, 480)
    assert boxes[1]["label"] == "1"


def test_malformed_entries_are_skipped():
    parsed = {"bboxes": [[0, 0, 1], "nope", [0, 0, 2, 2]], "labels": ["a", "b", "c"]}
    boxes = florence_od_to_boxes(parsed, 640, 480)
    # Only the well-formed 4-tuple survives; label index still aligns to it.
    assert len(boxes) == 1
    assert boxes[0]["xyxy"] == [0.0, 0.0, 2.0, 2.0]


def test_non_dict_result_is_empty():
    assert florence_od_to_boxes(None, 640, 480) == []
    assert florence_od_to_boxes([], 640, 480) == []


# --- VlmEngine task selection + status ----------------------------------------


def test_detect_task_defaults_to_od(monkeypatch):
    engine = VlmEngine(_settings(monkeypatch))
    assert engine.detect_task == "<OD>"


def test_detect_task_switches_to_open_vocab_with_classes(monkeypatch):
    engine = VlmEngine(_settings(monkeypatch, YOLO_CLASSES="person,backpack"))
    assert engine.detect_task == "<OPEN_VOCABULARY_DETECTION>"


def test_detect_task_explicit_override_wins(monkeypatch):
    engine = VlmEngine(
        _settings(monkeypatch, YOLO_CLASSES="person", VLM_DETECT_TASK="<CAPTION_TO_PHRASE_GROUNDING>")
    )
    assert engine.detect_task == "<CAPTION_TO_PHRASE_GROUNDING>"


def test_status_reports_disabled_defaults(monkeypatch):
    status = VlmEngine(_settings(monkeypatch)).status()
    assert status["enabled"] is False
    assert status["model"] == "microsoft/Florence-2-base"
    assert status["loaded"] is False
    assert status["detect_task"] == "<OD>"
    assert status["interval_sec"] == 3.0


def test_build_prompt_appends_classes_for_open_vocab(monkeypatch):
    engine = VlmEngine(_settings(monkeypatch, YOLO_CLASSES="person,backpack"))
    prompt = engine._build_prompt("<OPEN_VOCABULARY_DETECTION>")
    assert prompt == "<OPEN_VOCABULARY_DETECTION> person. backpack"


# --- VlmEngine.detect with a stubbed model ------------------------------------


class _FakeTorch:
    float16 = "float16"

    @staticmethod
    def inference_mode():
        return nullcontext()


class _FakeProcessor:
    def __init__(self):
        self.last_prompt = None

    def __call__(self, text, images, return_tensors):
        self.last_prompt = text
        return {"input_ids": "ids", "pixel_values": "pixels"}

    def batch_decode(self, generated_ids, skip_special_tokens):
        return ["<OD>decoded"]

    def post_process_generation(self, text, task, image_size):
        # Caption tasks return a plain string; detection tasks return the
        # parallel bboxes/labels dict — matching Florence-2's real contract.
        if "CAPTION" in task:
            return {task: "a tidy room with a wooden chair"}
        return {task: {"bboxes": [[1, 2, 3, 4]], "labels": ["chair"]}}


class _FakeModel:
    def generate(self, **kwargs):
        return [["ids"]]


def _jpeg_bytes(width=64, height=48):
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (30, 30, 30)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _stub_engine(monkeypatch, **env):
    engine = VlmEngine(_settings(monkeypatch, **env))
    processor = _FakeProcessor()
    # Pre-inject the loaded model so detect() skips _ensure_model's real import.
    engine._model = _FakeModel()
    engine._processor = processor
    engine._torch = _FakeTorch()
    engine._device = None
    engine._device_resolved = True
    return engine, processor


def test_detect_maps_stubbed_result_to_box_payload(monkeypatch):
    engine, _ = _stub_engine(monkeypatch)
    result = engine.detect(_jpeg_bytes(64, 48))
    assert result["width"] == 64
    assert result["height"] == 48
    assert result["boxes"] == [
        {"xyxy": [1.0, 2.0, 3.0, 4.0], "class_id": -1, "label": "chair",
         "confidence": VLM_BOX_CONFIDENCE, "track_id": None}
    ]
    assert engine.status()["last_boxes"] == 1


def test_detect_uses_open_vocab_prompt_when_classes_set(monkeypatch):
    engine, processor = _stub_engine(monkeypatch, YOLO_CLASSES="chair,lamp")
    engine.detect(_jpeg_bytes())
    assert processor.last_prompt == "<OPEN_VOCABULARY_DETECTION> chair. lamp"


# --- Phase 2: caption + analyze -----------------------------------------------


def test_caption_text_extracts_string():
    from app.vlm import _caption_text

    assert _caption_text("  hi  ") == "hi"
    assert _caption_text("") is None
    assert _caption_text({"<CAPTION>": "a dog"}) == "a dog"
    assert _caption_text({"bboxes": [[1, 2, 3, 4]]}) is None
    assert _caption_text(None) is None


def test_status_reports_caption_defaults(monkeypatch):
    status = VlmEngine(_settings(monkeypatch)).status()
    assert status["caption_enabled"] is True
    assert status["caption_task"] == "<MORE_DETAILED_CAPTION>"
    assert status["last_caption"] is None


def test_analyze_returns_boxes_and_caption(monkeypatch):
    engine, _ = _stub_engine(monkeypatch)  # captioning on by default
    result = engine.analyze(_jpeg_bytes())
    assert [box["label"] for box in result["boxes"]] == ["chair"]
    assert result["description"] == "a tidy room with a wooden chair"
    assert engine.status()["last_caption"] == "a tidy room with a wooden chair"


def test_analyze_skips_caption_when_disabled(monkeypatch):
    engine, _ = _stub_engine(monkeypatch, VLM_CAPTION="0")
    result = engine.analyze(_jpeg_bytes())
    assert result["boxes"]  # detection still runs
    assert result["description"] is None


def test_analyze_caption_failure_keeps_boxes(monkeypatch):
    engine, processor = _stub_engine(monkeypatch)

    def _boom(text, task, image_size):
        if "CAPTION" in task:
            raise RuntimeError("caption model exploded")
        return {task: {"bboxes": [[1, 2, 3, 4]], "labels": ["chair"]}}

    processor.post_process_generation = _boom
    result = engine.analyze(_jpeg_bytes())
    # Boxes survive a best-effort caption failure; the error is recorded.
    assert [box["label"] for box in result["boxes"]] == ["chair"]
    assert result["description"] is None
    assert "caption" in (engine.status()["last_error"] or "").lower()
