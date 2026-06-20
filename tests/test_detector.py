import pytest

from app.config import get_settings
from app.detector import (
    YoloDetector,
    clamp_xyxy,
    detection_error_payload,
    device_supports_half,
)


def _detector(monkeypatch):
    for name in (
        "DETECT_MODE",
        "YOLO_MODEL",
        "YOLO_MODEL_ACCURATE",
        "YOLO_CLASSES",
        "CLASSIFIER_MODEL",
        "CLASSIFIER_MIN_CONF",
    ):
        monkeypatch.delenv(name, raising=False)
    return YoloDetector(get_settings())


class _FakeWorldModel:
    def __init__(self):
        self.names = {0: "object"}
        self.applied = None

    def set_classes(self, classes):
        self.applied = list(classes)
        self.names = {index: name for index, name in enumerate(classes)}


class _FakeClosedModel:
    def __init__(self):
        self.names = {0: "person"}


class _FakeProbs:
    def __init__(self, top1, top1conf):
        self.top1 = top1
        self.top1conf = top1conf


class _FakeClsResult:
    def __init__(self, top1, top1conf):
        self.probs = _FakeProbs(top1, top1conf)


class _FakeClassifier:
    """Stand-in for an Ultralytics classification model.

    Returns one result per source crop with a fixed top-1 prediction so the
    second-stage classification path can be exercised without real weights.
    """

    def __init__(self, top1=0, top1conf=0.9, names=None):
        self.names = names or {0: "tabby cat", 1: "golden retriever"}
        self._top1 = top1
        self._top1conf = top1conf
        self.received = None

    def predict(self, **kwargs):
        self.received = kwargs
        sources = kwargs["source"]
        return [_FakeClsResult(self._top1, self._top1conf) for _ in sources]


def test_detector_defaults_to_fast_mode(monkeypatch):
    status = _detector(monkeypatch).status()
    assert status["mode"] == "fast"
    assert status["model"] == "yolov8s.pt"
    assert status["available_modes"] == ["fast", "accurate"]
    assert status["models"] == {"fast": "yolov8s.pt", "accurate": "yolov8x.pt"}
    assert status["configured_classes"] == []
    assert status["open_vocabulary"] is False
    assert status["loaded"] is False


def test_status_reports_configured_open_vocabulary_classes(monkeypatch):
    monkeypatch.delenv("DETECT_MODE", raising=False)
    monkeypatch.setenv("YOLO_CLASSES", "person, dog")
    status = YoloDetector(get_settings()).status()
    assert status["configured_classes"] == ["person", "dog"]
    # Not applied until a world model is actually loaded.
    assert status["open_vocabulary"] is False


def test_apply_open_vocabulary_sets_classes_on_world_models(monkeypatch):
    monkeypatch.setenv("YOLO_CLASSES", "cat, hat")
    detector = YoloDetector(get_settings())
    model = _FakeWorldModel()
    assert detector._apply_open_vocabulary(model) is True
    assert model.applied == ["cat", "hat"]


def test_apply_open_vocabulary_skips_closed_set_models(monkeypatch):
    monkeypatch.setenv("YOLO_CLASSES", "cat, hat")
    detector = YoloDetector(get_settings())
    assert detector._apply_open_vocabulary(_FakeClosedModel()) is False


def test_apply_open_vocabulary_noop_without_configured_classes(monkeypatch):
    monkeypatch.delenv("YOLO_CLASSES", raising=False)
    detector = YoloDetector(get_settings())
    model = _FakeWorldModel()
    assert detector._apply_open_vocabulary(model) is False
    assert model.applied is None


def test_set_mode_switches_active_model(monkeypatch):
    detector = _detector(monkeypatch)
    assert detector.set_mode("accurate") == "accurate"
    status = detector.status()
    assert status["mode"] == "accurate"
    assert status["model"] == "yolov8x.pt"


def test_set_mode_rejects_unknown_mode(monkeypatch):
    detector = _detector(monkeypatch)
    with pytest.raises(ValueError):
        detector.set_mode("ultra")


def test_update_config_applies_conf_img_and_models(monkeypatch):
    detector = _detector(monkeypatch)
    status = detector.update_config(
        {
            "mode": "accurate",
            "fast_model": "yolo11n.pt",
            "accurate_model": "yolo11x.pt",
            "conf_thresh": 0.5,
            "img_size": 640,
        }
    )
    assert status["mode"] == "accurate"
    assert status["models"] == {"fast": "yolo11n.pt", "accurate": "yolo11x.pt"}
    assert status["model"] == "yolo11x.pt"
    assert status["conf_thresh"] == 0.5
    assert status["img_size"] == 640


def test_update_config_swapping_model_drops_cached_weights(monkeypatch):
    detector = _detector(monkeypatch)
    detector._models["fast"] = _FakeClosedModel()
    detector._names_by_mode["fast"] = {0: "person"}
    detector._open_vocab_applied["fast"] = False

    detector.update_config({"fast_model": "best.pt"})

    assert "fast" not in detector._models
    assert "fast" not in detector._names_by_mode


def test_update_config_reapplies_classes_to_loaded_world_model(monkeypatch):
    detector = _detector(monkeypatch)
    model = _FakeWorldModel()
    detector._models["fast"] = model

    status = detector.update_config({"classes": "cat, dog"})

    assert model.applied == ["cat", "dog"]
    assert status["configured_classes"] == ["cat", "dog"]
    assert status["open_vocabulary"] is True


def test_update_config_accepts_classes_as_list(monkeypatch):
    detector = _detector(monkeypatch)
    status = detector.update_config({"classes": ["person", " hat "]})
    assert status["configured_classes"] == ["person", "hat"]


@pytest.mark.parametrize(
    "payload",
    [
        {"conf_thresh": 1.5},
        {"conf_thresh": "abc"},
        {"img_size": 16},
        {"img_size": "big"},
        {"mode": "ultra"},
        {"fast_model": "   "},
        {"classifier_min_conf": 1.5},
        {"classifier_min_conf": "abc"},
    ],
)
def test_update_config_rejects_invalid_values(monkeypatch, payload):
    detector = _detector(monkeypatch)
    with pytest.raises(ValueError):
        detector.update_config(payload)


def test_status_reports_classifier_disabled_by_default(monkeypatch):
    status = _detector(monkeypatch).status()
    assert status["classifier_model"] == ""
    assert status["classifier_enabled"] is False
    assert status["classifier_loaded"] is False
    assert status["classifier_min_conf"] == 0.0
    assert status["last_classifier_error"] is None


def test_update_config_sets_classifier_model_and_min_conf(monkeypatch):
    detector = _detector(monkeypatch)
    status = detector.update_config(
        {"classifier_model": "yolov8x-cls.pt", "classifier_min_conf": 0.3}
    )
    assert status["classifier_model"] == "yolov8x-cls.pt"
    assert status["classifier_enabled"] is True
    assert status["classifier_min_conf"] == 0.3


def test_update_config_swapping_classifier_drops_cached_model(monkeypatch):
    detector = _detector(monkeypatch)
    detector._classifier_name = "old-cls.pt"
    detector._classifier_model = _FakeClassifier()
    detector._classifier_names = {0: "tabby cat"}

    detector.update_config({"classifier_model": "new-cls.pt"})

    assert detector._classifier_model is None
    assert detector._classifier_names == {}


def test_update_config_empty_classifier_model_disables_it(monkeypatch):
    detector = _detector(monkeypatch)
    detector.update_config({"classifier_model": "yolov8x-cls.pt"})
    status = detector.update_config({"classifier_model": ""})
    assert status["classifier_model"] == ""
    assert status["classifier_enabled"] is False


def test_classify_boxes_attaches_top_species(monkeypatch):
    import numpy as np

    detector = _detector(monkeypatch)
    detector._classifier_name = "yolov8x-cls.pt"
    detector._classifier_model = _FakeClassifier(top1=1, top1conf=0.87)
    detector._classifier_names = {0: "tabby cat", 1: "golden retriever"}

    image = np.zeros((40, 40, 3), dtype=np.uint8)
    boxes = [{"xyxy": [0.0, 0.0, 20.0, 20.0], "class_id": 16, "label": "dog", "confidence": 0.8}]
    detector._classify_boxes(boxes, image, 40, 40)

    assert boxes[0]["species"] == "golden retriever"
    assert boxes[0]["species_confidence"] == 0.87
    assert boxes[0]["species_class_id"] == 1
    # The detection label is preserved alongside the new species fields.
    assert boxes[0]["label"] == "dog"


def test_classify_boxes_skips_species_below_min_conf(monkeypatch):
    import numpy as np

    detector = _detector(monkeypatch)
    detector._classifier_name = "yolov8x-cls.pt"
    detector._classifier_min_conf = 0.5
    detector._classifier_model = _FakeClassifier(top1=0, top1conf=0.3)
    detector._classifier_names = {0: "tabby cat"}

    image = np.zeros((40, 40, 3), dtype=np.uint8)
    boxes = [{"xyxy": [0.0, 0.0, 20.0, 20.0], "class_id": 15, "label": "cat", "confidence": 0.8}]
    detector._classify_boxes(boxes, image, 40, 40)

    assert "species" not in boxes[0]


def test_classify_boxes_skips_degenerate_crops(monkeypatch):
    import numpy as np

    detector = _detector(monkeypatch)
    detector._classifier_name = "yolov8x-cls.pt"
    classifier = _FakeClassifier()
    detector._classifier_model = classifier

    image = np.zeros((10, 10, 3), dtype=np.uint8)
    boxes = [{"xyxy": [5.0, 5.0, 5.0, 5.0], "class_id": 0, "label": "cat", "confidence": 0.8}]
    detector._classify_boxes(boxes, image, 10, 10)

    assert "species" not in boxes[0]
    # No valid crop means the classifier is never invoked.
    assert classifier.received is None


def test_clamp_xyxy_keeps_boxes_inside_image():
    assert clamp_xyxy([-5, 10, 120, 80], 100, 60) == [0.0, 10.0, 100.0, 60.0]


def test_clamp_xyxy_orders_reversed_points():
    assert clamp_xyxy([80, 50, 20, 10], 100, 60) == [20.0, 10.0, 80.0, 50.0]


def test_detection_error_payload_has_output_shape():
    payload = detection_error_payload(7, "bad frame")
    assert payload == {
        "frame_id": 7,
        "width": 0,
        "height": 0,
        "inference_ms": 0.0,
        "boxes": [],
        "error": "bad frame",
    }


def test_device_supports_half_only_for_cuda_targets():
    assert device_supports_half(0) is True
    assert device_supports_half("0") is True
    assert device_supports_half("cuda") is True
    assert device_supports_half("cuda:0") is True
    assert device_supports_half("cpu") is False
    assert device_supports_half(None) is False
