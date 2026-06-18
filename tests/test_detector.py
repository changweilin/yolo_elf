import pytest

from app.config import get_settings
from app.detector import (
    YoloDetector,
    clamp_xyxy,
    detection_error_payload,
    device_supports_half,
)


def _detector(monkeypatch):
    for name in ("DETECT_MODE", "YOLO_MODEL", "YOLO_MODEL_ACCURATE"):
        monkeypatch.delenv(name, raising=False)
    return YoloDetector(get_settings())


def test_detector_defaults_to_fast_mode(monkeypatch):
    status = _detector(monkeypatch).status()
    assert status["mode"] == "fast"
    assert status["model"] == "yolov8s.pt"
    assert status["available_modes"] == ["fast", "accurate"]
    assert status["models"] == {"fast": "yolov8s.pt", "accurate": "yolov8x.pt"}
    assert status["loaded"] is False


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
