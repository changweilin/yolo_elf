import sys
import types
from pathlib import Path

import pytest

from app.config import DETECT_TASKS, MAX_ACTIVE_TASKS, get_settings
from app.detector import (
    DecodedImage,
    YoloDetector,
    clamp_xyxy,
    detection_error_payload,
    device_supports_half,
    model_supports_end2end,
)


# Cache key for the default preset. Model caches are keyed by task (with the
# fast/accurate split kept only for "detect"), not by mode alone.
FAST = "detect:fast"


def _detector(monkeypatch):
    for name in (
        "DETECT_MODE",
        "DETECT_TASK",
        "DETECT_TASKS",
        "YOLO_MODEL",
        "YOLO_MODEL_ACCURATE",
        "YOLO_CLASSES",
        "YOLO_TRACK",
        "YOLO_TRACKER",
        "YOLO_END2END",
        "YOLO_MAX_DET",
        "YOLO_EXPORT",
        "CLASSIFIER_MODEL",
        "CLASSIFIER_MIN_CONF",
    ):
        monkeypatch.delenv(name, raising=False)
    return YoloDetector(get_settings())


def test_export_target_is_a_pure_path_decision():
    target = YoloDetector._export_target
    # export disabled -> load the name as-is
    assert target("yolov8s.pt", "") is None
    # a .pt gets the matching artifact suffix, directory preserved
    assert target("yolov8s.pt", "engine") == Path("yolov8s.engine")
    assert target("yolov8s.pt", "onnx") == Path("yolov8s.onnx")
    assert target("weights/yolov8x.pt", "onnx") == Path("weights/yolov8x.onnx")
    # an already-exported artifact is never re-exported
    assert target("yolov8s.engine", "engine") is None
    assert target("yolov8s.onnx", "engine") is None
    # unknown format -> no export
    assert target("yolov8s.pt", "trt") is None


def test_export_target_marks_a_forced_end2end_head(monkeypatch):
    target = YoloDetector._export_target
    # "auto" keeps the original filename layout, byte-for-byte.
    assert target("yolo26s.pt", "onnx", "auto") == Path("yolo26s.onnx")
    # A forced head gets its own artifact so flipping YOLO_END2END re-exports
    # instead of reusing a cache built with the other head.
    assert target("yolo26s.pt", "onnx", "on") == Path("yolo26s-e2eon.onnx")
    assert target("yolo26s.pt", "engine", "off") == Path("yolo26s-e2eoff.engine")
    # The directory is preserved and the marker never leaks into it.
    assert target("weights/yolo26x.pt", "onnx", "on") == Path("weights/yolo26x-e2eon.onnx")


def test_resolve_model_source_passthrough_when_disabled(monkeypatch):
    monkeypatch.delenv("YOLO_EXPORT", raising=False)
    detector = YoloDetector(get_settings())
    assert detector._resolve_model_source("yolov8s.pt") == "yolov8s.pt"


def test_resolve_model_source_reuses_cached_artifact(monkeypatch, tmp_path):
    monkeypatch.setenv("YOLO_EXPORT", "onnx")
    detector = YoloDetector(get_settings())
    source = tmp_path / "model.pt"
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"stub")
    # a cached export on disk is loaded without importing/exporting
    assert detector._resolve_model_source(str(source)) == str(artifact)
    assert detector._export_error is None


def test_resolve_model_source_falls_back_when_export_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("YOLO_EXPORT", "onnx")
    detector = YoloDetector(get_settings())

    class _BoomYOLO:
        def __init__(self, name):
            self.name = name

        def export(self, **kwargs):
            raise RuntimeError("no toolchain")

    fake = types.ModuleType("ultralytics")
    fake.YOLO = _BoomYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake)

    source = tmp_path / "model.pt"  # no cached artifact -> triggers the export attempt
    assert detector._resolve_model_source(str(source)) == str(source)
    assert "failed" in (detector._export_error or "")


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


class _FakeEnd2EndModel:
    """A YOLO26-style wrapper: the flag lives on the inner nn.Module."""

    class _Inner:
        end2end = True

    def __init__(self):
        self.names = {0: "person"}
        self.model = self._Inner()


class _FakeScalar:
    """Mimics the box-tensor cell chain: ``.detach().cpu().item()/.tolist()``."""

    def __init__(self, value):
        self._value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def item(self):
        return self._value

    def tolist(self):
        return self._value


class _FakeResultBox:
    def __init__(self, xyxy, class_id, confidence, track_id=None):
        self.xyxy = [_FakeScalar(list(xyxy))]
        self.cls = [_FakeScalar(class_id)]
        self.conf = [_FakeScalar(confidence)]
        # `boxes.id` is None until the tracker confirms the track (and always
        # when tracking is off), matching the Ultralytics contract.
        self.id = None if track_id is None else [_FakeScalar(track_id)]


class _FakeDetectResult:
    def __init__(self, boxes, masks=None, keypoints=None):
        self.boxes = boxes
        self.masks = masks
        self.keypoints = keypoints


class _FakeTensor:
    """Mimics a torch tensor's ``.detach().cpu().tolist()`` chain."""

    def __init__(self, value):
        self._value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self._value


class _FakeMasks:
    def __init__(self, polygons):
        self.xy = polygons


class _FakeKeypoints:
    def __init__(self, xy, conf=None):
        self.xy = _FakeTensor(xy)
        self.conf = None if conf is None else _FakeTensor(conf)


class _FakeObb:
    def __init__(self, quads, aligned, classes, confidences, ids=None):
        self.xyxyxyxy = _FakeTensor(quads)
        self.xyxy = _FakeTensor(aligned)
        self.cls = _FakeTensor(classes)
        self.conf = _FakeTensor(confidences)
        self.id = None if ids is None else _FakeTensor(ids)


class _FakeObbResult:
    def __init__(self, obb):
        self.obb = obb
        self.boxes = None


class _FakeRoutingModel:
    """Records whether detect() reached ``track`` vs ``predict``."""

    def __init__(self):
        self.predict_calls = 0
        self.track_calls = 0
        self.last_track_kwargs = None

    def predict(self, **kwargs):
        self.predict_calls += 1
        return ["predict"]

    def track(self, **kwargs):
        self.track_calls += 1
        self.last_track_kwargs = kwargs
        return ["track"]


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
    assert status["model"] == "yolo26s.pt"
    assert status["available_modes"] == ["fast", "accurate"]
    assert status["models"] == {"fast": "yolo26s.pt", "accurate": "yolo26x.pt"}
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
    assert status["model"] == "yolo26x.pt"


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
    detector._models[FAST] = _FakeClosedModel()
    detector._names_by_preset[FAST] = {0: "person"}
    detector._open_vocab_applied[FAST] = False

    detector.update_config({"fast_model": "best.pt"})

    assert FAST not in detector._models
    assert FAST not in detector._names_by_preset


def test_update_config_reapplies_classes_to_loaded_world_model(monkeypatch):
    detector = _detector(monkeypatch)
    model = _FakeWorldModel()
    detector._models[FAST] = model

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
        {"classifier_max_boxes": 0},
        {"classifier_max_boxes": "abc"},
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
    assert status["classifier_max_boxes"] == 5
    assert status["last_classifier_error"] is None


def test_update_config_sets_classifier_model_and_min_conf(monkeypatch):
    detector = _detector(monkeypatch)
    status = detector.update_config(
        {
            "classifier_model": "yolov8x-cls.pt",
            "classifier_min_conf": 0.3,
            "classifier_max_boxes": 8,
        }
    )
    assert status["classifier_model"] == "yolov8x-cls.pt"
    assert status["classifier_enabled"] is True
    assert status["classifier_min_conf"] == 0.3
    assert status["classifier_max_boxes"] == 8


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


def test_classify_boxes_only_classifies_largest_boxes(monkeypatch):
    import numpy as np

    detector = _detector(monkeypatch)
    detector._classifier_name = "yolov8x-cls.pt"
    detector._classifier_max_boxes = 5
    classifier = _FakeClassifier(top1=0, top1conf=0.9)
    detector._classifier_model = classifier
    detector._classifier_names = {0: "tabby cat"}

    image = np.zeros((200, 200, 3), dtype=np.uint8)
    # Seven boxes of strictly increasing area; the two smallest must be skipped.
    boxes = [
        {"xyxy": [0.0, 0.0, float(side), float(side)], "class_id": 0, "label": "cat",
         "confidence": 0.8}
        for side in (10, 20, 30, 40, 50, 60, 70)
    ]
    detector._classify_boxes(boxes, image, 200, 200)

    classified = [box for box in boxes if "species" in box]
    # Only the top-5 by area get a species; the 10px and 20px boxes do not.
    assert len(classified) == 5
    assert len(classifier.received["source"]) == 5
    assert "species" not in boxes[0]
    assert "species" not in boxes[1]
    assert all("species" in box for box in boxes[2:])


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


def test_status_reports_end2end_defaults(monkeypatch):
    status = _detector(monkeypatch).status()
    assert status["end2end"] == "auto"
    assert status["max_det"] == 300
    # Unknown until the weights are actually loaded.
    assert status["end2end_capable"] is None


def test_status_reports_end2end_capability_of_loaded_model(monkeypatch):
    detector = _detector(monkeypatch)

    detector._models[FAST] = _FakeEnd2EndModel()
    detector._end2end_native[FAST] = True
    assert detector.status()["end2end_capable"] is True

    # A closed-set YOLOv8-era model has no one-to-one head.
    detector._models[FAST] = _FakeClosedModel()
    detector._end2end_native[FAST] = False
    assert detector.status()["end2end_capable"] is False


def test_end2end_capability_survives_a_forced_off_prediction(monkeypatch):
    """Regression: `end2end=False` at predict time overwrites `model.end2end`.

    Reading the attribute back after inference reported "this model has no
    NMS-free head" for a YOLO26 checkpoint that plainly does, so the capability
    is sampled once at load time instead.
    """
    detector = _detector(monkeypatch)
    model = _FakeEnd2EndModel()
    detector._models[FAST] = model
    detector._end2end_native[FAST] = model_supports_end2end(model)

    # Ultralytics mutates the inner module when the kwarg is forced.
    detector.update_config({"end2end": "off"})
    model.model.end2end = False

    assert detector.status()["end2end_capable"] is True


def test_swapping_the_model_forgets_its_end2end_capability(monkeypatch):
    detector = _detector(monkeypatch)
    detector._models[FAST] = _FakeEnd2EndModel()
    detector._end2end_native[FAST] = True

    detector.update_config({"fast_model": "yolov8s.pt"})

    assert FAST not in detector._end2end_native
    assert detector.status()["end2end_capable"] is None


def test_update_config_sets_end2end_and_max_det(monkeypatch):
    detector = _detector(monkeypatch)
    status = detector.update_config({"end2end": "OFF", "max_det": 25})
    assert status["end2end"] == "off"
    assert status["max_det"] == 25


@pytest.mark.parametrize(
    "payload", [{"end2end": "yes"}, {"end2end": 1}, {"max_det": 0}, {"max_det": 1001},
                {"max_det": "many"}]
)
def test_update_config_rejects_invalid_end2end_and_max_det(monkeypatch, payload):
    detector = _detector(monkeypatch)
    with pytest.raises(ValueError):
        detector.update_config(payload)


def test_switching_end2end_keeps_cached_weights_when_export_is_off(monkeypatch):
    # Without export the head is a per-call predict kwarg, so there is nothing to
    # reload — dropping the weights would be a pointless stall.
    detector = _detector(monkeypatch)
    detector._models[FAST] = _FakeClosedModel()

    detector.update_config({"end2end": "on"})

    assert FAST in detector._models


def test_switching_end2end_drops_cached_weights_when_exporting(monkeypatch):
    # With export on, the head is baked into the artifact and each head has its
    # own filename, so the cache must go or the old artifact keeps serving.
    monkeypatch.setenv("YOLO_EXPORT", "onnx")
    detector = YoloDetector(get_settings())
    detector._models[FAST] = _FakeClosedModel()
    detector._loaded_sources[FAST] = "yolo26s.onnx"
    detector._tracker_models[(FAST, "cam2")] = _FakeClosedModel()

    detector.update_config({"end2end": "off"})

    assert detector._models == {}
    assert detector._loaded_sources == {}
    assert detector._tracker_models == {}
    # A no-op write must not throw the weights away.
    detector._models[FAST] = _FakeClosedModel()
    detector.update_config({"end2end": "off"})
    assert FAST in detector._models


def test_prediction_kwargs_omit_end2end_on_auto(monkeypatch):
    # "auto" must not pass the key at all: `end2end` only exists from ultralytics
    # 8.4, and an unknown predict key is a hard error there.
    kwargs = _detector(monkeypatch)._prediction_kwargs("src")
    assert "end2end" not in kwargs
    assert kwargs["max_det"] == 300


@pytest.mark.parametrize(("mode", "expected"), [("on", True), ("off", False)])
def test_prediction_kwargs_forward_forced_end2end(monkeypatch, mode, expected):
    detector = _detector(monkeypatch)
    detector.update_config({"end2end": mode, "max_det": 42})
    kwargs = detector._prediction_kwargs("src")
    assert kwargs["end2end"] is expected
    assert kwargs["max_det"] == 42


def test_tracking_path_carries_the_same_end2end_kwargs(monkeypatch):
    # The tracked path shares _prediction_kwargs, so the NMS-free head applies
    # to tracking too (Ultralytics tracks detect/segment/pose/obb results).
    detector = _detector(monkeypatch)
    detector._track_enabled = True
    detector.update_config({"end2end": "on", "max_det": 7})
    model = _FakeRoutingModel()

    detector._infer(model, "src")

    assert model.last_track_kwargs["end2end"] is True
    assert model.last_track_kwargs["max_det"] == 7


def test_status_reports_tracking_defaults(monkeypatch):
    status = _detector(monkeypatch).status()
    assert status["track_enabled"] is True
    assert status["tracker"] == "bytetrack.yaml"


def test_extract_boxes_includes_tracker_id(monkeypatch):
    detector = _detector(monkeypatch)
    detector._names = {0: "person"}
    result = _FakeDetectResult([_FakeResultBox([10, 20, 30, 40], 0, 0.91, track_id=7)])

    boxes = detector._extract_boxes(result, 100, 100)

    assert boxes[0]["track_id"] == 7
    assert boxes[0]["label"] == "person"
    assert boxes[0]["xyxy"] == [10.0, 20.0, 30.0, 40.0]


def test_extract_boxes_track_id_none_when_untracked(monkeypatch):
    detector = _detector(monkeypatch)
    detector._names = {5: "dog"}
    result = _FakeDetectResult([_FakeResultBox([0, 0, 10, 10], 5, 0.5, track_id=None)])

    boxes = detector._extract_boxes(result, 50, 50)

    assert boxes[0]["track_id"] is None


def test_infer_tracks_when_enabled(monkeypatch):
    detector = _detector(monkeypatch)
    detector._track_enabled = True
    detector._tracker = "bytetrack.yaml"
    model = _FakeRoutingModel()

    detector._infer(model, "src")

    assert model.track_calls == 1
    assert model.predict_calls == 0
    assert model.last_track_kwargs["persist"] is True
    assert model.last_track_kwargs["tracker"] == "bytetrack.yaml"
    assert model.last_track_kwargs["source"] == "src"


def test_infer_predicts_when_tracking_disabled(monkeypatch):
    detector = _detector(monkeypatch)
    detector._track_enabled = False
    model = _FakeRoutingModel()

    detector._infer(model, "src")

    assert model.predict_calls == 1
    assert model.track_calls == 0


def test_detector_defaults_to_the_detect_task(monkeypatch):
    status = _detector(monkeypatch).status()
    assert status["task"] == "detect"
    assert status["available_tasks"] == list(DETECT_TASKS)
    assert status["emits_boxes"] is True
    assert status["emits_raster"] is False
    assert status["task_models"]["segment"] == "yolo26s-seg.pt"
    assert status["task_models"]["openvocab"] == "yoloe-26s-seg.pt"


def test_set_task_rejects_unknown_task(monkeypatch):
    detector = _detector(monkeypatch)
    with pytest.raises(ValueError):
        detector.set_task("nonsense")


def test_single_task_status_keeps_the_original_shape(monkeypatch):
    # The multi-head fields are additive: a lone task still reports itself as
    # `task`, and `tasks` is just that one entry.
    status = _detector(monkeypatch).status()
    assert status["task"] == "detect"
    assert status["tasks"] == ["detect"]
    assert status["max_active_tasks"] == MAX_ACTIVE_TASKS


def test_set_tasks_runs_several_heads(monkeypatch):
    detector = _detector(monkeypatch)
    status = detector.update_config({"tasks": ["pose", "depth"]})

    assert status["tasks"] == ["pose", "depth"]
    # The first entry is the primary head every single-task client still reads.
    assert status["task"] == "pose"
    assert status["emits_boxes"] is True
    assert status["emits_raster"] is True


def test_set_tasks_rejects_two_rasters(monkeypatch):
    # Each raster repaints every pixel, so the second would just hide the first.
    detector = _detector(monkeypatch)
    with pytest.raises(ValueError):
        detector.set_tasks(["semantic", "depth"])


def test_set_task_collapses_back_to_one_head(monkeypatch):
    detector = _detector(monkeypatch)
    detector.set_tasks(["detect", "pose"])
    detector.set_task("obb")
    assert detector.tasks == ("obb",)


def test_loaded_requires_every_active_head(monkeypatch):
    detector = _detector(monkeypatch)
    detector.set_tasks(["detect", "pose"])
    detector._models[FAST] = _FakeClosedModel()
    assert detector.loaded is False
    detector._models["pose"] = _FakeClosedModel()
    assert detector.loaded is True


def test_detect_merges_every_head_into_one_payload(monkeypatch):
    # Two heads on one frame: the boxes concatenate, each tagged with the head
    # that produced it (track ids only mean anything within one head), and the
    # reported inference time is the sum of the passes.
    detector = _detector(monkeypatch)
    detector.set_tasks(["detect", "pose"])
    monkeypatch.setattr(
        detector, "_decode_jpeg", lambda _raw: DecodedImage(data="frame", width=100, height=80)
    )
    # detect() only runs heads whose weights are already resident.
    detector._models = {FAST: _FakeClosedModel(), "pose": _FakeClosedModel()}
    monkeypatch.setattr(detector, "_ensure_model", lambda task=None: task)
    results = {
        "detect": _FakeDetectResult([_FakeResultBox([1, 2, 3, 4], 0, 0.9, track_id=1)]),
        "pose": _FakeDetectResult([_FakeResultBox([5, 6, 7, 8], 0, 0.8, track_id=1)]),
    }
    monkeypatch.setattr(detector, "_infer", lambda _model, _src, task=None: [results[task]])
    detector._names_by_preset = {FAST: {0: "person"}, "pose": {0: "person"}}

    payload = detector.detect(b"jpeg", 7)

    assert payload["tasks"] == ["detect", "pose"]
    assert [box["task"] for box in payload["boxes"]] == ["detect", "pose"]
    assert set(payload["task_ms"]) == {"detect", "pose"}
    assert payload["inference_ms"] == pytest.approx(
        sum(payload["task_ms"].values()), abs=0.05
    )


def test_detect_omits_the_task_breakdown_for_one_head(monkeypatch):
    detector = _detector(monkeypatch)
    monkeypatch.setattr(
        detector, "_decode_jpeg", lambda _raw: DecodedImage(data="frame", width=10, height=10)
    )
    detector._models = {FAST: _FakeClosedModel()}
    monkeypatch.setattr(detector, "_ensure_model", lambda task=None: task)
    monkeypatch.setattr(detector, "_infer", lambda *_args, **_kwargs: [_FakeDetectResult([])])

    payload = detector.detect(b"jpeg", 1)

    assert payload["tasks"] == ["detect"]
    assert "task_ms" not in payload
    assert "pending_tasks" not in payload


class _FakeThread:
    """Records what would have been spawned instead of actually loading."""

    started: list = []

    def __init__(self, target=None, args=(), **_kwargs):
        self._target = target
        self._args = args

    def start(self):
        _FakeThread.started.append((self._target, self._args))


@pytest.fixture
def fake_threads(monkeypatch):
    _FakeThread.started = []
    monkeypatch.setattr("app.detector.threading.Thread", _FakeThread)
    return _FakeThread.started


def test_detect_never_loads_weights_on_the_worker_thread(monkeypatch, fake_threads):
    # The regression this guards: loading here blocks the one thread every
    # camera's frames pass through, so a first-use checkpoint download froze the
    # viewer on its last frame for the whole download.
    detector = _detector(monkeypatch)
    detector.set_tasks(["detect", "pose"])
    monkeypatch.setattr(
        detector, "_decode_jpeg", lambda _raw: DecodedImage(data="frame", width=100, height=80)
    )
    # Only the detect head is resident; pose is not.
    detector._models = {FAST: _FakeClosedModel()}
    detector._names_by_preset = {FAST: {0: "person"}}
    monkeypatch.setattr(detector, "_ensure_model", lambda task=None: task)
    monkeypatch.setattr(
        detector,
        "_infer",
        lambda _model, _src, task=None: [
            _FakeDetectResult([_FakeResultBox([1, 2, 3, 4], 0, 0.9)])
        ],
    )

    payload = detector.detect(b"jpeg", 3)

    # The frame still went out, carrying the head that was ready.
    assert payload["frame_id"] == 3
    assert [box["task"] for box in payload["boxes"]] == ["detect"]
    assert payload["pending_tasks"] == ["pose"]
    assert detector.status()["loading_tasks"] == ["pose"]
    assert len(fake_threads) == 1


def test_a_failed_load_is_not_retried_on_every_frame(monkeypatch, fake_threads):
    # A checkpoint that cannot be fetched has no cache to hit, so without the
    # cooldown every frame would start another download attempt.
    detector = _detector(monkeypatch)
    detector._request_background_load("detect")
    assert len(fake_threads) == 1

    target, args = fake_threads[0]
    monkeypatch.setattr(
        detector, "_ensure_model", lambda task=None: (_ for _ in ()).throw(RuntimeError("404"))
    )
    target(*args)

    detector._request_background_load("detect")
    assert len(fake_threads) == 1

    # Renaming the model is a new attempt, so the cooldown must not apply to it.
    detector._drop_preset(FAST)
    detector._request_background_load("detect")
    assert len(fake_threads) == 2


def test_raster_tasks_report_no_boxes(monkeypatch):
    status = _detector(monkeypatch).update_config({"task": "depth"})
    assert status["task"] == "depth"
    assert status["emits_boxes"] is False
    assert status["emits_raster"] is True


def test_each_task_gets_its_own_weights_cache(monkeypatch):
    # detect keeps the fast/accurate split; the other heads key on the task name,
    # so switching task must not evict the detect weights (or vice versa).
    detector = _detector(monkeypatch)
    assert detector._preset_key() == "detect:fast"
    detector.set_mode("accurate")
    assert detector._preset_key() == "detect:accurate"
    detector.set_task("pose")
    assert detector._preset_key() == "pose"

    detector._models["pose"] = _FakeClosedModel()
    detector._models["detect:accurate"] = _FakeClosedModel()
    detector.update_config({"task_models": {"pose": "custom-pose.pt"}})
    # Only the pose slot is dropped.
    assert "pose" not in detector._models
    assert "detect:accurate" in detector._models


def test_task_models_rejects_unknown_or_detect_keys(monkeypatch):
    detector = _detector(monkeypatch)
    with pytest.raises(ValueError):
        detector.update_config({"task_models": {"nonsense": "x.pt"}})
    # "detect" has its own fast/accurate pair and is not settable here.
    with pytest.raises(ValueError):
        detector.update_config({"task_models": {"detect": "x.pt"}})


def test_raster_tasks_never_track(monkeypatch):
    # Ultralytics raises for `mode=track` on semantic/depth, so those tasks must
    # take the predict path even with tracking switched on.
    detector = _detector(monkeypatch)
    detector._track_enabled = True
    model = _FakeRoutingModel()

    detector._infer(model, "src", "semantic")

    assert model.predict_calls == 1
    assert model.track_calls == 0


def test_open_vocabulary_task_forces_fp32(monkeypatch):
    # YOLOE mixes float32 text embeddings into a half-precision backbone and
    # dies with a dtype mismatch, so half is dropped for this task only.
    detector = _detector(monkeypatch)
    detector._half_enabled = True

    assert detector._half_for_task("detect") is True
    assert detector._half_for_task("openvocab") is False
    assert "half" not in detector._prediction_kwargs("src", "openvocab")
    assert detector._prediction_kwargs("src", "segment")["half"] is True


def test_extract_boxes_attaches_segmentation_masks(monkeypatch):
    detector = _detector(monkeypatch)
    result = _FakeDetectResult(
        [_FakeResultBox([10, 10, 40, 40], 0, 0.9)],
        masks=_FakeMasks([[(12, 12), (38, 14), (36, 38), (14, 36)]]),
    )

    boxes = detector._extract_boxes(result, 100, 100, {0: "person"}, "segment")

    assert boxes[0]["mask"] == [[12.0, 12.0], [38.0, 14.0], [36.0, 38.0], [14.0, 36.0]]


def test_mask_polygons_are_subsampled_not_truncated(monkeypatch):
    detector = _detector(monkeypatch)
    # A 500-point contour must come back capped but still spanning the shape —
    # taking a prefix would clip the polygon to one edge of the object.
    polygon = [(float(i), float(i)) for i in range(500)]
    simplified = detector._simplify_polygon(polygon, 1000, 1000)

    assert len(simplified) == 48
    assert simplified[0] == [0.0, 0.0]
    assert simplified[-1][0] > 480


def test_extract_boxes_attaches_pose_keypoints(monkeypatch):
    detector = _detector(monkeypatch)
    result = _FakeDetectResult(
        [_FakeResultBox([0, 0, 50, 50], 0, 0.9)],
        keypoints=_FakeKeypoints([[[10.0, 20.0], [30.0, 40.0]]], [[0.9, 0.1]]),
    )

    boxes = detector._extract_boxes(result, 100, 100, {0: "person"}, "pose")

    assert boxes[0]["keypoints"] == [[10.0, 20.0, 0.9], [30.0, 40.0, 0.1]]


def test_keypoints_default_to_full_confidence_without_scores(monkeypatch):
    detector = _detector(monkeypatch)
    result = _FakeDetectResult(
        [_FakeResultBox([0, 0, 50, 50], 0, 0.9)],
        keypoints=_FakeKeypoints([[[10.0, 20.0]]]),
    )

    boxes = detector._extract_boxes(result, 100, 100, {0: "person"}, "pose")

    assert boxes[0]["keypoints"] == [[10.0, 20.0, 1.0]]


def test_obb_boxes_carry_both_the_quad_and_an_axis_aligned_box(monkeypatch):
    detector = _detector(monkeypatch)
    result = _FakeObbResult(
        _FakeObb(
            quads=[[[10, 0], [30, 10], [20, 30], [0, 20]]],
            aligned=[[0, 0, 30, 30]],
            classes=[3],
            confidences=[0.77],
            ids=[9],
        )
    )

    boxes = detector._extract_boxes(result, 100, 100, {3: "ship"}, "obb")

    assert boxes[0]["obb"] == [10.0, 0.0, 30.0, 10.0, 20.0, 30.0, 0.0, 20.0]
    # The axis-aligned box is what zones / alerts / history consume, so it has to
    # be present and clamped exactly like a plain detection.
    assert boxes[0]["xyxy"] == [0.0, 0.0, 30.0, 30.0]
    assert boxes[0]["label"] == "ship"
    assert boxes[0]["track_id"] == 9


def test_obb_quads_are_not_clamped_to_the_frame(monkeypatch):
    # A rotated box that runs off the edge genuinely has corners outside it;
    # clamping them one by one would shear the rectangle into another shape.
    detector = _detector(monkeypatch)
    result = _FakeObbResult(
        _FakeObb(
            quads=[[[-25, 308], [11, 326], [38, 273], [2, 255]]],
            aligned=[[-25, 255, 38, 326]],
            classes=[0],
            confidences=[0.5],
        )
    )

    boxes = detector._extract_boxes(result, 100, 400, {0: "field"}, "obb")

    assert boxes[0]["obb"][0] == -25.0
    assert boxes[0]["xyxy"][0] == 0.0  # ...but the axis-aligned box still is


def test_obb_without_tracking_reports_null_ids(monkeypatch):
    detector = _detector(monkeypatch)
    result = _FakeObbResult(
        _FakeObb([[[0, 0], [1, 0], [1, 1], [0, 1]]], [[0, 0, 1, 1]], [0], [0.5])
    )
    boxes = detector._extract_boxes(result, 10, 10, {0: "x"}, "obb")
    assert boxes[0]["track_id"] is None


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
