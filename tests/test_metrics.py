from app.metrics import render_metrics


def _sample_status():
    return {
        "uptime_sec": 12.5,
        "camera_connected": True,
        "viewer_count": 2,
        "frames_received": 100,
        "frames_processed": 97,
        "frames_dropped": 3,
        "receive_fps": 9.8,
        "process_fps": 9.5,
        "queue_depth": 1,
        "last_frame_bytes": 45000,
        "last_inference_ms": 18.6,
        "last_queue_latency_ms": 4.0,
        "last_total_latency_ms": 25.0,
        "avg_inference_ms": 20.1,
        "avg_queue_latency_ms": 5.0,
        "avg_total_latency_ms": 27.0,
        "detector": {
            "loaded": True,
            "warmed_up": False,
            "conf_thresh": 0.2,
            "img_size": 1280,
            "mode": "fast",
            "model": "yolov8s.pt",
            "resolved_device": "cpu",
            "end2end": "auto",
            "end2end_capable": True,
            "max_det": 300,
        },
        "recordings": {"recordings_saved": 4, "bytes_saved": 999},
        "remote_storage": {
            "records_uploaded": 7,
            "records_failed": 1,
            "records_dropped": 2,
            "recordings_uploaded": 3,
            "queue_depth": 5,
        },
        "alerts": {
            "events_fired": 6,
            "webhook_sent": 5,
            "webhook_failed": 1,
            "rules": [{"name": "a"}, {"name": "b"}],
        },
        "zones": {"zones": [{"name": "door"}]},
        "events": {"sightings_written": 8, "active_sightings": 2},
    }


def _value_for(text, name):
    for line in text.splitlines():
        if line.startswith(name + " "):
            return line.split(" ", 1)[1]
    return None


def test_render_includes_help_type_and_values():
    text = render_metrics(_sample_status())

    assert "# HELP yolo_elf_frames_processed_total" in text
    assert "# TYPE yolo_elf_frames_processed_total counter" in text
    assert _value_for(text, "yolo_elf_frames_processed_total") == "97"
    assert _value_for(text, "yolo_elf_viewers") == "2"
    assert _value_for(text, "yolo_elf_alerts_fired_total") == "6"
    assert _value_for(text, "yolo_elf_active_sightings") == "2"
    # derived counts from list fields
    assert _value_for(text, "yolo_elf_alert_rules") == "2"
    assert _value_for(text, "yolo_elf_zones") == "1"
    # trailing newline per exposition convention
    assert text.endswith("\n")


def test_counters_and_gauges_have_correct_type_lines():
    text = render_metrics(_sample_status())
    assert "# TYPE yolo_elf_frames_dropped_total counter" in text
    assert "# TYPE yolo_elf_inference_ms gauge" in text
    assert "# TYPE yolo_elf_detector_loaded gauge" in text


def test_booleans_render_as_one_or_zero():
    text = render_metrics(_sample_status())
    assert _value_for(text, "yolo_elf_camera_connected") == "1"
    assert _value_for(text, "yolo_elf_detector_loaded") == "1"
    assert _value_for(text, "yolo_elf_detector_warmed_up") == "0"


def test_floats_keep_precision_integers_stay_integer():
    text = render_metrics(_sample_status())
    assert _value_for(text, "yolo_elf_inference_ms") == "18.6"
    assert _value_for(text, "yolo_elf_detector_conf_threshold") == "0.2"
    # integer-valued fields print without a trailing .0
    assert _value_for(text, "yolo_elf_last_frame_bytes") == "45000"


def test_detector_info_carries_low_cardinality_labels():
    text = render_metrics(_sample_status())
    info = next(
        line for line in text.splitlines() if line.startswith("yolo_elf_detector_info{")
    )
    assert 'mode="fast"' in info
    assert 'model="yolov8s.pt"' in info
    assert 'device="cpu"' in info
    assert 'end2end="auto"' in info
    assert info.endswith(" 1")
    # track_id must never become a label (cardinality guard)
    assert "track_id" not in text


def test_end2end_and_max_det_are_exposed_as_gauges():
    text = render_metrics(_sample_status())
    assert _value_for(text, "yolo_elf_detector_max_det") == "300"
    assert _value_for(text, "yolo_elf_detector_end2end_capable") == "1"
    # Not-yet-loaded weights report unknown as 0, matching detector_loaded.
    unloaded = render_metrics({"detector": {"end2end_capable": None}})
    assert _value_for(unloaded, "yolo_elf_detector_end2end_capable") == "0"


def test_missing_sections_default_to_zero():
    text = render_metrics({})
    assert _value_for(text, "yolo_elf_frames_processed_total") == "0"
    assert _value_for(text, "yolo_elf_active_sightings") == "0"
    assert _value_for(text, "yolo_elf_alert_rules") == "0"
    # info metric still emits with empty labels rather than crashing
    assert "yolo_elf_detector_info{" in text
