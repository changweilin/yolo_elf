from __future__ import annotations

from typing import Any

# Prometheus text exposition (version 0.0.4). Every value here is derived from
# the already-computed /api/status snapshot, so this module stays a pure
# formatter with no engine state of its own. Counters map to cumulative status
# fields (monotonic by construction — never instantaneous values), and the only
# labelled series is a low-cardinality detector "info" metric; per-object
# dimensions like track_id are deliberately never used as labels.

_PREFIX = "yolo_elf_"
CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def _num(value: Any) -> float:
    """Coerce a status field to a metric value; bools -> 1/0, junk -> 0."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _format_value(value: float) -> str:
    if value != value or value in (float("inf"), float("-inf")):
        return "0"
    if value.is_integer():
        return str(int(value))
    return repr(value)


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def render_metrics(status: dict[str, Any]) -> str:
    """Render an /api/status snapshot as Prometheus exposition text."""
    detector = status.get("detector") or {}
    recordings = status.get("recordings") or {}
    remote = status.get("remote_storage") or {}
    alerts = status.get("alerts") or {}
    zones = status.get("zones") or {}
    events = status.get("events") or {}
    cameras = status.get("cameras") or {}

    lines: list[str] = []

    def emit(
        name: str,
        mtype: str,
        help_text: str,
        value: Any,
        labels: dict[str, Any] | None = None,
    ) -> None:
        full = _PREFIX + name
        lines.append(f"# HELP {full} {help_text}")
        lines.append(f"# TYPE {full} {mtype}")
        rendered = _format_value(_num(value))
        if labels:
            pairs = ",".join(
                f'{key}="{_escape_label(str(val))}"' for key, val in labels.items()
            )
            lines.append(f"{full}{{{pairs}}} {rendered}")
        else:
            lines.append(f"{full} {rendered}")

    def emit_per_camera(name: str, mtype: str, help_text: str, field: str) -> None:
        """One series per camera under a single HELP/TYPE header.

        Cardinality is bounded by CAMERAS (MAX_CAMERAS caps it), which is why a
        camera label is safe here while per-object labels like track_id are not.
        """
        full = _PREFIX + name
        lines.append(f"# HELP {full} {help_text}")
        lines.append(f"# TYPE {full} {mtype}")
        for camera_id, stream in cameras.items():
            label = _escape_label(str(camera_id))
            lines.append(f'{full}{{camera="{label}"}} {_format_value(_num(stream.get(field)))}')

    # Pipeline / stream
    emit("uptime_seconds", "gauge", "Seconds since the server started.", status.get("uptime_sec"))
    emit("camera_connected", "gauge", "1 if a recorder is currently streaming, else 0.", status.get("camera_connected"))
    emit("viewers", "gauge", "Connected viewer clients.", status.get("viewer_count"))
    emit("frames_received_total", "counter", "Frames received from the recorder.", status.get("frames_received"))
    emit("frames_processed_total", "counter", "Frames run through detection.", status.get("frames_processed"))
    emit("frames_dropped_total", "counter", "Frames dropped by the single-slot queue.", status.get("frames_dropped"))
    emit("receive_fps", "gauge", "Frames received per second (lifetime average).", status.get("receive_fps"))
    emit("process_fps", "gauge", "Frames processed per second (lifetime average).", status.get("process_fps"))
    emit("queue_depth", "gauge", "Frames waiting in the detection queue.", status.get("queue_depth"))
    emit("last_frame_bytes", "gauge", "Size of the most recent frame in bytes.", status.get("last_frame_bytes"))
    emit("inference_ms", "gauge", "Inference time of the most recent frame (ms).", status.get("last_inference_ms"))
    emit("queue_latency_ms", "gauge", "Queue wait of the most recent frame (ms).", status.get("last_queue_latency_ms"))
    emit("total_latency_ms", "gauge", "End-to-end latency of the most recent frame (ms).", status.get("last_total_latency_ms"))
    emit("avg_inference_ms", "gauge", "Average inference time since start (ms).", status.get("avg_inference_ms"))
    emit("avg_queue_latency_ms", "gauge", "Average queue wait since start (ms).", status.get("avg_queue_latency_ms"))
    emit("avg_total_latency_ms", "gauge", "Average end-to-end latency since start (ms).", status.get("avg_total_latency_ms"))

    # Per-camera stream (the aggregate series above stay the server-wide roll-up)
    emit("cameras", "gauge", "Cameras configured in the registry.", len(cameras))
    emit_per_camera(
        "camera_connected", "gauge", "1 if this camera has a recorder streaming.", "camera_connected"
    )
    emit_per_camera(
        "camera_frames_received_total", "counter", "Frames received from this camera.", "frames_received"
    )
    emit_per_camera(
        "camera_frames_processed_total", "counter", "Frames of this camera run through detection.", "frames_processed"
    )
    emit_per_camera(
        "camera_frames_dropped_total", "counter", "Frames of this camera dropped by its single-slot queue.", "frames_dropped"
    )
    emit_per_camera(
        "camera_process_fps", "gauge", "Frames processed per second for this camera (lifetime average).", "process_fps"
    )
    emit_per_camera(
        "camera_inference_ms", "gauge", "Inference time of this camera's most recent frame (ms).", "last_inference_ms"
    )
    emit_per_camera(
        "camera_total_latency_ms", "gauge", "End-to-end latency of this camera's most recent frame (ms).", "last_total_latency_ms"
    )

    # Detector
    emit("detector_loaded", "gauge", "1 if the detection model is loaded, else 0.", detector.get("loaded"))
    emit("detector_warmed_up", "gauge", "1 if the model has completed warmup, else 0.", detector.get("warmed_up"))
    emit("detector_conf_threshold", "gauge", "Active detection confidence threshold.", detector.get("conf_thresh"))
    emit("detector_img_size", "gauge", "Active inference image size.", detector.get("img_size"))
    emit(
        "detector_info",
        "gauge",
        "Detector metadata; value is always 1, labels carry the active preset.",
        1,
        labels={
            "mode": detector.get("mode") or "",
            "model": detector.get("model") or "",
            "device": "" if detector.get("resolved_device") is None else detector.get("resolved_device"),
        },
    )

    # Recordings
    emit("recordings_saved_total", "counter", "Recordings saved locally.", recordings.get("recordings_saved"))
    emit("recording_bytes_saved_total", "counter", "Bytes written for local recordings.", recordings.get("bytes_saved"))

    # Remote storage
    emit("remote_records_uploaded_total", "counter", "Detection records uploaded to remote storage.", remote.get("records_uploaded"))
    emit("remote_records_failed_total", "counter", "Detection record uploads that failed.", remote.get("records_failed"))
    emit("remote_records_dropped_total", "counter", "Detection records dropped when the queue was full.", remote.get("records_dropped"))
    emit("remote_recordings_uploaded_total", "counter", "Recordings uploaded to remote storage.", remote.get("recordings_uploaded"))
    emit("remote_queue_depth", "gauge", "Items waiting in the remote-storage upload queue.", remote.get("queue_depth"))

    # Alerts
    emit("alerts_fired_total", "counter", "Alert events fired.", alerts.get("events_fired"))
    emit("alert_rules", "gauge", "Configured alert rules.", len(alerts.get("rules") or []))
    emit("alert_webhook_sent_total", "counter", "Alert webhooks delivered.", alerts.get("webhook_sent"))
    emit("alert_webhook_failed_total", "counter", "Alert webhook deliveries that failed.", alerts.get("webhook_failed"))

    # Zones. Zones are per-camera, so count every camera's polygons here — the
    # snapshot's flat `zones` list is only the requested (default) camera.
    zones_by_camera = zones.get("cameras") or {}
    emit(
        "zones",
        "gauge",
        "Configured ROI zones across every camera.",
        sum(len(items or []) for items in zones_by_camera.values())
        if zones_by_camera
        else len(zones.get("zones") or []),
    )

    # Events / sightings
    emit("sightings_written_total", "counter", "Sighting rows persisted to the event store.", events.get("sightings_written"))
    emit("active_sightings", "gauge", "Sightings currently tracked in memory.", events.get("active_sightings"))

    return "\n".join(lines) + "\n"
