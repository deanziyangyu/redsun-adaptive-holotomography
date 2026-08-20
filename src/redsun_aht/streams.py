"""Canonical Bluesky event-stream names owned by AHT."""

PREVIEW_STREAM = "aht_preview"
CALIBRATION_STREAM = "aht_calibration"
PRIMARY_STREAM = "primary"
PROCESSING_PROGRESS_STREAM = "aht_processing_progress"
PROCESSING_RESULT_STREAM = "aht_processing_result"

ALL_STREAMS = frozenset(
    {
        PREVIEW_STREAM,
        CALIBRATION_STREAM,
        PRIMARY_STREAM,
        PROCESSING_PROGRESS_STREAM,
        PROCESSING_RESULT_STREAM,
    }
)

__all__ = [
    "ALL_STREAMS",
    "CALIBRATION_STREAM",
    "PREVIEW_STREAM",
    "PRIMARY_STREAM",
    "PROCESSING_PROGRESS_STREAM",
    "PROCESSING_RESULT_STREAM",
]
