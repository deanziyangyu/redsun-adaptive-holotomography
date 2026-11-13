"""Event stream names shared by plans, presenters and storage.

Live visualisation travels as Event documents emitted by ``bps.monitor`` on a
camera's buffer signal, not as a direct signal subscription. That keeps what a
viewer shows inside the run's document sequence, so it can be traced, replayed
and written like any other stream. Plans emit these names; presenters route on
them.

The names are deliberately plain strings: a stream name travels in the start
document and in every descriptor, so renaming one is a data-compatibility
change, not a refactor.
"""

from __future__ import annotations

#: Preview frames, emitted by ``bps.monitor`` on a camera buffer signal.
LIVE_VIEW_STREAM = "live_view"

#: The acquisition stream a finite recording writes its stored frames to.
PRIMARY_STREAM = "primary"

#: The illumination pattern sequence a DPCT run records alongside its frames.
PATTERN_STREAM = "pattern"

__all__ = ["LIVE_VIEW_STREAM", "PATTERN_STREAM", "PRIMARY_STREAM"]
