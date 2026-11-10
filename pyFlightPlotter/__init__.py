from importlib.metadata import PackageNotFoundError, version as _version

from .core import FlightPlotterBase, Craft3D, Viewport, VideoViewport
from .crafts import Quadrotor, Tailsitter
from .cursor import BlittedCursor
from .style import local_rc

__all__ = [
    "FlightPlotterBase",
    "Craft3D",
    "Viewport",
    "VideoViewport",
    "Quadrotor",
    "Tailsitter",
    "BlittedCursor",
    "local_rc",
]

try:
    __version__ = _version("pyFlightPlotter")
except PackageNotFoundError:  # running from a source checkout without installing
    __version__ = "unknown"
