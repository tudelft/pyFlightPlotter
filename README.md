# pyFlightPlotter

[![PyPI](https://img.shields.io/pypi/v/pyFlightPlotter)](https://pypi.org/project/pyFlightPlotter/)
[![Python](https://img.shields.io/pypi/pyversions/pyFlightPlotter)](https://pypi.org/project/pyFlightPlotter/)
[![License: GPL v3+](https://img.shields.io/badge/license-GPL--3.0--or--later-blue)](https://github.com/tudelft/pyFlightPlotter/blob/main/LICENSE)

Interactive, configurable and three-dimensional flight data plotting, with little boilerplate.

Hover over a time-series plot and every connected window follows along: the cursor moves in sync in all
time-series axes, and a 3D viewport shows the vehicle's attitude, position, velocity, acceleration and
actuator state at that moment. You can also sync an onboard or external video to the log.

![Demo of pyFlightPlotter](https://raw.githubusercontent.com/tudelft/pyFlightPlotter/main/docs/images/demo.gif)

## Features

- **3D viewport** showing estimates, setpoints and measurements of
  - attitude (quaternions)
  - position, velocity and acceleration
  - rotor force commands
  - control surface deflections
- **Time-series plotting** in a configurable grid layout. `_plot_timeseries()` takes several line styles
  (light / solid / dashed / true value) and keeps colours consistent across series.
- **A synchronized vertical cursor** across all time-series axes, even when they're spread over several
  figures. The 3D viewport updates on mouse hover.
- **Playback**: press <kbd>Space</kbd> to play or pause the log in real time.
- **Video sync** (`VideoViewport`): show an MP4 next to the log, with an adjustable time offset.
- **Craft builder**: the `Craft3D` class has `addRotor` and `addSurface` methods to define a vehicle quickly.
  - single-axis surface tilt
  - dual-axis rotor tilt
  - presets included: `Quadrotor`, `Tailsitter`

## Installation

```sh
pip install pyFlightPlotter
```

With video support (`VideoViewport`, which needs OpenCV):

```sh
pip install "pyFlightPlotter[video]"
```

The latest development version from GitHub:

```sh
pip install "pyFlightPlotter @ git+https://github.com/tudelft/pyFlightPlotter"
```

Requires Python 3.10 or newer. The dependencies are `numpy`, `scipy` and `matplotlib`. You also need an
interactive matplotlib backend (e.g. QtAgg or TkAgg).

## Usage

You build a plotter by subclassing `FlightPlotterBase` and implementing `_populate()`, where you add the
subplots you want. `BlittedCursor` adds the synchronized cursor, and `connect_viewport()` links a 3D
`Viewport` (or a `VideoViewport`) to the mouse.

A minimal example ([`docs/examples/minimal_example.py`](https://github.com/tudelft/pyFlightPlotter/blob/main/docs/examples/minimal_example.py)):

```python
import numpy as np

from pyFlightPlotter import FlightPlotterBase, BlittedCursor, Viewport, Quadrotor

class FlightPlotterMinimalExample(FlightPlotterBase):
    def __init__(self, time, data):
        self.data = data
        super().__init__(time, name="MWE")
        self.define_layout(nrows=1, ncols=1)
        self.plot()

    def _populate(self):
        self._plot_timeseries(
            self.fig.add_subplot(self.gs[0, 0]),
            light=None,
            solid=[self.data[2], self.data[3]],
            dashed=[self.data[0], self.data[1]],
            series_labels=["Signal 1", "Signal 2"],
            style_labels=[None, "Estimated", "Setpoint"],
            title="Example Time Series",
            ylabel="Amplitude",
        )

if __name__ == "__main__":
    t = np.linspace(0, 2, 101)
    data = [
        np.sin(2*np.pi*0.5*t), np.sin(2*np.pi*0.5*t + 0.2),
        np.cos(2*np.pi*0.5*t + 0.4), np.cos(2*np.pi*0.5*t + 0.6),
    ]

    fp = FlightPlotterMinimalExample(t, data)
    bc = BlittedCursor(fp.all_axes, sharex=True)

    vp = Viewport(craft=Quadrotor(),
                  time=np.array([0., 1., 2.]),
                  att=np.array([[1., 0., 0., 0.], [0.7, 0., 0., 0.7], [0., 0., 0., 1.]]),
                  # ... many more data can be passed here, but attitude is required
                  interpolation="linear",
                  follow=False,
                  title="Viewport Example",
                  )

    # connect mouse hover
    fp.connect_viewport(vp)
```

Run it with `python -i` so the windows stay open.

For a full example with several figures, a tailsitter, and position/velocity/actuator data, see
[`docs/examples/complete_example.py`](https://github.com/tudelft/pyFlightPlotter/blob/main/docs/examples/complete_example.py).
For an example of hooking pyFlightPlotter into your own log-data structure, see
[indiflight](https://github.com/tudelft/indiflight/blob/global_rls/support/LogAnalysis/indiflightPlotter.py).

### Video synchronization

```python
from pyFlightPlotter import VideoViewport

vv = VideoViewport(time=t, mp4_path="flight.mp4", offset_s=0.0)
fp.connect_viewport(vv)
```

The video time is `data_time + offset`. You can set the offset with the slider or text box in the video
window, or step one frame at a time with the arrow keys. You can also load a file from inside the window.

### Controls

| Input | Action |
|---|---|
| Mouse hover over a time-series axis | Move the cursor; all connected viewports update |
| <kbd>Space</kbd> | Play / pause real-time playback |
| <kbd>←</kbd> / <kbd>→</kbd> (video window) | Shift the video offset by one frame |

### Conventions

Quaternions are scalar-first: `[w, x, y, z]`. Arrays are expected with time along the first axis, e.g.
`Nx4` for attitude and `Nx3` for position, velocity and acceleration. See the `Viewport` docstring for all
supported inputs.

## Development

```sh
git clone https://github.com/tudelft/pyFlightPlotter
cd pyFlightPlotter
pip install -e ".[video]"
python -i docs/examples/complete_example.py
```

## License

GPL-3.0-or-later. See [LICENSE](https://github.com/tudelft/pyFlightPlotter/blob/main/LICENSE).
