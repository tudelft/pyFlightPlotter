"""pyFlightPlotter.web: the complete example, in the browser.

    python docs/examples/web_example.py
"""
import numpy as np

from pyFlightPlotter import Tailsitter
from pyFlightPlotter.web import WebPlotter

# example data: a tailsitter flying a horizontal circle, yawing along
t = np.linspace(0, 20, 4001)
w = 2 * np.pi / 10                            # one lap per 10 s
yaw = w * t
pos = np.column_stack([2 * np.cos(yaw), 2 * np.sin(yaw), -1.5 + 0.2 * np.sin(3 * yaw)])
vel = np.gradient(pos, t, axis=0)
acc = np.gradient(vel, t, axis=0)
att = np.column_stack([np.cos(yaw / 2), 0 * t, 0 * t, np.sin(yaw / 2)])       # [w, x, y, z]
noise = 0.02 * np.random.default_rng(0).standard_normal((t.size, 3))
thrust = np.column_stack([0.6 + 0.1 * np.sin(5 * t), 0.6 + 0.1 * np.cos(5 * t)])
flaps = np.column_stack([0.3 * np.sin(2 * t), -0.3 * np.sin(2 * t)])

wp = WebPlotter(t, name="Flight Plotter Demo")

# signals by name ...
wp.add_signals({f"pos[{i}]": pos[:, i] for i in range(3)})
wp.add_signals({f"posMeas[{i}]": pos[:, i] + noise[:, i] for i in range(3)})
wp.add_signals({f"vel[{i}]": vel[:, i] for i in range(3)})

# ... are referenced in panels; arrays work as well. Arguments as _plot_timeseries.
wp.add_panel("Position", ylabel="Position [m]",
             light=[f"posMeas[{i}]" for i in range(3)],
             solid=[f"pos[{i}]" for i in range(3)],
             series_labels=["X", "Y", "Z"],
             style_labels=["Measured", "Estimate", None])
wp.add_panel("Velocity", ylabel="Velocity [m/s]",
             solid=[f"vel[{i}]" for i in range(3)],
             dashed=[("vel[0]", 1.1), None, None],          # (column, scale)
             series_labels=["X", "Y", "Z"],
             style_labels=[None, "Estimate", "Setpoint"])
wp.add_panel("Actuators", ylabel="[-] / [rad]", ylimits=(-1, 1),
             solid=[thrust[:, 0], thrust[:, 1], flaps[:, 0], flaps[:, 1]],
             series_labels=["Rotor 1", "Rotor 2", "Flap 1", "Flap 2"],
             style_labels=[None, "State", None])

# 3D view; arguments as pyFlightPlotter.Viewport
wp.set_viewport(Tailsitter(),
                att=att, pos=pos, posMeas=pos + noise, vel=vel, acc=acc,
                rotor=thrust, surface=flaps)

wp.add_event(5.0, "lap 1 half")
wp.add_event(10.0, "+LAP", kind="mode")

wp.show()
