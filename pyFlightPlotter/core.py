import numpy as np
import os
import importlib
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, Slider, TextBox

COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']

class FlightPlotterBase(object):
    def __init__(self, time, name="Flight Plotter"):
        self.t = np.asarray(time, dtype=float)
        # make sure time is a 1D array
        if self.t.ndim != 1:
            raise ValueError("time must be a 1D array")

        self.name = name
        self.all_axes = []

    def define_layout(self, figsize=(12, 8), nrows=3, ncols=3, width_ratios=None, height_ratios=None):
        if width_ratios is None:
            width_ratios = [1] * ncols
        if height_ratios is None:
            height_ratios = [1] * nrows

        self.fig = plt.figure(figsize=figsize)
        self.gs = GridSpec(nrows=nrows, ncols=ncols,
                           width_ratios=width_ratios,
                           height_ratios=height_ratios)

    def plot(self):
        self._populate()
        self._dress()
        # self.curser = BlittedCursor(self.all_axes, self.fig.canvas, sharex=True)
        self.fig.show()

    def _populate(self):
        raise NotImplementedError("Child class should implement this method to populate the plot.")

    def connect_viewport(self, viewport):
        self._add_callback('motion_notify_event', viewport.mouseHoverCallback)

    def _add_callback(self, event_type, callback):
        # e,g, motion_notify_event
        self.fig.canvas.mpl_connect(event_type, callback)

    def _find_axis(self, subplot_spec):
        axes = [ax for ax in self.fig.axes
                if ax.get_subplotspec() == subplot_spec]
        if len(axes) == 0:
            return None
        return axes[0]

    def _plot_timeseries(self, ax, light=None, solid=None, dashed=None, true_values=None, series_labels=[], style_labels=[None, None, None], title="", ylabel="", ylimits=(None, None)):
        if solid is None or len(solid) == 0:
            raise ValueError("At least one solid series must be provided.")
        if len(series_labels) != len(solid):
            raise ValueError("series_labels must have the same length as solid series.")
        if len(style_labels) != 3:
            raise ValueError("style_labels must have exactly 3 entries. Set to None if not needed.")
        if light is None or len(light) == 0:
            light = [None] * len(solid)
        if dashed is None or len(dashed) == 0:
            dashed = [None] * len(solid)
        if true_values is None or len(true_values) == 0:
            true_values = [None] * len(solid)
        lengths = np.array([len(solid), len(light), len(dashed), len(true_values), len(series_labels)])
        if not (lengths == lengths[0]).all():
            raise ValueError("light, solid, dashed, true_values and labels must all have the same length if given.")

        for i, series in enumerate(light):
            if series is not None:
                ax.plot(self.t, series, color=COLORS[i], alpha=0.3, lw=1.0, linestyle='-')

        for i, series in enumerate(solid):
            if series is not None:
                ax.plot(self.t, series, label=series_labels[i], color=COLORS[i], alpha=0.8, lw=1.0, linestyle='-')

        for i, series in enumerate(dashed):
            if series is not None:
                ax.plot(self.t, series, color=COLORS[i], lw=1.5, linestyle='--')

        for i, series in enumerate(true_values):
            if series is not None:
                ax.hlines(series, self.t[0], self.t[-1], color=COLORS[i], lw=1.0, linestyle=':')

        self.all_axes.append(ax)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylimits)

        ax.add_artist(ax.legend(loc='upper left'))
        ax.add_artist(self._generate_style_legend(ax, style_labels))

    def _generate_style_legend(self, ax, labels):
        linestyles = []
        if labels[0] is not None:
            linestyles.append(Line2D([0], [0], color='gray', alpha=0.3, lw=1.0, linestyle='-', label=labels[0]))
        if labels[1] is not None:
            linestyles.append(Line2D([0], [0], color='gray', alpha=0.8, lw=1.0, linestyle='-', label=labels[1]))
        if labels[2] is not None:
            linestyles.append(Line2D([0], [0], color='gray', alpha=1.0, lw=1.5, linestyle='--', label=labels[2]))

        leg_styles = ax.legend(handles=linestyles, title='Line Styles', loc='lower left')

        return leg_styles

    def _dress(self):
        for ax in self.all_axes:
            ax.grid(True)
            # only set time label if in lowest row
            if ax.get_subplotspec().is_last_row():
                ax.set_xlabel("Time [s]")

        self.fig.suptitle(self.name)

class Craft3D(object):
    """Base class that can generate x,y,z coordinates to visualize a craft
    
    This class contains a generate method that takes in an attitude quaternion
    and optional rotor and surface controls, and returns x,y,z coordinates that
    can be given to matplotlib for 3D plotting.

    This class can be extended to define specific craft geometries, see crafts.py

    Args:
        body_geometry: list of Nx3 arrays defining the body geometry

    Returns:
        Craft3D object

    """
    def __init__(self, body_geometry=[np.array([[0,0,0]])]):

        # check that array is Nx3
        self.Ng = len(body_geometry)
        self.geometry = []
        for element in body_geometry:
            self.geometry.append(np.asarray(element, dtype=float))
            if self.geometry[-1].ndim != 2 or self.geometry[-1].shape[1] != 3:
                raise ValueError("Base geometry must be a list of Nx3 arrays")

        self.rotors = []
        self.surfaces = []

    def addRotor(self,
                 xyz=[0, 0, 0], axis=[0, 0, -1], 
                 tilt_xyz=[0, 0, 0], tilt_axis=[1, 0, 0],
                 R=0.1, N=20):

        xyz = np.asarray(xyz, dtype=float)
        axis = np.asarray(axis, dtype=float)
        axis /= np.linalg.norm(axis)

        tilt_xyz = np.asarray(tilt_xyz, dtype=float)
        tilt_axis = np.asarray(tilt_axis, dtype=float)
        tilt_axis /= np.linalg.norm(tilt_axis)

        # compute cross tilt axis, and check that tilt_axis is not parallel to axis
        cross = np.cross(axis, tilt_axis)
        if np.linalg.norm(cross) < 1e-4:
            raise ValueError("tilt_axis cannot be parallel to axis")
        cross /= np.linalg.norm(cross)

        self.rotors.append({
            "xyz": xyz,
            "axis": axis,
            "tilt_xyz": tilt_xyz,
            "tilt_axis": tilt_axis,
            "tilt2_axis": cross,
            "R": R,
            "N": N,
        })

    def addSurface(self,
                   tilt_xyz=[0, 0, 0], tilt_axis=[0, 1, 0],
                   geometry=[np.array([0, 0, 0])]):
        tilt_xyz = np.asarray(tilt_xyz, dtype=float)
        tilt_axis = np.asarray(tilt_axis, dtype=float)
        geometry = [np.asarray(g, dtype=float) for g in geometry]

        self.surfaces.append({
            "tilt_xyz": tilt_xyz,
            "tilt_axis": tilt_axis,
            "geometry": geometry
        })

    def generate(self, quat, rotor_controls=None, surface_controls=None):
        if np.linalg.norm(quat) < 1e-6:
            quat = np.array([0, 0, 0, 1])
        body_rotation = R.from_quat(quat)

        xs, ys, zs = np.array([]), np.array([]), np.array([])
        for geo in self.geometry:
            geo_rotated = body_rotation.apply(geo)
            xs = np.concatenate((xs, geo_rotated[:, 0], np.array([np.nan])))
            ys = np.concatenate((ys, geo_rotated[:, 1], np.array([np.nan])))
            zs = np.concatenate((zs, geo_rotated[:, 2], np.array([np.nan])))

        if rotor_controls is None:
            rotor_controls = np.zeros((len(self.rotors), 3))
        elif rotor_controls.shape[0] != len(self.rotors):
            raise ValueError("rotor_controls must have the same length as the number of rotors")

        if surface_controls is None:
            surface_controls = np.zeros((len(self.surfaces),))
        elif surface_controls.shape[0] != len(self.surfaces):
            raise ValueError("surface_controls must have the same length as the number of surfaces")

        arrows = []
        for rotor, controls in zip(self.rotors, rotor_controls):
            # generate circle in the rotor plane by using xyz, axis
            Rr = rotor["R"]
            xyz = rotor["xyz"]
            axis = rotor["axis"]
            tilt_xyz = rotor["tilt_xyz"]

            tilt1_axis = rotor["tilt_axis"]
            tilt1_angle = controls[1]
            tilt1_rotation = R.from_rotvec(tilt1_axis*tilt1_angle)

            tilt2_axis = rotor["tilt2_axis"]
            tilt2_angle = controls[2]
            tilt2_rotation = R.from_rotvec(tilt2_axis*tilt2_angle)
            tilt_rotation = tilt2_rotation * tilt1_rotation

            N = rotor["N"]
            u = np.linspace(0, 2*np.pi, N)

            # generate circle in xy plane
            x = Rr * np.cos(u)
            y = Rr * np.sin(u)
            z = np.zeros_like(u)
            circle = np.vstack((x, y, z)).T

            # rotate circle to align with rotor axis
            circle_tilted = tilt_rotation.apply(circle)

            foot_xyz = xyz - tilt_xyz
            real_xyz = foot_xyz + tilt_rotation.apply(tilt_xyz)

            circle_tilted += real_xyz

            circle_rotated = body_rotation.apply(circle_tilted)


            xs = np.concatenate((xs, circle_rotated[:, 0], np.array([np.nan])))
            ys = np.concatenate((ys, circle_rotated[:, 1], np.array([np.nan])))
            zs = np.concatenate((zs, circle_rotated[:, 2], np.array([np.nan])))

            # arrow for thrust
            thrust = controls[0]
            real_axis = body_rotation.apply(tilt_rotation.apply(axis))
            rotated_real_xyz = body_rotation.apply(real_xyz)

            arrow_start = rotated_real_xyz
            arrow_end = rotated_real_xyz + real_axis * thrust * 2. * Rr  # scale thrust for visualization
            arrow_max = rotated_real_xyz + real_axis * 2. * Rr  # max arrow length for visualization

            arrows.append([arrow_start, arrow_end, arrow_max])

        for surface, control in zip(self.surfaces, surface_controls):
            geometry = surface["geometry"]
            tilt_xyz = surface["tilt_xyz"]
            tilt_axis = surface["tilt_axis"]
            tilt_angle = control
            tilt_rotation = R.from_rotvec(tilt_axis*tilt_angle)

            for geo in geometry:
                geo_tilted = tilt_rotation.apply(geo - tilt_xyz) + tilt_xyz
                geo_rotated = body_rotation.apply(geo_tilted)
                xs = np.concatenate((xs, geo_rotated[:, 0], np.array([np.nan])))
                ys = np.concatenate((ys, geo_rotated[:, 1], np.array([np.nan])))
                zs = np.concatenate((zs, geo_rotated[:, 2], np.array([np.nan])))

        return xs, ys, zs, arrows

class Viewport(object):
    """Generate an updatable 3D viewport for visualizing the state of a craft
    """
    def __init__(self, craft: Craft3D,
                 time,
                 att, attSet=None, attMeas=None,
                 pos=None, posSet=None, posMeas=None,
                 vel=None, velSet=None, velMeas=None,
                 acc=None, accSet=None, accMeas=None,
                 rotor=None, rotorSet=None,
                 surface=None, surfaceSet=None,
                 follow=False, interpolation="previous", title="Viewport"):
        """Initialize the viewport and open its plot window. Numpy arrays expected.

        Args:
            craft: Craft3D object defining the vehicle geometry
            time: 1D array of time stamps
            att: Nx4 array of attitude quaternions [w, x, y, z]
            attSet: Nx4 array of attitude setpoint quaternions [w, x, y, z]
            attMeas: Nx4 array of attitude measurement quaternions [w, x, y, z]
            pos: Nx3 array of position [x, y, z]
            posSet: Nx3 array of position setpoint [x, y, z]
            posMeas: Nx3 array of position measurement [x, y, z]
            vel: Nx3 array of velocity [vx, vy, vz]
            velSet: Nx3 array of velocity setpoint [vx, vy, vz]
            velMeas: Nx3 array of velocity measurement [vx, vy, vz]
            acc: Nx3 array of acceleration [ax, ay, az]
            accSet: Nx3 array of acceleration setpoint [ax, ay, az]
            accMeas: Nx3 array of acceleration measurement [ax, ay, az]
            rotor: NxMx3 array of rotor state for M rotors
            rotorSet: NxMx3 array of rotor controls (thrust, tilt1, tilt2) for M rotors
            surface: NxK array of surface state for K surfaces
            surfaceSet: NxK array of surface controls for K surfaces
            follow: if True, the camera will follow the vehicle position
            interpolation: interpolation method for data ("previous", "linear")
            title: title of the plot

        Returns:
            Viewport object
        """

        self.craft = craft
        self.name = title
        self.t = time
        self.follow = follow
        self.has_pos = pos is not None
        self.interpolation = interpolation

        if self.interpolation not in ["previous", "linear"]:
            raise ValueError("interpolation must be one of 'previous' or 'linear'")

        #%% define data series and their plotting styles
        # attitude
        self.att = {"att": {"raw": att[:, [1,2,3,0]], "style": "solid",  "color": COLORS[0], "marker": None, "width": 2.0, "label": "Attitude Estimate"}}
        if attSet is not None:
            self.att["attSet"] = {"raw": attSet[:, [1,2,3,0]], "style": "dashed", "color": COLORS[1], "marker": None, "width": 0.8,  "label": "Attitude Setpoint"}
        if attMeas is not None:
            self.att["attMeas"] = {"raw": attMeas[:, [1,2,3,0]], "style": "dashed", "color": COLORS[2], "marker": None, "width": 0.8,  "label": "Attitude Measurement"}

        # position
        self.pos = {}
        if pos is not None:
            self.pos["pos"] = {"raw": pos, "style": "solid",  "color": COLORS[0], "marker": ".", "width": 1.5, "label": "Position Estimate"}
        if posSet is not None:
            self.pos["posSet"] = {"raw": posSet, "style": "dashed", "color": COLORS[1], "marker": "o", "width": 1.0,  "label": "Position Setpoint"}
        if posMeas is not None:
            self.pos["posMeas"] = {"raw": posMeas, "style": "solid", "color": COLORS[2], "marker": "o", "width": 3.0, "label": "Position Measurement"}

        # velocity
        self.vel = {}
        if vel is not None:
            self.vel["vel"] = {"raw": vel, "style": "solid",  "color": COLORS[0], "marker": None, "width": 2.0, "label": "Velocity Estimate"}
        if velSet is not None:
            self.vel["velSet"] = {"raw": velSet, "style": "dashed", "color": COLORS[1], "marker": None, "width": 1.0,  "label": "Velocity Setpoint"}
        if velMeas is not None:
            self.vel["velMeas"] = {"raw": velMeas, "style": "dashed", "color": COLORS[2], "marker": None, "width": 1.0,  "label": "Velocity Measurement"}

        # acceleration
        self.acc = {}
        if acc is not None:
            self.acc["acc"] = {"raw": acc, "style": "solid",  "color": COLORS[3], "marker": None, "width": 2.0, "label": "Acceleration Estimate"}
        if accSet is not None:
            self.acc["accSet"] = {"raw": accSet, "style": "dashed", "color": COLORS[4], "marker": None, "width": 1.0,  "label": "Acceleration Setpoint"}
        if accMeas is not None:
            self.acc["accMeas"] = {"raw": accMeas, "style": "dashed", "color": COLORS[5], "marker": None, "width": 1.0,  "label": "Acceleration Measurement"}

        # controls
        self.controls = {}
        if rotor is not None:
            self.controls["rotor"] = {"raw": rotor, "style": "solid",  "color": COLORS[5], "marker": "s", "width": 3.0, "label": "Rotor States"}
        if rotorSet is not None:
            self.controls["rotorSet"] = {"raw": rotorSet, "style": "solid",  "color": COLORS[6], "marker": "s", "width": 3.0, "label": "Rotor Controls"}
        if surface is not None:
            self.controls["surface"] = {"raw": surface, "style": "solid",  "color": COLORS[7], "marker": None, "width": 3.0, "label": None}
        if surfaceSet is not None:
            self.controls["surfaceSet"] = {"raw": surfaceSet, "style": "solid",  "color": COLORS[7], "marker": None, "width": 3.0, "label": None}

        # collect all series
        self.series = {**self.att, **self.pos, **self.vel, **self.acc, **self.controls}

        #%% figure setup
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        # interpolators and initial plotting lines
        self._create_interpolators()

        for _, value in self.series.items():
            value['line'] = self.ax.plot(
                [np.nan], [np.nan], [np.nan],
                linestyle=value['style'],
                color=value['color'],
                lw=value['width'],
                label=value["label"])[0]

        # set view angle
        self.ax.view_init(elev=-25, azim=150, roll=180)

        # set limits
        self._update_limits()

        # legend, title and show
        self.ax.legend(loc='upper left')
        self.ax.set_title(self.name)
        self.fig.show()

    def _update_limits(self):
        if self.has_pos:
            minx, miny, minz = np.min(self.series["pos"]["raw"], axis=0)
            maxx, maxy, maxz = np.max(self.series["pos"]["raw"], axis=0)
        else:
            minx, miny, minz = 0, 0, 0
            maxx, maxy, maxz = 0, 0, 0

        self.ax.set_xlim(minx-0.5, maxx+0.5)
        self.ax.set_ylim(miny-0.5, maxy+0.5)
        self.ax.set_zlim(minz-0.5, maxz+0.5)

    def _create_interpolators(self):
        for _, value in self.series.items():
            value['interpolator'] = interp1d(
                self.t,
                np.moveaxis(value['raw'], 0, -1),
                kind=self.interpolation,
                bounds_error=False,
                fill_value=(value["raw"][0], value["raw"][-1]))

    def pushNewFrame(self, time,
                     att, attSet=None, attMeas=None,
                     pos=None, posSet=None, posMeas=None,
                     vel=None, velSet=None, velMeas=None,
                     acc=None, accSet=None, accMeas=None,
                     rotor=None, rotorSet=None,
                     surface=None, surfaceSet=None):
        """Push a new frame of data. All data that was not None at initialization must be given here as well
        """

        self.t = np.append(self.t, time)
        self.series["att"]['raw'] = np.vstack((self.series["att"]['raw'], att[[1,2,3,0]]))
        if attSet is not None:
            # accept errors if series was not given before, that's up to the user
            self.series["attSet"]['raw'] = np.vstack((self.series["attSet"]['raw'], attSet[[1,2,3,0]]))
        if attMeas is not None:
            self.series["attMeas"]['raw'] = np.vstack((self.series["attMeas"]['raw'], attMeas[[1,2,3,0]]))
        if pos is not None:
            self.series["pos"]['raw'] = np.vstack((self.series["pos"]['raw'], pos))
        if posSet is not None:
            self.series["posSet"]['raw'] = np.vstack((self.series["posSet"]['raw'], posSet))
        if posMeas is not None:
            self.series["posMeas"]['raw'] = np.vstack((self.series["posMeas"]['raw'], posMeas))
        if vel is not None:
            self.series["vel"]['raw'] = np.vstack((self.series["vel"]['raw'], vel))
        if velSet is not None:
            self.series["velSet"]['raw'] = np.vstack((self.series["velSet"]['raw'], velSet))
        if velMeas is not None:
            self.series["velMeas"]['raw'] = np.vstack((self.series["velMeas"]['raw'], velMeas))
        if acc is not None:
            self.series["acc"]['raw'] = np.vstack((self.series["acc"]['raw'], acc))
        if accSet is not None:
            self.series["accSet"]['raw'] = np.vstack((self.series["accSet"]['raw'], accSet))
        if accMeas is not None:
            self.series["accMeas"]['raw'] = np.vstack((self.series["accMeas"]['raw'], accMeas))
        if rotor is not None:
            self.series["rotor"]['raw'] = np.vstack((self.series["rotor"]['raw'], rotor[np.newaxis]))
        if rotorSet is not None:
            self.series["rotorSet"]['raw'] = np.vstack((self.series["rotorSet"]['raw'], rotorSet[np.newaxis]))
        if surface is not None:
            self.series["surface"]['raw'] = np.vstack((self.series["surface"]['raw'], surface))
        if surfaceSet is not None:
            self.series["surfaceSet"]['raw'] = np.vstack((self.series["surfaceSet"]['raw'], surfaceSet))

        self._create_interpolators()
        self._update_limits()

    def showAtTime(self, time=None):
        """Update the viewport to show the state at a specific time

        Args:
            time (float): time stamp to show

        Returns:
            None
        """

        if time is None:
            time = self.t[-1]

        # remove old lines
        for _, ser in self.series.items():
            try:
                ser['line'].remove()
            except ValueError:
                pass

        # get interpolates
        interpolates = {}
        for ser in self.series.keys():
            interpolates[ser] = self.series[ser]['interpolator'](time)

        # plot attitude
        for ser in [x for x in ["att", "attSet", "attMeas"] if x in self.series.keys()]:
            # invoke craft to get the rotated geometry
            if ser == "attSet":
                rotor_controls = interpolates["rotorSet"] if "rotorSet" in interpolates.keys() else None
                surface_controls = interpolates["surfaceSet"] if "surfaceSet" in interpolates.keys() else None
            elif ser == "att":
                rotor_controls = interpolates["rotor"] if "rotor" in interpolates.keys() else None
                surface_controls = interpolates["surface"] if "surface" in interpolates.keys() else None
            else:
                # never draw rotors or surface motion for measured attitude as it 
                # would suggest that these are also measured
                rotor_controls = None
                surface_controls = None
                if self.follow:
                    # skip all of this entirely, as it would change the limits
                    continue

            xs, ys, zs, q = self.craft.generate(interpolates[ser],
                                                rotor_controls=rotor_controls,
                                                surface_controls=surface_controls,
                                                )

            qs = np.empty((0, 3))
            for arrow in q:
                arrow_start, arrow_end, arrow_max = arrow
                qs = np.concatenate((qs,
                                     arrow_start[np.newaxis],
                                     arrow_end[np.newaxis],
                                     np.array([[np.nan, np.nan, np.nan]]),
                                     arrow_max[np.newaxis],
                                     np.array([[np.nan, np.nan, np.nan]]),
                                     ))

            if "pos" in interpolates.keys():
                qs += interpolates["pos"]

            # translate geometry, if necessary
            if "pos" in interpolates.keys():
                if ser == "attMeas" and "posMeas" in interpolates.keys():
                    # offset by measured position
                    xs += interpolates["posMeas"][0]
                    ys += interpolates["posMeas"][1]
                    zs += interpolates["posMeas"][2]
                else:
                    # offset by estimator position
                    xs += interpolates["pos"][0]
                    ys += interpolates["pos"][1]
                    zs += interpolates["pos"][2]

            # update plot line
            self.series[ser]['line'] = self.ax.plot(xs, ys, zs,
                linestyle=self.series[ser]['style'],
                color=self.series[ser]['color'],
                lw=self.series[ser]['width'])[0]

            # add arrows for rotors
            if ser == "attSet" and len(q) > 0 and "rotorSet" in self.series.keys():
                self.series["rotorSet"]['line'] = self.ax.plot(qs[:, 0], qs[:, 1], qs[:, 2],
                             linestyle=self.series["rotorSet"]['style'],
                             color=self.series["rotorSet"]['color'],
                             marker=self.series["rotorSet"]['marker'],
                             markersize=1,
                             lw=self.series["rotorSet"]['width'])[0]

            if ser == "att" and len(q) > 0 and "rotor" in self.series.keys():
                self.series["rotor"]['line'] = self.ax.plot(qs[:, 0], qs[:, 1], qs[:, 2],
                             linestyle=self.series["rotor"]['style'],
                             color=self.series["rotor"]['color'],
                             marker=self.series["rotor"]['marker'],
                             markersize=1,
                             lw=self.series["rotor"]['width'])[0]

        if not self.follow:
            # scatter plot for position
            for ser in [x for x in ["pos", "posSet", "posMeas"] if x in self.series.keys()]:
                self.series[ser]['line'] = self.ax.scatter(
                    interpolates[ser][0],
                    interpolates[ser][1],
                    interpolates[ser][2],
                    linestyle=self.series[ser]['style'],
                    color=self.series[ser]['color'],
                    lw=self.series[ser]['width'],
                    marker=self.series[ser]['marker'],
                    facecolor='none',
                    s=50*self.series[ser]['width'])

            # line to show velocity
            for ser in [x for x in ["vel", "velSet", "velMeas"] if x in self.series.keys()]:
                if ser == "velMeas" and "posMeas" in interpolates.keys():
                    offset = interpolates["posMeas"]
                elif "pos" in interpolates.keys():
                    offset = interpolates["pos"]
                else:
                    offset = np.array([0, 0, 0])

                self.series[ser]['line'] = self.ax.plot(
                    [offset[0], 0.2*interpolates[ser][0] + offset[0]],
                    [offset[1], 0.2*interpolates[ser][1] + offset[1]],
                    [offset[2], 0.2*interpolates[ser][2] + offset[2]],
                    linestyle=self.series[ser]['style'],
                    color=self.series[ser]['color'],
                    lw=self.series[ser]['width'])[0]

            # line to show acceleration
            for ser in [x for x in ["acc", "accSet", "accMeas"] if x in self.series.keys()]:
                if ser == "accMeas" and "posMeas" in interpolates.keys():
                    offset = interpolates["posMeas"]
                elif "pos" in interpolates.keys():
                    offset = interpolates["pos"]
                else:
                    offset = np.array([0, 0, 0])

                self.series[ser]['line'] = self.ax.plot(
                    [offset[0], 0.1*interpolates[ser][0] + offset[0]],
                    [offset[1], 0.1*interpolates[ser][1] + offset[1]],
                    [offset[2], 0.1*interpolates[ser][2] + offset[2]],
                    linestyle=self.series[ser]['style'],
                    color=self.series[ser]['color'],
                    lw=self.series[ser]['width'])[0]

        # make sure all the axes are equal
        if self.follow:
            self.ax.set_aspect('equal', adjustable='datalim')
        else:
            self.ax.set_aspect('equal', adjustable='box')

        # finally, update the canvas
        self.fig.canvas.draw()

    def mouseHoverCallback(self, event):
        """Callback function to update the viewport on mouse hover

        Args:
            event (matplotlib.backend_bases.MouseEvent): mouse event

        Returns:
            None
        """
        if event.xdata is not None:
            self.showAtTime(event.xdata)


class VideoViewport(object):
    """MP4 viewport synchronized to the FlightPlotter timeline.

    The mapping from data time to video time is:
        video_time = data_time + time_offset
    """

    def __init__(self, time, title="Video Viewport", mp4_path=None, offset_s=0.0):
        self.t = np.asarray(time, dtype=float)
        if self.t.ndim != 1:
            raise ValueError("time must be a 1D array")

        self.name = title
        self.time_offset_s = float(offset_s)
        self.current_data_time = float(self.t[0]) if len(self.t) > 0 else 0.0

        self._video_path = None
        self._video_filename = "<no video loaded>"
        self._video_capture = None
        self._fps = None
        self._frame_count = None
        self._duration_s = None
        self._backend = None
        self._last_frame_id = None
        self._last_frame_rgb = None
        self._last_rendered_frame_id = None

        self.fig = plt.figure(figsize=(10, 7))
        self.gs = GridSpec(nrows=10, ncols=12, figure=self.fig)

        self.ax_video = self.fig.add_subplot(self.gs[0:8, :])
        self.ax_video.set_axis_off()
        self._imshow = None

        self._overlay = self.ax_video.text(
            0.01,
            0.99,
            "",
            transform=self.ax_video.transAxes,
            va='top',
            ha='left',
            color='white',
            fontsize=10,
            bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.25')
        )

        self.ax_open = self.fig.add_subplot(self.gs[8, 0:2])
        self.ax_load = self.fig.add_subplot(self.gs[8, 2:4])
        self.ax_path = self.fig.add_subplot(self.gs[8, 4:12])
        self.ax_offset_slider = self.fig.add_subplot(self.gs[9, 0:10])
        self.ax_offset_box = self.fig.add_subplot(self.gs[9, 10:12])

        self.btn_open = Button(self.ax_open, "Open MP4")
        self.btn_load = Button(self.ax_load, "Load")
        self.path_box = TextBox(self.ax_path, "Path", initial="")

        self.offset_slider = Slider(
            self.ax_offset_slider,
            "Offset [s]",
            valmin=-30.0,
            valmax=30.0,
            valinit=self.time_offset_s,
            valstep=None,
        )
        self.offset_box = TextBox(self.ax_offset_box, "", initial=f"{self.time_offset_s:.3f}")

        self.btn_open.on_clicked(self._on_open_clicked)
        self.btn_load.on_clicked(self._on_load_clicked)
        self.offset_slider.on_changed(self._on_offset_slider_changed)
        self.offset_box.on_submit(self._on_offset_box_submit)
        self.path_box.on_submit(self._on_path_submit)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)

        self.ax_video.set_title(self.name)
        self._set_placeholder("No video loaded")
        self.fig.show()

        if mp4_path is not None:
            self.path_box.set_val(str(mp4_path))
            self.load(mp4_path)

    def _set_placeholder(self, text):
        h, w = 360, 640
        placeholder = np.zeros((h, w, 3), dtype=np.uint8)
        if self._imshow is None:
            self._imshow = self.ax_video.imshow(placeholder)
        else:
            self._imshow.set_data(placeholder)
        self._update_overlay(frame_id=None, video_time=None, data_time=self.current_data_time, message=text)
        self.fig.canvas.draw_idle()

    def _try_open_dialog(self):
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            selected = filedialog.askopenfilename(
                title="Select MP4 file",
                filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")],
            )
            root.destroy()
            return selected
        except Exception:
            return None

    def _on_open_clicked(self, _event):
        selected = self._try_open_dialog()
        if selected:
            self.path_box.set_val(selected)
            self.load(selected)

    def _on_load_clicked(self, _event):
        path = self.path_box.text.strip()
        if path:
            self.load(path)

    def _on_path_submit(self, text):
        path = text.strip()
        if path:
            self.load(path)

    def _on_offset_slider_changed(self, value):
        self.time_offset_s = float(value)
        self.offset_box.set_val(f"{self.time_offset_s:.3f}")
        self.showAtDataTime(self.current_data_time)

    def _on_offset_box_submit(self, text):
        try:
            offset = float(text)
        except ValueError:
            self.offset_box.set_val(f"{self.time_offset_s:.3f}")
            return

        self.time_offset_s = offset
        slider_min = float(self.offset_slider.valmin)
        slider_max = float(self.offset_slider.valmax)
        if offset < slider_min or offset > slider_max:
            span = max(1.0, abs(offset) + 1.0)
            self.offset_slider.valmin = -span
            self.offset_slider.valmax = span
            self.offset_slider.ax.set_xlim(self.offset_slider.valmin, self.offset_slider.valmax)
        self.offset_slider.set_val(self.time_offset_s)

    def _frame_time_step(self):
        # Use video frame period if available; otherwise fall back to a sensible default.
        fps = self._fps if self._fps is not None and self._fps > 0 else 30.0
        return 1.0 / fps

    def _set_offset(self, offset):
        self.time_offset_s = float(offset)
        slider_min = float(self.offset_slider.valmin)
        slider_max = float(self.offset_slider.valmax)
        if self.time_offset_s < slider_min or self.time_offset_s > slider_max:
            span = max(1.0, abs(self.time_offset_s) + 1.0)
            self.offset_slider.valmin = -span
            self.offset_slider.valmax = span
            self.offset_slider.ax.set_xlim(self.offset_slider.valmin, self.offset_slider.valmax)
        self.offset_slider.set_val(self.time_offset_s)

    def _on_key_press(self, event):
        if event.key not in ("left", "right"):
            return

        step = self._frame_time_step()
        if event.key == "left":
            self._set_offset(self.time_offset_s - step)
        elif event.key == "right":
            self._set_offset(self.time_offset_s + step)

    def _ensure_backend(self):
        if self._backend is not None:
            return

        try:
            cv2 = importlib.import_module('cv2')
            self._backend = 'cv2'
            self._cv2 = cv2
        except ImportError:
            raise ImportError("OpenCV (cv2) is required for VideoViewport. Install python package 'opencv-python'.")

    def load(self, path):
        self._ensure_backend()

        if self._video_capture is not None:
            self._video_capture.release()
            self._video_capture = None

        resolved = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(resolved):
            self._set_placeholder(f"File not found: {resolved}")
            return False

        if not resolved.lower().endswith('.mp4'):
            self._set_placeholder("Selected file is not an MP4")
            return False

        cap = self._cv2.VideoCapture(resolved)
        if not cap.isOpened():
            self._set_placeholder(f"Could not open video: {resolved}")
            cap.release()
            return False

        fps = float(cap.get(self._cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0

        frame_count = int(cap.get(self._cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            frame_count = None

        self._video_capture = cap
        self._video_path = resolved
        self._video_filename = os.path.basename(resolved)
        self._fps = fps
        self._frame_count = frame_count
        self._duration_s = (frame_count / fps) if frame_count is not None else None
        self._last_frame_id = None
        self._last_frame_rgb = None
        self._last_rendered_frame_id = None

        if self._duration_s is not None:
            self.offset_slider.valmin = -self._duration_s
            self.offset_slider.ax.set_xlim(self.offset_slider.valmin, self.offset_slider.valmax)
            if self.time_offset_s < self.offset_slider.valmin:
                self.time_offset_s = self.offset_slider.valmin
                self.offset_box.set_val(f"{self.time_offset_s:.3f}")
            self.offset_slider.set_val(self.time_offset_s)

        self.showAtDataTime(self.current_data_time)
        return True

    def _read_frame(self, frame_id):
        if self._video_capture is None:
            return None

        if self._frame_count is not None:
            frame_id = int(np.clip(frame_id, 0, self._frame_count - 1))
        else:
            frame_id = max(0, int(frame_id))

        # Reuse the last decoded frame and avoid expensive seeking when stepping forward.
        if self._last_frame_id == frame_id and self._last_frame_rgb is not None:
            return self._last_frame_rgb

        if self._last_frame_id is None or frame_id != (self._last_frame_id + 1):
            self._video_capture.set(self._cv2.CAP_PROP_POS_FRAMES, frame_id)

        ok, frame = self._video_capture.read()
        if not ok or frame is None:
            return None

        frame_rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        self._last_frame_id = frame_id
        self._last_frame_rgb = frame_rgb
        return frame_rgb

    def _update_overlay(self, frame_id, video_time, data_time, message=None):
        lines = [
            f"file: {self._video_filename}",
            f"frame: {frame_id if frame_id is not None else '-'}",
            f"video time: {video_time:.3f} s" if video_time is not None else "video time: -",
            f"data time: {data_time:.3f} s",
            f"offset: {self.time_offset_s:+.3f} s",
        ]
        if message:
            lines.append(message)
        self._overlay.set_text("\n".join(lines))

    def showAtDataTime(self, data_time):
        self.current_data_time = float(data_time)

        if self._video_capture is None:
            self._update_overlay(frame_id=None, video_time=None, data_time=self.current_data_time, message="No video loaded")
            self.fig.canvas.draw_idle()
            return

        video_time = self.current_data_time + self.time_offset_s
        video_time = max(0.0, video_time)
        if self._duration_s is not None:
            video_time = min(video_time, max(0.0, self._duration_s - 1.0 / self._fps))

        frame_id = int(round(video_time * self._fps))
        if self._frame_count is not None:
            frame_id = int(np.clip(frame_id, 0, self._frame_count - 1))

        if self._last_rendered_frame_id == frame_id and self._imshow is not None:
            self._update_overlay(frame_id=frame_id, video_time=video_time, data_time=self.current_data_time)
            self.fig.canvas.draw_idle()
            return

        frame = self._read_frame(frame_id)
        if frame is None:
            self._update_overlay(frame_id=frame_id, video_time=video_time, data_time=self.current_data_time, message="Failed to decode frame")
            self.fig.canvas.draw_idle()
            return

        if self._imshow is None:
            self._imshow = self.ax_video.imshow(frame)
        else:
            self._imshow.set_data(frame)

        self._last_rendered_frame_id = frame_id
        self._update_overlay(frame_id=frame_id, video_time=video_time, data_time=self.current_data_time)
        self.fig.canvas.draw_idle()

    def mouseHoverCallback(self, event):
        """Update video on hover over a linked FlightPlotter timeline."""
        if event.xdata is not None:
            self.showAtDataTime(event.xdata)
