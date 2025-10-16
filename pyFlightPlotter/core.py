import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

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

    def _plot_timeseries(self, ax, light=None, solid=None, dashed=None, series_labels=[], style_labels=[None, None, None], title="", ylabel="", ylimits=(None, None)):
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
        lengths = np.array([len(solid), len(light), len(dashed), len(series_labels)])
        if not (lengths == lengths[0]).all():
            raise ValueError("light, solid, dashed and labels must all have the same length if given.")

        for i, series in enumerate(light):
            if series is not None:
                ax.plot(self.t, series, color=COLORS[i], alpha=0.3, lw=1.0, linestyle='-')

        for i, series in enumerate(solid):
            ax.plot(self.t, series, label=series_labels[i], color=COLORS[i], alpha=0.8, lw=1.0, linestyle='-')

        for i, series in enumerate(dashed):
            if series is not None:
                ax.plot(self.t, series, color=COLORS[i], lw=1.5, linestyle='--')

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
                 rotorSet=None, surfaceSet=None,
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
            rotorSet: NxMx3 array of rotor controls (thrust, tilt1, tilt2) for M rotors
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
        if rotorSet is not None:
            self.controls["rotorSet"] = {"raw": rotorSet, "style": "solid",  "color": COLORS[6], "marker": "s", "width": 3.0, "label": "Rotor Controls"}
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
                     rotorSet=None, surfaceSet=None):
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
        if rotorSet is not None:
            self.series["rotorSet"]['raw'] = np.vstack((self.series["rotorSet"]['raw'], rotorSet[np.newaxis]))
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
            if ser in ["att", "attSet"]:
                rotor_controls = interpolates["rotorSet"] if "rotorSet" in interpolates.keys() else None
                surface_controls = interpolates["surfaceSet"] if "surfaceSet" in interpolates.keys() else None
            else:
                # never draw rotors or surface motion for measured attitude as it 
                # would suggest that these are also measured
                rotor_controls = None
                surface_controls = None

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
            if ser == "att" and len(q) > 0 and "rotorSet" in self.series.keys():
                self.series["rotorSet"]['line'] = self.ax.plot(qs[:, 0], qs[:, 1], qs[:, 2],
                             linestyle=self.series["rotorSet"]['style'],
                             color=self.series["rotorSet"]['color'],
                             marker=self.series["rotorSet"]['marker'],
                             markersize=1,
                             lw=self.series["rotorSet"]['width'])[0]

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
