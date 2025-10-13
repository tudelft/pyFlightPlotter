import numpy as np

# configure matplotlib with local style
import matplotlib.pyplot as plt
from pyFlightPlotter.style import local_rc
plt.rcParams.update(local_rc)

# import stuff
from pyFlightPlotter import FlightPlotterBase, BlittedCursor, Viewport, Tailsitter

#%% derived FlightPlotter class

# create a derived class to implement _populate, using your own data structure
# In this example, we'll just use a list of numpy arrays
class FlightPlotterCompleteExample(FlightPlotterBase):
    def __init__(self, time, data, name="Demo"):
        # store your data in the instance
        self.data = data

        # initialize parent class with the time array
        super().__init__(time, name=name)

        # define a simple 3x2 layout
        self.define_layout(figsize=(14, 8), nrows=3, ncols=2,
                           width_ratios=[1, 1],
                           height_ratios=[1, 1, 1])

        # invoke plotting
        self.plot()

    def _populate(self):
        # use the provided _plot_timeseries function to create subplots
        self._plot_timeseries(
            self.fig.add_subplot(self.gs[0, 0]),    # gridspec generated from define_layout
            light=[self.data[0], self.data[1]],     # thin line
            solid=[self.data[2], self.data[3]],     # solid line
            dashed=None,                            # in this case, no dashed line
            series_labels=["Signal 1", "Signal 2"], # these must match the length of the array given to light, solid, dashed
            style_labels=["Raw", "Filtered", None], # must be len==3: (light, solid, dashed)
            title="Example Time Series",            # title over the subplot
            ylabel="Amplitude",                     # y-axis label
        )

        # another one
        self._plot_timeseries(
            self.fig.add_subplot(self.gs[0, 1]),
            light=None,
            solid=[self.data[4], self.data[5]],
            dashed=[self.data[6], self.data[7]],
            series_labels=["Signal A", "Signal B"],
            style_labels=[None, "Estimate", "Setpoint"],
            title="Another Time Series",
            ylabel="Value",
        )


#%%  use it
if __name__ == "__main__":
    # example data
    t = np.linspace(0, 5, 101)
    data = [
        np.sin(2*np.pi*0.5*t), np.sin(2*np.pi*0.5*t + 0.2),
        np.sin(2*np.pi*0.5*t + 0.4), np.sin(2*np.pi*0.5*t + 0.6),
        np.cos(2*np.pi*0.5*t), np.cos(2*np.pi*0.5*t + 0.2),
        np.cos(2*np.pi*0.5*t + 0.4), np.cos(2*np.pi*0.5*t + 0.6),
    ]

    # create flight plotter instances
    fp = FlightPlotterCompleteExample(t, data, name="Flight Plotter Demo")
    fp2 = FlightPlotterCompleteExample(t, data, name="Another Instance")

    # create vertical cursor that is synced between all axes
    bc = BlittedCursor(fp.all_axes + fp2.all_axes,
                       sharex=True                   # if True, zoom and pan is synced between axes
                       )


    #%% Viewport example

    # create a tailsitter: 2 rotors, 2 control surfaces
    # qd = Quadrotor()
    ts = Tailsitter()

    # fill with some example data. Note: only att are strictly required
    vp = Viewport(ts,
                  time=np.array([0., 1., 2.]),
                  att=np.array([[1., 0., 0., 0.], [1., 0., 0., 0.], [0., 0., 0., 1.]]),
                  attSet=np.array([[1., 0., 0., 0.], [1., 0., 0., 0.1], [0., 0., 0., 1.]]),
                  attMeas=np.array([[1., 0., 0., 0.], [1., 0., 0., -0.1], [0., 0., 0., 1.]]),
                  pos=np.array([[0., 0., 0.], [0.2, 0.2, 0.2], [0.4, 0.4, 0.4]]),
                  posSet=np.array([[0.4, 0.4, 0.4], [0.4, 0.4, 0.4], [0.4, 0.4, 0.4]]),
                  posMeas=np.array([[0., 0., 0.], [0.1, 0.1, 0.1], [0.4, 0.4, 0.4]]),
                  vel=3*np.array([[0., 0., 0.], [0.2, 0.2, 0.2], [0.4, 0.4, 0.4]]),
                  velSet=3*np.array([[0.4, 0.4, 0.4], [0.4, 0.4, 0.4], [0.4, 0.4, 0.4]]),
                  velMeas=3*np.array([[0., 0., 0.], [0.1, 0.1, 0.1], [0.4, 0.4, 0.4]]),
                  acc=5*np.array([[0., 0., 0.], [0.2, -0.2, -0.2], [0.4, -0.4, -0.4]]),
                  accSet=5*np.array([[0.4, 0.4, 0.4], [0.4, -0.4, -0.4], [0.4, -0.4, -0.4]]),
                  accMeas=5*np.array([[0., 0., 0.], [0.1, -0.1, -0.1], [0.4, -0.4, -0.4]]),
                  rotorSet=np.array([
                      [[1., 0., 0.], [1., 0., 0.]], # 0: thrust, 1: tilt1 angle radians, 2: tilt2 angle (perpto tilt1)
                      [[0.5, 0.5, 0.], [0.5, 0., 0.5]],
                      [[1., 0.2, 0.2], [1., -0.2, -0.2]]
                  ]),
                  surfaceSet=np.array([[0., 0.], [0.5, -0.5], [1.0, -0.5]]), # tilt angles in radians
                  interpolation="previous", # also linear possible
                  follow=False, # if True, the camera will follow the vehicle position
                  title="Viewport Demo",
                  )

    # you can also dynamically push frames to the data. Useful for real-time visualization
    for i in range(50):
        vp.pushNewFrame(
            time=2. + 0.1*i,
            att=np.array([np.cos(0.05*i), 0., 0., np.sin(0.05*i)]),   # quaternion, format [w, x, y, z]
            attSet=np.array([np.cos(0.05*i), 0., 0., np.sin(0.05*i)]),
            attMeas=np.array([np.cos(0.05*i), 0., 0., np.sin(0.05*i)]),
            pos=np.array([0.4+0.02*i, 0.4+0.02*i, 0.4+0.02*i]),
            posSet=np.array([0.4, 0.4, 0.4]),
            posMeas=np.array([0.4+0.01*i, 0.4+0.01*i, 0.4+0.01*i]),
            vel=3*np.array([0.4, 0.4, 0.4]),
            velSet=3*np.array([0.4, 0.4, 0.4]),
            velMeas=3*np.array([0.4+0.01*i, 0.4+0.01*i, 0.4+0.01*i]),
            acc=5*np.array([0.4, -0.4, -0.4]),
            accSet=5*np.array([0.4, -0.4, -0.4]),
            accMeas=5*np.array([0.4+0.01*i, -0.4-0.01*i, -0.4-0.01*i]),
            rotorSet=np.array([
                [1., 0.2*np.sin(0.1*i), 0.2*np.cos(0.1*i)], # 0: thrust, 1: tilt1 angle radians, 2: tilt2 angle (perpto tilt1)
                [1., -0.2*np.sin(0.1*i), -0.2*np.cos(0.1*i)]
            ]),
            surfaceSet=np.array([1.+0.1*np.sin(0.1*i), -1.+0.1*np.cos(0.1*i)]), # tilt angles in radians
        )

        if i % 3 == 0:
            vp.showAtTime() # no argument --> show latest
            plt.pause(0.01) # pause to allow GUI to update

    # connect mouse hover event, so that hovering in the FlightPlotter updates the Viewport
    fp.connect_viewport(vp)
    fp2.connect_viewport(vp)