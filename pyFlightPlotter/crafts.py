import numpy as np

from .core import Craft3D

class Quadrotor(Craft3D):
    def __init__(self, l=0.2, R=0.1):
        body = np.array([
            [    l,   -l/4,     0.],
            [    l,   +l/4,     0.],
            [    l+l/4,   0,     0.],
            [    l,   -l/4,     0.],
        ])
        geometry = [body]
        super().__init__(body_geometry=geometry)

        arms = np.array([
            [ -l, +l, 0.],
            [ +l, +l, 0.],
            [ -l, -l, 0.],
            [ +l, -l, 0.],
        ])

        for arm in arms:
            self.addRotor(xyz=arm, axis=[0, 0, -1], R=R)

class Tailsitter(Craft3D):
    def __init__(self, l=0.3, R=0.1):
        body = np.array([
            [    0,   2*l,     l],
            [    0,   2*l,     0.],
            [    0,    0.,   -1*l],
            [    l,    0.,   -1*l],
            [    0,    0.,   -1*l],
            [    0,  -2*l,   0.],
            [    0,  -2*l,   l],
            [    0,   2*l,   l],
        ])
        geometry = [body]
        super().__init__(body_geometry=geometry)

        self.addRotor(xyz=[0, -l, -l], axis=[0, 0, -1], R=R)
        self.addRotor(xyz=[0, +l, -l], axis=[0, 0, -1], R=R)

        surface = np.array([
            [0, -2.*l, 1.0*l],
            [0, -2.*l, 1.5*l],
            [0, -0.5*l, 1.5*l],
            [0, -0.5*l, 1.0*l],
        ])
        self.addSurface(
            tilt_xyz=[0., 0., 1.*l], tilt_axis=[0., +1., 0.], geometry=[surface]
        )
        self.addSurface(
            tilt_xyz=[0., 0., 1.*l], tilt_axis=[0., +1., 0.], geometry=[surface*np.array([1., -1., 1.])]
        )
