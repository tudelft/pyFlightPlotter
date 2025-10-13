from matplotlib import pyplot as plt

local_rc = plt.rcParams.copy()
local_rc.update({
#    "text.usetex": True,
#    "font.family": "Helvetica",
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.grid": True,
    "axes.grid.which": 'both',
    "grid.linestyle": '--',
    "grid.alpha": 0.7,
    "axes.labelsize": 10,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.loc": 'upper left',
    "legend.fontsize": 8,
    'figure.subplot.bottom': 0.06,
    'figure.subplot.left': 0.05,
    'figure.subplot.right': 0.95,
    'figure.subplot.top': 0.925,
    'figure.subplot.wspace': 0.3,
    'figure.subplot.hspace': 0.433,
})
