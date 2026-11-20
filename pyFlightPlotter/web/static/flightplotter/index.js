// flightplotter: synchronized time-series and 3D craft plots in the browser,
// the JS counterpart of pyFlightPlotter. Needs "uplot" and "three" resolvable
// (bundler, or an import map), plus style.css and uplot/dist/uPlot.min.css.
export { FlightPlotter } from "./plotter.js";
export { Dataset } from "./dataset.js";
export { Craft3D, quadrotor, tailsitter, multirotor, noseArrow } from "./crafts.js";
export { COLORS } from "./plots.js";
export { welch } from "./psd.js";
export { rotateByQuaternion, lowpassFiltfilt, derivative, sampleRate, bitChangeEvents } from "./signal.js";
