# Notes for agents

Facts about this repo that aren't obvious from the code. User-facing usage is in `README.md`.

## Two front ends, one data model
- `pyFlightPlotter/` (matplotlib) and `pyFlightPlotter/web/` (browser) are separate implementations of the same
  ideas. `WebPlotter.add_panel` takes the same arguments as `_plot_timeseries`, and `set_viewport` takes the same
  arguments as `Viewport`. Keep the two signatures in step when changing either one.
- The library is domain-agnostic. It knows nothing about Indiflight, Betaflight or BFL files, and the browser version has
  no file loading or drag-and-drop. Consumers decode, scale to SI and choose the layout. Known consumers:
  `support/LogAnalysis/webPlotter.py` in tudelft/indiflight, and the Flight Plotter tab in
  tudelft/indiflight-configurator (`src/js/indiflight/flightPlotterLayout.js`). Don't add consumer-specific code here.

## JS package (`pyFlightPlotter/web/static/flightplotter/`)
- Plain ES modules with no build step. Bundlers consume them through the root `package.json` (npm name
  `flightplotter`; `files` restricts the published contents to that folder). The configurator installs it as a git
  dependency on this branch, so a push is a release for it. It picks up changes only after `npm install` updates the
  locked commit.
- `uplot` is a dependency and `three` a peer dependency. Import them only by bare specifier (`"uplot"`, `"three"`,
  `"three/addons/..."`). The Python-served page resolves those through the import map in `static/index.html`; bundlers
  resolve them through `node_modules`.
- `viewport.js` (three.js) is loaded with a dynamic `import()` only when a viewport is configured, so plots-only users
  don't pay for three.js. Keep it that way.
- All CSS is scoped under `.fp-root` and themed with `--fp-*` variables (`.fp-dark` for dark). The host also has to load
  `uplot/dist/uPlot.min.css`.
- `FlightPlotter.toolbarSlot` is an empty element in the toolbar that hosts fill with their own controls (the configurator
  teleports its file selector into it).
- Performance matters, because logs reach about 10^6 samples × 300 columns:
  - Store columns as `Float32Array`; only time is `Float64Array`.
  - Coalesce redraws to one per animation frame.
  - Never rebuild uPlot instances on hover or playback.
  - The 3D view renders only when something has changed.

## Python ↔ page protocol
- `WebPlotter.show()` runs a singleton `ThreadingHTTPServer` on 127.0.0.1 that serves `static/`. Each plot is a session
  under `/s/<id>/`, made of `meta.json` (panels, viewport, column table, events) and `data.bin`.
- In `data.bin`, the float64 columns come first and the float32 columns after them, so every column stays aligned. The
  time column is named `__time__`. `main.js` reads `?src=/s/<id>/` and builds a `Dataset` from the buffer without copying.
- `meta.json` must be strict JSON: `_jsonable` replaces NaN and inf with null and converts numpy scalars.

## Vendored copies (`static/vendor/`)
- These are verbatim copies of uPlot 1.6.32 (`uPlot.esm.js`, `uPlot.min.css`) and three.js r184
  (`three.module.min.js`, `three.core.min.js`, and the used addons from `examples/jsm/`). The vendored set lets the
  Python page work offline.
- When updating them, copy the files from `node_modules/<pkg>/dist` or `build/` and keep each `LICENSE`. Addons import
  `"three"` by bare specifier, so they don't need edits.

## Packaging
- `pyproject.toml` uses namespace discovery so that the `static/` asset folders are data-only packages. Each folder needs
  its own `package-data` entry. Listing packages by hand brings back setuptools' "Package would be ignored" warning.
  When you add a new asset folder, add a `package-data` line for it, then check that `pip wheel --no-deps .` lists the
  new files.
