# Bilevel Planning Visualizer — Frontend

React + Vite frontend for the bilevel planning graph visualizer. Pairs with
the Flask backend at `bilevel_planning.visualizer.app`.

## Prerequisites

- Node.js 18+
- The Python backend running on `http://localhost:5001` (see the module
  docstring on `bilevel_planning.visualizer.app.run_webapp`).

## Development

```bash
npm install
npm run dev
```

Vite serves the app on `http://localhost:3000` and proxies `/api/*`
requests to the backend on port 5001.

## Production build

```bash
npm run build
```

Outputs a static bundle to `dist/`.

## Installing a renderer

The visualizer won't display state images until the backend has a
`render_state` callable. Two ways to supply one:

- **Browser**: expand the "Python renderer" pane in the header, edit the
  source, and click "Apply renderer". The source is POSTed to
  `/api/set_renderer` and `exec`'d in the backend process, so any package
  installed in the backend's Python environment is importable. Runs
  arbitrary Python locally — don't expose the backend to untrusted
  networks.
- **Launcher script**: call `bilevel_planning.visualizer.app.run_webapp`
  from your own Python with `render_state_fn=<your callable>`. Useful
  when the environment is expensive to construct and you want it ready
  before the server boots.
