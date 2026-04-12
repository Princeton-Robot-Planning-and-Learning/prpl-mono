# Bilevel Planning Visualizer — Frontend

React + Vite frontend for the bilevel planning graph visualizer. Built
once with `npm run build`, then served at `/` by the Flask backend at
`bilevel_planning.visualizer.app`. There's only one process at runtime —
this directory only matters during installation and frontend development.

## Prerequisites

- Node.js 18+

## One-time build

```bash
npm ci
npm run build
```

This produces `dist/`, which the Python backend serves as static files.
After this, users only ever run `python -m bilevel_planning.visualizer`
(or `run_webapp` from a script). No npm process is needed at runtime.

## Frontend development mode

If you're editing the React code, Vite's dev server gives you hot
reload:

```bash
npm run dev
```

Vite serves on `http://localhost:3000` and proxies `/api/*` to the
Python backend on `http://localhost:5001`. Run the backend separately
in another terminal while you're iterating.

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
