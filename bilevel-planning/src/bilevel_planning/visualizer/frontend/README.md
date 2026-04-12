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
After this, users only ever run `python -m bilevel_planning.visualizer`.
No npm process is needed at runtime.

## Frontend development mode

If you're editing the React code, Vite's dev server gives you hot
reload:

```bash
npm run dev
```

Vite serves on `http://localhost:3000` and proxies `/api/*` to the
Python backend on `http://localhost:5001`. Run the backend separately
in another terminal while you're iterating.

## Supplying the renderer

The visualizer won't display state images until you give the backend a
`render_state` callable. The header has a **Python renderer** pane
(expanded by default) with a textarea — edit the source so
`render_state(state)` returns an HxWx3 uint8 RGB array, then click
**Apply renderer**. The source is POSTed to `/api/set_renderer` and
`exec`'d in the backend process, so any package installed in the
backend's Python environment is importable.

The arbitrary-`exec` surface is the reason the backend binds to
`127.0.0.1`. Don't put it behind a reverse proxy exposing it to untrusted
networks.
