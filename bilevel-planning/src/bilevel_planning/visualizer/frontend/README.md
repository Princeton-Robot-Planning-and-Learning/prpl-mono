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

## Planned follow-ups

- Move topology loading to the backend so users upload only one artifact
  (a single pickle) instead of a JSON + pickle pair.
- Add an in-browser Python editor pane that defines `render_state_fn`
  and posts it to `/api/set_renderer`, removing the need for a bespoke
  launcher script per environment.
