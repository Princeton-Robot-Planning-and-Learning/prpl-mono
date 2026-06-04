# Bilevel Planning Visualizer — Frontend

React + Vite frontend for the bilevel planning graph visualizer. Served
at `/` by the Flask backend at `bilevel_planning.visualizer.app`. There's
only one process at runtime — this directory only matters when you're
editing the frontend.

## Users don't need Node

The built bundle (`dist/`) is committed to the repo, so anyone who has
installed the Python package can run `python -m
bilevel_planning.visualizer` directly. **No Node, npm, or build step is
required to use the visualizer.** This directory matters only to
maintainers editing the React source.

## Rebuilding the bundle (maintainers)

Requires Node.js 18+. After changing any frontend source, rebuild and
commit the updated `dist/`:

```bash
../../../../scripts/build_frontend.sh   # from this directory
```

or equivalently `npm ci && npm run build`. Committing the result is what
keeps the no-Node-required guarantee true for users.

## Frontend development mode

If you're editing the React code, Vite's dev server gives you hot
reload:

```bash
npm run dev
```

Vite serves on `http://localhost:3000` and proxies `/api/*` to the
Python backend on `http://localhost:5001`. Run the backend separately
in another terminal (with `--bundle`/`--renderer`) while you're iterating.

## The bundle and renderer

The frontend has no upload or code-entry UI: the backend is launched with
a bundle and a renderer file already chosen, then this frontend fetches
`/api/graph` on load and renders states via `/api/visualize_state` when
you click a node. See the backend module `bilevel_planning.visualizer.app`
for how the bundle and `render_state` file are loaded.

The renderer file is `exec`'d in the backend process, which is why the
backend binds to `127.0.0.1`. Don't put it behind a reverse proxy
exposing it to untrusted networks.
