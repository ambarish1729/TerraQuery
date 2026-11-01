# Frontend

A Vite + React 19 + Tailwind single-page app that talks to the backend proxy at `/api`.

- Upload a satellite image to get an automatic caption, then ask free-form questions in the chat panel.
- Answers render inline: captions, grounded boxes, and mask overlays for filtering queries.
- Canvas tools: rotate, flip, zoom, and **crop**. A crop is sent as a new image, so you can query a sub-region.
- Sessions, chat history and cached images are kept in `localStorage`, so you can reopen earlier work after a restart.
- No inference happens in the browser. Every model call goes through `POST /api/handle`.

## Develop

```bash
cd frontend
npm install
npm run dev -- --host        # http://localhost:5173
```

The dev server expects the backend at the same origin under `/api`. Either run it behind Nginx (see [`docs/deployment.md`](../docs/deployment.md)) or add a Vite proxy.

## Build

```bash
npm run build                # outputs to frontend/dist/
```

Serve `dist/` with Nginx and proxy `/api/` to the backend on `127.0.0.1:8000`.

## Layout

```
frontend/
├── public/            # logos (served as-is)
├── src/
│   ├── App.jsx        # entire UI: upload, canvas tools, chat, session store
│   ├── main.jsx
│   └── index.css      # Tailwind entry
├── index.html
├── tailwind.config.js
└── vite.config.js
```
