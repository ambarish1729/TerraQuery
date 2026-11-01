# Backend

A thin FastAPI gateway (`root_path=/api`) between the web UI and the GPU services. It routes each request to the right task, forwards it to the VQA service, and post-processes the result: it draws boxes with PIL, normalises yes/no answers, and converts pixel distances to metres when a spatial resolution is supplied.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/handle` | Main entry point. Multipart `image`, `prompt`, `context` (chat history). Classifies the task (`CAPTION`, `VQA_ATTRIBUTE`, `VQA_NUMERICAL`, `VQA_BINARY`, `VQA_FILTERING`), calls the matching upstream service, and returns `{routed_to, result}`. |
| `POST` | `/api/eval` | JSON benchmark mode. Runs the caption, grounding and VQA queries in the payload directly, without the router. |
| `GET` | `/api/health` | Liveness check. |

## Configure

```bash
cp backend/.env.example backend/.env
```

| Variable | Default in `.env.example` | Notes |
|---|---|---|
| `API_MODE` | `custom` | `custom` posts to the VQA service router. `openai` uses an OpenAI-compatible endpoint at `ROUTER_URL` and needs `pip install openai`. |
| `ROUTER_URL` | `http://localhost:8001/router` | |
| `CAPTION_URL` | `http://localhost:8001/caption` | |
| `GROUND_URL` | `http://localhost:8001/bbox` | |
| `VQA_URL` | `http://localhost:8001/vqa` | `/attribute`, `/numerical`, `/binary` or `/filtering` is appended per request |

## Run

```bash
pip install fastapi uvicorn httpx python-dotenv pillow python-multipart
uvicorn api:app --host 0.0.0.0 --port 8000 --root-path /api      # from backend/
```

For production, see the gunicorn and systemd setup in [`docs/deployment.md`](../docs/deployment.md).
