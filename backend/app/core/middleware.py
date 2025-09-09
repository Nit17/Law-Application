from __future__ import annotations
import json, time
from pathlib import Path
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "token_usage.jsonl"

class TokenUsageMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response: Response | None = None
        error = None
        try:
            response = await call_next(request)
            return response
        except Exception as e:  # pragma: no cover
            error = str(e)
            raise
        finally:
            try:
                record = {
                    "ts": time.time(),
                    "path": request.url.path,
                    "method": request.method,
                    "status": getattr(response, 'status_code', None),
                    "duration_ms": round((time.time() - start) * 1000, 2),
                    "error": error,
                }
                with LOG_FILE.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
            except Exception:
                pass
