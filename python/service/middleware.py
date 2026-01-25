"""
生产级中间件：Request ID、可观测性

- RequestIDMiddleware：为每个请求注入 X-Request-ID，便于日志与追踪关联
"""
from __future__ import annotations

import uuid
from contextvars import ContextVar
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

REQUEST_ID_CTX_KEY = "request_id"
request_id_ctx: ContextVar[str] = ContextVar(REQUEST_ID_CTX_KEY, default="")


def get_request_id() -> str:
    """获取当前请求的 Request ID（在请求生命周期内有效）。"""
    return request_id_ctx.get()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """为每个请求注入 X-Request-ID；若请求头已带则复用，否则生成 UUID。"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        token = request_id_ctx.set(rid)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            request_id_ctx.reset(token)
