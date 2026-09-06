"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

from src.api.errors import APIError, error_payload
from src.api.routes import router
from src.api.schemas import ErrorResponse
from src.core.execution import ResearchExecutionService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = ResearchExecutionService()
    app.state.execution_service = service
    try:
        yield
    finally:
        service.close()


app = FastAPI(
    title="Autonomous Research & Report Agent API",
    description="API for autonomous web research, analysis, report writing, and self-critique.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(APIError)
async def api_error_handler(_: Request, exc: APIError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=error_payload(exc))


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    safe_errors = [
        {
            "loc": error.get("loc", []),
            "msg": error.get("msg", "Invalid request"),
            "type": error.get("type", "validation_error"),
        }
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={
            "error": "ValidationError",
            "message": "Request validation failed.",
            "detail": str(safe_errors),
            "report_id": None,
        },
    )


@app.exception_handler(Exception)
async def unexpected_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled API exception", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected internal error occurred.",
            "detail": "The request could not be completed.",
            "report_id": None,
        },
    )


app.include_router(router)
