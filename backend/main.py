import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.chat import router as chat_router
from core.config import get_settings
from core.logging_config import setup_logging

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info("Starting Document and Web Assistant API")

    # Warn at startup about missing optional API keys
    import os
    if not os.getenv("TAVILY_API_KEY"):
        logger.warning("TAVILY_API_KEY not set — web search will be unavailable")
    if not os.getenv("GROQ_API_KEY"):
        logger.warning("GROQ_API_KEY not set — LLM calls will fail")

    app = FastAPI(title="Document and Web Assistant API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(chat_router)
    return app


app = create_app()
