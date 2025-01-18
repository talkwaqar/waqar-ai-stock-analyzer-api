from functools import lru_cache
from src.app.core.config import get_settings
from src.app.services.langchain_service import StockAnalyzer

settings = get_settings()

@lru_cache()
def get_stock_analyzer() -> StockAnalyzer:
    """Get a cached instance of StockAnalyzer"""
    return StockAnalyzer(
        openai_api_key=settings.OPENAI_API_KEY,
        finnhub_api_key=settings.FINNHUB_API_KEY,
        gemini_api_key=settings.GEMINI_API_KEY,
        default_llm=settings.DEFAULT_LLM
    ) 