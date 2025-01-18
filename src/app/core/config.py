from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    PROJECT_NAME: str = "Stock Earnings Analyzer"
    OPENAI_API_KEY: str
    FINNHUB_API_KEY: str
    GEMINI_API_KEY: str
    DEFAULT_LLM: str = "gemini"  # Supported values: "gemini" or "openai"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def supported_llms(self) -> dict:
        """Return dictionary of supported LLMs"""
        return {
            "openai": "OpenAI GPT-4",
            "gemini": "Google Gemini Pro"
        }

    def validate_llm(self, llm: str) -> bool:
        """Validate if the LLM is supported"""
        return llm in self.supported_llms

@lru_cache()
def get_settings():
    return Settings() 