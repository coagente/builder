"""
Configuración Simple para Despiece-Bot
"""

from pydantic_settings import BaseSettings
from pydantic import Field
import os


class Settings(BaseSettings):
    """Configuración simple de la aplicación"""
    
    # API Configuration
    app_name: str = "Despiece-Bot Simple AI"
    version: str = "1.0.0-simple"
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Google AI
    google_ai_api_key: str = Field(..., env="GOOGLE_AI_API_KEY")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
