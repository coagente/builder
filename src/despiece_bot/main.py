"""
Despiece-Bot - FastAPI Main Application
Integración Simple: Google GenAI + CrewAI + DSPy
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any
import asyncio
from loguru import logger

from .config import settings
from .hello_ai import test_ai_stack, HelloAIIntegration

# Configurar logging
logger.add("logs/app.log", rotation="500 MB", level=settings.log_level)

# Crear aplicación FastAPI
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Sistema simple de AI con Google GenAI + CrewAI + DSPy",
    debug=settings.debug
)


@app.get("/")
async def root():
    """Endpoint raíz con información básica"""
    return {
        "message": "🤖 Bienvenido a Despiece-Bot Simple AI",
        "version": settings.version,
        "environment": settings.environment,
        "ai_stack": ["Google GenAI", "CrewAI", "DSPy"],
        "endpoints": {
            "health": "/health",
            "test_all": "/test-ai-stack",
            "test_genai": "/test-genai",
            "test_crewai": "/test-crewai", 
            "test_dspy": "/test-dspy"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "service": settings.app_name,
            "version": settings.version,
            "environment": settings.environment,
            "timestamp": "2025-01-01T00:00:00Z"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Service unhealthy")


@app.get("/test-ai-stack")
async def test_all_ai():
    """Test todos los frameworks AI juntos"""
    try:
        logger.info("🚀 Iniciando test completo del AI Stack...")
        
        results = await test_ai_stack()
        
        return {
            "success": True,
            "message": "✅ AI Stack test completado exitosamente!",
            "results": results,
            "frameworks_tested": ["google_genai", "crewai", "dspy"]
        }
    except Exception as e:
        logger.error(f"Error en test AI stack: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error testing AI stack: {str(e)}"
        )


@app.get("/test-genai")
async def test_google_genai():
    """Test solo Google GenAI"""
    try:
        logger.info("🧠 Testing Google GenAI...")
        
        integration = HelloAIIntegration()
        result = await integration.google_genai_hello()
        
        return {
            "success": True,
            "framework": "Google GenAI",
            "result": result
        }
    except Exception as e:
        logger.error(f"Error en Google GenAI: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error with Google GenAI: {str(e)}"
        )


@app.get("/test-crewai")
async def test_crewai():
    """Test solo CrewAI"""
    try:
        logger.info("👥 Testing CrewAI...")
        
        integration = HelloAIIntegration()
        result = integration.crewai_hello()
        
        return {
            "success": True,
            "framework": "CrewAI",
            "result": result
        }
    except Exception as e:
        logger.error(f"Error en CrewAI: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error with CrewAI: {str(e)}"
        )


@app.get("/test-dspy")
async def test_dspy():
    """Test solo DSPy"""
    try:
        logger.info("🔬 Testing DSPy...")
        
        integration = HelloAIIntegration()
        result = integration.dspy_hello()
        
        return {
            "success": True,
            "framework": "DSPy",
            "result": result
        }
    except Exception as e:
        logger.error(f"Error en DSPy: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error with DSPy: {str(e)}"
        )


# Event handlers
@app.on_event("startup")
async def startup_event():
    """Eventos de inicio"""
    logger.info(f"🚀 Iniciando {settings.app_name} v{settings.version}")
    logger.info(f"🔧 Ambiente: {settings.environment}")
    logger.info(f"🤖 AI Stack: Google GenAI + CrewAI + DSPy")


@app.on_event("shutdown")
async def shutdown_event():
    """Eventos de cierre"""
    logger.info("🛑 Cerrando Despiece-Bot Simple AI")


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handler global para excepciones"""
    logger.error(f"Excepción global: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "Something went wrong"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
