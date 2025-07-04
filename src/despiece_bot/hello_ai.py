"""
Hola Mundo - Integración Google GenAI + CrewAI + DSPy
"""

import asyncio
from typing import Dict, Any
import google.genai as genai
from crewai import Agent, Task, Crew
import dspy
from loguru import logger

from .config import settings


class HelloAIIntegration:
    """Clase que integra los tres frameworks de AI"""
    
    def __init__(self):
        self.setup_google_genai()
        self.setup_dspy()
        
    def setup_google_genai(self):
        """Configurar Google GenAI"""
        try:
            genai.configure(api_key=settings.google_ai_api_key)
            self.gemini_client = genai.GenerativeModel('gemini-pro')
            logger.info("✅ Google GenAI configurado correctamente")
        except Exception as e:
            logger.error(f"❌ Error configurando Google GenAI: {e}")
            raise
    
    def setup_dspy(self):
        """Configurar DSPy con Gemini"""
        try:
            # Configurar DSPy para usar Gemini
            lm = dspy.Google(model="gemini-pro", api_key=settings.google_ai_api_key)
            dspy.settings.configure(lm=lm)
            logger.info("✅ DSPy configurado correctamente")
        except Exception as e:
            logger.error(f"❌ Error configurando DSPy: {e}")
            raise
    
    async def google_genai_hello(self) -> str:
        """Test básico de Google GenAI"""
        try:
            prompt = "Di 'Hola desde Google GenAI' y explica qué eres en una línea"
            response = await self.gemini_client.generate_content_async(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error en Google GenAI: {e}")
            return f"Error: {str(e)}"
    
    def dspy_hello(self) -> str:
        """Test básico de DSPy"""
        try:
            # Definir una signature simple
            class HelloSignature(dspy.Signature):
                """Generar un saludo desde DSPy"""
                input: str = dspy.InputField()
                output: str = dspy.OutputField()
            
            # Crear un predictor
            hello_predictor = dspy.Predict(HelloSignature)
            
            # Ejecutar
            result = hello_predictor(input="Di hola desde DSPy y explica qué haces")
            return result.output
        except Exception as e:
            logger.error(f"Error en DSPy: {e}")
            return f"Error: {str(e)}"
    
    def crewai_hello(self) -> str:
        """Test básico de CrewAI"""
        try:
            # Definir un agente simple
            greeting_agent = Agent(
                role="Saludo Specialist",
                goal="Crear saludos amigables y explicar CrewAI",
                backstory="Eres un agente especializado en crear saludos creativos",
                verbose=True,
                allow_delegation=False
            )
            
            # Definir una tarea
            greeting_task = Task(
                description="Di 'Hola desde CrewAI' y explica qué es CrewAI en una línea",
                agent=greeting_agent,
                expected_output="Un saludo seguido de una explicación corta de CrewAI"
            )
            
            # Crear el crew
            crew = Crew(
                agents=[greeting_agent],
                tasks=[greeting_task],
                verbose=True
            )
            
            # Ejecutar
            result = crew.kickoff()
            return str(result)
        except Exception as e:
            logger.error(f"Error en CrewAI: {e}")
            return f"Error: {str(e)}"
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Ejecutar todos los tests de los frameworks"""
        logger.info("🚀 Iniciando tests de AI Stack...")
        
        results = {}
        
        # Test Google GenAI
        logger.info("🧠 Testing Google GenAI...")
        results["google_genai"] = await self.google_genai_hello()
        
        # Test DSPy
        logger.info("🔬 Testing DSPy...")
        results["dspy"] = self.dspy_hello()
        
        # Test CrewAI
        logger.info("👥 Testing CrewAI...")
        results["crewai"] = self.crewai_hello()
        
        logger.info("✅ Todos los tests completados!")
        return results


# Función de conveniencia
async def test_ai_stack() -> Dict[str, Any]:
    """Función principal para testear el stack AI"""
    integration = HelloAIIntegration()
    return await integration.run_all_tests()
