# =============================================================================
# DESPIECE-BOT SIMPLE AI - Makefile
# =============================================================================

.PHONY: help setup build dev dev-detached stop logs health test-ai clean

# Default target
.DEFAULT_GOAL := help

help: ## 📋 Mostrar comandos disponibles
	@echo "🤖 Despiece-Bot Simple AI - Comandos:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""

setup: ## 🚀 Setup inicial (copia env y prepara proyecto)
	@echo "🚀 Configurando Despiece-Bot Simple AI..."
	@if [ ! -f .env.local ]; then \
		cp .env.example .env.local; \
		echo "✅ Archivo .env.local creado"; \
		echo "📝 IMPORTANTE: Edita .env.local y agrega tu GOOGLE_AI_API_KEY"; \
	else \
		echo "⚠️  .env.local ya existe"; \
	fi
	@mkdir -p data logs
	@echo "📁 Directorios data/ y logs/ creados"
	@echo ""
	@echo "🔑 Próximo paso:"
	@echo "   1. Obtén tu API key: https://makersuite.google.com/app/apikey"
	@echo "   2. Edita .env.local y agrega: GOOGLE_AI_API_KEY=tu_api_key"
	@echo "   3. Ejecuta: make dev"

build: ## 🐳 Construir imagen Docker
	@echo "🐳 Construyendo imagen Docker..."
	docker-compose build
	@echo "✅ Build completado"

dev: ## 🔥 Ejecutar en desarrollo con logs
	@echo "🔥 Iniciando servidor de desarrollo..."
	docker-compose -f docker-compose.dev.yml up

dev-detached: ## �� Ejecutar en desarrollo (background)
	@echo "🌙 Iniciando servidor en background..."
	docker-compose -f docker-compose.dev.yml up -d
	@echo "✅ Servidor ejecutándose en: http://localhost:8000"
	@echo "🧪 Probar AI Stack: http://localhost:8000/test-ai-stack"

stop: ## 🛑 Parar servicios
	@echo "🛑 Parando servicios..."
	docker-compose -f docker-compose.dev.yml down
	docker-compose down

logs: ## 📋 Ver logs del contenedor
	@echo "📋 Mostrando logs..."
	docker-compose -f docker-compose.dev.yml logs -f

health: ## 🏥 Verificar estado de la aplicación
	@echo "🏥 Verificando salud de la aplicación..."
	@curl -s http://localhost:8000/health | python -m json.tool 2>/dev/null || echo "❌ Servicio no disponible en http://localhost:8000"

test-ai: ## 🧪 Probar integración completa del AI Stack
	@echo "🧪 Probando AI Stack (Google GenAI + CrewAI + DSPy)..."
	@curl -s http://localhost:8000/test-ai-stack | python -m json.tool 2>/dev/null || echo "❌ AI Stack no disponible"

clean: ## 🧹 Limpiar contenedores, volúmenes e imágenes
	@echo "🧹 Limpiando proyecto..."
	docker-compose -f docker-compose.dev.yml down -v
	docker-compose down -v
	docker system prune -f
	@echo "✅ Limpieza completada"

# Aliases para comandos comunes
up: dev ## Alias para 'dev'
down: stop ## Alias para 'stop'
restart: stop dev ## Reiniciar servicios
