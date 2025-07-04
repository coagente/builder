# =============================================================================
# DESPIECE-BOT - Makefile para automatización
# =============================================================================
# Basado en PRD v2.1 - Arquitectura Final Completa

.PHONY: help install install-dev clean lint format test build dev prod logs monitor backup deploy

# Variables
PYTHON := python3.11
PIP := pip
DOCKER_COMPOSE_DEV := docker-compose -f docker-compose.dev.yml
DOCKER_COMPOSE_PROD := docker-compose -f docker-compose.yml
PROJECT_NAME := despiece-bot
VERSION := 1.0.0

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[0;37m
RESET := \033[0m

# Default target
.DEFAULT_GOAL := help

# =============================================================================
# HELP
# =============================================================================

help: ## 📚 Mostrar ayuda
	@echo "$(CYAN)=============================================================================$(RESET)"
	@echo "$(CYAN)🏗️  DESPIECE-BOT - Sistema Multi-Agente para Construcción$(RESET)"
	@echo "$(CYAN)=============================================================================$(RESET)"
	@echo ""
	@echo "$(YELLOW)📋 Comandos disponibles:$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BLUE)🚀 Quick Start:$(RESET)"
	@echo "  $(WHITE)1.$(RESET) make install       # Instalar dependencias"
	@echo "  $(WHITE)2.$(RESET) make setup         # Configurar entorno"
	@echo "  $(WHITE)3.$(RESET) make dev           # Ejecutar en desarrollo"
	@echo ""

# =============================================================================
# INSTALLATION & SETUP
# =============================================================================

install: ## 📦 Instalar dependencias de producción
	@echo "$(BLUE)📦 Instalando dependencias de producción...$(RESET)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✅ Dependencias instaladas correctamente$(RESET)"

install-dev: ## 🔧 Instalar dependencias de desarrollo
	@echo "$(BLUE)🔧 Instalando dependencias de desarrollo...$(RESET)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .
	@echo "$(GREEN)✅ Dependencias de desarrollo instaladas$(RESET)"

setup: install-dev ## ⚙️ Configurar entorno de desarrollo completo
	@echo "$(BLUE)⚙️ Configurando entorno de desarrollo...$(RESET)"
	@if [ ! -f .env.local ]; then \
		echo "$(YELLOW)📝 Creando archivo .env.local desde plantilla...$(RESET)"; \
		cp env.example .env.local; \
		echo "$(YELLOW)⚠️  Recuerda configurar tus API keys en .env.local$(RESET)"; \
	fi
	pre-commit install
	@mkdir -p logs data/lancedb storage/{uploads,documents,temp,processed} cache/dspy
	@echo "$(GREEN)✅ Entorno configurado correctamente$(RESET)"

# =============================================================================
# CODE QUALITY
# =============================================================================

lint: ## 🔍 Ejecutar linting
	@echo "$(BLUE)🔍 Ejecutando linting...$(RESET)"
	flake8 src/ tests/
	mypy src/ tests/
	bandit -r src/
	@echo "$(GREEN)✅ Linting completado$(RESET)"

format: ## 🎨 Formatear código
	@echo "$(BLUE)🎨 Formateando código...$(RESET)"
	black src/ tests/
	isort src/ tests/
	@echo "$(GREEN)✅ Código formateado$(RESET)"

format-check: ## 🔎 Verificar formato sin cambios
	@echo "$(BLUE)🔎 Verificando formato...$(RESET)"
	black --check src/ tests/
	isort --check-only src/ tests/
	@echo "$(GREEN)✅ Formato verificado$(RESET)"

security: ## 🔒 Ejecutar escaneo de seguridad
	@echo "$(BLUE)🔒 Ejecutando escaneo de seguridad...$(RESET)"
	bandit -r src/
	safety check
	pip-audit
	@echo "$(GREEN)✅ Escaneo de seguridad completado$(RESET)"

pre-commit: ## 🎯 Ejecutar todos los hooks de pre-commit
	@echo "$(BLUE)🎯 Ejecutando hooks de pre-commit...$(RESET)"
	pre-commit run --all-files
	@echo "$(GREEN)✅ Pre-commit hooks completados$(RESET)"

# =============================================================================
# TESTING
# =============================================================================

test: ## 🧪 Ejecutar todos los tests
	@echo "$(BLUE)🧪 Ejecutando tests...$(RESET)"
	pytest tests/ -v --tb=short
	@echo "$(GREEN)✅ Tests completados$(RESET)"

test-unit: ## 🔬 Ejecutar tests unitarios
	@echo "$(BLUE)🔬 Ejecutando tests unitarios...$(RESET)"
	pytest tests/unit/ -v -m "unit"
	@echo "$(GREEN)✅ Tests unitarios completados$(RESET)"

test-integration: ## 🔗 Ejecutar tests de integración
	@echo "$(BLUE)🔗 Ejecutando tests de integración...$(RESET)"
	pytest tests/integration/ -v -m "integration"
	@echo "$(GREEN)✅ Tests de integración completados$(RESET)"

test-e2e: ## 🎭 Ejecutar tests end-to-end
	@echo "$(BLUE)🎭 Ejecutando tests end-to-end...$(RESET)"
	pytest tests/e2e/ -v -m "e2e"
	@echo "$(GREEN)✅ Tests e2e completados$(RESET)"

test-cov: ## 📊 Ejecutar tests con coverage
	@echo "$(BLUE)📊 Ejecutando tests con coverage...$(RESET)"
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)✅ Coverage report generado en htmlcov/$(RESET)"

test-fast: ## ⚡ Ejecutar solo tests rápidos
	@echo "$(BLUE)⚡ Ejecutando tests rápidos...$(RESET)"
	pytest tests/ -v -m "not slow" --tb=short
	@echo "$(GREEN)✅ Tests rápidos completados$(RESET)"

test-ai: ## 🤖 Ejecutar tests que requieren AI/LLM
	@echo "$(BLUE)🤖 Ejecutando tests de AI...$(RESET)"
	@echo "$(YELLOW)⚠️  Estos tests requieren API keys válidas$(RESET)"
	pytest tests/ -v -m "ai" --tb=short
	@echo "$(GREEN)✅ Tests de AI completados$(RESET)"

# =============================================================================
# DOCKER OPERATIONS
# =============================================================================

build: ## 🏗️ Construir imágenes Docker
	@echo "$(BLUE)🏗️ Construyendo imágenes Docker...$(RESET)"
	$(DOCKER_COMPOSE_DEV) build
	@echo "$(GREEN)✅ Imágenes construidas$(RESET)"

dev: ## 🚀 Ejecutar en modo desarrollo
	@echo "$(BLUE)🚀 Iniciando en modo desarrollo...$(RESET)"
	$(DOCKER_COMPOSE_DEV) up --build
	@echo "$(GREEN)✅ Servidor de desarrollo iniciado$(RESET)"

dev-detached: ## 🌙 Ejecutar en modo desarrollo (background)
	@echo "$(BLUE)🌙 Iniciando en modo desarrollo (background)...$(RESET)"
	$(DOCKER_COMPOSE_DEV) up --build -d
	@echo "$(GREEN)✅ Servidor ejecutándose en background$(RESET)"

prod: ## 🏭 Ejecutar en modo producción
	@echo "$(BLUE)🏭 Iniciando en modo producción...$(RESET)"
	$(DOCKER_COMPOSE_PROD) up -d
	@echo "$(GREEN)✅ Servidor de producción iniciado$(RESET)"

stop: ## ⏹️ Parar servicios
	@echo "$(BLUE)⏹️ Parando servicios...$(RESET)"
	$(DOCKER_COMPOSE_DEV) stop
	$(DOCKER_COMPOSE_PROD) stop
	@echo "$(GREEN)✅ Servicios parados$(RESET)"

down: ## ⬇️ Parar y remover contenedores
	@echo "$(BLUE)⬇️ Parando y removiendo contenedores...$(RESET)"
	$(DOCKER_COMPOSE_DEV) down
	$(DOCKER_COMPOSE_PROD) down
	@echo "$(GREEN)✅ Contenedores removidos$(RESET)"

logs: ## 📋 Ver logs de servicios
	@echo "$(BLUE)📋 Mostrando logs...$(RESET)"
	$(DOCKER_COMPOSE_DEV) logs -f

logs-api: ## 📋 Ver logs del API Gateway
	@echo "$(BLUE)📋 Mostrando logs del API Gateway...$(RESET)"
	$(DOCKER_COMPOSE_DEV) logs -f api-gateway

logs-db: ## 📋 Ver logs de la base de datos
	@echo "$(BLUE)📋 Mostrando logs de PostgreSQL...$(RESET)"
	$(DOCKER_COMPOSE_DEV) logs -f postgres

# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

db-migrate: ## 🗄️ Ejecutar migraciones de base de datos
	@echo "$(BLUE)🗄️ Ejecutando migraciones...$(RESET)"
	alembic upgrade head
	@echo "$(GREEN)✅ Migraciones completadas$(RESET)"

db-migrate-create: ## 📝 Crear nueva migración
	@echo "$(BLUE)📝 Creando nueva migración...$(RESET)"
	@read -p "Nombre de la migración: " name; \
	alembic revision --autogenerate -m "$$name"
	@echo "$(GREEN)✅ Migración creada$(RESET)"

db-seed: ## 🌱 Cargar datos de prueba
	@echo "$(BLUE)🌱 Cargando datos de prueba...$(RESET)"
	$(PYTHON) scripts/seed_database.py
	@echo "$(GREEN)✅ Datos de prueba cargados$(RESET)"

db-reset: ## 🔄 Resetear base de datos
	@echo "$(RED)⚠️  ADVERTENCIA: Esto eliminará todos los datos$(RESET)"
	@read -p "¿Estás seguro? (y/N): " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		alembic downgrade base; \
		alembic upgrade head; \
		echo "$(GREEN)✅ Base de datos reseteada$(RESET)"; \
	else \
		echo "$(YELLOW)❌ Operación cancelada$(RESET)"; \
	fi

# =============================================================================
# DEVELOPMENT UTILITIES
# =============================================================================

shell: ## 🐚 Abrir shell interactivo
	@echo "$(BLUE)🐚 Abriendo shell interactivo...$(RESET)"
	$(PYTHON) -c "from src.despiece_bot.core.config import settings; import IPython; IPython.embed()"

jupyter: ## 📊 Iniciar Jupyter Lab
	@echo "$(BLUE)📊 Iniciando Jupyter Lab...$(RESET)"
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

dspy-optimize: ## 🧠 Optimizar módulos DSPy
	@echo "$(BLUE)🧠 Optimizando módulos DSPy...$(RESET)"
	$(PYTHON) scripts/optimize_dspy.py
	@echo "$(GREEN)✅ Optimización DSPy completada$(RESET)"

# =============================================================================
# MONITORING & OBSERVABILITY
# =============================================================================

monitor: ## 📊 Abrir dashboards de monitoreo
	@echo "$(BLUE)📊 Dashboards de monitoreo:$(RESET)"
	@echo "$(CYAN)  • Grafana:    http://localhost:3000$(RESET)"
	@echo "$(CYAN)  • Prometheus: http://localhost:9090$(RESET)"
	@echo "$(CYAN)  • API Docs:   http://localhost:8000/docs$(RESET)"
	@echo "$(CYAN)  • Redis:      http://localhost:8001$(RESET)"

health: ## 🏥 Verificar salud del sistema
	@echo "$(BLUE)🏥 Verificando salud del sistema...$(RESET)"
	curl -f http://localhost:8000/health || echo "$(RED)❌ API Gateway no disponible$(RESET)"
	$(DOCKER_COMPOSE_DEV) ps

metrics: ## 📈 Mostrar métricas del sistema
	@echo "$(BLUE)📈 Mostrando métricas...$(RESET)"
	curl -s http://localhost:9090/api/v1/query?query=up | jq .

# =============================================================================
# BACKUP & RESTORE
# =============================================================================

backup: ## 💾 Crear backup de la base de datos
	@echo "$(BLUE)💾 Creando backup...$(RESET)"
	./scripts/backup.sh
	@echo "$(GREEN)✅ Backup completado$(RESET)"

restore: ## 🔄 Restaurar backup
	@echo "$(BLUE)🔄 Restaurando backup...$(RESET)"
	@read -p "Archivo de backup: " backup_file; \
	./scripts/restore.sh "$$backup_file"
	@echo "$(GREEN)✅ Backup restaurado$(RESET)"

# =============================================================================
# DEPLOYMENT
# =============================================================================

deploy-staging: ## 🚀 Desplegar a staging
	@echo "$(BLUE)🚀 Desplegando a staging...$(RESET)"
	./scripts/deploy.sh staging
	@echo "$(GREEN)✅ Desplegado a staging$(RESET)"

deploy-prod: ## 🏭 Desplegar a producción
	@echo "$(RED)⚠️  DESPLEGANDO A PRODUCCIÓN$(RESET)"
	@read -p "¿Confirmas el despliegue a producción? (y/N): " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		./scripts/deploy.sh production; \
		echo "$(GREEN)✅ Desplegado a producción$(RESET)"; \
	else \
		echo "$(YELLOW)❌ Despliegue cancelado$(RESET)"; \
	fi

# =============================================================================
# CLEANUP
# =============================================================================

clean: ## 🧹 Limpiar archivos temporales
	@echo "$(BLUE)🧹 Limpiando archivos temporales...$(RESET)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/
	@echo "$(GREEN)✅ Limpieza completada$(RESET)"

clean-docker: ## 🐳 Limpiar contenedores y volúmenes
	@echo "$(BLUE)🐳 Limpiando Docker...$(RESET)"
	$(DOCKER_COMPOSE_DEV) down -v
	docker system prune -f
	@echo "$(GREEN)✅ Docker limpio$(RESET)"

clean-all: clean clean-docker ## 🧹 Limpieza completa
	@echo "$(GREEN)✅ Limpieza completa finalizada$(RESET)"

# =============================================================================
# INFORMATION
# =============================================================================

version: ## ℹ️ Mostrar información de versión
	@echo "$(CYAN)=============================================================================$(RESET)"
	@echo "$(CYAN)📋 INFORMACIÓN DEL PROYECTO$(RESET)"
	@echo "$(CYAN)=============================================================================$(RESET)"
	@echo "$(WHITE)Proyecto:$(RESET) $(PROJECT_NAME)"
	@echo "$(WHITE)Versión:$(RESET) $(VERSION)"
	@echo "$(WHITE)Python:$(RESET) $(shell $(PYTHON) --version)"
	@echo "$(WHITE)Docker:$(RESET) $(shell docker --version)"
	@echo "$(WHITE)Docker Compose:$(RESET) $(shell docker-compose --version)"
	@echo "$(CYAN)=============================================================================$(RESET)"

status: ## 📊 Mostrar estado actual
	@echo "$(BLUE)📊 Estado actual del sistema:$(RESET)"
	@echo ""
	@echo "$(WHITE)🐳 Contenedores Docker:$(RESET)"
	$(DOCKER_COMPOSE_DEV) ps
	@echo ""
	@echo "$(WHITE)💾 Espacio en disco:$(RESET)"
	df -h | grep -E "(Filesystem|/dev/)"
	@echo ""
	@echo "$(WHITE)🧠 Memoria:$(RESET)"
	free -h 