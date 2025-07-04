#!/bin/bash
# =============================================================================
# DESPIECE-BOT - Health Check Script
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
API_URL="http://localhost:8000"
POSTGRES_CONTAINER="despiece-postgres-dev"
REDIS_CONTAINER="despiece-redis-dev"
API_CONTAINER="despiece-api-dev"

echo -e "${BLUE}🏥 DESPIECE-BOT Health Check${NC}"
echo "=================================="

# Function to check if container is running
check_container() {
    local container_name=$1
    local service_name=$2
    
    if docker ps --format "table {{.Names}}" | grep -q "^${container_name}$"; then
        echo -e "${GREEN}✅ ${service_name}: Container running${NC}"
        return 0
    else
        echo -e "${RED}❌ ${service_name}: Container not running${NC}"
        return 1
    fi
}

# Main health checks
echo -e "\n${BLUE}📦 Checking Docker Containers...${NC}"
errors=0

# Check containers
check_container "${API_CONTAINER}" "FastAPI" || ((errors++))
check_container "${POSTGRES_CONTAINER}" "PostgreSQL" || ((errors++))
check_container "${REDIS_CONTAINER}" "Redis" || ((errors++))

# Summary
echo -e "\n${BLUE}📊 Health Check Summary${NC}"
echo "=================================="

if [ $errors -eq 0 ]; then
    echo -e "${GREEN}🎉 All services are healthy!${NC}"
    exit 0
else
    echo -e "${RED}⚠️  Found ${errors} issue(s)${NC}"
    exit 1
fi
