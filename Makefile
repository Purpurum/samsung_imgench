# ============================================================
# Makefile: удобные команды для работы с проектом
# ============================================================

.PHONY: help build up down restart logs init-hdfs generate-sample run test clean

help:
	@echo "Satlas Image Enhancement — доступные команды:"
	@echo ""
	@echo "  make build          — собрать Docker-образы"
	@echo "  make up             — поднять весь кластер"
	@echo "  make down           — остановить и удалить контейнеры"
	@echo "  make restart        — перезапустить стек"
	@echo "  make logs           — смотреть логи всех сервисов"
	@echo "  make init-hdfs      — создать директории /data/* в HDFS"
	@echo "  make generate-sample — сгенерировать тестовый снимок"
	@echo "  make run            — запустить пайплайн на тестовом снимке"
	@echo "  make test           — прогнать unit-тесты внутри контейнера"
	@echo "  make clean          — удалить контейнеры + volumes"
	@echo ""
	@echo "WebUI после 'make up':"
	@echo "  Jupyter Lab   — http://localhost:8888"
	@echo "  Spark Master  — http://localhost:8080"
	@echo "  HDFS NameNode — http://localhost:9870"

build:
	docker compose -f docker/docker-compose.yml build

up:
	docker compose -f docker/docker-compose.yml up -d
	@echo "Ждём инициализации сервисов (30 сек)..."
	@sleep 30
	@$(MAKE) init-hdfs
	@echo ""
	@echo "✅ Кластер запущен. Jupyter: http://localhost:8888"

down:
	docker compose -f docker/docker-compose.yml down

restart: down up

logs:
	docker compose -f docker/docker-compose.yml logs -f --tail=100

init-hdfs:
	bash scripts/init_hdfs.sh

generate-sample:
	docker exec satlas-jupyter python /app/scripts/generate_sample.py \
		--output /app/data/samples/sat_001.png --size 1536

run: generate-sample
	bash scripts/run_pipeline.sh /app/data/samples/sat_001.png SAT_DEMO_$(shell date -u +%Y%m%d_%H%M%S)

test:
	docker exec satlas-jupyter bash -c "cd /app && PYTHONPATH=. python -m pytest tests/ -v"

clean:
	docker compose -f docker/docker-compose.yml down -v
	@echo "🧹 Контейнеры и volumes удалены."
