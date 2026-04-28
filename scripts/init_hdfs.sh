#!/bin/bash
# ============================================================
# Создаёт рабочие директории в HDFS.
# Выполняется один раз после старта кластера.
#
# Использование:
#   docker exec -it satlas-namenode bash /scripts/init_hdfs.sh
# или из хоста:
#   bash scripts/init_hdfs.sh
# ============================================================

set -euo pipefail

echo "[init_hdfs] Проверяю готовность NameNode..."
until docker exec satlas-namenode hdfs dfsadmin -report > /dev/null 2>&1; do
    echo "  NameNode ещё не готов, жду 3 сек..."
    sleep 3
done

echo "[init_hdfs] Создаю структуру директорий..."
docker exec satlas-namenode hdfs dfs -mkdir -p /data/input
docker exec satlas-namenode hdfs dfs -mkdir -p /data/output
docker exec satlas-namenode hdfs dfs -mkdir -p /data/metadata

echo "[init_hdfs] Устанавливаю открытые права (учебная среда)..."
docker exec satlas-namenode hdfs dfs -chmod -R 777 /data

echo "[init_hdfs] Готово. Структура HDFS:"
docker exec satlas-namenode hdfs dfs -ls -R /data
