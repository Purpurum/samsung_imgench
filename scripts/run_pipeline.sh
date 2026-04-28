#!/bin/bash
# ============================================================
# Удобная обёртка над spark-submit — запускает пайплайн из
# контейнера jupyter с правильными параметрами.
#
# Использование:
#   bash scripts/run_pipeline.sh <image_path_in_container> <image_id>
#
# Пример:
#   bash scripts/run_pipeline.sh /app/data/samples/sat_001.png SAT_20240420_001
# ============================================================

set -euo pipefail

IMAGE_PATH="${1:-/app/data/samples/sat_001.png}"
IMAGE_ID="${2:-SAT_$(date -u +%Y%m%d_%H%M%S)}"
CONFIG="${3:-/app/config/settings.yaml}"

echo "=================================================="
echo "Запуск пайплайна:"
echo "  image:    ${IMAGE_PATH}"
echo "  image_id: ${IMAGE_ID}"
echo "  config:   ${CONFIG}"
echo "=================================================="

docker exec -i satlas-jupyter bash -lc "
    export CLASSPATH=\$(/opt/hadoop/bin/hadoop classpath --glob)
    /usr/local/spark/bin/spark-submit \
        --master spark://spark-master:7077 \
        --conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000 \
        --conf spark.executor.memory=2g \
        --conf spark.executor.cores=2 \
        --conf spark.pyspark.python=python3 \
        /app/src/main.py \
            --image '${IMAGE_PATH}' \
            --image-id '${IMAGE_ID}' \
            --config '${CONFIG}'
"
