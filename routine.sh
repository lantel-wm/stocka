#!/bin/bash

# 默认配置参数
DB_PATH="${DB_PATH:-data/stock.db}"
DATE="${DATE:-}"
FACTOR_NAME="${FACTOR_NAME:-alpha158}"
MODEL_PATH="${MODEL_PATH:-ckpt/lightgbm_model_2005_2021.pkl}"
TOP_N="${TOP_N:-50}"
LOG_DIR="${LOG_DIR:-logs}"
CACHE_DIR="${CACHE_DIR:-signals}"
SKIP_CACHE="${SKIP_CACHE:-false}"

# 生成带日期时间的日志文件名
LOG_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export STOCKA_LOG_FILE="${LOG_DIR}/routine_${LOG_TIMESTAMP}.log"

# 支持通过命令行参数覆盖
while [[ $# -gt 0 ]]; do
    case $1 in
        --db-path)
            DB_PATH="$2"
            shift 2
            ;;
        --date)
            DATE="$2"
            shift 2
            ;;
        --factor-name)
            FACTOR_NAME="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --top-n)
            TOP_N="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --cache-dir)
            CACHE_DIR="$2"
            shift 2
            ;;
        --skip-cache)
            SKIP_CACHE="true"
            shift
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 获取模型名称（不含路径和扩展名）
MODEL_FILENAME=$(basename "$MODEL_PATH")
MODEL_NAME="${MODEL_FILENAME%.*}"

# 确定预测日期（用于缓存文件名）
if [ -n "$DATE" ]; then
    # 将 YYYY-MM-DD 转换为 YYYYMMDD
    PREDICT_DATE=$(echo "$DATE" | tr -d '-')
else
    # 获取最新交易日
    PREDICT_DATE=$(python -c "from quant_framework import DataHandler; dh = DataHandler('$DB_PATH'); print(dh.get_latest_trade_date().strftime('%Y%m%d'))" 2>/dev/null || echo "")
fi

# 构建缓存文件路径
if [ -n "$PREDICT_DATE" ]; then
    CACHE_FILENAME="${PREDICT_DATE}_${MODEL_NAME}_top${TOP_N}.csv"
    CACHE_FILE="${CACHE_DIR}/${CACHE_FILENAME}"
else
    echo "警告: 无法确定预测日期，缓存已禁用"
    CACHE_FILE=""
fi

# 检查缓存（如果未禁用缓存且缓存文件路径有效）
if [ "$SKIP_CACHE" != "true" ] && [ -n "$CACHE_FILE" ] && [ -f "$CACHE_FILE" ]; then
    echo "===== 命中预测缓存 ====="
    echo "日期: $PREDICT_DATE"
    echo "模型: $MODEL_NAME"
    echo "TOP-N: $TOP_N"
    echo ""
    echo "预测结果："
    cat "$CACHE_FILE"
    echo ""
    echo "===== 缓存文件: $CACHE_FILE ====="
    exit 0
fi

# 确保目录存在
mkdir -p "$LOG_DIR"
mkdir -p "$CACHE_DIR"

echo "===== 启动 routine 任务 ====="
echo "数据库: $DB_PATH"
echo "日期: $DATE"
echo "因子: $FACTOR_NAME"
echo "模型: $MODEL_PATH"
echo "TOP-N: $TOP_N"
echo "日志文件: $STOCKA_LOG_FILE"
echo ""

# 构建 date 参数（如果 DATE 非空）
DATE_ARG=""
if [ -n "$DATE" ]; then
    DATE_ARG="--date $DATE"
fi

# 更新数据
python -m quant_framework.cli data update --db-path "$DB_PATH" $DATE_ARG

# 计算因子
python -m quant_framework.cli factor calculate --db-path "$DB_PATH" --factor-name "$FACTOR_NAME"

# 策略预测（输出会自动保存到缓存文件）
if [ -n "$CACHE_FILE" ]; then
    python -m quant_framework.cli strategy predict --db-path "$DB_PATH" --model-path "$MODEL_PATH" $DATE_ARG --top-n "$TOP_N" --output "$CACHE_FILE"

    # 输出预测结果
    if [ -f "$CACHE_FILE" ]; then
        echo ""
        echo "===== 预测结果 ====="
        cat "$CACHE_FILE"
    fi
else
    python -m quant_framework.cli strategy predict --db-path "$DB_PATH" --model-path "$MODEL_PATH" $DATE_ARG --top-n "$TOP_N"
fi

echo ""
echo "===== routine 任务完成 ====="