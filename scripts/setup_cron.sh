#!/bin/bash
#
# Stocka 定时任务配置脚本
#
# 配置每日16:30运行routine并发送通知
#

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 默认配置
RUN_TIME="16:30"
WORK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_CMD="${WORK_DIR}/venv/bin/python"
LOG_FILE="${WORK_DIR}/logs/cron.log"

# 帮助信息
show_help() {
    cat << EOF
Stocka 定时任务配置脚本

用法: $0 [选项]

选项:
    -t, --time TIME         设置运行时间 (默认: 16:30)
    -w, --work-dir DIR      设置工作目录 (默认: ${WORK_DIR})
    -p, --python PATH       Python解释器路径 (默认: ${PYTHON_CMD})
    -r, --remove            移除定时任务
    -s, --status            查看当前定时任务状态
    -h, --help              显示帮助信息

示例:
    $0                      # 使用默认配置
    $0 -t 17:00             # 设置为17:00运行
    $0 --remove             # 移除定时任务

EOF
}

# 检查参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--time)
            RUN_TIME="$2"
            shift 2
            ;;
        -w|--work-dir)
            WORK_DIR="$2"
            shift 2
            ;;
        -p|--python)
            PYTHON_CMD="$2"
            shift 2
            ;;
        -r|--remove)
            ACTION="remove"
            shift
            ;;
        -s|--status)
            ACTION="status"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}未知选项: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# 默认值
ACTION="${ACTION:-setup}"

# 解析时间
HOUR=$(echo "$RUN_TIME" | cut -d: -f1)
MINUTE=$(echo "$RUN_TIME" | cut -d: -f2)

# 验证时间格式
if ! [[ "$HOUR" =~ ^[0-9]+$ ]] || ! [[ "$MINUTE" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}错误: 时间格式不正确，应为 HH:MM${NC}"
    exit 1
fi

if [ "$HOUR" -lt 0 ] || [ "$HOUR" -gt 23 ] || [ "$MINUTE" -lt 0 ] || [ "$MINUTE" -gt 59 ]; then
    echo -e "${RED}错误: 时间范围不正确${NC}"
    exit 1
fi

# Cron命令
CRON_CMD="${MINUTE} ${HOUR} * * 1-5 cd ${WORK_DIR} && ${PYTHON_CMD} -m quant_framework.scheduler.trading_scheduler >> ${LOG_FILE} 2>&1"
CRON_COMMENT="# Stocka 每日选股任务"

# 检查当前定时任务
show_status() {
    echo -e "${YELLOW}当前用户的定时任务:${NC}"
    crontab -l 2>/dev/null | grep -E "(Stocka|quant_framework)" || echo "未找到 Stocka 相关任务"
    echo ""
    echo -e "${YELLOW}完整 crontab:${NC}"
    crontab -l 2>/dev/null || echo "当前用户没有定时任务"
}

# 移除定时任务
remove_cron() {
    echo -e "${YELLOW}正在移除 Stocka 定时任务...${NC}"

    # 获取当前crontab（去掉Stocka相关任务）
    crontab -l 2>/dev/null | grep -v "quant_framework" | grep -v "Stocka" | crontab -

    echo -e "${GREEN}✓ 定时任务已移除${NC}"
}

# 设置定时任务
setup_cron() {
    echo -e "${YELLOW}正在配置 Stocka 定时任务...${NC}"
    echo ""

    # 检查工作目录
    if [ ! -d "$WORK_DIR" ]; then
        echo -e "${RED}错误: 工作目录不存在: ${WORK_DIR}${NC}"
        exit 1
    fi
    echo -e "工作目录: ${GREEN}${WORK_DIR}${NC}"

    # 检查Python
    if [ ! -f "$PYTHON_CMD" ]; then
        # 尝试使用系统Python
        if command -v python3 &> /dev/null; then
            PYTHON_CMD="python3"
            echo -e "Python: ${GREEN}${PYTHON_CMD}${NC} (使用系统Python)"
        else
            echo -e "${RED}错误: 未找到Python解释器${NC}"
            exit 1
        fi
    else
        echo -e "Python: ${GREEN}${PYTHON_CMD}${NC}"
    fi

    # 检查调度器模块
    SCHEDULER_MODULE="${WORK_DIR}/quant_framework/scheduler/trading_scheduler.py"
    if [ ! -f "$SCHEDULER_MODULE" ]; then
        echo -e "${RED}错误: 调度器模块不存在: ${SCHEDULER_MODULE}${NC}"
        exit 1
    fi

    # 创建日志目录
    mkdir -p "$(dirname "$LOG_FILE")"
    echo -e "日志文件: ${GREEN}${LOG_FILE}${NC}"

    # 测试调度器
    echo ""
    echo -e "${YELLOW}测试调度器...${NC}"
    if ! cd "$WORK_DIR" && $PYTHON_CMD -c "from quant_framework.scheduler import TradingScheduler; print('导入成功')" 2>/dev/null; then
        echo -e "${RED}错误: 无法导入 TradingScheduler，请检查环境${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ 调度器模块测试通过${NC}"

    # 移除旧任务
    echo ""
    echo -e "${YELLOW}移除旧任务（如果有）...${NC}"
    crontab -l 2>/dev/null | grep -v "quant_framework" | grep -v "Stocka" > /tmp/crontab.tmp

    # 添加新任务
    echo "" >> /tmp/crontab.tmp
    echo "$CRON_COMMENT" >> /tmp/crontab.tmp
    echo "$CRON_CMD" >> /tmp/crontab.tmp

    # 安装新crontab
    crontab /tmp/crontab.tmp
    rm /tmp/crontab.tmp

    echo ""
    echo -e "${GREEN}✓ 定时任务配置成功！${NC}"
    echo ""
    echo -e "${YELLOW}任务详情:${NC}"
    echo "  运行时间: 周一至周五 ${HOUR}:${MINUTE}"
    echo "  工作日:   周一至周五"
    echo "  工作目录: ${WORK_DIR}"
    echo "  执行命令: ${PYTHON_CMD} -m quant_framework.scheduler.trading_scheduler"
    echo "  日志文件: ${LOG_FILE}"
    echo ""
    echo -e "${YELLOW}当前定时任务:${NC}"
    crontab -l | grep -A1 "Stocka"
    echo ""
    echo -e "${YELLOW}提示:${NC}"
    echo "  - 使用 'crontab -l' 查看所有定时任务"
    echo "  - 使用 'tail -f ${LOG_FILE}' 查看运行日志"
    echo "  - 使用 '$0 --remove' 移除定时任务"
    echo ""
    echo -e "${GREEN}配置完成！系统将在交易日自动运行并推送选股结果。${NC}"
}

# 主逻辑
case $ACTION in
    status)
        show_status
        ;;
    remove)
        remove_cron
        ;;
    setup)
        setup_cron
        ;;
esac
