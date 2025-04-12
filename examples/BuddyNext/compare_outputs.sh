#!/bin/bash

# 设置颜色输出
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color
YELLOW='\033[1;33m'
BLUE='\033[0;34m'

# 检查命令行参数
if [ $# -ne 2 ]; then
    echo -e "${YELLOW}Usage: $0 <command1> <command2>${NC}"
    echo "Example: $0 'next-reduce-sum-run' 'next-reduce-sum-vec-manual-run'"
    exit 1
fi

CMD1="$1"
CMD2="$2"

# 创建临时文件存储输出
OUTPUT1=$(mktemp)
OUTPUT2=$(mktemp)
PROCESSED1=$(mktemp)
PROCESSED2=$(mktemp)
TIME1=$(mktemp)
TIME2=$(mktemp)

echo "Running first command: make $CMD1"
make $CMD1 > "$OUTPUT1" 2>/dev/null

echo "Running second command: make $CMD2"
make $CMD2 > "$OUTPUT2" 2>/dev/null

# 使用grep提取Memref数据行，并处理输出（移除base@部分）
grep "data =" "$OUTPUT1" | sed 's/base@ = [^[:space:]]*/base@ = <addr>/g' > "$PROCESSED1"
grep "data =" "$OUTPUT2" | sed 's/base@ = [^[:space:]]*/base@ = <addr>/g' > "$PROCESSED2"

# 提取时间数据（包括科学计数法格式）
grep -o '[0-9]\+\.[0-9]\+e[-+]\?[0-9]\+' "$OUTPUT1" > "$TIME1"
grep -o '[0-9]\+\.[0-9]\+e[-+]\?[0-9]\+' "$OUTPUT2" > "$TIME2"

# 比较数据输出
echo -e "\n${BLUE}Comparing output data:${NC}"
if diff "$PROCESSED1" "$PROCESSED2" > /dev/null; then
    echo -e "${GREEN}✓ Outputs match! Both versions produce the same results.${NC}"
else
    echo -e "${RED}✗ Outputs differ! Found differences:${NC}"
    echo "----------------------------------------"
    diff "$PROCESSED1" "$PROCESSED2"
    echo "----------------------------------------"
fi

# 比较执行时间
echo -e "\n${BLUE}Comparing execution times:${NC}"
TIME1_VAL=$(cat "$TIME1")
TIME2_VAL=$(cat "$TIME2")

if [ -n "$TIME1_VAL" ] && [ -n "$TIME2_VAL" ]; then
    # 将科学计数法转换为小数
    TIME1_SEC=$(printf "%.9f" $TIME1_VAL)
    TIME2_SEC=$(printf "%.9f" $TIME2_VAL)
    
    echo "First version ($CMD1): ${TIME1_SEC} seconds"
    echo "Second version ($CMD2): ${TIME2_SEC} seconds"
    
    # 计算加速比
    if [ $(echo "$TIME1_SEC > 0" | bc -l) -eq 1 ]; then
        SPEEDUP=$(echo "scale=2; $TIME1_SEC/$TIME2_SEC" | bc -l)
        if [ $(echo "$SPEEDUP > 1" | bc -l) -eq 1 ]; then
            echo -e "${GREEN}Second version is ${SPEEDUP}x faster!${NC}"
        else
            SLOWDOWN=$(echo "scale=2; 1/$SPEEDUP" | bc -l)
            echo -e "${RED}Second version is ${SLOWDOWN}x slower.${NC}"
        fi
    fi
else
    echo -e "${RED}Could not extract timing information from one or both outputs${NC}"
fi

# 清理临时文件
rm "$OUTPUT1" "$OUTPUT2" "$PROCESSED1" "$PROCESSED2" "$TIME1" "$TIME2" 