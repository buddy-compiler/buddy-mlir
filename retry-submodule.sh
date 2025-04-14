#!/bin/bash

# 配置参数
max_retries=15          # 最大重试次数
retry_delay=3          # 初始重试间隔（秒）
enable_backoff=true    # 启用指数退避（每次延迟翻倍）

# 递归更新子模块（强制模式）
retry_count=0
while [[ $retry_count -le $max_retries ]]; do
  echo "▶ 尝试更新子模块 ($((retry_count+1))/$((max_retries+1)))"
  
  # 执行命令
  if git submodule update --init --force --recursive; then
    echo "✅ 更新成功！"
    exit 0
  else
    exit_code=$?
    
    # 计算下次等待时间
    if [[ "$enable_backoff" == "true" ]]; then
      delay=$(( retry_delay * 2 ** retry_count ))
    else
      delay=$retry_delay
    fi

    # 判断是否继续重试
    if [[ $retry_count -lt $max_retries ]]; then
      echo "⏳ 将在 $delay 秒后重试 (退出码: $exit_code)..."
      sleep $delay
      retry_count=$((retry_count+1))
    else
      echo "❌ 已达最大重试次数 ($max_retries 次)"
      exit $exit_code
    fi
  fi
done
