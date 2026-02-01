#!/bin/bash

# 1. 配置：进入目录并指定密钥
cd /home/zhonghua/Filt_Event
export GIT_SSH_COMMAND="ssh -i ~/.ssh/github_ed25519"

echo "--- 正在备份 KM2A 磁单极子搜索代码 ---"

# 2. 检查是否有修改
if [ -z "$(git status --porcelain)" ]; then
    echo "☕ 代码没有变动，无需更新。"
    exit 0
fi

# 3. 执行提交
git add .
MSG=${1:-"Update: $(date +'%Y-%m-%d %H:%M')"}
git commit -m "$MSG"

# 4. 推送到远程
git push origin main

if [ $? -eq 0 ]; then
    echo "✅ 同步成功！代码已上云。"
else
    echo "❌ 同步失败，请检查网络。"
fi