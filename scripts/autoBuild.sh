#!/bin/bash
###################################################################################
# 脚本名称: autoBuildForJenkins.sh
# 描述: 自动下载代码、更新代码并启动训练流程
# 用法: bash autoBuildForJenkins.sh
# 作者: DnaZen Team
###################################################################################

# 设置错误时退出
set -e

###################################################################################
# 1. 配置参数
###################################################################################
# 代码仓库地址
REPO_URL="git@github.com:oomics/dnazen.git"
# 本地代码目录 - 使用当前工作目录
CODE_DIR="$(pwd)"
# 主分支名称
MAIN_BRANCH="zeq"

###################################################################################
# 2. 下载或更新代码
###################################################################################
echo "===== 开始代码准备 ====="

# 检查git是否安装
if ! command -v git &> /dev/null; then
    echo "错误: git未安装，请先安装git"
    exit 1
fi

# 检查当前目录是否为git仓库
if [ -d ".git" ]; then
    echo "当前目录是git仓库，检查更新..."
    
    # 保存当前commit哈希值
    if git rev-parse HEAD &>/dev/null; then
        CURRENT_COMMIT=$(git rev-parse HEAD)
        
        # 获取远程最新代码信息
        git fetch origin
        
        # 获取远程最新commit哈希值
        if git rev-parse origin/$MAIN_BRANCH &>/dev/null; then
            LATEST_COMMIT=$(git rev-parse origin/$MAIN_BRANCH)
            
            # 比较当前commit和最新commit
            if [ "$CURRENT_COMMIT" = "$LATEST_COMMIT" ]; then
                echo "代码已是最新版本，无需更新"
            else
                echo "发现新版本，当前版本: ${CURRENT_COMMIT:0:8}，最新版本: ${LATEST_COMMIT:0:8}"
                echo "正在更新代码..."
                
                # 检查是否有未提交的更改
                if [ -n "$(git status --porcelain)" ]; then
                    echo "警告: 本地有未提交的更改，将被覆盖"
                    # 保存未提交的更改到stash
                    git stash
                    HAS_STASH=true
                fi
                
                # 更新代码
                git checkout $MAIN_BRANCH
                git pull origin $MAIN_BRANCH
                
                # 如果有stash，尝试恢复
                if [ "$HAS_STASH" = true ]; then
                    echo "尝试恢复本地更改..."
                    if git stash apply; then
                        echo "本地更改已恢复"
                    else
                        echo "恢复本地更改失败，请手动检查git stash"
                    fi
                fi
                
                echo "代码已更新到最新版本: $(git rev-parse HEAD | cut -c1-8)"
            fi
        else
            echo "警告: 无法获取远程分支 $MAIN_BRANCH 的信息"
        fi
    else
        echo "警告: 当前目录是git仓库但无法获取commit信息"
    fi
else
    echo "当前目录不是git仓库，初始化仓库..."
    
    # 备份当前目录中的文件
    TEMP_DIR=$(mktemp -d)
    echo "备份当前文件到: $TEMP_DIR"
    cp -r * "$TEMP_DIR/" 2>/dev/null || true
    
    # 初始化git仓库
    git init
    git remote add origin "$REPO_URL"
    
    # 拉取代码
    echo "拉取代码..."
    git fetch origin
    git checkout -b $MAIN_BRANCH
    git pull origin $MAIN_BRANCH
    
    # 恢复备份的文件
    echo "恢复备份文件..."
    cp -r "$TEMP_DIR"/* . 2>/dev/null || true
    
    echo "代码已初始化并更新"
fi

echo "===== 代码准备完成 ====="

###################################################################################
# 3. 准备数据和配置
###################################################################################
echo "===== 开始准备数据和配置 ====="

cd scripts

cp ~/dnazen/data/downloads/* ../data/downloads/
# 执行数据准备脚本
if [ -f "./run.sh" ]; then
    echo "执行数据准备脚本..."
    bash ./run.sh
    echo "数据准备完成"
else
    echo "警告: 未找到数据准备脚本 ./run.sh"
fi

echo "===== 数据和配置准备完成 ====="

###################################################################################
# 4. 启动训练流程
###################################################################################
echo "===== 开始启动训练流程 ====="

bash  ./prepare_data_and_config.sh --train-ngram    --experiment 4
#bash  ./prepare_data_and_config.sh --coverage-analysis   --experiment 4
bash ./prepare_data_and_config.sh --tokenize-train  
#bash ./prepare_data_and_config.sh --tokenize-dev  

bash  ./prepare_data_and_config.sh --prepare-dataset   --experiment 4
bash  ./prepare_data_and_config.sh --pretrain   --experiment 4


echo "===== 训练流程执行完成 ====="

echo "所有任务已完成"
exit 0
