#!/bin/bash
# 训练流水线主脚本
# 用法: ./run_training.sh [--skip-eval] [--skip-register]

set -e

# 配置
VERSION=$(date +%Y%m%d)
MODEL_DIR="/app/model"
S3_BUCKET="${S3_BUCKET:-reckit-models}"
SKIP_EVAL=false
SKIP_REGISTER=false

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        --skip-register)
            SKIP_REGISTER=true
            shift
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "开始训练流水线: $VERSION"
echo "=========================================="

# 1. 样本生成（可选，若数据已就绪可跳过）
if [ -n "$GENERATE_DATA_SCRIPT" ] && [ -f "$GENERATE_DATA_SCRIPT" ]; then
    echo "[1/5] 生成训练数据..."
    python "$GENERATE_DATA_SCRIPT" --dt "$VERSION" || {
        echo "⚠️  数据生成失败，使用现有数据"
    }
else
    echo "[1/5] 跳过数据生成（使用现有数据）"
fi

# 2. 训练 XGBoost
echo "[2/5] 训练 XGBoost 模型..."
cd /app
python train/train_xgb.py --version "$VERSION" || {
    echo "❌ XGBoost 训练失败"
    exit 1
}

# 3. 训练 DeepFM
echo "[3/5] 训练 DeepFM 模型..."
python train/train_deepfm.py --version "$VERSION" || {
    echo "❌ DeepFM 训练失败"
    exit 1
}

# 4. 评估（可选）
if [ "$SKIP_EVAL" = false ]; then
    echo "[4/5] 评估模型..."
    CURRENT_VERSION=$(cat "$MODEL_DIR/.current_version" 2>/dev/null || echo "")
    if [ -z "$CURRENT_VERSION" ]; then
        echo "⚠️  未找到当前版本，跳过评估门控"
    else
        python scripts/evaluate.py --version "$VERSION" --current-version "$CURRENT_VERSION" || {
            echo "❌ 评估未通过，不发布"
            exit 1
        }
    fi
else
    echo "[4/5] 跳过评估"
fi

# 5. 注册模型（可选）
if [ "$SKIP_REGISTER" = false ]; then
    echo "[5/5] 注册模型..."
    python scripts/register_model.py --version "$VERSION" || {
        echo "⚠️  模型注册失败，但训练已完成"
    }
    
    # 更新当前版本
    echo "$VERSION" > "$MODEL_DIR/.current_version"
    echo "✅ 当前版本已更新: $VERSION"
else
    echo "[5/5] 跳过模型注册"
fi

echo "=========================================="
echo "✅ 训练流水线完成: $VERSION"
echo "=========================================="
