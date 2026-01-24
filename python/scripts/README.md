# 训练自动化脚本

本目录包含训练流程自动化的脚本，实现「最小可落地方案」。

## 脚本说明

### 1. run_training.sh

训练流水线主脚本，串联数据生成、训练、评估、注册等步骤。

**用法**:
```bash
./run_training.sh [--skip-eval] [--skip-register]
```

**参数**:
- `--skip-eval`: 跳过评估门控
- `--skip-register`: 跳过模型注册

**环境变量**:
- `S3_BUCKET`: S3 桶名（可选）
- `MODEL_DIR`: 模型目录（默认 `/app/model`）
- `GENERATE_DATA_SCRIPT`: 数据生成脚本路径（可选）

**示例**:
```bash
# 完整流程
./run_training.sh

# 跳过评估（用于实验）
./run_training.sh --skip-eval

# 仅训练，不注册
./run_training.sh --skip-register
```

---

### 2. evaluate.py

模型评估与门控脚本，判断新模型是否优于当前线上模型。

**用法**:
```bash
python scripts/evaluate.py --version <新版本> --current-version <当前版本> [--model-dir <模型目录>]
```

**门控规则**:
- AUC 至少不劣化 1% (`new_auc >= current_auc * 0.99`)
- LogLoss 不增加 1% (`new_logloss <= current_logloss * 1.01`)

**示例**:
```bash
python scripts/evaluate.py \
  --version 20250123 \
  --current-version 20250122 \
  --model-dir /app/model
```

**输出**:
- 通过门控: 退出码 0
- 未通过门控: 退出码 1，打印失败原因

---

### 3. register_model.py

模型注册脚本，将训练好的模型打包上传到对象存储，并在本地记录版本信息。

**用法**:
```bash
python scripts/register_model.py --version <版本> [--model-dir <模型目录>] [--s3-bucket <S3桶名>]
```

**功能**:
1. 打包模型文件（`xgb_model.json`、`feature_meta.json`、`deepfm_model.pt` 等）
2. 上传到 S3/OSS（如果配置了 `--s3-bucket`）
3. 写入本地注册中心（`model/registry.json`）

**示例**:
```bash
# 仅本地注册
python scripts/register_model.py --version 20250123

# 注册并上传到 S3
python scripts/register_model.py \
  --version 20250123 \
  --s3-bucket reckit-models
```

**输出文件**:
- `model/model-{version}.tar.gz`: 模型压缩包
- `model/registry.json`: 本地注册中心（JSON 格式）

---

### 4. deploy.py

模型部署脚本，从对象存储拉取指定版本的模型，更新到推理服务，并触发 reload。

**用法**:
```bash
python scripts/deploy.py --version <版本> [--model-dir <模型目录>] [--service-url <服务URL>] [--s3-bucket <S3桶名>]
```

**功能**:
1. 从 S3 拉取模型（如果配置了）
2. 解压模型到 `model/` 目录
3. 更新 `.current_version` 文件
4. 触发推理服务 `/reload` 端点（如果配置了 `--service-url`）

**示例**:
```bash
# 从 S3 拉取并部署
python scripts/deploy.py \
  --version 20250123 \
  --service-url http://reckit-inference:8080 \
  --s3-bucket reckit-models

# 仅本地部署（不触发 reload）
python scripts/deploy.py --version 20250123
```

**注意**:
- 若未配置 `--service-url`，需要手动触发 reload 或重启服务
- 手动触发: `curl -X POST http://localhost:8080/reload`

---

## 完整流程示例

### 1. 训练流水线（CronJob / 手动）

```bash
# 执行完整训练流程
cd /app
./scripts/run_training.sh
```

流程：
1. 数据生成（可选）
2. 训练 XGBoost
3. 训练 DeepFM
4. 评估门控
5. 模型注册

### 2. 手动部署（若评估通过）

```bash
# 部署新模型
python scripts/deploy.py \
  --version 20250123 \
  --service-url http://reckit-inference:8080 \
  --s3-bucket reckit-models
```

### 3. 验证部署

```bash
# 检查服务状态
curl http://localhost:8080/health

# 检查模型版本
curl http://localhost:8080/ | jq .model_version

# 测试预测
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"features_list": [{"item_ctr": 0.15, ...}]}'
```

---

## 依赖

### Python 包

- `requests`: HTTP 请求（用于触发 reload）
- `boto3`: AWS S3 客户端（可选，用于 S3 上传/下载）

安装：
```bash
pip install requests boto3
```

### 环境变量（可选）

- `AWS_ACCESS_KEY_ID`: AWS 访问密钥
- `AWS_SECRET_ACCESS_KEY`: AWS 密钥
- `AWS_REGION`: AWS 区域（默认 `us-east-1`）

---

## 文件结构

```
python/
├── scripts/
│   ├── run_training.sh      # 训练流水线主脚本
│   ├── evaluate.py          # 评估门控
│   ├── register_model.py    # 模型注册
│   ├── deploy.py            # 模型部署
│   └── README.md           # 本文件
├── model/
│   ├── .current_version    # 当前版本（文本文件）
│   ├── registry.json       # 模型注册中心（JSON）
│   ├── model-{version}.tar.gz  # 模型压缩包
│   └── ...
└── ...
```

---

## 故障排查

### 1. 训练失败

```bash
# 查看训练日志
kubectl logs <training-pod>

# 检查数据文件
ls -lh /app/data/

# 检查模型目录权限
ls -ld /app/model
```

### 2. 评估失败

```bash
# 检查模型文件是否存在
ls -lh /app/model/*.json

# 检查版本文件
cat /app/model/.current_version

# 手动运行评估
python scripts/evaluate.py --version 20250123 --current-version 20250122
```

### 3. 部署失败

```bash
# 检查 S3 连接（如果使用）
aws s3 ls s3://reckit-models/

# 检查服务 URL
curl http://reckit-inference:8080/health

# 手动触发 reload
curl -X POST http://reckit-inference:8080/reload
```

---

## 与 K8s 集成

详见 `k8s/README.md`。
