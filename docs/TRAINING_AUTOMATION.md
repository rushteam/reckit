# 工业级训练流程自动化指南

本文档说明如何构建**工业级自动化流水线**，涵盖数据准备、训练编排、模型版本管理、评估门控、发布部署、监控反馈等全流程。

---

## 一、整体架构：从手动到自动化

工业级训练自动化将「数据 → 训练 → 评估 → 发布 → 监控」串联成一条可调度、可复现、可回滚的流水线。

```
[数据产出] → [样本构建] → [训练] → [评估] → [模型注册] → [发布/回滚] → [监控告警]
     ↑                                                                        |
     └──────────────────────── 反馈 / 重训触发 ────────────────────────────────┘
```

### 当前工程现状

当前工程已具备以下基础：

- **训练脚本**：`python/train/train_xgb.py`、`train_deepfm.py`、`train_item2vec.py`
- **模型输出**：`python/model/` 目录，支持版本化（`--version` 参数）
- **推理服务**：`python/service/server.py`、`deepfm_server.py`，Docker 化部署
- **特征配置**：`python/train/features.py`，定义 `FEATURE_COLUMNS`
- **示例数据**：`python/data/` 目录，支持自动生成

**自动化目标**：在上述基础上，增加调度编排、数据流水线、模型注册、评估门控、自动发布、监控反馈等环节。

---

## 二、自动化流程各环节

### 2.1 触发方式

| 方式 | 适用场景 | 常见实现 | 推荐场景 |
|------|----------|----------|----------|
| **定时触发** | 日更/周更模型（CTR、排序、召回等） | Cron、Airflow DAG、Kubernetes CronJob | XGBoost、DeepFM 日级重训 |
| **事件触发** | 数据就绪后立刻训练 | 数据湖/数仓触发器 → 消息队列 → 训练任务 | 新数据分区就绪后自动训练 |
| **手动触发** | 实验、紧急重训、回滚 | CI 手动跑、Airflow 手动触发、运维平台按钮 | 实验性模型、紧急修复 |
| **监控驱动** | 指标退化到阈值自动重训 | 监控系统 → 告警 → 触发训练流水线 | 线上指标劣化自动重训 |

**当前工程推荐**：
- **起步阶段**：定时触发（Cron / K8s CronJob），每日凌晨执行
- **进阶阶段**：事件触发（数据就绪后自动训练）+ 监控驱动（指标劣化自动重训）

---

### 2.2 数据流水线

工业级场景中，训练数据不是静态的 `python/data/train_data.csv`，而是：

- **数据源**：日志/数仓（点击、曝光、转化等行为表）
- **特征平台**：Feature Store（用户、物品、上下文特征，离线 + 在线一致）
- **样本组装**：按日期、场景拉取特征 + 标签，生成 `train_data.csv`、`behavior.csv`、`corpus.txt`

#### 自动化要点

1. **分区产出**：按 `dt=yyyy-mm-dd` 产出日表，训练任务只消费「已就绪」分区
2. **特征一致性**：样本列、特征名、预处理与 `train/features.py`、`data/README.md` 约定一致，避免线上线下不一致
3. **可复现性**：数据版本（分区、表版本）、随机种子固定，便于复现和对比实验

#### 实现方式

在训练流水线中增加「样本生成」步骤：

```python
# 示例：样本生成脚本（可集成到 Airflow DAG）
def generate_training_data(dt: str, output_path: str):
    """
    从数仓/特征平台拉取数据，生成训练样本
    
    Args:
        dt: 数据日期，如 '2025-01-23'
        output_path: 输出路径，如 'data/train_data.csv'
    """
    # 1. 从 Hive/Spark 读取行为日志（dt 分区）
    # 2. 关联特征平台获取用户/物品特征
    # 3. 组装样本（特征列与 FEATURE_COLUMNS 对齐）
    # 4. 写入 output_path
    pass
```

训练脚本（`train_xgb.py`、`train_deepfm.py`、`train_item2vec.py`）保持不变，只需从流水线产出的数据路径读取即可。

---

### 2.3 训练编排

**推荐做法**：使用 **DAG 编排器** 将「数据准备 → 多模型训练 → 评估 → 注册」串联，而非单机裸跑 Cron。

| 方案 | 特点 | 适用场景 |
|------|------|----------|
| **Airflow** | DAG 可视化、重试、依赖、告警成熟 | 已有数据团队、多用 Python |
| **Kubeflow Pipelines** | 跑在 K8s，适合 GPU 训练、分布式 | 深度模型、大规模训练 |
| **MLflow Projects + 自建调度** | 模型管理强，调度需自己接 | 已有 MLflow，想轻量级 |
| **GitHub Actions / GitLab CI** | 与代码绑定，跑训练、构建镜像 | 小团队、模型更新不频繁 |
| **K8s CronJob** | 无 DAG，简单定时跑容器 | 单模型、日更、无复杂 DAG |

#### 当前工程推荐方案

**方案 A：K8s CronJob（简单版）**

```yaml
# k8s/training-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: reckit-training
spec:
  schedule: "0 2 * * *"  # 每日凌晨 2 点
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: training
            image: reckit-training:latest
            command:
            - /bin/sh
            - -c
            - |
              cd /app
              python train/train_xgb.py --version $(date +%Y%m%d)
              python train/train_deepfm.py --version $(date +%Y%m%d)
              # 上传模型到 S3/OSS
              aws s3 cp model/ s3://reckit-models/$(date +%Y%m%d)/ --recursive
          volumes:
          - name: data
            persistentVolumeClaim:
              claimName: training-data-pvc
          - name: model
            persistentVolumeClaim:
              claimName: model-pvc
```

**方案 B：Airflow DAG（进阶版）**

```python
# airflow/dags/reckit_training.py
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'reckit_training',
    default_args=default_args,
    description='Reckit 模型训练流水线',
    schedule_interval='0 2 * * *',  # 每日凌晨 2 点
    catchup=False,
)

# Task 1: 样本生成（可选，若数据已就绪可跳过）
generate_data = BashOperator(
    task_id='generate_training_data',
    bash_command='python scripts/generate_data.py --dt {{ ds }}',
    dag=dag,
)

# Task 2: XGBoost 训练
train_xgb = BashOperator(
    task_id='train_xgb',
    bash_command='cd /app && python train/train_xgb.py --version {{ ds_nodash }}',
    dag=dag,
)

# Task 3: DeepFM 训练
train_deepfm = BashOperator(
    task_id='train_deepfm',
    bash_command='cd /app && python train/train_deepfm.py --version {{ ds_nodash }}',
    dag=dag,
)

# Task 4: 评估
evaluate = BashOperator(
    task_id='evaluate_models',
    bash_command='python scripts/evaluate.py --version {{ ds_nodash }}',
    dag=dag,
)

# Task 5: 模型注册（仅当评估通过）
register = BashOperator(
    task_id='register_models',
    bash_command='python scripts/register_model.py --version {{ ds_nodash }}',
    dag=dag,
)

# 依赖关系
generate_data >> [train_xgb, train_deepfm] >> evaluate >> register
```

---

### 2.4 模型版本与注册

工业级不会直接覆盖 `python/model/xgb_model.json`，而是：

- **版本化**：每次训练产出带版本，如 `v1.0.0`、`train_20250123`，存到对象存储（S3/OSS）或模型库
- **注册中心**：使用 **MLflow Model Registry**、**自建 DB + 存储** 等记录：版本、指标、训练时间、数据分区、触发人
- **可回滚**：线上用「当前版本」指针，出问题可快速切回上一版本

#### 实现方式

当前工程的训练脚本已支持 `--version` 参数，可在此基础上增加：

1. **版本号生成**：流水线中自动生成版本号（日期、Git SHA、递增序号等）
2. **模型上传**：训练完成后，将 `model/` 目录打包上传到对象存储
3. **注册记录**：在注册中心（DB/MLflow）记录版本信息

```python
# 示例：模型注册脚本
import boto3
import json
from datetime import datetime

def register_model(version: str, model_dir: str, metrics: dict):
    """
    注册模型到对象存储和注册中心
    
    Args:
        version: 模型版本，如 '20250123'
        model_dir: 模型目录，如 'model/'
        metrics: 评估指标，如 {'auc': 0.85, 'logloss': 0.32}
    """
    # 1. 打包模型
    import tarfile
    tar_path = f'model-{version}.tar.gz'
    with tarfile.open(tar_path, 'w:gz') as tar:
        tar.add(model_dir, arcname='.')
    
    # 2. 上传到 S3/OSS
    s3 = boto3.client('s3')
    s3.upload_file(tar_path, 'reckit-models', f'{version}/model.tar.gz')
    
    # 3. 写入注册中心（DB/MLflow）
    registry_record = {
        'version': version,
        'metrics': metrics,
        'train_time': datetime.now().isoformat(),
        'data_partition': version,  # 假设版本号即数据分区
        'status': 'staging',  # staging / production / archived
    }
    # 写入 DB 或 MLflow
    # save_to_registry(registry_record)
```

---

### 2.5 评估与上线门控

训练完成后不能直接替换线上，需要 **自动评估 + 门控**：

1. **离线指标**：AUC、GAUC、LogLoss 等（当前工程训练脚本已有验证集评估）
2. **规则门控**：
   - 离线指标优于当前线上版本才允许推进发布
   - 或「至少不劣化且通过回归测试」
3. **在线 A/B**：新模型先灰度 5% 流量，看 CTR/CVR 等，再全量

#### 实现方式

在训练流水线中增加「评估门控」步骤：

```python
# 示例：评估门控脚本
import json
import subprocess

def evaluate_and_gate(new_version: str, current_version: str) -> bool:
    """
    评估新模型并判断是否通过门控
    
    Returns:
        True: 通过门控，可以发布
        False: 未通过门控，不发布
    """
    # 1. 加载新模型和当前线上模型
    # 2. 在相同验证集上评估
    new_metrics = evaluate_model(new_version)
    current_metrics = evaluate_model(current_version)
    
    # 3. 门控规则
    if new_metrics['auc'] < current_metrics['auc'] * 0.99:  # 至少不劣化 1%
        print(f"❌ 门控失败: AUC 劣化 ({new_metrics['auc']} < {current_metrics['auc']})")
        return False
    
    if new_metrics['logloss'] > current_metrics['logloss'] * 1.01:  # LogLoss 不增加 1%
        print(f"❌ 门控失败: LogLoss 增加 ({new_metrics['logloss']} > {current_metrics['logloss']})")
        return False
    
    print(f"✅ 门控通过: AUC={new_metrics['auc']}, LogLoss={new_metrics['logloss']}")
    return True
```

在 Airflow DAG 中，只有评估通过才执行「模型注册」和「发布」任务。

---

### 2.6 发布与运行

#### 发布方式

1. **模型文件更新**（推荐）：
   - 从对象存储拉取指定版本到推理服务的 `model/` 目录
   - 推理服务支持热加载（无需重启）或滚动重启容器

2. **镜像重建**：
   - 将新模型打包进 Docker 镜像
   - 滚动更新推理服务容器

#### 当前工程实现

当前工程的推理服务（`service/server.py`、`deepfm_server.py`）从 `model/` 目录加载模型。发布流程：

```bash
# 1. 从对象存储拉取模型
aws s3 cp s3://reckit-models/20250123/model.tar.gz /tmp/
tar -xzf /tmp/model.tar.gz -C /app/model/

# 2. 更新推理服务（方式 A：热加载，若支持）
curl -X POST http://inference-service/reload

# 2. 更新推理服务（方式 B：滚动重启）
kubectl rollout restart deployment/reckit-inference
```

#### 灰度发布

```yaml
# k8s/inference-deployment.yaml（支持灰度）
apiVersion: apps/v1
kind: Deployment
metadata:
  name: reckit-inference
spec:
  replicas: 10
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2
      maxUnavailable: 1
  template:
    spec:
      containers:
      - name: inference
        image: reckit-inference:latest
        volumeMounts:
        - name: model
          mountPath: /app/model
        env:
        - name: MODEL_VERSION
          value: "20250123"  # 从 ConfigMap 读取
```

---

### 2.7 监控与反馈

#### 推理监控

- **性能指标**：QPS、延迟、错误率、超时
- **业务指标**：CTR、CVR、停留时长等，按模型版本/分桶统计
- **告警**：指标异常时触发告警

#### 数据闭环

监控指标进入数仓/数据湖，作为下一轮 **样本** 和 **评估** 的输入。若指标退化超过阈值，可 **自动告警** 乃至 **触发重训流水线**。

```python
# 示例：监控驱动重训
def check_model_health():
    """检查模型健康度，必要时触发重训"""
    metrics = get_realtime_metrics(model_version='20250123')
    
    if metrics['ctr'] < baseline_ctr * 0.95:  # CTR 下降 5%
        print("⚠️ 模型指标劣化，触发重训")
        trigger_training_pipeline()
```

---

## 三、最小可落地方案

在不改动当前工程训练脚本的前提下，可先实现「最小自动化闭环」：

### 3.1 目录结构

```
reckit/
├── python/
│   ├── train/          # 训练脚本（保持不变）
│   ├── service/        # 推理服务（保持不变）
│   ├── model/          # 模型输出（保持不变）
│   └── scripts/        # 新增：自动化脚本
│       ├── generate_data.py      # 样本生成（可选）
│       ├── evaluate.py           # 评估门控
│       ├── register_model.py     # 模型注册
│       └── deploy.py              # 发布部署
├── k8s/                # 新增：K8s 配置
│   ├── training-cronjob.yaml     # 训练 CronJob
│   └── inference-deployment.yaml  # 推理服务部署
└── airflow/            # 新增：Airflow DAG（可选）
    └── dags/
        └── reckit_training.py
```

### 3.2 单脚本流程（Cron / K8s CronJob）

```bash
#!/bin/bash
# scripts/run_training.sh

set -e

VERSION=$(date +%Y%m%d)
MODEL_DIR="/app/model"
S3_BUCKET="reckit-models"

echo "开始训练流水线: $VERSION"

# 1. 样本生成（若需要）
# python scripts/generate_data.py --dt $VERSION

# 2. 训练
cd /app
python train/train_xgb.py --version $VERSION
python train/train_deepfm.py --version $VERSION

# 3. 评估
python scripts/evaluate.py --version $VERSION --current-version $(cat $MODEL_DIR/.current_version)

# 4. 注册（仅当评估通过）
if [ $? -eq 0 ]; then
    python scripts/register_model.py --version $VERSION
    echo $VERSION > $MODEL_DIR/.current_version
    echo "✅ 训练完成: $VERSION"
else
    echo "❌ 评估未通过，不发布"
    exit 1
fi
```

### 3.3 评估脚本示例

```python
# scripts/evaluate.py
import argparse
import json
import sys

def evaluate_and_gate(new_version: str, current_version: str) -> bool:
    """评估并门控"""
    # 加载新模型指标（从训练日志或单独评估）
    with open(f'model/{new_version}_metrics.json') as f:
        new_metrics = json.load(f)
    
    # 加载当前模型指标
    with open(f'model/{current_version}_metrics.json') as f:
        current_metrics = json.load(f)
    
    # 门控规则
    if new_metrics['auc'] < current_metrics['auc'] * 0.99:
        print(f"❌ AUC 劣化: {new_metrics['auc']} < {current_metrics['auc']}")
        return False
    
    print(f"✅ 门控通过: AUC={new_metrics['auc']}")
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', required=True)
    parser.add_argument('--current-version', required=True)
    args = parser.parse_args()
    
    if not evaluate_and_gate(args.version, args.current_version):
        sys.exit(1)
```

---

## 四、进阶方案：Airflow 完整 DAG

### 4.1 DAG 结构

```
generate_data (可选)
    ↓
train_xgb ──┐
            ├──→ evaluate ──→ register ──→ deploy
train_deepfm ──┘
```

### 4.2 完整 DAG 代码

见「2.3 训练编排」章节的 Airflow 示例。

---

## 五、与当前工程的对应关系

| 环节 | 当前工程 | 自动化补充 |
|------|----------|------------|
| **数据** | `python/data/` 示例 + 脚本内 `generate_*` | 数仓/特征平台 → 产出 `train_data` / `behavior` / `corpus`，再给训练用 |
| **训练** | `train_xgb.py`、`train_deepfm.py`、`train_item2vec.py` | Airflow/K8s CronJob 定时或事件触发；读流水线产出的数据，写版本化 `model/` |
| **特征** | `train/features.py`、`FEATURE_COLUMNS` | 与 Feature Store / 样本生成逻辑对齐，保证线上线下一致 |
| **模型输出** | `model/*.json`、`*.pt`、`*_meta.json` 等 | 版本化 + 上传对象存储 + 模型注册 |
| **服务** | `Dockerfile`、`docker-compose`、`service/server.py` 等 | 从模型存储拉取指定版本 → 发布 → 健康检查、灰度、回滚 |
| **评估** | 训练脚本内验证集 | 单独 evaluate 脚本 + 门控逻辑，再决定是否发布 |

---

## 六、总结

工业级训练自动化的核心要素：

1. **触发**：定时 / 事件 / 监控驱动，而非长期手工执行
2. **数据**：可复现、版本化的样本与特征，与 `features.py`、`data/README.md` 一致
3. **编排**：Airflow / K8s 等将数据 → 训练 → 评估 → 注册 → 发布串联成 DAG
4. **版本与门控**：模型版本化、注册、评估通过再发布，支持回滚
5. **监控与闭环**：推理与业务指标监控，结果反哺样本与重训触发

当前工程的 `train_*.py`、`model/`、Docker 已具备「可被自动化调用」的基础。补齐 **调度**、**数据来源**、**版本与发布**、**评估门控** 后，即可演进为工业级训练自动化流程。

---

## 七、参考资源

- [Airflow 官方文档](https://airflow.apache.org/docs/)
- [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [Kubernetes CronJob](https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/)
