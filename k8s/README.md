# Kubernetes 部署配置

本目录包含 Reckit 训练和推理服务的 Kubernetes 配置。

## 文件说明

- **training-cronjob.yaml**: 训练任务 CronJob，每日凌晨 2 点自动执行
- **inference-deployment.yaml**: 推理服务 Deployment，支持模型热加载（reload）

## 前置要求

1. **PersistentVolumeClaim (PVC)**:
   - `training-data-pvc`: 训练数据存储
   - `model-pvc`: 模型文件存储（训练和推理共享）

2. **Docker 镜像**:
   - `reckit-training:latest`: 训练镜像（包含训练脚本和依赖）
   - `reckit-inference:latest`: 推理镜像（包含推理服务）

## 部署步骤

### 1. 创建 PVC

```bash
# 创建训练数据 PVC
kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: training-data-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 10Gi
EOF

# 创建模型 PVC
kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 5Gi
EOF
```

### 2. 部署训练 CronJob

```bash
kubectl apply -f training-cronjob.yaml
```

查看 CronJob 状态：

```bash
kubectl get cronjob reckit-training
kubectl get jobs -l job-name=reckit-training-*
```

### 3. 部署推理服务

```bash
kubectl apply -f inference-deployment.yaml
```

查看服务状态：

```bash
kubectl get deployment reckit-inference
kubectl get pods -l app=reckit-inference
kubectl get service reckit-inference
```

### 4. 测试推理服务

```bash
# 获取服务地址
SERVICE_IP=$(kubectl get service reckit-inference -o jsonpath='{.spec.clusterIP}')

# 健康检查
curl http://$SERVICE_IP:8080/health

# 预测
curl -X POST http://$SERVICE_IP:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features_list": [
      {
        "item_ctr": 0.15,
        "item_cvr": 0.08,
        "item_price": 99.0,
        "user_age": 25.0,
        "user_gender": 1.0,
        "cross_age_x_ctr": 3.75,
        "cross_gender_x_price": 99.0
      }
    ]
  }'
```

## 模型热加载（Reload）

推理服务支持通过 `/reload` 端点热加载新模型，无需重启服务。

### 方式 1: 通过 API 触发

```bash
# 获取 Pod 名称
POD_NAME=$(kubectl get pods -l app=reckit-inference -o jsonpath='{.items[0].metadata.name}')

# 端口转发
kubectl port-forward $POD_NAME 8080:8080

# 触发 reload
curl -X POST http://localhost:8080/reload
```

### 方式 2: 使用部署脚本

```bash
# 在训练 Pod 中执行
kubectl exec -it <training-pod> -- python scripts/deploy.py \
  --version 20250123 \
  --service-url http://reckit-inference:8080
```

## 手动触发训练

```bash
# 创建一次性 Job
kubectl create job --from=cronjob/reckit-training manual-training-$(date +%s)
```

## 查看日志

```bash
# 训练任务日志
kubectl logs -l job-name=reckit-training-* --tail=100

# 推理服务日志
kubectl logs -l app=reckit-inference --tail=100 -f
```

## 配置说明

### 训练 CronJob

- **调度**: 每日凌晨 2 点 (`0 2 * * *`)
- **资源**: 2-4Gi 内存，1-2 CPU
- **环境变量**:
  - `S3_BUCKET`: S3 桶名（可选）
  - `MODEL_DIR`: 模型目录

### 推理服务

- **副本数**: 3
- **资源**: 1-2Gi 内存，0.5-1 CPU
- **健康检查**: `/health` 端点
- **模型热加载**: `/reload` 端点
- **环境变量**:
  - `MODEL_VERSION`: 从 ConfigMap 读取

## 故障排查

1. **训练任务失败**:
   ```bash
   kubectl describe job <job-name>
   kubectl logs <pod-name>
   ```

2. **推理服务无法启动**:
   ```bash
   kubectl describe pod <pod-name>
   kubectl logs <pod-name>
   ```

3. **模型文件不存在**:
   检查 PVC 挂载和模型文件路径

4. **Reload 失败**:
   检查模型文件权限和服务日志
