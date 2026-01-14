# Reckit 示例代码

本目录包含 Reckit 的各种使用示例。

## 示例列表

### 1. basic - 基础示例

演示如何使用 Pipeline 构建一个完整的推荐系统，包括：
- Store 的使用（MemoryStore）
- 多路召回（Fanout）
- 特征注入
- LR 模型排序
- 多样性重排

运行：
```bash
go run ./examples/basic
```

### 2. config - 配置化 Pipeline

演示如何从 YAML 配置文件加载 Pipeline，无需修改代码即可调整推荐策略。

运行：
```bash
go run ./examples/config
```

**注意**：从项目根目录运行，配置文件路径为 `examples/config/pipeline.example.yaml`

### 3. dsl - Label DSL 表达式

演示如何使用 Label DSL 解释器进行策略表达式判断，包括：
- 字符串比较和包含
- 数值比较
- 逻辑运算符
- 存在性检查

运行：
```bash
go run ./examples/dsl
```

### 4. personalization - 千人千面个性化推荐

演示如何使用特征注入节点实现千人千面推荐，包括：
- 用户特征提取
- 物品特征注入
- 交叉特征生成
- 个性化排序

运行：
```bash
go run ./examples/personalization
```

### 5. rpc_xgb - Python XGBoost 模型调用

演示如何调用 Python 训练的 XGBoost 模型进行排序，实现端到端闭环。

**前置条件**：
1. 训练模型：`cd python && python train/train_xgb.py`
2. 启动服务：`cd python && uvicorn service.server:app --host 0.0.0.0 --port 8080`

运行：
```bash
# 在另一个终端
go run ./examples/rpc_xgb
```

## 如何运行所有示例

```bash
# 基础示例
go run ./examples/basic

# 配置化 Pipeline
go run ./examples/config

# DSL 表达式
go run ./examples/dsl

# 千人千面推荐
go run ./examples/personalization

# Python XGBoost 模型调用（需要先启动 Python 服务）
go run ./examples/rpc_xgb
```
