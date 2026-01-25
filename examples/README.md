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

### 5. feature_version - 特征版本管理

演示如何实现特征版本管理，包括：
- 多版本特征存储（v1, v2）
- 版本化特征服务
- 灰度发布（流量分配）
- 版本降级策略
- 版本元数据管理

运行：
```bash
go run ./examples/feature_version
```

**特性**：
- 支持在存储 key 中包含版本号（如 `user:features:v2:42`）
- 根据用户 ID 哈希自动选择版本（灰度发布）
- 新版本失败时自动降级到旧版本
- 版本元数据管理（版本信息、特征列表、状态）

### 6. rpc_xgb - Python XGBoost 模型调用

演示如何调用 Python 训练的 XGBoost 模型进行排序，实现端到端闭环。

**前置条件**：
1. 训练模型：`cd python && python train/train_xgb.py`
2. 启动服务：`cd python && uvicorn service.server:app --host 0.0.0.0 --port 8080`

运行：
```bash
# 在另一个终端
go run ./examples/rpc_xgb
```

### 7. deepfm - Python DeepFM 模型调用

演示如何调用 PyTorch 训练的 DeepFM 模型进行排序。

**前置条件**：
1. 训练模型：`cd python && python train/train_deepfm.py --version v1.0.0`
2. 启动服务：`cd python && uvicorn service.deepfm_server:app --host 0.0.0.0 --port 8080`

运行：
```bash
# 在另一个终端
go run ./examples/deepfm
```

**DeepFM 特点**：
- 自动学习特征交互（FM 二阶 + DNN 高阶）
- 适合 CTR 预估、广告打分场景
- 无需手动构造交叉特征

### 8. full_recommendation_system - 完整推荐系统示例 ⭐

**最完整的推荐系统示例**，展示工业级推荐系统的完整流程：

- ✅ **多路召回**：UserHistory（7天点击历史）+ I2I（协同过滤）+ Content（category匹配）+ Hot（热门兜底）
- ✅ **过滤策略**：黑名单 + 用户拉黑 + 已曝光（7天窗口）
- ✅ **特征工程**：用户特征（age, gender, region）+ 物品特征（category, price, ctr, cvr）+ 交叉特征
- ✅ **排序模型**：RPC XGBoost（或本地 LR 备用）
- ✅ **重排策略**：多样性重排（按 category）

**行为数据时间窗口**：
- 浏览 (view): 1-3 天
- 点击 (click): 7-30 天（本示例使用 7 天）
- 点赞 (like): 30-90 天
- 曝光过滤: 7 天

运行：
```bash
# 方式 1: 直接运行（使用本地 LR）
go run ./examples/full_recommendation_system

# 方式 2: 使用 RPC XGBoost（需要先启动 Python 服务）
cd python
python train/train_xgb.py
uvicorn service.server:app --host 0.0.0.0 --port 8080
# 在另一个终端
go run ./examples/full_recommendation_system
```

**详细说明**：见 `examples/full_recommendation_system/README.md`

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

# 特征版本管理
go run ./examples/feature_version

# Python XGBoost 模型调用（需要先启动 Python 服务）
go run ./examples/rpc_xgb

# Python DeepFM 模型调用（需要先启动 Python 服务）
go run ./examples/deepfm

# 完整推荐系统示例（推荐！）
go run ./examples/full_recommendation_system
```
