# 完整推荐系统示例

本示例展示了一个**完整的推荐系统 Pipeline**，涵盖：

- ✅ **多路召回**：UserHistory + I2I + Content + Hot
- ✅ **过滤策略**：黑名单 + 用户拉黑 + 已曝光
- ✅ **特征工程**：用户特征 + 物品特征 + 交叉特征
- ✅ **排序模型**：RPC XGBoost（或本地 LR）
- ✅ **重排策略**：多样性重排

## 功能说明

### 1. 召回策略

| 召回源 | 时间窗口 | TopK | 优先级 | 说明 |
|--------|----------|------|--------|------|
| **UserHistory** | 7 天 | 50 | 1（最高） | 基于用户最近点击历史 |
| **I2IRecall** | - | 30 | 2 | 基于物品相似度（协同过滤） |
| **ContentRecall** | - | 20 | 3 | 基于 category 匹配 |
| **Hot** | - | 20 | 4（最低） | 热门物品兜底 |

### 2. 过滤策略

- **黑名单过滤**：过滤全局黑名单物品
- **用户拉黑过滤**：过滤用户拉黑的物品
- **已曝光过滤**：过滤最近 7 天内已曝光的物品

### 3. 特征工程

**用户特征**（`user_*` 前缀）：
- `user_age`: 用户年龄
- `user_gender`: 用户性别（male=1, female=2, unknown=0）
- `user_region`: 用户地区（beijing=1, shanghai=2, ...）
- `user_interest_*`: 用户兴趣偏好（从 `UserProfile.Interests` 提取）

**物品特征**（`item_*` 前缀）：
- `item_category`: 物品类别（tech=1, game=2, ...）
- `item_price`: 物品价格
- `item_ctr`: 物品点击率
- `item_cvr`: 物品转化率

**交叉特征**（`cross_*` 前缀）：
- `cross_age_x_category`: 年龄 × 类别
- `cross_gender_x_price`: 性别 × 价格
- `cross_region_x_category`: 地区 × 类别

### 4. 排序模型

- **RPC XGBoost**（推荐）：通过 HTTP 调用 Python XGBoost 服务
- **本地 LR**（备用）：如果 RPC 服务不可用，使用本地 LR 模型

### 5. 重排策略

- **多样性重排**：按 `category` 保证结果多样性，TopK=20

## 运行方式

### 方式 1: 直接运行（使用本地 LR）

```bash
cd examples/full_recommendation_system
go run main.go
```

### 方式 2: 使用 RPC XGBoost 服务

1. **启动 XGBoost 服务**：

```bash
cd python
python train/train_xgb.py  # 训练模型
uvicorn service.server:app --host 0.0.0.0 --port 8080  # 启动服务
```

2. **修改代码**：确保 `main.go` 中的 RPC 地址正确：

```go
rpcModel := model.NewRPCModel(
    "xgboost",
    "http://localhost:8080/predict", // 确保地址正确
    5*time.Second,
)
```

3. **运行示例**：

```bash
go run main.go
```

## 数据流说明

### 1. 召回阶段

```
用户请求
  ↓
Fanout（多路并发召回）
  ├─ UserHistory (点击历史，7天) → [item_1, item_2, item_3]
  ├─ I2IRecall (物品相似度) → [item_10, item_20, ...]
  ├─ ContentRecall (category匹配) → [item_5, item_6, ...]
  └─ Hot (热门) → [item_10, item_11, ...]
  ↓
合并去重（优先级策略）
  → [item_1, item_2, item_3, item_10, item_20, ...]
```

### 2. 过滤阶段

```
召回结果
  ↓
FilterNode
  ├─ 黑名单过滤 → 移除 item_99, item_98
  ├─ 用户拉黑过滤 → 移除用户拉黑的物品
  └─ 已曝光过滤 → 移除最近 7 天已曝光的物品
  ↓
过滤后结果
  → [item_1, item_2, item_3, ...]
```

### 3. 特征注入阶段

```
过滤后结果
  ↓
enrichItemFeatures（物品基础特征）
  → 注入 item_ctr, item_cvr, category, price
  ↓
EnrichNode（用户特征 + 交叉特征）
  ├─ 用户特征：user_age, user_gender, user_region
  ├─ 物品特征：item_category, item_price, item_ctr, item_cvr
  └─ 交叉特征：cross_age_x_category, cross_gender_x_price, ...
  ↓
完整特征向量
  → {user_age: 25.0, item_ctr: 0.15, cross_age_x_category: 25.0, ...}
```

### 4. 排序阶段

```
特征向量
  ↓
RPCNode（XGBoost 预测）
  → 批量预测，得到 score
  ↓
排序结果（按 score 降序）
  → [item_2(0.85), item_1(0.82), item_3(0.78), ...]
```

### 5. 重排阶段

```
排序结果
  ↓
Diversity（多样性重排）
  → 按 category 保证多样性，TopK=20
  ↓
最终推荐结果
  → [item_2, item_5, item_1, item_6, ...]
```

## 存储格式

### Redis 存储示例

```redis
# 用户历史（按行为类型）
user:history:user_123:click -> ["item_1", "item_2", "item_3"]
user:history:user_123:view  -> ["item_4", "item_5"]
user:history:user_123:like  -> ["item_6"]

# 协同过滤
cf:user:user_123 -> {"item_1": 1.0, "item_2": 2.0, ...}
cf:item:item_1   -> {"user_123": 1.0, "user_456": 1.0, ...}

# 内容特征
content:item:item_1 -> {"category": 1.0, "price": 99.0, "ctr": 0.15}
content:user:user_123 -> {"tech": 0.8, "game": 0.6, ...}

# 热门物品
hot:feed -> ZSet{item_10: 100.0, item_11: 95.0, ...}

# 曝光历史
user:exposed:user_123 -> [{"item_id": "item_10", "timestamp": 1705996800}, ...]

# 黑名单
blacklist:items -> ["item_99", "item_98"]

# 特征
user:features:user_123 -> {"user_age": 25.0, "user_gender": 1.0, ...}
item:features:item_1   -> {"item_category": 1.0, "item_price": 99.0, ...}
```

## 行为数据时间窗口

| 行为类型 | 时间窗口 | 说明 |
|----------|----------|------|
| **浏览 (view)** | 1-3 天 | 时效性强，用户兴趣变化快 |
| **点击 (click)** | 7-30 天 | 明确兴趣，可保留更久 |
| **点赞 (like)** | 30-90 天 | 强烈偏好，长期有效 |

本示例中：
- **UserHistory (click)**: 7 天
- **曝光过滤**: 7 天

## 扩展建议

### 1. 添加更多召回源

```go
fanout.Sources = append(fanout.Sources,
    // Embedding 召回
    &recall.EmbRecall{
        Store:  vectorAdapter,
        TopK:   30,
        Metric: "cosine",
    },
    // Word2Vec 召回
    &recall.Word2VecRecall{
        Model:  word2vecModel,
        Store:  word2vecStore,
        TopK:   20,
        Mode:   "text",
    },
)
```

### 2. 添加实时特征

```go
enrichNode.FeatureService = featureService // 已支持实时特征
// 实时特征从 realtime:features:{userID}:{itemID} 获取
```

### 3. 使用 DeepFM 排序

```go
rpcModel := model.NewRPCModel(
    "deepfm",
    "http://deepfm-service:8080/predict",
    10*time.Second,
)
```

### 4. 添加自定义过滤

```go
filterNode.Filters = append(filterNode.Filters,
    &CustomFilter{
        // 例如：过滤已购买、过滤下架物品等
    },
)
```

## 注意事项

1. **特征一致性**：训练和在线特征必须一致（特征名、特征值、预处理方式）
2. **行为数据写入**：实际应用中需要将用户行为写入 Store（本示例使用 Mock 数据）
3. **曝光记录**：推荐结果返回后，需要记录曝光（代码中有注释示例）
4. **RPC 服务**：如果使用 RPC 排序，确保 Python 服务已启动且地址正确

## 相关文档

- [推荐系统整体方案设计](../../docs/RECOMMENDATION_SYSTEM_DESIGN.md)
- [召回算法文档](../../docs/RECALL_ALGORITHMS.md)
- [排序模型文档](../../docs/RANK_MODELS.md)
- [特征一致性](../../docs/FEATURE_CONSISTENCY.md)
