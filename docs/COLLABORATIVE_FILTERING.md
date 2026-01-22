# 协同过滤算法实现

## 概述

本项目实现了两种经典的协同过滤推荐算法：
- **用户协同过滤（User-Based Collaborative Filtering, User-CF）**：u2u → u2i 工程拆分
- **物品协同过滤（Item-Based Collaborative Filtering, Item-CF）**：i2i 召回

两种算法都实现了 `recall.Source` 接口，可以无缝集成到推荐系统的 Pipeline 中。

## 工程抽象：u2i 和 i2i

### u2i（User → Item）

**工程含义**（不是算法名）：
- u2i 是工程抽象，表示"直接给用户算候选物品集合"
- u2i ≠ 算法，u2i = 召回方向

**常见 u2i 实现方式**：
| 方法 | 本质 |
|------|------|
| User-CF | 用户相似 → 物品 |
| MF | 用户隐向量 → 物品 |
| Embedding | User Emb → ANN |
| Rule | 用户规则 |

### i2i（Item → Item）

**核心理解**：
- "我看了这个，还可能看什么"
- 物品 → 物品的召回方向

**常见 i2i 实现**：
| 方法 | 说明 |
|------|------|
| Item-CF | 共现统计 |
| Embedding i2i | item 向量 |
| Graph i2i | 物品图 |

**工业价值**：
- 最稳健 Recall
- 实时性好
- 可缓存

## 算法原理

### 用户协同过滤（User-Based CF）

**核心思想**："兴趣相似的用户，喜欢相似的物品"

**算法流程**：
1. 用户 → 行为向量（点击/收藏/购买）
2. 计算用户相似度（Cosine / Pearson）
3. 找 TopK 相似用户
4. 推荐这些用户喜欢但目标用户未见过的物品

**工程特征**：
| 维度 | 情况 |
|------|------|
| 实时性 | 较差（用户变化快） |
| 计算复杂度 | 高（用户数大） |
| 可解释性 | 强 |
| 冷启动 | 差 |

**工程使用现状**：
- ❌ 几乎不直接在线用
- ✅ 离线分析 / 冷启动补充

**在 Reckit 中的位置**：
- 离线产出 u2u / u2i 结果
- 作为 Recall Node（u2u → u2i 工程拆分）
- Label: `recall.u2i`

### 物品协同过滤（Item-Based CF / i2i）

**核心思想**："被同一批用户喜欢的物品，相互相似"

**算法流程**：
1. 构建物品 → 用户倒排表
2. 计算物品相似度
3. 对用户历史行为物品，取相似物品集合

**工程特征**：
| 维度 | 情况 |
|------|------|
| 实时性 | 好 |
| 计算复杂度 | 可控 |
| 可解释性 | 强 |
| 稳定性 | 高 |

**工业地位**：
- 工业级召回的"常青树"
- 电商、内容流、短视频都在用
- 可直接线上使用

**在 Reckit 中的位置**：
- 核心 Recall Node（i2iRecall）
- Label: `recall.i2i`

**使用场景**：
- 输入：用户最近点击 items
- 输出：相似 items
- "我看了这个，还可能看什么"

## 核心组件

### 1. CFStore 接口

**位置**: `recall/collaborative_filtering.go`

**定义**:
```go
type CFStore interface {
    GetUserItems(ctx context.Context, userID int64) (map[int64]float64, error)
    GetItemUsers(ctx context.Context, itemID int64) (map[int64]float64, error)
    GetAllUsers(ctx context.Context) ([]int64, error)
    GetAllItems(ctx context.Context) ([]int64, error)
}
```

**职责**:
- 提供用户-物品交互数据的存储和检索
- 支持双向查询（用户→物品，物品→用户）
- 可扩展实现（内存、Redis、MySQL 等）

### 2. UserBasedCF（u2u → u2i）

**配置参数**:
- `Store`: CFStore 接口实现
- `TopKSimilarUsers`: 计算相似度时考虑的 TopK 个相似用户（默认 50）
- `TopKItems`: 最终返回的 TopK 个物品（默认 20）
- `SimilarityMetric`: 相似度度量方式，支持 `cosine`（余弦相似度）和 `pearson`（皮尔逊相关系数）
- `MinCommonItems`: 两个用户至少需要有多少个共同交互物品才计算相似度（默认 2）

**使用示例**:
```go
// 用户协同过滤（u2u → u2i）
userCF := &recall.UserBasedCF{
    Store:            cfStore,
    TopKSimilarUsers: 10,
    TopKItems:        5,
    SimilarityMetric:  "cosine",
    MinCommonItems:   2,
}

items, err := userCF.Recall(ctx, rctx)
// Label: recall.u2i
```

**注意**：由于实时性较差、计算复杂度高，建议离线计算 u2u 相似度，在线直接使用。

### 3. ItemBasedCF / I2IRecall（i2i）

**配置参数**:
- `Store`: CFStore 接口实现
- `TopKSimilarItems`: 计算相似度时考虑的 TopK 个相似物品（默认 100）
- `TopKItems`: 最终返回的 TopK 个物品（默认 20）
- `SimilarityMetric`: 相似度度量方式，支持 `cosine` 和 `pearson`
- `MinCommonUsers`: 两个物品至少需要有多少个共同交互用户才计算相似度（默认 2）
- `UserHistoryKey`: 从 RecommendContext 获取用户历史物品的 key（可选）

**使用示例**:
```go
// 方式1：使用 ItemBasedCF
itemCF := &recall.ItemBasedCF{
    Store:            cfStore,
    TopKSimilarItems: 10,
    TopKItems:        5,
    SimilarityMetric:  "cosine",
    MinCommonUsers:   2,
}

// 方式2：使用 I2IRecall 别名（更符合工业习惯）
i2i := &recall.I2IRecall{
    Store:            cfStore,
    TopKSimilarItems: 10,
    TopKItems:        5,
    SimilarityMetric:  "cosine",
    MinCommonUsers:   2,
}

items, err := i2i.Recall(ctx, rctx)
// Label: recall.i2i
```

**工业最佳实践**：
- 离线计算物品相似度矩阵，在线直接查表
- 支持缓存，实时性好
- 可直接线上使用

## 存储实现

### MemoryCFStore

**位置**: `recall/cf_store_adapter.go`

基于内存的实现，适用于测试和小规模数据。

**使用示例**:
```go
cfStore := recall.NewMemoryCFStore()
cfStore.AddInteraction(userID, itemID, score)
```

### StoreCFAdapter

**位置**: `recall/cf_store_adapter.go`

基于 `store.Store` 接口的适配器，可以从 Redis、MySQL 等存储中读取数据。

**存储格式**:
- 用户物品交互：`{KeyPrefix}:user:{userID}` → JSON map[itemID]score
- 物品用户交互：`{KeyPrefix}:item:{itemID}` → JSON map[userID]score
- 所有用户列表：`{KeyPrefix}:users` → JSON []int64
- 所有物品列表：`{KeyPrefix}:items` → JSON []int64

**使用示例**:
```go
adapter := recall.NewStoreCFAdapter(redisStore, "cf")
userCF := &recall.UserBasedCF{Store: adapter}
```

## 相似度计算

### 余弦相似度（Cosine Similarity）

**公式**: `cos(θ) = (A·B) / (||A|| * ||B||)`

**特点**:
- 范围: [-1, 1]，值越大越相似
- 适用于方向相似性判断
- 对向量长度不敏感

### 皮尔逊相关系数（Pearson Correlation）

**公式**: `r = Σ((Xi - X̄)(Yi - Ȳ)) / √(Σ(Xi - X̄)² * Σ(Yi - Ȳ)²)`

**特点**:
- 范围: [-1, 1]，值越大越相似
- 考虑了均值，对评分偏差不敏感
- 适用于评分数据

## 集成到 Pipeline

协同过滤召回源可以与其他召回源组合使用：

```go
pipeline := &pipeline.Pipeline{
    Nodes: []pipeline.Node{
        &recall.Fanout{
            Sources: []recall.Source{
                &recall.Hot{...},           // 热门召回
                &recall.UserHistory{...},   // 用户历史召回
                &recall.UserBasedCF{...},   // 用户协同过滤
                &recall.ItemBasedCF{...},   // 物品协同过滤
                &recall.ANN{...},           // Embedding 召回
            },
            Dedup:         true,
            MergeStrategy: "priority",
        },
        // 其他节点...
    },
}
```

## 输出结果

召回结果包含以下信息：

- **Item ID**: 物品标识
- **Score**: 推荐分数（基于相似度加权计算）
- **Labels**:
  - `recall_source`: 
    - UserBasedCF: `"u2i"` (u2u → u2i)
    - ItemBasedCF/I2IRecall: `"i2i"` (工业标准)
  - `cf_metric`: 使用的相似度度量方式（"cosine" 或 "pearson"）

## 性能优化建议

### 1. 离线计算相似度矩阵

对于物品协同过滤，可以离线计算物品相似度矩阵，在线直接查表：

```go
// 离线：计算所有物品对的相似度，存储到 Redis
// 在线：直接查询相似物品列表
```

### 2. 使用向量数据库

对于大规模数据，可以使用向量数据库（如 Faiss、Milvus）加速相似度计算。

### 3. 缓存策略

- 缓存用户相似度计算结果
- 缓存热门物品的相似物品列表
- 设置合理的缓存 TTL

### 4. 采样优化

- 对于用户协同过滤，可以只考虑活跃用户
- 对于物品协同过滤，可以只考虑热门物品

## 使用示例

完整示例请参考：`examples/collaborative_filtering/main.go`

运行示例：
```bash
go run ./examples/collaborative_filtering
```

## 注意事项

1. **冷启动问题**：
   - 新用户：没有历史数据，无法使用协同过滤
   - 新物品：没有交互数据，无法计算相似度
   - 解决方案：结合热门召回、内容召回等

2. **数据稀疏性**：
   - 用户-物品交互矩阵通常很稀疏
   - 需要设置合理的 `MinCommonItems` 和 `MinCommonUsers` 阈值

3. **可扩展性**：
   - 用户协同过滤：计算复杂度 O(用户数)
   - 物品协同过滤：计算复杂度 O(物品数)
   - 大规模场景建议使用物品协同过滤或离线计算

4. **实时性**：
   - 当前实现为在线计算，适合中小规模数据
   - 大规模场景建议离线计算相似度，在线查表

## 扩展方向

1. **矩阵分解**：实现 SVD、NMF 等矩阵分解算法
2. **深度学习**：实现 Neural Collaborative Filtering (NCF)
3. **混合推荐**：结合内容特征和协同过滤
4. **时间衰减**：考虑时间因素，给近期交互更高权重
