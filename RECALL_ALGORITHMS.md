# Reckit 召回算法支持情况

## 算法支持概览

| 算法 | Reckit Node | 阶段 | 支持状态 | 实现文件 | Label |
|------|-------------|------|----------|----------|-------|
| User-CF | U2IRecall | Recall | ✅ 已支持 | `recall/collaborative_filtering.go` | `recall.u2i` |
| Item-CF | I2IRecall | Recall | ✅ 已支持 | `recall/collaborative_filtering.go` | `recall.i2i` |
| MF / ALS | MFRecall | Recall | ✅ 已支持 | `recall/matrix_factorization.go` | `recall.mf` |
| Embedding | EmbRecall | Recall | ✅ 已支持 | `recall/ann.go` | `recall.emb` |
| Content | ContentRecall | Recall | ✅ 已支持 | `recall/content.go` | `recall.content` |

**所有算法均已实现并支持！** ✅

## 详细说明

### 1. User-CF → U2IRecall ✅

**实现类**: `UserBasedCF` / `U2IRecall`

**核心思想**: "兴趣相似的用户，喜欢相似的物品"

**使用示例**:
```go
u2i := &recall.U2IRecall{
    Store:            cfStore,
    TopKSimilarUsers: 10,
    TopKItems:        5,
    SimilarityMetric: "cosine",
}
```

**Label**: `recall.u2i`

### 2. Item-CF → I2IRecall ✅

**实现类**: `ItemBasedCF` / `I2IRecall`

**核心思想**: "被同一批用户喜欢的物品，相互相似"

**使用示例**:
```go
i2i := &recall.I2IRecall{
    Store:            cfStore,
    TopKSimilarItems: 10,
    TopKItems:        5,
    SimilarityMetric: "cosine",
}
```

**Label**: `recall.i2i`

### 3. MF / ALS → MFRecall ✅

**实现类**: `MFRecall`

**核心思想**: 将用户-物品交互矩阵分解为用户隐向量和物品隐向量，预测分数 = 用户隐向量 · 物品隐向量

**算法类型**:
- MF (Matrix Factorization): 基础矩阵分解
- ALS (Alternating Least Squares): 交替最小二乘法
- SVD (Singular Value Decomposition): 奇异值分解

**使用示例**:
```go
mfRecall := &recall.MFRecall{
    Store: mfStore,
    TopK:  20,
    UserVectorKey: "user_vector", // 从 Context 获取
}
```

**存储适配器**:
```go
mfStore := recall.NewStoreMFAdapter(memStore, "mf")
```

**Label**: `recall.mf`

### 4. Embedding → EmbRecall ✅

**实现类**: `ANN` / `EmbRecall`

**核心思想**: 基于 Embedding 向量的近似最近邻检索

**使用示例**:
```go
embRecall := &recall.EmbRecall{
    Store:      vectorStore,
    TopK:       20,
    Metric:     "cosine",
}
```

**Label**: `recall.emb`

### 5. Content → ContentRecall ✅

**实现类**: `ContentRecall`

**核心思想**: "用户喜欢具有某些特征的物品，推荐具有相似特征的其他物品"

**使用示例**:
```go
contentRecall := &recall.ContentRecall{
    Store:            contentStore,
    TopK:             20,
    SimilarityMetric: "cosine", // cosine / jaccard
    UserPreferencesKey: "user_preferences",
}
```

**存储适配器**:
```go
contentStore := recall.NewStoreContentAdapter(memStore, "content")
```

**Label**: `recall.content`

## 存储接口

### CFStore（协同过滤）
- `GetUserItems`: 获取用户交互物品
- `GetItemUsers`: 获取物品交互用户
- `GetAllUsers`: 获取所有用户
- `GetAllItems`: 获取所有物品

### MFStore（矩阵分解）
- `GetUserVector`: 获取用户隐向量
- `GetItemVector`: 获取物品隐向量
- `GetAllItemVectors`: 获取所有物品隐向量

### VectorStore（Embedding）
- `GetVector`: 获取单个物品向量
- `ListVectors`: 获取所有物品向量

### ContentStore（内容推荐）
- `GetItemFeatures`: 获取物品特征
- `GetUserPreferences`: 获取用户偏好
- `GetAllItems`: 获取所有物品

## 集成到 Pipeline

所有召回算法都可以集成到 Pipeline 中：

```go
pipeline := &pipeline.Pipeline{
    Nodes: []pipeline.Node{
        &recall.Fanout{
            Sources: []recall.Source{
                &recall.U2IRecall{...},      // User-CF
                &recall.I2IRecall{...},      // Item-CF
                &recall.MFRecall{...},       // MF/ALS
                &recall.EmbRecall{...},      // Embedding
                &recall.ContentRecall{...},   // Content
            },
            MergeStrategy: "priority",
        },
        // 其他节点...
    },
}
```

## 工程特征对比

| 算法 | 实时性 | 计算复杂度 | 可解释性 | 冷启动 | 工业使用 |
|------|--------|------------|----------|--------|----------|
| User-CF | 较差 | 高 | 强 | 差 | 离线分析 |
| Item-CF | 好 | 可控 | 强 | 中等 | 工业常青树 |
| MF/ALS | 好 | 低 | 中等 | 中等 | 广泛使用 |
| Embedding | 好 | 中等 | 弱 | 好 | 主流方案 |
| Content | 好 | 中等 | 强 | 好 | 冷启动首选 |
