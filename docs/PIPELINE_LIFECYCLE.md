# Pipeline 生命周期分析

本文档分析 Pipeline 在 Web 服务中的初始化与运行逻辑，区分常驻部分和请求生命周期部分。

---

## 核心概念

### 常驻部分（初始化一次，服务启动时创建）

**特点**：
- 不涉及请求特定的数据
- 不涉及请求特定的配置
- 可以在服务启动时初始化，多个请求共享
- 通常是接口、服务连接、模型权重等

### 请求生命周期部分（每次请求创建）

**特点**：
- 包含请求特定的上下文信息
- 包含请求特定的数据
- 每次请求都需要重新创建或加载
- 通常是用户信息、实时特征、召回结果等

---

## Pipeline 结构分析

### Pipeline 结构体

```go
type Pipeline struct {
    Nodes []Node      // 常驻：Node 实例列表
    Hooks []PipelineHook  // 常驻：Hook 实例列表
}
```

**生命周期**：✅ **常驻**
- `Nodes` 和 `Hooks` 在服务启动时初始化一次
- 所有请求共享同一个 Pipeline 实例

---

## Node 依赖分析

### 1. 召回节点（Recall）

#### `recall.Hot`

```go
type Hot struct {
    Store core.Store  // ✅ 常驻：存储连接
    Key   string      // ✅ 常驻：配置（存储 key）
    IDs   []string    // ✅ 常驻：fallback 数据
}
```

**生命周期**：
- ✅ **常驻**：`Store`、`Key`、`IDs` 都是常驻的
- 🔄 **请求时**：从 Store 读取热门物品列表（数据加载）

#### `recall.Fanout`

```go
type Fanout struct {
    Sources []Source          // ✅ 常驻：召回源列表
    Dedup   bool              // ✅ 常驻：配置
    MergeStrategy MergeStrategy  // ✅ 常驻：合并策略
    ErrorHandler ErrorHandler    // ✅ 常驻：错误处理策略
    Timeout time.Duration     // ✅ 常驻：超时配置
}
```

**生命周期**：
- ✅ **常驻**：所有字段都是常驻的
- 🔄 **请求时**：调用各个 Source 的 `Recall` 方法（数据加载）

#### `recall.U2IRecall` / `recall.I2IRecall`

```go
type UserBasedCF struct {
    Store core.Store              // ✅ 常驻：存储连接
    SimilarityCalculator SimilarityCalculator  // ✅ 常驻：相似度计算器
    Config RecallConfig          // ✅ 常驻：配置接口
    TopKSimilarUsers int         // ✅ 常驻：配置
    TopKItems int                // ✅ 常驻：配置
}
```

**生命周期**：
- ✅ **常驻**：所有字段都是常驻的
- 🔄 **请求时**：从 Store 读取用户/物品交互数据，计算相似度（数据加载）

#### `recall.ANN`

```go
type ANN struct {
    Store VectorStore            // ✅ 常驻：向量存储连接
    TopK int                     // ✅ 常驻：配置
    Metric string                // ✅ 常驻：配置
    UserVector []float64         // ❌ 请求时：用户向量（每次请求不同）
    UserVectorExtractor func(...) // ✅ 常驻：提取器函数（逻辑）
}
```

**生命周期**：
- ✅ **常驻**：`Store`、`TopK`、`Metric`、`UserVectorExtractor` 是常驻的
- 🔄 **请求时**：`UserVector` 是请求特定的，需要从 `RecommendContext` 提取

---

### 2. 过滤节点（Filter）

#### `filter.FilterNode`

```go
type FilterNode struct {
    Filters []Filter  // ✅ 常驻：过滤器列表
}
```

**生命周期**：
- ✅ **常驻**：`Filters` 列表是常驻的
- 🔄 **请求时**：调用各个 Filter 的 `ShouldFilter` 方法（数据加载）

#### `filter.BlacklistFilter`

```go
type BlacklistFilter struct {
    Store core.Store  // ✅ 常驻：存储连接
    Key   string      // ✅ 常驻：配置
}
```

**生命周期**：
- ✅ **常驻**：`Store` 和 `Key` 是常驻的
- 🔄 **请求时**：从 Store 读取黑名单数据（数据加载）

---

### 3. 特征注入节点（Feature Enrichment）

#### `feature.EnrichNode`

```go
type EnrichNode struct {
    FeatureService FeatureService  // ✅ 常驻：特征服务连接
    UserFeatureExtractor func(...)  // ✅ 常驻：提取器函数（逻辑）
    ItemFeatureExtractor func(...)  // ✅ 常驻：提取器函数（逻辑）
    UserFeaturePrefix string        // ✅ 常驻：配置
    ItemFeaturePrefix string        // ✅ 常驻：配置
}
```

**生命周期**：
- ✅ **常驻**：所有字段都是常驻的
- 🔄 **请求时**：从 `FeatureService` 加载用户特征、物品特征（数据加载）

---

### 4. 排序节点（Rank）

#### `rank.LRNode`

```go
type LRNode struct {
    Model RankModel      // ✅ 常驻：模型权重
    SortStrategy SortStrategy  // ✅ 常驻：排序策略
}
```

**生命周期**：
- ✅ **常驻**：`Model`（包含权重）和 `SortStrategy` 是常驻的
- 🔄 **请求时**：使用模型对 items 进行预测和排序（计算）

#### `rank.RPCNode`

```go
type RPCNode struct {
    Model RankModel  // ✅ 常驻：RPC 模型客户端（连接）
}
```

**生命周期**：
- ✅ **常驻**：`Model`（RPC 客户端）是常驻的
- 🔄 **请求时**：调用远程模型服务进行预测（数据加载）

---

### 5. 重排节点（ReRank）

#### `rerank.Diversity`

```go
type Diversity struct {
    LabelKey string  // ✅ 常驻：配置
}
```

**生命周期**：
- ✅ **常驻**：`LabelKey` 是常驻的
- 🔄 **请求时**：基于 items 的 Label 进行多样性重排（计算）

---

## 请求生命周期部分

### RecommendContext

```go
type RecommendContext struct {
    UserID   string           // 🔄 请求时：用户 ID
    DeviceID string           // 🔄 请求时：设备 ID
    Scene    string           // 🔄 请求时：场景
    User     *UserProfile     // 🔄 请求时：用户画像
    UserProfile map[string]any  // 🔄 请求时：用户画像（map 形式）
    Labels   map[string]Label  // 🔄 请求时：用户标签
    Realtime map[string]any   // 🔄 请求时：实时特征
    Params   map[string]any   // 🔄 请求时：请求参数
}
```

**生命周期**：🔄 **请求时**
- 每次请求都需要创建新的 `RecommendContext`
- 包含请求特定的用户信息、实时特征等

---

## Web 服务使用模式

### 初始化阶段（服务启动时）

```go
// 1. 初始化常驻资源
store := store.NewRedisStore("localhost:6379", 0)
featureService := feature.NewFeatureServiceFactory().CreateFull(store, ...)
mlService := service.NewTorchServeClient(...)

// 2. 创建常驻的 Node 实例
hotRecall := &recall.Hot{
    Store: store,  // 常驻
    Key:   "hot:feed",  // 常驻配置
}

enrichNode := &feature.EnrichNode{
    FeatureService: featureService,  // 常驻
    UserFeaturePrefix: "user_",  // 常驻配置
}

rankNode := &rank.RPCNode{
    Model: model.NewRPCModel("pytorch", "http://...", ...),  // 常驻
}

// 3. 构建常驻的 Pipeline
pipeline := &pipeline.Pipeline{
    Nodes: []pipeline.Node{
        &recall.Fanout{Sources: []recall.Source{hotRecall}},
        enrichNode,
        rankNode,
    },
    Hooks: []pipeline.PipelineHook{
        &LoggingHook{},  // 常驻
    },
}
```

### 请求处理阶段（每次请求）

```go
func HandleRecommendRequest(w http.ResponseWriter, r *http.Request) {
    // 1. 创建请求上下文（请求生命周期）
    rctx := &core.RecommendContext{
        UserID: getUserID(r),
        Scene:  getScene(r),
        UserProfile: loadUserProfile(getUserID(r)),  // 从存储加载
        Realtime: getRealtimeFeatures(r),  // 实时特征
    }
    
    // 2. 执行 Pipeline（使用常驻的 pipeline 实例）
    items, err := pipeline.Run(r.Context(), rctx, nil)
    
    // 3. 返回结果
    // ...
}
```

---

## 数据加载时机

### 常驻数据（初始化时加载）

- ❌ **不推荐**：在初始化时加载所有数据到内存
- ✅ **推荐**：初始化连接和配置，数据按需加载

### 请求时数据加载

以下数据在请求时从存储加载：

1. **用户画像**：从 `UserProfile` 存储加载
2. **热门物品列表**：从 `Store` 读取（`recall.Hot`）
3. **用户/物品交互数据**：从 `Store` 读取（协同过滤）
4. **特征数据**：从 `FeatureService` 加载（用户特征、物品特征）
5. **黑名单数据**：从 `Store` 读取（过滤器）
6. **向量数据**：从 `VectorStore` 加载（ANN 召回）

---

## 最佳实践

### 1. 初始化阶段

**重要原则**：初始化阶段**不包含用户参数**，只包含常驻资源和配置。

```go
// ✅ 正确：初始化常驻资源（不包含用户参数）
func InitPipeline() *pipeline.Pipeline {
    // 初始化存储连接（常驻）
    store := store.NewRedisStore("localhost:6379", 0)
    
    // 初始化特征服务（常驻）
    featureService := feature.NewFeatureServiceFactory().CreateFull(store, ...)
    
    // 初始化模型客户端（常驻）
    mlService := service.NewTorchServeClient(...)
    
    // 创建 Node 实例（常驻，不包含用户特定配置）
    nodes := []pipeline.Node{
        &recall.Hot{Store: store, Key: "hot:feed"},  // Key 是场景配置，不是用户参数
        &feature.EnrichNode{FeatureService: featureService},
        &rank.RPCNode{Model: mlService},
    }
    
    // 返回 Pipeline 实例（常驻）
    return &pipeline.Pipeline{Nodes: nodes}
}
```

**允许的配置参数**（非用户特定）：
- ✅ 场景配置：`Key: "hot:feed"`（不同场景可以有不同的 Pipeline）
- ✅ 超时配置：`Timeout: 5 * time.Second`
- ✅ TopK 配置：`TopK: 20`（召回数量）
- ✅ 特征前缀：`UserFeaturePrefix: "user_"`（特征命名规则）
- ✅ 模型端点：`Endpoint: "http://..."`（服务地址）

**不允许的参数**（用户特定）：
- ❌ 用户 ID：应该在 `RecommendContext` 中
- ❌ 用户画像：应该在 `RecommendContext` 中
- ❌ 用户标签：应该在 `RecommendContext` 中
- ❌ 实时特征：应该在 `RecommendContext` 中

### 2. 请求处理阶段

**重要原则**：所有用户相关的参数和数据都在请求时通过 `RecommendContext` 传入。

```go
// ✅ 正确：每次请求创建上下文（包含所有用户参数）
func HandleRequest(pipeline *pipeline.Pipeline, userID string, scene string) ([]*core.Item, error) {
    // 创建请求上下文（请求生命周期，包含用户参数）
    rctx := &core.RecommendContext{
        UserID: userID,  // 用户参数
        Scene:  scene,   // 场景参数（请求级别）
    }
    
    // 加载请求特定的数据（用户相关）
    userProfile, _ := loadUserProfile(userID)  // 从存储加载用户画像
    rctx.User = userProfile
    
    // 加载实时特征（用户相关）
    rctx.Realtime = map[string]any{
        "hour":   time.Now().Hour(),
        "device": getDeviceFromRequest(r),
    }
    
    // 加载用户标签（用户相关）
    rctx.Labels = loadUserLabels(userID)
    
    // 执行 Pipeline（使用常驻实例，传入用户上下文）
    return pipeline.Run(context.Background(), rctx, nil)
}
```

**用户参数的位置**：
- ✅ **RecommendContext**：所有用户相关的参数都在这里
- ✅ **请求时加载**：用户画像、实时特征、标签等都在请求时加载
- ❌ **不在 Pipeline 初始化时**：Pipeline 初始化不包含任何用户参数

### 3. 避免的错误

```go
// ❌ 错误：在请求时创建 Pipeline
func HandleRequest(userID string) {
    // 每次请求都创建 Pipeline，浪费资源
    pipeline := &pipeline.Pipeline{...}
    pipeline.Run(...)
}

// ❌ 错误：在初始化时加载所有数据
func InitPipeline() {
    // 加载所有用户画像到内存（不推荐）
    allUsers := loadAllUserProfiles()
    // ...
}

// ❌ 错误：在初始化时包含用户参数
func InitPipeline(userID string) {  // ❌ 不应该有用户参数
    // 用户相关的配置不应该在初始化时传入
    // ...
}

// ❌ 错误：在 Node 初始化时包含用户数据
func InitPipeline() {
    nodes := []pipeline.Node{
        &recall.Hot{
            Store: store,
            Key:   "hot:feed",
            UserID: "user_123",  // ❌ 不应该包含用户 ID
        },
    }
}
```

---

## 总结

### 常驻部分（初始化一次，不包含用户参数）

| 组件 | 说明 | 是否包含用户参数 |
|------|------|----------------|
| `Pipeline` 结构体 | Node 和 Hook 列表 | ❌ 否 |
| `Store` 连接 | Redis、Memory 等存储连接 | ❌ 否 |
| `FeatureService` | 特征服务连接 | ❌ 否 |
| `MLService` | 模型服务客户端连接 | ❌ 否 |
| `VectorStore` | 向量存储连接 | ❌ 否 |
| `Model` 权重 | LR、DNN 等模型的权重 | ❌ 否 |
| `MergeStrategy` | 合并策略实例 | ❌ 否 |
| `SortStrategy` | 排序策略实例 | ❌ 否 |
| `SimilarityCalculator` | 相似度计算器实例 | ❌ 否 |
| 场景配置参数 | Key、TopK、Timeout、特征前缀等 | ❌ 否（场景配置，非用户参数） |

### 请求生命周期部分（每次请求，包含所有用户参数）

| 组件 | 说明 | 是否包含用户参数 |
|------|------|----------------|
| `RecommendContext` | 用户上下文信息 | ✅ 是（UserID、Scene 等） |
| `UserProfile` | 用户画像数据 | ✅ 是（用户特定） |
| `Realtime` | 实时特征数据 | ✅ 是（请求时特征） |
| `Labels` | 用户标签 | ✅ 是（用户特定） |
| `Params` | 请求参数 | ✅ 是（请求特定） |
| `items []*core.Item` | 召回和处理的物品列表 | ✅ 是（请求结果） |
| 从存储加载的数据 | 热门列表、交互数据、特征数据等 | ✅ 是（请求时加载） |

### 关键原则

1. **初始化阶段**：
   - ✅ 只包含常驻资源和配置
   - ❌ **不包含任何用户参数**
   - ✅ 可以包含场景配置（如 `Key: "hot:feed"`）

2. **请求处理阶段**：
   - ✅ 所有用户参数通过 `RecommendContext` 传入
   - ✅ 用户数据在请求时从存储加载
   - ✅ 使用常驻的 Pipeline 实例执行

---

## 相关文档

- [架构设计文档](./ARCHITECTURE.md) - 架构设计说明
- [Pipeline 功能特性](./PIPELINE_FEATURES.md) - Pipeline 功能介绍
- [接口与实现完整分析](./INTERFACES_AND_IMPLEMENTATIONS.md) - 接口使用手册
