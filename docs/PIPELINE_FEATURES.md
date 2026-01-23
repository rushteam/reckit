# Pipeline 功能特性

本文档介绍 Reckit Pipeline 支持的功能特性。

---

## 核心功能

### 1. 召回（Recall）

- **多路并发召回**：`Fanout` 支持并发执行多个召回源
- **多种召回算法**：热门、协同过滤、内容推荐、向量检索等
- **召回结果合并**：支持 First、Union、Priority 等合并策略
- **超时控制和限流保护**：防止单个召回源影响整体性能

### 2. 过滤（Filter）

- **黑名单过滤**：支持物品黑名单
- **用户拉黑过滤**：支持用户级别的拉黑
- **已曝光过滤**：支持时间窗口内的曝光过滤
- **组合过滤器**：`FilterNode` 支持多个过滤器组合

### 3. 特征注入（Feature Enrichment）

- **用户特征注入**：自动注入用户特征（带 `user_` 前缀）
- **物品特征注入**：自动注入物品特征（带 `item_` 前缀）
- **交叉特征生成**：自动生成交叉特征（带 `cross_` 前缀）
- **批量特征获取**：支持批量获取特征，提高性能

### 4. 排序（Rank）

- **多种排序模型**：LR、DNN、DIN、Wide&Deep、Two Tower、RPC
- **排序策略**：支持 ScoreDesc、ScoreAsc、MultiField 等策略
- **批量预测支持**：支持批量预测，提高性能

### 5. 重排（ReRank）

- **多样性重排**：`Diversity` 支持基于标签的多样性重排
- **Top-N 截断**：`TopNNode` 支持截取前 N 个物品

### 6. Pipeline Hook

- **执行前后 Hook**：支持在执行前后插入自定义逻辑
- **应用场景**：日志、监控、缓存等

---

## 完整流程示例

```go
pipeline := &pipeline.Pipeline{
    Nodes: []pipeline.Node{
        // 1. 召回：拉取 item_id 列表
        &recall.Fanout{
            Sources: []recall.Source{
                &recall.Hot{Store: redisStore, Key: "hot:feed"},
            },
        },
        // 2. 过滤
        &filter.FilterNode{...},
        // 3. 特征注入：批量查询特征（Redis）
        &feature.EnrichNode{
            FeatureService: featureService,
        },
        // 4. 排序：批量推理（通过 TorchServe 客户端）
        &rank.RPCNode{
            Model: model.NewRPCModel("pytorch", "http://torchserve:8080/predictions/model_name", 5*time.Second),
        },
        // 5. Top-N：截取 Top 20
        &rerank.TopNNode{N: 20},
    },
}
```

---

## 支持的功能

| 功能 | 支持情况 | 说明 |
|------|---------|------|
| 拉取召回 item_id 列表 | ✅ 完全支持 | Source 接口返回 Item 列表 |
| Redis 特征查询 | ✅ 完全支持 | StoreFeatureProvider + RedisStore |
| Doris 特征查询 | ⚠️ 可扩展 | 需实现 FeatureProvider 接口 |
| 批量特征获取 | ✅ 完全支持 | BatchGetItemFeatures + PredictBatch |
| PyTorch/TorchScript 推理 | ✅ 完全支持 | TorchServeClient 提供完整支持 |
| 排序 | ✅ 完全支持 | SortStrategy 接口 |
| Top-N 截断 | ✅ 完全支持 | TopNNode 提供完整支持 |

---

## 扩展支持

可以通过实现 `feature.FeatureProvider` 接口支持其他数据源（如 Doris）：

```go
type DorisFeatureProvider struct {
    client *doris.Client
}

func (p *DorisFeatureProvider) BatchGetItemFeatures(ctx context.Context, itemIDs []string) (map[string]map[string]float64, error) {
    // 实现 Doris 批量查询
}
```

---

## 相关文档

- [接口与实现完整分析](./INTERFACES_AND_IMPLEMENTATIONS.md) - 所有接口的详细定义
- [架构设计文档](./ARCHITECTURE.md) - 架构设计说明
- [可扩展性指南](./EXTENSIBILITY.md) - 如何扩展功能
- [召回算法文档](./RECALL_ALGORITHMS.md) - 召回算法介绍
- [排序模型文档](./RANK_MODELS.md) - 排序模型介绍
