# Pipeline 功能分析报告

## 概述

本文档分析 Reckit Pipeline 对推荐系统常见流程的支持情况，以及特定流程的实现能力。

## 推荐系统常见流程支持情况

### ✅ 已支持的流程

1. **召回（Recall）**
   - ✅ 多路并发召回（Fanout）
   - ✅ 多种召回算法（热门、协同过滤、内容推荐、向量检索等）
   - ✅ 召回结果合并策略（First、Union、Priority）
   - ✅ 超时控制和限流保护

2. **过滤（Filter）**
   - ✅ 黑名单过滤
   - ✅ 用户拉黑过滤
   - ✅ 已曝光过滤
   - ✅ 组合过滤器（FilterNode）

3. **特征注入（Feature Enrichment）**
   - ✅ 用户特征注入
   - ✅ 物品特征注入
   - ✅ 交叉特征生成
   - ✅ 批量特征获取

4. **排序（Rank）**
   - ✅ 多种排序模型（LR、DNN、DIN、Wide&Deep、Two Tower、RPC）
   - ✅ 排序策略（ScoreDesc、ScoreAsc、MultiField）
   - ✅ 批量预测支持

5. **重排（ReRank）**
   - ✅ 多样性重排（Diversity）

6. **Pipeline Hook**
   - ✅ 执行前后 Hook（日志、监控、缓存等）

## 特定流程支持分析

### 1. 拉取召回 item_id 列表 ✅ **完全支持**

**实现方式：**
- `recall.Source` 接口的 `Recall()` 方法返回 `[]*core.Item`
- 每个 `Item` 包含 `ID` 字段（string 类型）
- `Fanout` 支持并发执行多个召回源，自动合并结果

**代码位置：**
```12:14:recall/source.go
type Source interface {
	Name() string
	Recall(ctx context.Context, rctx *core.RecommendContext) ([]*core.Item, error)
}
```

**示例：**
```go
fanout := &recall.Fanout{
    Sources: []recall.Source{
        &recall.Hot{Store: redisStore, Key: "hot:feed"},
        &recall.U2IRecall{...},
    },
}
items, _ := fanout.Recall(ctx, rctx)
// items 是 []*core.Item，每个 item 有 ID
```

### 2. 查询特征（Redis / Doris） ⚠️ **部分支持**

#### Redis ✅ **完全支持**

**实现方式：**
- `store.RedisStore` 提供 Redis 存储能力
- `feature.StoreFeatureProvider` 将 Store 适配为 FeatureProvider
- 支持批量获取特征（`BatchGetItemFeatures`）

**代码位置：**
```76:87:feature/store_provider.go
func (p *StoreFeatureProvider) GetUserFeatures(ctx context.Context, userID string) (map[string]float64, error) {
	key := fmt.Sprintf("%s%s", p.keyPrefix.User, userID)
	data, err := p.store.Get(ctx, key)
	if err != nil {
		if err == store.ErrNotFound {
			return nil, ErrFeatureNotFound
		}
		return nil, err
	}

	return p.serializer.Deserialize(data)
}
```

**使用示例：**
```go
redisStore, _ := store.NewRedisStore("localhost:6379", 0)
provider := feature.NewStoreFeatureProvider(redisStore, feature.KeyPrefix{
    User: "user:features:",
    Item: "item:features:",
})
featureService := feature.NewBaseFeatureService(provider)
```

#### Doris ❌ **不支持（但可扩展）**

**现状：**
- 当前没有 Doris 的直接支持
- 但可以通过实现 `feature.FeatureProvider` 接口扩展

**扩展方式：**
```go
type DorisFeatureProvider struct {
    client *doris.Client
}

func (p *DorisFeatureProvider) BatchGetItemFeatures(ctx context.Context, itemIDs []string) (map[string]map[string]float64, error) {
    // 实现 Doris 查询逻辑
    query := fmt.Sprintf("SELECT item_id, feature_name, feature_value FROM features WHERE item_id IN (%s)", ...)
    // ...
}
```

### 3. 拼 batch ✅ **完全支持**

**实现方式：**
- `FeatureService` 接口提供批量方法：
  - `BatchGetUserFeatures(ctx, userIDs []string)`
  - `BatchGetItemFeatures(ctx, itemIDs []string)`
  - `BatchGetRealtimeFeatures(ctx, pairs []UserItemPair)`
- `EnrichNode` 自动使用批量获取
- `RPCModel.PredictBatch()` 支持批量预测

**代码位置：**
```144:156:feature/enrich.go
	// 批量获取物品特征（如果使用 FeatureService）
	var itemFeaturesMap map[string]map[string]float64
	if n.FeatureService != nil {
		itemIDs := make([]string, 0, len(items))
		for _, item := range items {
			if item != nil {
				itemIDs = append(itemIDs, item.ID)
			}
		}
		if len(itemIDs) > 0 {
			itemFeaturesMap, _ = n.FeatureService.BatchGetItemFeatures(ctx, itemIDs)
		}
	}
```

**批量预测：**
```51:111:model/rpc.go
// PredictBatch 调用远程模型服务进行批量预测。
// 请求格式（JSON）：
//
//	{"features_list": [{"ctr": 0.15, "cvr": 0.08, ...}, ...]}
//
// 响应格式（JSON）：
//
//	{"scores": [0.85, 0.72, ...]}
func (m *RPCModel) PredictBatch(featuresList []map[string]float64) ([]float64, error) {
	// ... 批量预测实现
}
```

### 4. PyTorch / TorchScript 推理 ✅ **完全支持**

#### 通过 RPC 方式 ✅ **支持**
#### 通过 TorchServe 客户端 ✅ **完全支持**

**实现方式：**
- `service.TorchServeClient` 提供 TorchServe REST API 客户端
- 支持批量预测
- 支持多种响应格式自动解析
- 支持认证和超时控制

**代码位置：**
```34:50:service/torchserve.go
type TorchServeClient struct {
	// Endpoint 服务端点
	// REST: "http://localhost:8080"
	Endpoint string

	// ModelName 模型名称
	ModelName string

	// ModelVersion 模型版本（可选，TorchServe 通过模型版本管理）
	ModelVersion string

	// Timeout 超时时间
	Timeout time.Duration

	// Auth 认证信息
	Auth *AuthConfig

	// httpClient HTTP 客户端
	httpClient *http.Client
}
```

**使用示例：**
```go
// 方式1：直接使用 TorchServeClient
torchService := service.NewTorchServeClient(
    "http://localhost:8080",
    "my_model",
    service.WithTorchServeVersion("1.0"),
    service.WithTorchServeTimeout(30*time.Second),
)

// 方式2：通过工厂方法
config := &service.ServiceConfig{
    Type:        service.ServiceTypeTorchServe,
    Endpoint:    "http://localhost:8080",
    ModelName:   "my_model",
    ModelVersion: "1.0",
    Timeout:     30,
}
torchService, _ := service.NewMLService(config)

// 方式3：通过 RPCModel（兼容旧代码）
rpcModel := model.NewRPCModel("pytorch_model", "http://torchserve:8080/predictions/model_name", 5*time.Second)
rpcNode := &rank.RPCNode{Model: rpcModel}
```

#### 本地 PyTorch 推理 ❌ **不支持**

**现状：**
- 当前没有 Go 端的 PyTorch 本地推理支持
- 需要通过 TorchServe 服务化部署

**建议：**
- 使用 TorchServe 部署 PyTorch/TorchScript 模型（推荐）
- 或通过 `model.RPCModel` 调用自定义 PyTorch 服务

### 5. 排序 Top-N ✅ **完全支持**

**实现方式：**
- ✅ 支持排序：`SortStrategy` 接口提供多种排序策略
- ✅ **支持显式 Top-N 截断**：`rerank.TopNNode` 提供 Top-N 截断功能

**代码位置：**
```13:17:rank/lr_node.go
// SortStrategy 是排序策略接口，用于自定义物品排序逻辑。
type SortStrategy interface {
	// Sort 对物品列表进行排序（原地排序）
	Sort(items []*core.Item)
}
```

**TopNNode 实现：**
```1:50:rerank/topn.go
package rerank

import (
	"context"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pipeline"
)

// TopNNode 是一个 Top-N 截断节点，用于在排序后截取前 N 个物品。
// 通常在排序（Rank）节点之后使用，用于限制返回结果数量。
type TopNNode struct {
	// N 要保留的物品数量（Top N）
	// 如果 N <= 0，则返回所有物品（不截断）
	// 如果 N > len(items)，则返回所有物品
	N int
}

func (n *TopNNode) Process(
	_ context.Context,
	_ *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	// 如果 N <= 0，不截断，返回所有物品
	if n.N <= 0 {
		return items, nil
	}

	// 如果物品数量小于等于 N，直接返回
	if len(items) <= n.N {
		return items, nil
	}

	// 截取前 N 个物品
	return items[:n.N], nil
}
```

**使用示例：**
```go
pipeline := &pipeline.Pipeline{
    Nodes: []pipeline.Node{
        &rank.LRNode{...},        // 排序
        &rerank.TopNNode{N: 20},  // 截取 Top 20
        &rerank.Diversity{...},   // 多样性重排（可选）
    },
}
```

## 完整流程示例

### 支持的流程（当前实现）

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
            FeatureService: featureService, // 使用 Redis 特征服务
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

### 缺失的功能

1. **Doris 特征查询**
   - 需要实现 `feature.FeatureProvider` 接口
   - 或扩展 `feature.StoreFeatureProvider` 支持 Doris

2. **本地 PyTorch 推理**
   - 需要 Go 的 PyTorch 绑定（如 `gorgonia` 或 CGO 调用）
   - 推荐通过 TorchServe 服务化（已支持）

## 改进建议

### 优先级 1：Top-N 节点 ✅ **已实现**

TopNNode 已完整实现，支持：
- Top-N 截断功能
- 边界情况处理（N <= 0 或 N > len(items)）
- 与 Pipeline 无缝集成

详见：`rerank/topn.go`

### 优先级 2：Doris 特征提供者

```go
// feature/doris_provider.go
type DorisFeatureProvider struct {
    client *doris.Client
    // ...
}

func (p *DorisFeatureProvider) BatchGetItemFeatures(ctx context.Context, itemIDs []string) (map[string]map[string]float64, error) {
    // 实现 Doris 批量查询
}
```

### 优先级 3：TorchServe 客户端 ✅ **已实现**

TorchServe REST API 客户端已完整实现，支持：
- 批量预测
- 多种响应格式自动解析
- 认证和超时控制
- 健康检查

详见：`service/torchserve.go`

## 总结

| 功能 | 支持情况 | 说明 |
|------|---------|------|
| 拉取召回 item_id 列表 | ✅ 完全支持 | Source 接口返回 Item 列表 |
| Redis 特征查询 | ✅ 完全支持 | StoreFeatureProvider + RedisStore |
| Doris 特征查询 | ❌ 不支持（可扩展） | 需实现 FeatureProvider 接口 |
| 拼 batch | ✅ 完全支持 | BatchGetItemFeatures + PredictBatch |
| PyTorch/TorchScript 推理 | ✅ 完全支持 | TorchServeClient 提供完整支持 |
| PyTorch 本地推理 | ❌ 不支持 | 需通过 TorchServe 服务化（推荐） |
| 排序 | ✅ 完全支持 | SortStrategy 接口 |
| Top-N 截断 | ✅ 完全支持 | TopNNode 提供完整支持 |

**总体评价：**
- Pipeline 架构设计良好，支持推荐系统的主要流程
- 批量处理和特征查询能力完善
- TorchServe 客户端已完整实现，支持 PyTorch/TorchScript 模型推理
- TopNNode 已完整实现，支持 Top-N 截断功能
- 缺少 Doris 支持，但易于扩展（实现 FeatureProvider 接口）
- PyTorch 本地推理需通过 TorchServe 服务化（推荐方式）
