# Reckit 架构设计文档

## 设计原则

### 1. 通用框架，高度解耦
- **框架层**：使用 `string` 作为通用的 ID 类型，支持 UUID、数字 ID、以及任何字符串格式。
- **不可修改**：框架代码保持通用，具体项目通过实现接口或注入依赖来扩展逻辑。

### 2. 设计模式充分运用

#### 策略模式（Strategy Pattern）
- `FeatureProvider`：不同的特征提供策略（Redis、Feast、Memory）。
- `RankModel`：不同的排序模型策略（LR、DNN、Wide&Deep）。
- `recall.Source`：不同的召回算法策略。

#### 适配器模式（Adapter Pattern）
- `FeatureServiceAdapter`：适配 Feast 到 FeatureService。
- `VectorStoreAdapter`：适配 ANNService 到 VectorStore。

#### 工厂模式（Factory Pattern）
- `FeatureServiceFactory`：创建不同类型的特征服务。
- `MLServiceFactory`：创建不同类型的 ML 服务。

#### 装饰器模式（Decorator Pattern）
- `FeatureCache`：为特征服务添加缓存。
- `FeatureMonitor`：为特征服务添加监控。
- `FallbackStrategy`：为特征服务添加降级。

### 3. 高内聚低耦合（DDD 原则）

#### 领域层（Domain Layer）
- `core`：核心领域模型（RecommendContext, UserProfile, Item）。
- `pipeline`：推荐流水线抽象。
- `feature`：特征服务领域抽象。
- `recall`：召回领域抽象。
- `rank`：排序领域抽象。

#### 基础设施层（Infrastructure Layer）
- `store`：KV 存储实现。
- `vector`：向量库集成（Milvus）。
- `service`：机器学习服务集成（TF Serving）。
- `feast`：Feature Store 集成。

## ID 设计

所有实体 ID（用户 ID、物品 ID）均统一使用 `string` 类型。

**设计理由**：
1. **通用性**：无需修改代码即可支持所有 ID 格式。
2. **简单性**：直接使用原生类型，降低学习和使用成本。
3. **互操作性**：JSON 序列化、存储索引、API 调用均有良好支持。

## 架构层次

```
┌─────────────────────────────────────────┐
│         应用层（Application）            │
│  - 具体的业务逻辑 & 推荐流水线配置         │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│         领域层（Domain）                 │
│  - core: Context, UserProfile, Item     │
│  - pipeline: Pipeline, Node            │
│  - feature: FeatureService              │
│  - recall: Source                       │
│  - rank: RankModel                      │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│      基础设施层（Infrastructure）        │
│  - store: Redis, Memory                │
│  - vector: Milvus                       │
│  - service: TF Serving, ANN Service    │
│  - feast: Feast Client                  │
└─────────────────────────────────────────┘
```

## 扩展性示例

### 自定义召回算法

具体项目只需实现 `recall.Source` 接口：

```go
type MyRecall struct {}
func (r *MyRecall) Name() string { return "my_recall" }
func (r *MyRecall) Recall(ctx context.Context, rctx *core.RecommendContext) ([]*core.Item, error) {
    // 业务逻辑
}
```

### 自定义排序特征注入

通过 `EnrichNode` 的选项注入自定义逻辑：

```go
node := &feature.EnrichNode{
    UserFeatureExtractor: func(rctx *core.RecommendContext) map[string]float64 {
        // 自定义提取逻辑
    },
}
```
