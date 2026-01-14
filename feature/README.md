# 特征服务（Feature Service）

特征服务提供了统一的特征获取接口，支持用户特征、物品特征、实时特征的获取，采用高度抽象、高内聚低耦合的设计。

## 设计模式

### 1. 策略模式（Strategy Pattern）
- **FeatureProvider**：不同的特征源实现（Redis、HTTP、Memory）实现此接口
- **FeatureStore**：特征存储抽象，支持多种存储后端

### 2. 装饰器模式（Decorator Pattern）
- **缓存装饰器**：`FeatureCache` 为特征服务添加缓存能力
- **监控装饰器**：`FeatureMonitor` 为特征服务添加监控能力
- **降级装饰器**：`FallbackStrategy` 为特征服务添加降级能力

### 3. 工厂模式（Factory Pattern）
- **FeatureServiceFactory**：统一创建不同类型的特征服务

### 4. 适配器模式（Adapter Pattern）
- **StoreFeatureProvider**：将 `store.Store` 适配为 `FeatureProvider` 接口

### 5. 函数式选项模式（Functional Options Pattern）
- **ServiceOption**：通过选项函数配置特征服务

## 核心接口

### FeatureService
统一的特征服务接口，提供用户特征、物品特征、实时特征的获取能力。

```go
type FeatureService interface {
    GetUserFeatures(ctx context.Context, userID int64) (map[string]float64, error)
    BatchGetUserFeatures(ctx context.Context, userIDs []int64) (map[int64]map[string]float64, error)
    GetItemFeatures(ctx context.Context, itemID int64) (map[string]float64, error)
    BatchGetItemFeatures(ctx context.Context, itemIDs []int64) (map[int64]map[string]float64, error)
    GetRealtimeFeatures(ctx context.Context, userID, itemID int64) (map[string]float64, error)
    BatchGetRealtimeFeatures(ctx context.Context, pairs []UserItemPair) (map[UserItemPair]map[string]float64, error)
}
```

### FeatureProvider
特征提供者接口，不同的特征源实现此接口。

```go
type FeatureProvider interface {
    GetUserFeatures(ctx context.Context, userID int64) (map[string]float64, error)
    BatchGetUserFeatures(ctx context.Context, userIDs []int64) (map[int64]map[string]float64, error)
    // ... 其他方法
}
```

### FeatureCache
特征缓存接口，为特征服务添加缓存能力。

```go
type FeatureCache interface {
    GetUserFeatures(ctx context.Context, userID int64) (map[string]float64, bool)
    SetUserFeatures(ctx context.Context, userID int64, features map[string]float64, ttl time.Duration)
    // ... 其他方法
}
```

### FeatureMonitor
特征监控接口，用于监控特征质量、分布、缺失率等。

```go
type FeatureMonitor interface {
    RecordFeatureUsage(ctx context.Context, featureName string, value float64)
    RecordFeatureMissing(ctx context.Context, featureName string, entityType string, entityID int64)
    RecordFeatureError(ctx context.Context, featureName string, err error)
    GetFeatureStats(ctx context.Context, featureName string) (*FeatureStats, error)
}
```

### FallbackStrategy
降级策略接口，当特征服务不可用时提供降级方案。

```go
type FallbackStrategy interface {
    GetUserFeatures(ctx context.Context, userID int64, rctx *core.RecommendContext) (map[string]float64, error)
    GetItemFeatures(ctx context.Context, itemID int64, item *core.Item) (map[string]float64, error)
    GetRealtimeFeatures(ctx context.Context, userID, itemID int64, rctx *core.RecommendContext, item *core.Item) (map[string]float64, error)
}
```

## 使用示例

### 1. 基础使用（从 Store 创建）

```go
// 初始化 Store
redisStore, _ := store.NewRedisStore("localhost:6379", 0)
defer redisStore.Close()

// 创建特征服务
factory := feature.NewFeatureServiceFactory()
featureService := factory.CreateFromStore(redisStore)

// 获取用户特征
userFeatures, err := featureService.GetUserFeatures(ctx, 42)
if err != nil {
    log.Printf("获取用户特征失败: %v", err)
}

// 批量获取物品特征
itemIDs := []int64{1, 2, 3, 4, 5}
itemFeatures, err := featureService.BatchGetItemFeatures(ctx, itemIDs)
if err != nil {
    log.Printf("批量获取物品特征失败: %v", err)
}
```

### 2. 带缓存的特征服务

```go
factory := feature.NewFeatureServiceFactory()
featureService := factory.CreateWithCache(
    redisStore,
    10000,              // 缓存大小
    5*time.Minute,      // 缓存 TTL
)
```

### 3. 带监控的特征服务

```go
factory := feature.NewFeatureServiceFactory()
featureService := factory.CreateWithMonitor(
    redisStore,
    1000,  // 每个特征保留的最大样本数
)

// 获取特征统计信息
stats, err := monitor.GetFeatureStats(ctx, "user_age")
if err == nil {
    log.Printf("特征统计: 均值=%.2f, 标准差=%.2f, 使用次数=%d",
        stats.Mean, stats.Std, stats.UsageCount)
}
```

### 4. 带降级策略的特征服务

```go
factory := feature.NewFeatureServiceFactory()
featureService := factory.CreateWithFallback(redisStore)

// 当特征服务不可用时，自动降级到从 RecommendContext 和 Item 提取基础特征
```

### 5. 完整配置（缓存 + 监控 + 降级）

```go
factory := feature.NewFeatureServiceFactory()
featureService := factory.CreateFull(
    redisStore,
    10000,              // 缓存大小
    5*time.Minute,      // 缓存 TTL
    1000,               // 监控样本数
)
```

### 6. 在 EnrichNode 中使用

```go
// 创建特征服务
factory := feature.NewFeatureServiceFactory()
featureService := factory.CreateFull(redisStore, 10000, 5*time.Minute, 1000)

// 在 EnrichNode 中使用
enrichNode := &feature.EnrichNode{
    FeatureService: featureService,  // 使用特征服务
    UserFeaturePrefix:  "user_",
    ItemFeaturePrefix:  "item_",
    CrossFeaturePrefix: "cross_",
}

// 传统模式（向后兼容）
enrichNode := &feature.EnrichNode{
    UserFeatureExtractor: customUserExtractor,  // 自定义提取器
    ItemFeatureExtractor: customItemExtractor,
    UserFeaturePrefix:  "user_",
    ItemFeaturePrefix:  "item_",
    CrossFeaturePrefix: "cross_",
}
```

### 7. 手动配置（使用选项模式）

```go
// 创建缓存
cache := feature.NewMemoryFeatureCache(10000, 5*time.Minute)
defer cache.Close()

// 创建监控
monitor := feature.NewMemoryFeatureMonitor(1000)
defer monitor.Close()

// 创建降级策略
fallback := feature.NewDefaultFallbackStrategy()

// 创建特征提供者
provider := feature.NewStoreFeatureProvider(redisStore, feature.KeyPrefix{})

// 创建特征服务（使用选项模式）
featureService := feature.NewBaseFeatureService(
    provider,
    feature.WithCache(cache, 5*time.Minute),
    feature.WithMonitor(monitor),
    feature.WithFallback(fallback),
)
```

## 特征存储格式

### Redis 存储格式

特征以 JSON 格式存储在 Redis 中：

```
user:features:42 -> {"age": 25.0, "gender": 1.0, "city": "beijing"}
item:features:1 -> {"ctr": 0.15, "cvr": 0.08, "price": 99.0}
realtime:features:42:1 -> {"click_count": 5.0, "view_count": 10.0}
```

### 特征序列化

默认使用 JSON 序列化，可以通过 `WithSerializer` 自定义序列化器：

```go
provider := feature.NewStoreFeatureProvider(redisStore, keyPrefix)
provider.WithSerializer(customSerializer)  // 自定义序列化器（如 MsgPack）
```

## 监控和统计

### 特征统计信息

```go
monitor := feature.NewMemoryFeatureMonitor(1000)
stats, err := monitor.GetFeatureStats(ctx, "user_age")
if err == nil {
    log.Printf("特征: %s", stats.FeatureName)
    log.Printf("使用次数: %d", stats.UsageCount)
    log.Printf("缺失次数: %d", stats.MissingCount)
    log.Printf("错误次数: %d", stats.ErrorCount)
    log.Printf("均值: %.2f", stats.Mean)
    log.Printf("标准差: %.2f", stats.Std)
    log.Printf("P50: %.2f, P95: %.2f, P99: %.2f", stats.P50, stats.P95, stats.P99)
}
```

## 扩展指南

### 实现自定义 FeatureProvider

```go
type CustomFeatureProvider struct {
    // 自定义字段
}

func (p *CustomFeatureProvider) Name() string {
    return "custom"
}

func (p *CustomFeatureProvider) GetUserFeatures(ctx context.Context, userID int64) (map[string]float64, error) {
    // 实现特征获取逻辑
}

// ... 实现其他方法

// 使用
provider := &CustomFeatureProvider{}
service := feature.NewBaseFeatureService(provider)
```

### 实现自定义 FeatureCache

```go
type CustomFeatureCache struct {
    // 自定义缓存实现
}

func (c *CustomFeatureCache) GetUserFeatures(ctx context.Context, userID int64) (map[string]float64, bool) {
    // 实现缓存获取逻辑
}

// ... 实现其他方法

// 使用
cache := &CustomFeatureCache{}
service := feature.NewBaseFeatureService(provider, feature.WithCache(cache, 5*time.Minute))
```

### 实现自定义 FeatureMonitor

```go
type PrometheusFeatureMonitor struct {
    // Prometheus 指标
}

func (m *PrometheusFeatureMonitor) RecordFeatureUsage(ctx context.Context, featureName string, value float64) {
    // 记录到 Prometheus
}

// ... 实现其他方法

// 使用
monitor := &PrometheusFeatureMonitor{}
service := feature.NewBaseFeatureService(provider, feature.WithMonitor(monitor))
```

## 最佳实践

1. **批量获取优先**：使用 `BatchGet*` 方法减少网络往返
2. **启用缓存**：对频繁访问的特征启用缓存
3. **监控特征质量**：使用 FeatureMonitor 监控特征分布和缺失率
4. **降级策略**：配置降级策略，确保特征服务不可用时系统仍能运行
5. **特征版本管理**：在存储 key 中包含特征版本，支持特征热更新

## 性能优化

1. **本地缓存**：使用 `MemoryFeatureCache` 减少对远程存储的访问
2. **批量操作**：使用批量接口减少网络往返
3. **连接池**：配置 Redis 连接池优化性能
4. **异步获取**：非关键特征可以异步获取

## 与现有系统集成

### 更新 EnrichNode

`EnrichNode` 已更新，支持使用 `FeatureService`：

```go
// 新方式：使用特征服务
enrichNode := &feature.EnrichNode{
    FeatureService: featureService,
    UserFeaturePrefix:  "user_",
    ItemFeaturePrefix:  "item_",
    CrossFeaturePrefix: "cross_",
}

// 旧方式：仍然支持（向后兼容）
enrichNode := &feature.EnrichNode{
    UserFeatureExtractor: customExtractor,
    UserFeaturePrefix:  "user_",
    ItemFeaturePrefix:  "item_",
    CrossFeaturePrefix: "cross_",
}
```
