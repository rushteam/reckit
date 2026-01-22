# 特征版本管理示例

本示例演示如何在 Reckit 中实现特征版本管理，支持多版本特征存储、灰度发布、版本降级等功能。

## 功能特性

### 1. 多版本特征存储

不同版本的特征存储在独立的 key 空间中：

- **v1 版本**：`user:features:v1:{userID}`
- **v2 版本**：`user:features:v2:{userID}`

这样可以同时维护多个版本的特征，互不干扰。

### 2. 版本化特征服务

`VersionedFeatureService` 实现了 `feature.FeatureService` 接口，支持：

- 根据配置自动选择版本
- 灰度发布（流量分配）
- 版本降级（新版本失败时回退到旧版本）

### 3. 灰度发布

通过流量分配实现平滑的版本切换：

```go
config := &VersionConfig{
    DefaultVersion: "v2",
    TrafficSplit: map[string]float64{
        "v2": 0.7, // 70% 流量使用 v2
        "v1": 0.3, // 30% 流量使用 v1（灰度）
    },
}
```

系统会根据用户 ID 的哈希值自动分配流量。

### 4. 版本元数据管理

`FeatureVersionRegistry` 管理版本元数据：

- 版本号
- 创建时间
- 描述信息
- 特征列表
- 状态（active, deprecated, archived）

### 5. 版本降级

当新版本的特征服务失败时，自动降级到旧版本：

```go
features, err := service.GetUserFeatures(ctx, userID)
if err != nil {
    // 自动降级到默认版本
    return fallbackService.GetUserFeatures(ctx, userID)
}
```

## 使用示例

### 基本使用

```go
// 1. 创建版本化特征服务
versionedService := createVersionedFeatureService(store)

// 2. 在 EnrichNode 中使用
enrichNode := &feature.EnrichNode{
    FeatureService: versionedService,
    UserFeaturePrefix:  "user_",
    ItemFeaturePrefix:  "item_",
    CrossFeaturePrefix: "cross_",
}

// 3. 在 Pipeline 中使用
p := &pipeline.Pipeline{
    Nodes: []pipeline.Node{
        &recall.Fanout{...},
        enrichNode,
        &rank.LRNode{...},
    },
}
```

### 版本选择

系统会根据用户 ID 的哈希值自动选择版本：

```go
// 用户 "42" 可能使用 v2 版本
// 用户 "100" 可能使用 v1 版本（灰度）
features, _ := versionedService.GetUserFeatures(ctx, "42")
```

### 版本信息查询

```go
registry := versionedService.registry
meta, _ := registry.GetVersion("v2")
fmt.Printf("版本: %s\n", meta.Version)
fmt.Printf("描述: %s\n", meta.Description)
fmt.Printf("特征列表: %v\n", meta.Features)
```

## 运行示例

```bash
# 使用内存 Store（默认）
go run ./examples/feature_version

# 使用 Redis Store（需要启动 Redis）
redis-server
go run ./examples/feature_version
```

## 版本管理最佳实践

### 1. 版本命名

- 使用语义化版本：`v1`, `v2`, `v3`
- 或使用时间戳：`20240101`, `20240201`

### 2. 存储隔离

不同版本使用不同的 key prefix，避免冲突：

```go
v1KeyPrefix := feature.KeyPrefix{
    User: "user:features:v1:",
    Item: "item:features:v1:",
}

v2KeyPrefix := feature.KeyPrefix{
    User: "user:features:v2:",
    Item: "item:features:v2:",
}
```

### 3. 灰度发布流程

1. **准备新版本**：部署新版本特征到存储
2. **小流量验证**：设置 10% 流量使用新版本
3. **逐步扩大**：验证无问题后逐步增加到 50%、100%
4. **下线旧版本**：确认新版本稳定后下线旧版本

### 4. 版本降级策略

- 新版本失败时自动降级到旧版本
- 记录降级事件，便于监控和排查
- 设置告警，及时发现版本问题

### 5. 版本元数据

维护版本元数据，包括：

- 版本号
- 创建时间
- 特征列表
- 状态（active, deprecated, archived）
- 描述信息

### 6. 缓存隔离

不同版本使用不同的缓存 key，避免缓存污染：

```go
// v1 版本缓存：cache:user:features:v1:{userID}
// v2 版本缓存：cache:user:features:v2:{userID}
```

## 扩展

### 自定义版本选择策略

可以实现自定义的版本选择逻辑：

```go
func (v *VersionedFeatureService) selectVersion(userID string) string {
    // 自定义逻辑：根据用户属性、A/B 测试分组等选择版本
    if isVIPUser(userID) {
        return "v2" // VIP 用户优先使用新版本
    }
    return v.config.DefaultVersion
}
```

### 特征兼容性转换

如果需要在新旧版本之间转换特征格式：

```go
func migrateFeatures(oldFeatures map[string]float64, version string) map[string]float64 {
    newFeatures := make(map[string]float64)
    
    // 特征名称映射（v1 -> v2）
    featureMapping := map[string]string{
        "user_age":     "age",
        "user_gender":  "gender",
        "user_city":    "location",
    }
    
    for oldKey, newKey := range featureMapping {
        if val, ok := oldFeatures[oldKey]; ok {
            newFeatures[newKey] = val
        }
    }
    
    return newFeatures
}
```

## 注意事项

1. **版本切换需要平滑**：避免突然切换导致特征不一致
2. **监控版本使用情况**：记录各版本的调用量、成功率、延迟等
3. **特征兼容性**：确保新旧版本的特征格式兼容，或提供转换逻辑
4. **缓存管理**：不同版本的特征需要独立的缓存空间
5. **数据一致性**：确保同一用户在不同版本间特征的一致性
