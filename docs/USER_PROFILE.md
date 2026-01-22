# 用户画像支持情况

## 概述

**一句话定义**：用户画像 = 推荐 Pipeline 的"全局上下文 + 特征源 + 决策信号"

用户画像不是某一个 Node，而是：
- 被所有 Node 共享
- 驱动 Recall / Rank / ReRank
- 可以被 Label 打标、回写、持续演进

## 架构设计

### 核心对象

```
UserProfile
   ↓
RecommendContext
   ↓
Pipeline
   ↓
Recall / Rank / ReRank Nodes
```

### 文件结构

```
core/
├── user_profile.go  # UserProfile 结构体定义
├── context.go       # RecommendContext（包含 UserProfile）
└── item.go          # Item（支持 Labels）
```

## UserProfile 结构

### 定义

```go
type UserProfile struct {
    UserID int64

    // 静态属性（冷启动 / 基础过滤）
    Gender   string
    Age      int
    Location string

    // 兴趣画像（长期）- Recall / Rank 核心
    Interests map[string]float64 // category -> weight

    // 行为统计（短期）- 实时调权
    RecentClicks   []int64
    RecentImpress []int64

    // 偏好信号
    PreferTags map[string]float64

    // 控制与实验（策略切换）
    Buckets map[string]string // AB / 实验桶

    // 元数据
    UpdateTime time.Time
}
```

### 设计要点

| 维度 | 作用 |
|------|------|
| 静态属性 | 冷启动 / 基础过滤 |
| 长期兴趣 | Recall / Rank 核心 |
| 短期行为 | 实时调权 |
| 实验桶 | 策略切换 |
| 可更新 | Online Learning |

### 核心方法

```go
// 创建用户画像
userProfile := core.NewUserProfile(userID)

// 更新兴趣（支持 Online Learning）
userProfile.UpdateInterest("tech", 0.8)

// 添加行为记录
userProfile.AddRecentClick(itemID, maxSize)
userProfile.AddRecentImpress(itemID, maxSize)

// 实验桶管理
userProfile.SetBucket("diversity", "strong")
bucket := userProfile.GetBucket("diversity")

// 兴趣查询
if userProfile.HasInterest("tech", 0.5) {
    // 用户对科技感兴趣
}
weight := userProfile.GetInterestWeight("tech")
```

## RecommendContext 升级

### 新结构

```go
type RecommendContext struct {
    UserID   int64
    DeviceID string
    Scene    string

    // UserProfile 是强类型用户画像（推荐使用）
    User *UserProfile

    // UserProfileMap 是向后兼容的 map 形式（保留）
    UserProfile map[string]any

    // Labels 是用户级标签，可驱动整个 Pipeline 行为
    Labels map[string]utils.Label

    Realtime map[string]any
    Params   map[string]any
}
```

### 兼容方法

```go
// 获取用户画像（兼容方法）
userProfile := rctx.GetUserProfile()

// 用户级 Label 操作
rctx.PutLabel("user_type", utils.Label{Value: "active", Source: "system"})
label, ok := rctx.GetLabel("user_type")
```

## 用户画像驱动 Recall

### 示例：用户兴趣驱动的召回增强

```go
type userProfileDrivenRecall struct{}

func (n *userProfileDrivenRecall) Process(
    ctx context.Context,
    rctx *core.RecommendContext,
    items []*core.Item,
) ([]*core.Item, error) {
    if rctx.User == nil {
        return items, nil
    }

    for _, item := range items {
        // 检查物品类别是否匹配用户兴趣
        if category, ok := item.Labels["category"]; ok {
            if rctx.User.HasInterest(category.Value, 0.5) {
                // 用户偏好放大
                weight := rctx.User.GetInterestWeight(category.Value)
                item.Score += weight * 0.5
                item.PutLabel("user_interest", utils.Label{
                    Value:  category.Value,
                    Source: n.Name(),
                })
            }
        }
    }
    return items, nil
}
```

## 用户画像驱动 Rank

### 示例：用户兴趣加权的排序

```go
type userProfileDrivenRank struct{}

func (n *userProfileDrivenRank) Process(
    ctx context.Context,
    rctx *core.RecommendContext,
    items []*core.Item,
) ([]*core.Item, error) {
    if rctx.User == nil {
        return items, nil
    }

    for _, item := range items {
        // 用户兴趣加权
        if category, ok := item.Labels["category"]; ok {
            if weight := rctx.User.GetInterestWeight(category.Value); weight > 0 {
                item.Score *= (1 + weight)
                item.PutLabel("user_interest_boost", utils.Label{
                    Value:  category.Value,
                    Source: n.Name(),
                })
            }
        }
    }
    return items, nil
}
```

## 用户画像驱动 ReRank

### 示例：实验桶驱动的多样性

```go
type userProfileDrivenRerank struct{}

func (n *userProfileDrivenRerank) Process(
    ctx context.Context,
    rctx *core.RecommendContext,
    items []*core.Item,
) ([]*core.Item, error) {
    if rctx.User == nil {
        return items, nil
    }

    // 根据实验桶调整多样性
    if rctx.User.GetBucket("diversity") == "strong" {
        for _, item := range items {
            item.Score *= 0.7 // 多样性惩罚
            item.PutLabel("diversity_boost", utils.Label{
                Value:  "strong",
                Source: n.Name(),
            })
        }
    }
    return items, nil
}
```

## SetBucket 实验桶使用指南

### 基本用法

`SetBucket` 用于设置实验桶（A/B 测试和策略切换），是用户画像中用于控制策略切换的核心机制。

#### 1. 设置实验桶

```go
// 创建用户画像
userProfile := core.NewUserProfile(userID)

// 设置实验桶
// 参数：key（实验名称），value（实验组/策略版本）
userProfile.SetBucket("diversity", "strong")    // 多样性策略：强多样性
userProfile.SetBucket("recall", "v2")            // 召回策略：版本2
userProfile.SetBucket("rank", "deep_model")     // 排序策略：深度模型
userProfile.SetBucket("rerank", "diversity_v1") // 重排策略：多样性版本1
```

#### 2. 获取实验桶值

```go
// 在 Node 中获取实验桶值
diversityStrategy := rctx.User.GetBucket("diversity")
recallVersion := rctx.User.GetBucket("recall")
rankStrategy := rctx.User.GetBucket("rank")
```

### 在 Node 中使用

#### Recall Node：根据实验桶选择召回策略

```go
type recallWithBucket struct{}

func (r *recallWithBucket) Recall(
    ctx context.Context,
    rctx *core.RecommendContext,
) ([]*core.Item, error) {
    // 根据实验桶选择召回策略
    recallVersion := rctx.User.GetBucket("recall")
    
    var items []*core.Item
    switch recallVersion {
    case "v2":
        // 使用新版本召回：更多个性化
        items = personalizedRecall()
    case "v1":
        // 使用旧版本召回：热门召回
        items = hotRecall()
    default:
        // 默认策略
        items = defaultRecall()
    }
    
    return items, nil
}
```

#### Rank Node：根据实验桶选择排序策略

```go
type rankWithBucket struct{}

func (r *rankWithBucket) Process(
    ctx context.Context,
    rctx *core.RecommendContext,
    items []*core.Item,
) ([]*core.Item, error) {
    // 根据实验桶选择排序策略
    rankStrategy := rctx.User.GetBucket("rank")
    
    for _, item := range items {
        switch rankStrategy {
        case "deep_model":
            // 使用深度模型排序
            item.Score = deepModelPredict(item)
        case "lr_model":
            // 使用 LR 模型排序
            item.Score = lrModelPredict(item)
        default:
            // 默认排序
            item.Score = defaultScore(item)
        }
    }
    
    return items, nil
}
```

#### ReRank Node：根据实验桶调整多样性

```go
type rerankWithBucket struct{}

func (r *rerankWithBucket) Process(
    ctx context.Context,
    rctx *core.RecommendContext,
    items []*core.Item,
) ([]*core.Item, error) {
    // 根据实验桶调整多样性
    diversityStrategy := rctx.User.GetBucket("diversity")
    
    switch diversityStrategy {
    case "strong":
        // 强多样性：降低相似物品的分数
        for _, item := range items {
            item.Score *= 0.7
        }
    case "weak":
        // 弱多样性：保持原分数
        // 不做调整
    default:
        // 默认策略
    }
    
    return items, nil
}
```

### 使用场景

#### 1. A/B 测试：不同用户使用不同策略版本

```go
// 根据用户 ID 分桶进行 A/B 测试
if userID % 2 == 0 {
    userProfile.SetBucket("recall", "v2")  // 实验组：50% 用户
} else {
    userProfile.SetBucket("recall", "v1")  // 对照组：50% 用户
}

// 在 Recall Node 中根据实验桶选择策略
recallVersion := rctx.User.GetBucket("recall")
if recallVersion == "v2" {
    // 使用新策略
} else {
    // 使用旧策略
}
```

#### 2. 策略切换：根据用户特征选择策略

```go
// 根据用户年龄选择多样性策略
if userProfile.Age > 30 {
    userProfile.SetBucket("diversity", "strong")  // 年龄大的用户使用强多样性
} else {
    userProfile.SetBucket("diversity", "weak")   // 年轻用户使用弱多样性
}

// 根据用户活跃度选择排序策略
if userProfile.GetInterestWeight("active") > 0.8 {
    userProfile.SetBucket("rank", "deep_model")   // 活跃用户使用深度模型
} else {
    userProfile.SetBucket("rank", "lr_model")    // 普通用户使用 LR 模型
}
```

#### 3. 灰度发布：逐步切换策略

```go
// 灰度发布：10% 用户使用新模型
if userID % 100 < 10 {
    userProfile.SetBucket("rank", "new_model")   // 10% 用户使用新模型
} else {
    userProfile.SetBucket("rank", "old_model")   // 90% 用户使用旧模型
}

// 逐步扩大灰度范围
// 第一阶段：1% 用户
// 第二阶段：10% 用户
// 第三阶段：50% 用户
// 第四阶段：100% 用户
```

#### 4. 多实验并行：同时进行多个 A/B 测试

```go
// 同时进行多个实验
userProfile.SetBucket("recall", "v2")        // 召回实验
userProfile.SetBucket("rank", "deep_model")  // 排序实验
userProfile.SetBucket("diversity", "strong") // 多样性实验

// 在各自的 Node 中独立判断
// Recall Node: 根据 "recall" 实验桶选择策略
// Rank Node: 根据 "rank" 实验桶选择策略
// ReRank Node: 根据 "diversity" 实验桶选择策略
```

### 完整示例

完整示例请参考：`examples/bucket_usage/main.go`

运行示例：
```bash
go run ./examples/bucket_usage
```

输出示例：
```
=== 实验桶使用示例 ===
多样性策略: strong
召回策略: v2
排序策略: deep_model
重排策略: diversity_v1
使用召回策略 v2：个性化召回
使用强多样性策略

=== 推荐结果 ===
1. 物品 1 (分数: 0.6300)
   - 策略: recall_v2|rank_deep_model|rerank_diversity_strong
2. 物品 2 (分数: 0.6300)
   - 策略: recall_v2|rank_deep_model|rerank_diversity_strong
3. 物品 3 (分数: 0.6300)
   - 策略: recall_v2|rank_deep_model|rerank_diversity_strong
```

### 最佳实践

1. **实验桶命名规范**
   - 使用有意义的 key：`recall`、`rank`、`diversity` 等
   - value 使用版本号或策略名称：`v1`、`v2`、`strong`、`weak` 等

2. **实验桶设置时机**
   - 在用户画像构建时设置（离线/实时）
   - 在请求入口处根据用户特征设置
   - 支持从外部实验平台获取并设置

3. **实验桶使用原则**
   - 每个 Node 独立判断自己的实验桶
   - 避免实验桶之间的相互影响
   - 记录实验桶信息到 Label 中，便于分析

4. **实验桶持久化**
   - 可以将实验桶信息存储到用户画像存储中
   - 支持实验桶的版本管理和回滚
   - 记录实验桶变更历史

### 总结

- **SetBucket(key, value)**：设置实验桶，key 为实验名称，value 为实验组/策略版本
- **GetBucket(key)**：获取实验桶值，返回对应的策略版本
- **主要用途**：A/B 测试、策略切换、灰度发布、多实验并行
- **使用位置**：在 Recall、Rank、ReRank 等 Node 中使用
- **优势**：灵活控制不同用户使用不同策略，便于进行实验和迭代

## Label 回写和 Online Learning

### 用户点击回写

```go
// 用户点击后
clickedItemID := items[0].ID
rctx.PutLabel(fmt.Sprintf("user.click.%d", clickedItemID), utils.Label{
    Value:  "1",
    Source: "feedback",
})

// 更新用户行为记录
userProfile.AddRecentClick(clickedItemID, 100)
```

### Online Learning：更新兴趣

```go
// 根据点击更新兴趣
if category, ok := items[0].Labels["category"]; ok {
    currentWeight := userProfile.GetInterestWeight(category.Value)
    newWeight := currentWeight + 0.1 // 点击后增加 0.1
    if newWeight > 1.0 {
        newWeight = 1.0
    }
    userProfile.UpdateInterest(category.Value, newWeight)
}
```

## 使用示例

完整示例请参考：`examples/user_profile/main.go`

运行示例：
```bash
go run ./examples/user_profile
```

## 向后兼容

为了保持向后兼容，`RecommendContext` 同时支持：

1. **强类型 UserProfile**（推荐）：
   ```go
   rctx := &core.RecommendContext{
       User: core.NewUserProfile(userID),
   }
   ```

2. **Map 形式 UserProfileMap**（兼容）：
   ```go
   rctx := &core.RecommendContext{
       UserProfile: map[string]any{
           "age": 25,
           "gender": "male",
       },
   }
   ```

3. **兼容方法**：
   ```go
   userProfile := rctx.GetUserProfile() // 自动兼容两种形式
   ```

## 用户级 Labels

`RecommendContext.Labels` 支持用户级标签，可驱动整个 Pipeline 行为：

```go
// 设置用户级标签
rctx.PutLabel("user_type", utils.Label{Value: "active", Source: "system"})
rctx.PutLabel("price_sensitive", utils.Label{Value: "false", Source: "system"})

// 在 Node 中使用
if label, ok := rctx.GetLabel("user_type"); ok && label.Value == "active" {
    // 活跃用户特殊处理
}
```

## 工程实践

### 1. 用户画像构建

- **离线构建**：基于历史行为统计长期兴趣
- **实时更新**：基于实时行为更新短期行为
- **Online Learning**：根据反馈持续更新兴趣权重

### 2. 用户画像使用

- **Recall 阶段**：根据兴趣过滤和加权
- **Rank 阶段**：根据兴趣调整排序分数
- **ReRank 阶段**：根据实验桶调整策略

### 3. Label 回写

- **行为回写**：点击、曝光等行为记录到 `RecentClicks`、`RecentImpress`
- **兴趣更新**：根据行为反馈更新 `Interests`
- **标签记录**：记录到 `Context.Labels` 供后续使用

## 扩展方向

1. **用户画像服务化**：独立的用户画像服务
2. **实时画像更新**：基于流式计算的实时画像
3. **画像版本管理**：支持画像版本和 A/B 测试
4. **画像存储优化**：支持 Redis、MySQL 等持久化存储
