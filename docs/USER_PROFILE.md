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
    UserID string // 使用 string 类型（通用，支持所有 ID 格式）

    // 静态属性（冷启动 / 基础过滤）
    Gender   string
    Age      int
    Location string

    // 兴趣画像（长期）- Recall / Rank 核心
    Interests map[string]float64 // category -> weight

    // 行为统计（短期）- 实时调权
    RecentClicks   []string
    RecentImpress []string

    // 偏好信号
    PreferTags map[string]float64

    // 控制与实验（策略切换）
    Buckets map[string]string // AB / 实验桶

    // 扩展字段（用户自定义属性）
    // 用于存储框架未定义的用户属性，支持任意类型
    Extras map[string]any

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

// 扩展属性（用户自定义）
userProfile.SetExtra("vip_level", 3)
userProfile.SetExtra("preferred_price_range", "100-500")
userProfile.SetExtra("custom_tags", []string{"tech", "gaming"})

// 获取扩展属性
vipLevel, _ := userProfile.GetExtraFloat64("vip_level")
priceRange, _ := userProfile.GetExtraString("preferred_price_range")
tags, _ := userProfile.GetExtra("custom_tags").([]string)
```

## RecommendContext

`RecommendContext` 是推荐请求的上下文，支持强类型对象和 map 形式。

### 结构

```go
type RecommendContext struct {
    UserID   string // 使用 string 类型（通用，支持所有 ID 格式）
    DeviceID string
    Scene    string

    // User 是强类型用户画像
    User *UserProfile

    // UserProfile 是 map 形式，用于快速原型或动态属性
    UserProfile map[string]any

    // Labels 是用户级标签，可驱动整个 Pipeline 行为
    Labels map[string]utils.Label

    Realtime map[string]any
    Params   map[string]any
}
```

### 使用方式

1. **强类型 UserProfile**（推荐）：
   ```go
   rctx := &core.RecommendContext{
       User: core.NewUserProfile(userID),
   }
   ```

2. **Map 形式 UserProfile**：
   ```go
   rctx := &core.RecommendContext{
       UserProfile: map[string]any{
           "age": 25,
           "gender": "male",
       },
   }
   ```

3. **统一获取方法**：
   ```go
   userProfile := rctx.GetUserProfile() // 自动适配两种形式
   ```

### 用户级 Label 操作

```go
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

## 扩展属性（Extras）

`UserProfile` 支持通过 `Extras` 字段存储用户自定义属性，无需修改框架代码即可扩展。

### 基本用法

```go
// 设置扩展属性
userProfile := core.NewUserProfile("user_123")
userProfile.SetExtra("vip_level", 3)
userProfile.SetExtra("preferred_price_range", "100-500")
userProfile.SetExtra("custom_tags", []string{"tech", "gaming"})
userProfile.SetExtra("purchase_history_count", 150)

// 获取扩展属性
vipLevel, _ := userProfile.GetExtraFloat64("vip_level")
priceRange, _ := userProfile.GetExtraString("preferred_price_range")
tags, _ := core.GetExtraAs[[]string](userProfile, "custom_tags")  // 泛型：精确类型匹配
purchaseCount, _ := userProfile.GetExtraInt("purchase_history_count")
```

### 扩展方向

1. **用户画像服务化**：独立的用户画像服务
2. **实时画像更新**：基于流式计算的实时画像
3. **画像版本管理**：支持画像版本和 A/B 测试
4. **画像存储优化**：支持 Redis、MySQL 等持久化存储
5. **扩展属性**：通过 `Extras` 字段支持用户自定义属性
