# 双塔模型搭建指南

## 概述

根据模型选型表，双塔模型（Two-Tower）是召回阶段的首选模型，具有以下特点：

- **核心优点**：线上极速，Item 向量可预计算存入 Redis/Faiss，Golang 仅需做向量点积
- **核心缺点**：无法捕捉 User 和 Item 之间的细粒度特征交互
- **适用场景**：大规模初筛、跨域召回
- **输入**：用户特征、物品特征
- **输出**：相似度分数或 Embedding 向量

### 快速参考：Embedding 持久化策略

| Embedding 类型 | 是否需要持久化 | 原因 | 推荐方案 |
|---------------|---------------|------|---------|
| **Item Embedding** | ✅ **必须持久化** | 物品特征稳定，可离线批量计算 | Redis/Faiss 存储，离线更新 |
| **User Embedding** | ❌ **通常不需要** | 用户特征变化频繁，需要实时计算 | 实时计算（< 10ms） |
| **User Embedding** | ⚠️ **可选缓存** | 极高 QPS 场景，延迟要求 < 1ms | 短期缓存（5-30分钟 TTL） |

**核心原则**：
- **Item Embedding**：离线预计算 + 持久化存储（核心优化）
- **User Embedding**：实时计算（默认）或短期缓存（优化）

## 架构设计

### 模型结构

```
用户特征 (User Features)
    ↓
User Tower (DNN) → User Embedding (维度: embeddingDim)
                                    ↓
物品特征 (Item Features)           相似度计算 (内积/余弦)
    ↓                              ↓
Item Tower (DNN) → Item Embedding (维度: embeddingDim)
                                    ↓
                              Sigmoid → 输出分数
```

### 核心组件

1. **User Tower（用户塔）**：学习用户表示（User Embedding）
2. **Item Tower（物品塔）**：学习物品表示（Item Embedding）
3. **相似度计算**：User Embedding 和 Item Embedding 的内积/余弦相似度

## 一、用户塔（User Tower）搭建

### 1.1 用户特征设计

用户特征应该包含以下维度：

#### 基础特征
- **用户ID**：`user_id`（数值化或哈希）
- **年龄**：`user_age`
- **性别**：`user_gender`（0/1 或 one-hot）
- **地理位置**：`user_location`（城市ID、省份ID等）

#### 行为统计特征
- **点击率**：`user_ctr`（用户历史点击率）
- **转化率**：`user_cvr`（用户历史转化率）
- **活跃度**：`user_active_days`（最近N天活跃天数）
- **行为频次**：`user_click_count_7d`、`user_view_count_7d`等

#### 兴趣特征
- **兴趣标签**：`user_interest_tech`、`user_interest_game`等（多值特征）
- **偏好类别**：`user_prefer_category_1`、`user_prefer_category_2`等
- **历史行为序列**：`user_recent_clicks`（可选，如果使用序列特征）

#### 实时特征
- **当前时间**：`user_hour`、`user_day_of_week`等
- **设备信息**：`user_device_type`、`user_os`等
- **场景信息**：`user_scene`（feed、search、detail等）

### 1.2 用户特征提取

在 Reckit 中，用户特征通过 `EnrichNode` 自动提取：

```go
// 方式1：使用默认提取器（从 UserProfile 提取）
enrichNode := &feature.EnrichNode{
    UserFeaturePrefix: "user_",  // 用户特征前缀
}

// 方式2：使用自定义提取器
enrichNode := &feature.EnrichNode{
    UserFeaturePrefix: "user_",
    UserFeatureExtractor: func(rctx *core.RecommendContext) map[string]float64 {
        features := make(map[string]float64)
        
        // 从 UserProfile 提取
        if rctx.User != nil {
            features["age"] = float64(rctx.User.Age)
            if rctx.User.Gender == "male" {
                features["gender"] = 1.0
            } else {
                features["gender"] = 0.0
            }
            
            // 兴趣特征
            for tag, score := range rctx.User.Interests {
                features["interest_"+tag] = score
            }
        }
        
        // 从 Realtime 提取实时特征
        if rctx.Realtime != nil {
            if hour, ok := rctx.Realtime["hour"].(float64); ok {
                features["hour"] = hour
            }
        }
        
        return features
    },
}

// 方式3：使用特征服务（推荐）
featureService := feature.NewFeatureService(...)
enrichNode := &feature.EnrichNode{
    FeatureService: featureService,
    UserFeaturePrefix: "user_",
}
```

### 1.3 User Tower 结构设计

User Tower 是一个 DNN 网络，典型结构：

```go
// 创建 User Tower
userTowerLayers := []int{128, 64, 32}  // 层结构
embeddingDim := 32                     // 最终 Embedding 维度

twoTowerModel := model.NewTwoTowerModel(
    userTowerLayers,  // User Tower 层结构
    []int{128, 64, 32}, // Item Tower 层结构
    embeddingDim,     // Embedding 维度
)
```

**层结构设计建议**：
- **输入层**：根据用户特征数量确定（例如：128维）
- **隐藏层**：通常2-3层，每层逐渐降维（例如：128 → 64 → 32）
- **输出层**：Embedding 维度（例如：32维）

**设计原则**：
- 第一层应该足够宽，能够容纳所有用户特征
- 逐层降维，提取抽象的用户表示
- 最后一层输出固定维度的 Embedding

## 二、物品塔（Item Tower）搭建

### 2.1 物品特征设计

物品特征应该包含以下维度：

#### 基础特征
- **物品ID**：`item_id`（数值化或哈希）
- **类别**：`item_category`（类别ID）
- **价格**：`item_price`
- **发布时间**：`item_publish_time`（时间戳或相对时间）

#### 统计特征
- **点击率**：`item_ctr`（物品历史点击率）
- **转化率**：`item_cvr`（物品历史转化率）
- **曝光量**：`item_impression_count`
- **点击量**：`item_click_count`
- **收藏数**：`item_favorite_count`
- **评分**：`item_rating`（平均评分）

#### 内容特征
- **标题长度**：`item_title_len`
- **描述长度**：`item_desc_len`
- **标签**：`item_tag_1`、`item_tag_2`等（多值特征）
- **作者信息**：`item_author_id`、`item_author_followers`等

#### 实时特征
- **当前热度**：`item_hot_score`（实时计算的热度分数）
- **趋势特征**：`item_trend_score`（上升/下降趋势）

### 2.2 物品特征提取

在 Reckit 中，物品特征通过 `EnrichNode` 自动提取：

```go
// 方式1：使用默认提取器（从 Item.Features 提取）
enrichNode := &feature.EnrichNode{
    ItemFeaturePrefix: "item_",  // 物品特征前缀
}

// 方式2：使用自定义提取器
enrichNode := &feature.EnrichNode{
    ItemFeaturePrefix: "item_",
    ItemFeatureExtractor: func(item *core.Item) map[string]float64 {
        features := make(map[string]float64)
        
        // 从 item.Features 提取
        if item.Features != nil {
            features["ctr"] = item.Features["ctr"]
            features["cvr"] = item.Features["cvr"]
            features["price"] = item.Features["price"]
        }
        
        // 从 item.Meta 提取
        if item.Meta != nil {
            if category, ok := item.Meta["category"].(string); ok {
                // 类别特征数值化
                features["category"] = hashCategory(category)
            }
        }
        
        return features
    },
}

// 方式3：使用特征服务（推荐）
featureService := feature.NewFeatureService(...)
enrichNode := &feature.EnrichNode{
    FeatureService: featureService,
    ItemFeaturePrefix: "item_",
}
```

### 2.3 Item Tower 结构设计

Item Tower 是一个 DNN 网络，典型结构：

```go
// 创建 Item Tower
itemTowerLayers := []int{128, 64, 32}  // 层结构
embeddingDim := 32                     // 最终 Embedding 维度（与 User Tower 一致）

twoTowerModel := model.NewTwoTowerModel(
    []int{128, 64, 32}, // User Tower 层结构
    itemTowerLayers,    // Item Tower 层结构
    embeddingDim,       // Embedding 维度
)
```

**层结构设计建议**：
- **输入层**：根据物品特征数量确定（例如：128维）
- **隐藏层**：通常2-3层，每层逐渐降维（例如：128 → 64 → 32）
- **输出层**：Embedding 维度（必须与 User Tower 一致，例如：32维）

**设计原则**：
- 第一层应该足够宽，能够容纳所有物品特征
- 逐层降维，提取抽象的物品表示
- 最后一层输出固定维度的 Embedding（与 User Tower 维度一致）

## 三、完整示例

### 3.1 创建双塔模型

```go
package main

import (
    "context"
    "time"
    
    "github.com/rushteam/reckit/core"
    "github.com/rushteam/reckit/feature"
    "github.com/rushteam/reckit/model"
    "github.com/rushteam/reckit/pipeline"
    "github.com/rushteam/reckit/rank"
    "github.com/rushteam/reckit/recall"
)

func main() {
    ctx := context.Background()
    
    // 1. 创建用户画像
    userProfile := core.NewUserProfile("user_123")
    userProfile.Age = 25
    userProfile.Gender = "male"
    userProfile.UpdateInterest("tech", 0.8)
    userProfile.UpdateInterest("game", 0.6)
    
    rctx := &core.RecommendContext{
        UserID: "user_123",
        Scene:  "feed",
        User:   userProfile,
        Realtime: map[string]any{
            "hour": float64(time.Now().Hour()),
        },
    }
    
    // 2. 创建双塔模型
    twoTowerModel := model.NewTwoTowerModel(
        []int{128, 64, 32}, // User Tower: 128 → 64 → 32
        []int{128, 64, 32}, // Item Tower: 128 → 64 → 32
        32,                 // Embedding 维度
    )
    twoTowerModel.SimilarityType = "dot" // 使用内积（或 "cosine" 使用余弦）
    
    // 3. 创建特征注入节点
    enrichNode := &feature.EnrichNode{
        UserFeaturePrefix: "user_",  // 用户特征前缀
        ItemFeaturePrefix: "item_",  // 物品特征前缀
    }
    
    // 4. 创建 Pipeline
    p := &pipeline.Pipeline{
        Nodes: []pipeline.Node{
            // 召回
            &recall.Fanout{
                Sources: []recall.Source{
                    &recall.Hot{IDs: []string{"item_1", "item_2", "item_3"}},
                },
                Dedup: true,
            },
            // 特征注入
            enrichNode,
            // 排序（双塔模型）
            &rank.TwoTowerNode{Model: twoTowerModel},
        },
    }
    
    // 5. 运行 Pipeline
    items, err := p.Run(ctx, rctx, nil)
    if err != nil {
        panic(err)
    }
    
    // 6. 输出结果
    for i, item := range items {
        fmt.Printf("%d. Item %s (Score: %.4f)\n", i+1, item.ID, item.Score)
    }
}
```

### 3.2 特征命名规范

为了确保双塔模型正确区分用户特征和物品特征，必须遵循命名规范：

- **用户特征**：必须以 `user_` 开头（例如：`user_age`、`user_gender`）
- **物品特征**：必须以 `item_` 开头（例如：`item_ctr`、`item_price`）

特征注入后，`item.Features` 中会包含：

```go
item.Features = map[string]float64{
    // 用户特征（带 user_ 前缀）
    "user_age": 25.0,
    "user_gender": 1.0,
    "user_interest_tech": 0.8,
    "user_hour": 14.0,
    
    // 物品特征（带 item_ 前缀）
    "item_ctr": 0.15,
    "item_cvr": 0.08,
    "item_price": 99.9,
    "item_category": 1.0,
}
```

### 3.3 模型预测流程

双塔模型的预测流程：

1. **特征提取**：从 `item.Features` 中提取 `user_*` 和 `item_*` 特征
2. **User Tower 前向传播**：用户特征 → User Embedding
3. **Item Tower 前向传播**：物品特征 → Item Embedding
4. **相似度计算**：User Embedding · Item Embedding（内积）或余弦相似度
5. **Sigmoid 激活**：将相似度转换为概率分数

## 四、工程优化建议

### 4.1 离线预计算 Item Embedding

根据模型选型表，双塔模型的核心优势是 **Item 向量可预计算存入 Redis/Faiss**。

**优化方案**：

1. **离线计算**：使用训练好的 Item Tower，批量计算所有物品的 Embedding
2. **存储**：将 Item Embedding 存入 Redis 或向量数据库（如 Faiss、Milvus）
3. **在线推理**：
   - 实时计算 User Embedding（用户特征变化频繁）
   - 从 Redis/Faiss 检索 Item Embedding
   - 计算相似度（向量点积）

```go
// 伪代码示例
// 离线：预计算 Item Embedding
func PrecomputeItemEmbeddings(items []*core.Item) {
    for _, item := range items {
        itemFeatures := extractItemFeatures(item)
        itemEmb := itemTower.Forward(itemFeatures)  // Item Tower 前向传播
        redis.Set("item_emb:"+item.ID, itemEmb)     // 存入 Redis
    }
}

// 在线：快速推理
func Predict(userFeatures map[string]float64, itemID string) float64 {
    // 1. 实时计算 User Embedding
    userEmb := userTower.Forward(userFeatures)
    
    // 2. 从 Redis 获取 Item Embedding
    itemEmb := redis.Get("item_emb:" + itemID)
    
    // 3. 计算相似度
    similarity := dotProduct(userEmb, itemEmb)
    
    // 4. Sigmoid 激活
    return sigmoid(similarity)
}
```

### 4.2 User Embedding 持久化策略

**核心结论：User Embedding 通常不需要持久化，但可以根据业务场景选择性使用。**

#### 4.2.1 通常不需要持久化的原因

1. **用户特征变化频繁**
   - 实时行为（点击、浏览）会改变用户特征
   - 兴趣标签、偏好会随时间变化
   - 实时特征（当前时间、设备信息）每次请求都不同

2. **计算成本相对较低**
   - User Tower 通常比 Item Tower 小（用户特征维度 < 物品特征维度）
   - 单次 User Embedding 计算耗时通常在毫秒级（< 10ms）
   - 相比 Item Embedding 检索，计算成本可接受

3. **存储成本高**
   - 用户数量通常远大于物品数量（百万/千万级用户 vs 万/十万级物品）
   - 持久化需要大量存储空间
   - 需要实时更新机制，增加系统复杂度

#### 4.2.2 什么情况下可以考虑持久化

以下场景可以考虑持久化 User Embedding：

1. **用户特征相对稳定**
   - 用户画像变化不频繁（例如：年龄、性别、地理位置）
   - 主要使用静态特征，实时特征占比低

2. **QPS 极高，延迟要求极低**
   - 需要 < 1ms 的响应时间
   - 即使毫秒级计算也会成为瓶颈

3. **批量召回场景**
   - 一次计算 User Embedding，用于召回大量物品
   - 同一用户的多次请求可以复用 Embedding

4. **用户数量相对较少**
   - 用户规模在万级或十万级
   - 存储成本可控

#### 4.2.3 持久化实现方案

如果决定持久化 User Embedding，可以采用以下方案：

**方案1：短期缓存（推荐）**

```go
// 使用 Redis 缓存 User Embedding（TTL: 5-30分钟）
func GetUserEmbeddingWithCache(userID string, userFeatures map[string]float64) ([]float64, error) {
    // 1. 尝试从缓存获取
    cacheKey := "user_emb:" + userID
    if cached, err := redis.Get(cacheKey); err == nil {
        return cached, nil
    }
    
    // 2. 缓存未命中，实时计算
    userEmb := userTower.Forward(userFeatures)
    
    // 3. 写入缓存（TTL: 10分钟）
    redis.Set(cacheKey, userEmb, 10*60) // 10分钟过期
    
    return userEmb, nil
}
```

**优点**：
- 平衡性能和实时性
- 自动过期，无需手动更新
- 实现简单

**缺点**：
- 用户特征变化后，需要等待缓存过期

**方案2：特征版本化缓存**

```go
// 基于特征版本号的缓存
func GetUserEmbeddingWithVersionCache(
    userID string, 
    userFeatures map[string]float64,
    featureVersion string, // 特征版本号
) ([]float64, error) {
    // 缓存 Key 包含特征版本号
    cacheKey := fmt.Sprintf("user_emb:%s:v%s", userID, featureVersion)
    
    if cached, err := redis.Get(cacheKey); err == nil {
        return cached, nil
    }
    
    // 计算并缓存
    userEmb := userTower.Forward(userFeatures)
    redis.Set(cacheKey, userEmb, 30*60) // 30分钟过期
    
    return userEmb, nil
}
```

**优点**：
- 特征变化时自动失效（版本号变化）
- 缓存命中率高

**缺点**：
- 需要维护特征版本号

**方案3：增量更新**

```go
// 定期批量更新 User Embedding
func BatchUpdateUserEmbeddings(userIDs []string) {
    for _, userID := range userIDs {
        userFeatures := getUserFeatures(userID)
        userEmb := userTower.Forward(userFeatures)
        redis.Set("user_emb:"+userID, userEmb, 24*60*60) // 24小时过期
    }
}

// 在线：优先使用缓存，缓存未命中时实时计算
func GetUserEmbedding(userID string, userFeatures map[string]float64) ([]float64, error) {
    cacheKey := "user_emb:" + userID
    if cached, err := redis.Get(cacheKey); err == nil {
        return cached, nil
    }
    
    // 缓存未命中，实时计算（并异步更新缓存）
    userEmb := userTower.Forward(userFeatures)
    go redis.Set(cacheKey, userEmb, 24*60*60) // 异步更新
    
    return userEmb, nil
}
```

**优点**：
- 大部分请求命中缓存，性能好
- 缓存未命中时仍能实时计算

**缺点**：
- 需要离线批量更新任务
- 系统复杂度增加

#### 4.2.4 推荐策略

**默认策略：不持久化，实时计算**

```go
// 推荐：实时计算 User Embedding
func Predict(userFeatures map[string]float64, itemID string) float64 {
    // 实时计算 User Embedding（通常 < 10ms）
    userEmb := userTower.Forward(userFeatures)
    
    // 从 Redis 获取 Item Embedding（预计算）
    itemEmb := redis.Get("item_emb:" + itemID)
    
    // 计算相似度
    return sigmoid(dotProduct(userEmb, itemEmb))
}
```

**优化策略：短期缓存（5-30分钟）**

```go
// 优化：短期缓存 User Embedding
func PredictWithCache(userID string, userFeatures map[string]float64, itemID string) float64 {
    // 尝试从缓存获取（TTL: 10分钟）
    userEmb := GetUserEmbeddingWithCache(userID, userFeatures)
    
    // 从 Redis 获取 Item Embedding
    itemEmb := redis.Get("item_emb:" + itemID)
    
    // 计算相似度
    return sigmoid(dotProduct(userEmb, itemEmb))
}
```

#### 4.2.5 性能对比

| 方案 | 延迟 | 存储成本 | 实时性 | 复杂度 | 推荐场景 |
|------|------|---------|--------|--------|----------|
| 实时计算 | 5-10ms | 无 | 最高 | 低 | **默认方案** |
| 短期缓存（10分钟） | 1-2ms | 中等 | 高 | 低 | **推荐优化** |
| 长期缓存（24小时） | 1-2ms | 高 | 中 | 中 | 用户特征稳定 |
| 离线批量更新 | 1-2ms | 高 | 低 | 高 | 大规模场景 |

### 4.3 批量推理优化

对于大规模召回场景，可以使用批量推理：

```go
// 批量计算 User Embedding（一次计算，多次使用）
userEmb := userTower.Forward(userFeatures)

// 批量检索 Item Embeddings
itemIDs := []string{"item_1", "item_2", "item_3", ...}
itemEmbs := redis.MGet("item_emb:", itemIDs)

// 批量计算相似度
scores := make([]float64, len(itemEmbs))
for i, itemEmb := range itemEmbs {
    scores[i] = sigmoid(dotProduct(userEmb, itemEmb))
}
```

### 4.4 特征工程优化

1. **特征归一化**：确保数值特征在合理范围内（例如：0-1 或标准化）
2. **特征选择**：选择对模型有贡献的特征，避免噪声特征
3. **特征版本管理**：使用特征版本管理，确保训练和推理特征一致

```go
// 使用特征标准化器
scaler := feature.NewFeatureScaler()
normalizedFeatures := scaler.Normalize(rawFeatures)
```

### 4.5 模型训练建议

虽然 Reckit 是 Golang 推理库，但模型训练通常在 Python 中进行：

1. **训练框架**：使用 PyTorch 或 TensorFlow 训练双塔模型
2. **负采样**：使用负采样策略（例如：随机负采样、困难负采样）
3. **损失函数**：使用对比学习损失（例如：InfoNCE、Triplet Loss）
4. **模型导出**：将训练好的模型权重导出为 ONNX 或自定义格式
5. **模型加载**：在 Golang 中加载模型权重（需要实现模型加载逻辑）

## 五、实现细节说明

### 5.1 当前实现的简化

当前 Reckit 中的双塔模型实现是**简化版本**，主要用于演示和快速原型开发：

```go
// 当前实现（简化版）
func (m *TwoTowerModel) getUserEmbedding(userFeatures map[string]float64) ([]float64, error) {
    score, err := m.UserTower.Predict(userFeatures)
    // 简化处理：将 score 转换为 embedding
    emb := make([]float64, m.EmbeddingDim)
    for i := range emb {
        emb[i] = score * 0.1 // 简单的映射
    }
    return emb, nil
}
```

**实际生产环境应该**：
1. 从 DNN 的**中间层**（最后一层隐藏层）获取 Embedding，而不是从最终输出
2. 使用训练好的模型权重，而不是随机初始化
3. 支持从文件或模型服务加载预训练权重

### 5.2 改进建议

如果需要生产级实现，建议：

1. **修改 DNN 模型**：支持从中间层获取输出
   ```go
   // 建议：在 DNNModel 中添加方法
   func (m *DNNModel) GetEmbedding(features map[string]float64, layerIndex int) ([]float64, error) {
       // 前向传播到指定层，返回该层的输出（不经过激活）
       // ...
   }
   ```

2. **使用预训练模型**：从训练好的模型加载权重
   ```go
   // 从文件加载权重
   func LoadTwoTowerModel(weightPath string) (*TwoTowerModel, error) {
       // 加载 User Tower 和 Item Tower 的权重
       // ...
   }
   ```

3. **集成模型服务**：使用 TensorFlow Serving 或 TorchServe
   ```go
   // 使用模型服务获取 Embedding
   userEmb, err := mlService.GetUserEmbedding(userFeatures)
   itemEmb, err := mlService.GetItemEmbedding(itemFeatures)
   ```

## 六、注意事项

### 6.1 特征一致性

- **训练和推理特征必须一致**：确保训练时使用的特征与在线推理时一致
- **特征顺序**：如果使用固定顺序的特征向量，确保顺序一致
- **缺失值处理**：统一处理缺失值（填充0、均值、或特殊标记）

### 6.2 Embedding 维度

- **User Tower 和 Item Tower 的 Embedding 维度必须一致**
- **维度选择**：通常选择 32、64、128 等维度，根据业务需求平衡精度和性能

### 6.3 相似度计算方式

- **内积（dot）**：计算速度快，适合大规模场景
- **余弦相似度（cosine）**：归一化后计算，对向量长度不敏感

```go
twoTowerModel.SimilarityType = "dot"    // 内积（推荐）
twoTowerModel.SimilarityType = "cosine" // 余弦相似度
```

### 6.4 模型局限性

根据模型选型表，双塔模型的核心缺点是 **无法捕捉 User 和 Item 之间的细粒度特征交互**。

如果需要捕捉细粒度交互，可以考虑：
- **精排阶段**：使用 DeepFM、DIN 等模型（支持特征交互）
- **两阶段推荐**：召回使用双塔模型，精排使用 DeepFM/DIN

## 七、相关文档

- [模型选型指南](./MODEL_SELECTION.md) - 双塔模型选型说明
- [排序模型文档](./RANK_MODELS.md) - 双塔模型使用示例
- [特征处理文档](./FEATURE_PROCESSING.md) - 特征工程指南
- [特征一致性文档](./FEATURE_CONSISTENCY.md) - 训练与在线一致性

## 八、示例代码

完整示例请参考：
- `examples/rank_models/main.go` - 双塔模型基础示例
- `examples/feature_service/main.go` - 特征服务示例
- `examples/feature_processing/main.go` - 特征处理示例
