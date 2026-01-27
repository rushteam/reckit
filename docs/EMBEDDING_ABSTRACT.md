# Embedding 能力抽象

## 概述

项目提供了基于 Embedding 向量的近似最近邻（ANN）召回能力，支持通过向量相似度计算实现个性化物品召回。

## 核心组件

### 1. ANN 召回器 (`recall.ANN`)

**位置**: `recall/ann.go`

**功能**:
- 基于用户向量和物品向量的相似度计算进行召回
- 支持多种相似度度量方式（余弦相似度、欧氏距离）
- 支持 TopK 检索
- 可集成到 Pipeline 中作为召回源

**核心结构**:
```go
type ANN struct {
    Store      VectorStore  // 向量存储接口
    Key        string       // 向量索引 key（预留）
    UserVector []float64    // 用户向量（可选，优先使用）
    TopK       int          // 返回 TopK 相似物品
    Metric     string       // 距离度量：cosine / euclidean
    
    // 用户向量提取器（可选）
    UserVectorExtractor func(rctx *core.RecommendContext) []float64
}
```

### 2. VectorStore 接口

**位置**: `recall/ann.go`

**定义**:
```go
type VectorStore interface {
    GetVector(ctx context.Context, itemID int64) ([]float64, error)
    ListVectors(ctx context.Context) (map[int64][]float64, error)
}
```

**职责**:
- 提供物品向量的存储和检索能力
- 支持单个向量查询和批量向量查询
- 可扩展实现（Redis、内存、向量数据库如 Faiss、Milvus 等）

## 用户向量获取策略

支持三种方式获取用户向量，优先级从高到低：

1. **直接提供** (`UserVector`): 在创建 ANN 时直接指定用户向量
2. **自定义提取器** (`UserVectorExtractor`): 通过函数从 `RecommendContext` 中提取
3. **默认提取** (`rctx.UserProfile["user_vector"]`): 从上下文的用户画像中获取

**示例**:
```go
// 方式1: 直接提供
ann := &recall.ANN{
    UserVector: []float64{0.1, 0.2, 0.3},
    // ...
}

// 方式2: 自定义提取器
ann := &recall.ANN{
    UserVectorExtractor: func(rctx *core.RecommendContext) []float64 {
        // 自定义提取逻辑
        return extractUserVector(rctx)
    },
    // ...
}

// 方式3: 从 Context 获取
rctx := &core.RecommendContext{
    UserProfile: map[string]any{
        "user_vector": []float64{0.1, 0.2, 0.3},
    },
}
```

## 相似度计算

### 支持的度量方式

1. **余弦相似度** (`cosine`): 默认方式
   - 计算公式: `cos(θ) = (A·B) / (||A|| * ||B||)`
   - 范围: [-1, 1]，值越大越相似
   - 适用于方向相似性判断

2. **欧氏距离** (`euclidean`): 
   - 计算公式: `distance = sqrt(Σ(Ai - Bi)²)`
   - 转换为相似度: `similarity = 1 / (1 + distance)`
   - 适用于绝对距离判断

### 实现细节

- 余弦相似度: `cosineSimilarity()` 函数
- 欧氏距离: `euclideanDistance()` 函数
- TopK 选择: 使用选择排序（生产环境建议使用堆优化）

## 集成方式

### 作为召回源使用

ANN 实现了 `recall.Source` 接口，可以：

#### 1. 基础用法（单独使用）

```go
import (
    "context"
    "time"
    
    "github.com/rushteam/reckit/core"
    "github.com/rushteam/reckit/recall"
    "github.com/rushteam/reckit/vector"
)

// 创建 Milvus 服务（扩展包）
import milvus "github.com/rushteam/reckit/ext/vector/milvus"
// 安装：go get github.com/rushteam/reckit/ext/vector/milvus

milvusService := milvus.NewMilvusService(
    "localhost:19530",
    milvus.WithMilvusAuth("root", "Milvus"),
    milvus.WithMilvusDatabase("recommend"),
    milvus.WithMilvusTimeout(30),
)
defer milvusService.Close()

// 创建适配器
adapter := vector.NewVectorStoreAdapter(milvusService, "items")

// 创建推荐上下文（包含用户向量）
userVector := []float64{0.1, 0.2, 0.3, ...} // 用户向量
rctx := &core.RecommendContext{
    UserID: "user_123",
    Scene:  "feed",
    UserProfile: map[string]any{
        "user_vector": userVector,
    },
}

// 创建 ANN 召回
ann := &recall.ANN{
    Store:      adapter,
    TopK:       20,
    Metric:     "cosine",
    UserVector: userVector, // 直接提供用户向量
}

// 执行召回
ctx := context.Background()
items, err := ann.Recall(ctx, rctx)
if err != nil {
    log.Fatal(err)
}

// 使用召回结果
for _, item := range items {
    fmt.Printf("Item: %s, Score: %.4f\n", item.ID, item.Score)
}
```

#### 2. 集成到 Fanout（多路召回）

```go
import (
    "github.com/rushteam/reckit/recall"
    "github.com/rushteam/reckit/vector"
)

// 创建向量服务适配器（扩展包）
import milvus "github.com/rushteam/reckit/ext/vector/milvus"
// 安装：go get github.com/rushteam/reckit/ext/vector/milvus

milvusService := milvus.NewMilvusService("localhost:19530")
defer milvusService.Close()
adapter := vector.NewVectorStoreAdapter(milvusService, "items")

// 创建多种召回源
hotRecall := &recall.Hot{IDs: []string{"1", "2", "3", "4", "5"}}
annRecall := &recall.ANN{
    Store:      adapter,
    TopK:       20,
    Metric:     "cosine",
    UserVector: userVector,
}
i2iRecall := &recall.I2IRecall{
    Store:            cfStore,
    TopKSimilarItems: 10,
    TopKItems:        20,
    SimilarityCalculator: &recall.CosineSimilarity{},
    Config:            config,
}

// 创建 Fanout（多路并发召回）
fanout := &recall.Fanout{
    Sources: []recall.Source{
        hotRecall,   // 热门召回（优先级 0）
        annRecall,   // Embedding 召回（优先级 1）
        i2iRecall,   // Item-CF 召回（优先级 2）
    },
    Dedup:         true,
    Timeout:       2 * time.Second,
    MaxConcurrent: 5,
    MergeStrategy: &recall.PriorityMergeStrategy{},
}

// 执行多路召回
items, err := fanout.Process(ctx, rctx, nil)
```

#### 3. 集成到 Pipeline（完整推荐流程）

```go
import (
    "github.com/rushteam/reckit/core"
    "github.com/rushteam/reckit/pipeline"
    "github.com/rushteam/reckit/recall"
    "github.com/rushteam/reckit/vector"
)

// 创建向量服务（扩展包）
import milvus "github.com/rushteam/reckit/ext/vector/milvus"
// 安装：go get github.com/rushteam/reckit/ext/vector/milvus

milvusService := milvus.NewMilvusService("localhost:19530")
defer milvusService.Close()
adapter := vector.NewVectorStoreAdapter(milvusService, "items")

// 创建推荐上下文
rctx := &core.RecommendContext{
    UserID: "user_123",
    Scene:  "feed",
    UserProfile: map[string]any{
        "user_vector": userVector,
        "age":          25.0,
        "gender":       1.0,
    },
    Realtime: map[string]any{
        "hour":   float64(time.Now().Hour()),
        "device": "mobile",
    },
}

// 创建 Pipeline
p := &pipeline.Pipeline{
    Nodes: []pipeline.Node{
        // 多路召回
        &recall.Fanout{
            Sources: []recall.Source{
                &recall.Hot{IDs: []string{"1", "2", "3"}},
                &recall.ANN{
                    Store:      adapter,
                    TopK:       20,
                    Metric:     "cosine",
                    UserVector: userVector,
                },
            },
            Dedup:         true,
            Timeout:       2 * time.Second,
            MaxConcurrent: 5,
            MergeStrategy: &recall.PriorityMergeStrategy{},
        },
        // 过滤节点
        &filter.FilterNode{
            Filters: []filter.Filter{
                filter.NewUserBlockFilter(nil, "user:block"),
                filter.NewExposedFilter(nil, "user:exposed", 7*24*3600),
            },
        },
        // 特征注入
        &feature.EnrichNode{
            UserFeaturePrefix:  "user_",
            ItemFeaturePrefix:  "item_",
            CrossFeaturePrefix: "cross_",
        },
        // 排序节点
        &rank.LRNode{
            Bias: 0.0,
            Weights: map[string]float64{
                "ctr": 1.2,
                "cvr": 0.8,
            },
        },
        // 重排节点
        &rerank.Diversity{LabelKey: "category"},
    },
}

// 运行 Pipeline
items, err := p.Run(ctx, rctx, nil)
```

#### 4. 使用 Milvus 服务的完整示例

```go
package main

import (
    "context"
    "fmt"
    "time"

    "github.com/rushteam/reckit/core"
    "github.com/rushteam/reckit/pipeline"
    "github.com/rushteam/reckit/recall"
    "github.com/rushteam/reckit/vector"
)

func main() {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    // 1. 创建 Milvus 服务（扩展包）
    import milvus "github.com/rushteam/reckit/ext/vector/milvus"
    // 安装：go get github.com/rushteam/reckit/ext/vector/milvus

    milvusService := milvus.NewMilvusService(
        "localhost:19530",
        milvus.WithMilvusAuth("root", "Milvus"),
        milvus.WithMilvusDatabase("recommend"),
        milvus.WithMilvusTimeout(30),
    )
    defer milvusService.Close()

    // 2. 检查并创建集合
    collectionName := "items"
    dimension := 128
    exists, _ := milvusService.HasCollection(ctx, collectionName)
    if !exists {
        milvusService.CreateCollection(ctx, &vector.CreateCollectionRequest{
            Name:      collectionName,
            Dimension: dimension,
            Metric:    "cosine",
        })
    }

    // 3. 插入向量（示例）
    vectors := [][]float64{
        generateRandomVector(dimension),
        generateRandomVector(dimension),
        generateRandomVector(dimension),
    }
    itemIDs := []string{"1", "2", "3"}
    milvusService.Insert(ctx, &vector.InsertRequest{
        Collection: collectionName,
        Vectors:    vectors,
        IDs:        itemIDs,
    })

    // 4. 创建适配器
    adapter := vector.NewVectorStoreAdapter(milvusService, collectionName)

    // 5. 创建推荐上下文
    userVector := generateRandomVector(dimension)
    rctx := &core.RecommendContext{
        UserID: "1",
        Scene:  "feed",
        UserProfile: map[string]any{
            "user_vector": userVector,
        },
    }

    // 6. 使用 ANN 召回
    ann := &recall.ANN{
        Store:      adapter,
        TopK:       10,
        Metric:     "cosine",
        UserVector: userVector,
    }

    items, err := ann.Recall(ctx, rctx)
    if err != nil {
        fmt.Printf("ANN 召回失败: %v\n", err)
        return
    }

    fmt.Printf("ANN 召回成功，返回 %d 个物品\n", len(items))
    for i, item := range items {
        fmt.Printf("  %d. 物品 %s (分数: %.4f)\n", i+1, item.ID, item.Score)
    }

    // 7. Pipeline 集成
    p := &pipeline.Pipeline{
        Nodes: []pipeline.Node{
            &recall.Fanout{
                Sources: []recall.Source{ann},
                Dedup:   true,
            },
        },
    }

    items, err = p.Run(ctx, rctx, nil)
    if err != nil {
        fmt.Printf("Pipeline 执行失败: %v\n", err)
    } else {
        fmt.Printf("Pipeline 执行成功，返回 %d 个物品\n", len(items))
    }
}

func generateRandomVector(dimension int) []float64 {
    vec := make([]float64, dimension)
    for i := range vec {
        vec[i] = float64(i%10) / 10.0
    }
    return vec
}
```

#### 5. 使用自定义用户向量提取器

```go
// 自定义用户向量提取器
userVectorExtractor := func(rctx *core.RecommendContext) []float64 {
    // 从多个来源组合用户向量
    if rctx.UserProfile == nil {
        return nil
    }
    
    // 方式1: 直接从 UserProfile 获取
    if uv, ok := rctx.UserProfile["user_vector"]; ok {
        if vec, ok := uv.([]float64); ok {
            return vec
        }
    }
    
    // 方式2: 从特征计算用户向量
    // 例如：使用用户特征（年龄、性别等）生成向量
    age, _ := rctx.UserProfile["age"].(float64)
    gender, _ := rctx.UserProfile["gender"].(float64)
    
    // 这里可以使用模型或规则生成向量
    return []float64{age / 100.0, gender, ...}
}

// 使用自定义提取器
ann := &recall.ANN{
    Store:              adapter,
    TopK:               20,
    Metric:             "cosine",
    UserVectorExtractor: userVectorExtractor, // 使用自定义提取器
}
```

## 输出结果

召回结果包含以下信息：

- **Item ID**: 物品标识
- **Score**: 相似度分数
- **Labels**:
  - `recall_source`: "ann"
  - `ann_metric`: 使用的度量方式（"cosine" 或 "euclidean"）

## 扩展点

### 1. VectorStore 实现

可以扩展实现不同的向量存储后端：

- **内存实现**: 适用于小规模数据或测试
- **Redis 实现**: 利用 Redis 的 Hash 或 String 存储向量
- **专业向量数据库**: 集成 Faiss、Milvus、Qdrant 等，支持高效 ANN 检索

### 2. 优化方向

当前实现为简化版本，生产环境可优化：

- **索引优化**: 使用向量索引（如 HNSW、IVF）替代全量扫描
- **TopK 优化**: 使用堆排序替代选择排序
- **并发优化**: 支持向量检索的并发处理
- **缓存优化**: 对常用向量进行缓存

## 使用场景

1. **个性化推荐**: 基于用户 Embedding 召回相似物品
2. **内容相似性**: 基于物品 Embedding 进行相似物品推荐
3. **冷启动**: 新用户/新物品的向量化召回
4. **多模态推荐**: 支持文本、图像等 Embedding 的召回

## 注意事项

1. **向量维度**: 确保用户向量和物品向量维度一致
2. **向量归一化**: 建议对向量进行归一化处理，特别是使用余弦相似度时
3. **性能考虑**: 
   - 优先使用 `Search` 方法（高性能，支持向量索引）
   - 如果 `Search` 失败，会回退到暴力搜索（适用于小规模数据）
   - 大规模数据建议使用专业向量数据库（如 Milvus）
4. **向量更新**: 需要建立向量更新机制，保证向量数据的时效性
5. **接口限制**: 
   - `VectorStoreAdapter.GetVector` 和 `ListVectors` 返回 `ErrNotSupported`
   - 原因：向量数据库通常不支持这些操作
   - 解决方案：使用 `Search` 方法进行向量检索

## 参考资源

- **完整示例**: `examples/milvus_ann/main.go`
- **向量服务文档**: `vector/README.md`
- **召回算法文档**: `RECALL_ALGORITHMS.md`