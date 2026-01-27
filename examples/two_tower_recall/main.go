package main

import (
	"context"
	"fmt"
	"time"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/feature"
	"github.com/rushteam/reckit/pipeline"
	"github.com/rushteam/reckit/recall"
	"github.com/rushteam/reckit/service"
	"github.com/rushteam/reckit/store"
	milvus "github.com/rushteam/reckit/ext/vector/milvus"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// ========== 1. 创建用户画像 ==========
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

	// ========== 2. 创建特征服务 ==========
	// 方式1：使用存储特征提供者（示例，从内存存储提取）
	// 实际使用时，应该使用 Redis、HTTP 等特征服务
	// 注意：store.NewMemoryStore() 返回的是 core.Store 接口的实现
	memStore := store.NewMemoryStore()
	storeProvider := feature.NewStoreFeatureProvider(memStore, feature.KeyPrefix{})
	featureService := feature.NewBaseFeatureService(storeProvider)

	// 方式2：使用 HTTP 特征服务
	// httpProvider := feature.NewHTTPFeatureProvider("http://localhost:8080/features", 5*time.Second)
	// featureService := feature.NewBaseFeatureService(httpProvider)

	// 方式3：使用 Redis 特征服务
	// redisProvider := feature.NewRedisFeatureProvider(...)
	// featureService := feature.NewBaseFeatureService(redisProvider)

	// ========== 3. 创建用户塔推理服务 ==========
	// 方式1：使用 TorchServe（推荐，支持 PyTorch 模型）
	// 注意：service.NewTorchServeClient() 返回的是 core.MLService 接口的实现
	userTowerService := service.NewTorchServeClient(
		"http://localhost:8080", // TorchServe REST API 端点
		"user_tower",            // 模型名称
		service.WithTorchServeTimeout(5*time.Second),
	)

	// 方式2：使用 TensorFlow Serving
	// userTowerService := service.NewTFServingClient(...)

	// 方式3：使用自定义 ONNX Runtime 服务
	// userTowerService := service.NewCustomMLService(...)

	// ========== 4. 创建向量检索服务 ==========
	// 方式1：使用 Milvus（推荐）
	milvusService := milvus.NewMilvusService(
		"localhost:19530", // Milvus 地址
		milvus.WithMilvusDatabase("default"),
		milvus.WithMilvusTimeout(10),
	)

	// 方式2：使用 Faiss（需要 CGO 绑定）
	// faissService := vector.NewFaissService(...)

	// ========== 5. 使用向量服务 ==========
	// MilvusService 直接实现了 core.VectorService 接口（通过接口组合）
	// 可以直接作为 core.VectorService 使用，无需包装器
	var vectorService core.VectorService = milvusService

	// ========== 6. 创建双塔召回源 ==========
	twoTowerRecall := recall.NewTwoTowerRecall(
		featureService,
		userTowerService,
		vectorService,
		recall.WithTwoTowerTopK(100), // 返回 Top 100
		recall.WithTwoTowerCollection("item_embeddings"), // 向量数据库集合名称
		recall.WithTwoTowerMetric("inner_product"),       // 使用内积（适合双塔模型）
	)

	// ========== 7. 在 Pipeline 中使用 ==========
	p := &pipeline.Pipeline{
		Nodes: []pipeline.Node{
			// 双塔召回
			&recall.Fanout{
				Sources: []recall.Source{
					twoTowerRecall,
					// 可以与其他召回源组合
					&recall.Hot{IDs: []string{"item_1", "item_2", "item_3"}},
				},
				Dedup: true,
			},
			// 后续可以添加过滤、排序等节点
		},
	}

	// ========== 8. 运行 Pipeline ==========
	items, err := p.Run(ctx, rctx, nil)
	if err != nil {
		fmt.Printf("Pipeline error: %v\n", err)
		return
	}

	// ========== 9. 输出结果 ==========
	fmt.Printf("=== 双塔召回结果 ===\n")
	fmt.Printf("用户: %s\n", rctx.UserID)
	fmt.Printf("召回物品数量: %d\n\n", len(items))

	for i, item := range items {
		if i >= 10 { // 只显示前 10 个
			break
		}
		fmt.Printf("%d. Item %s (Score: %.4f)", i+1, item.ID, item.Score)
		if item.Labels != nil {
			if source, ok := item.Labels["recall_source"]; ok {
				fmt.Printf(" [%s]", source.Value)
			}
		}
		fmt.Println()
	}
}

// ========== 完整流程说明 ==========
//
// 线上推理流程：
//
// 1. 获取用户特征 (Golang 强项：高并发)
//    features := featureService.GetUserFeatures(uid)
//
// 2. 运行用户塔推理 (通过 TorchServe/ONNX Runtime)
//    userVector := userTowerService.Predict(features)
//
// 3. 向量检索 (通过 Milvus/Faiss)
//    itemIDs := vectorService.Search(userVector, 100)
//
// 4. 返回召回结果
//    return itemIDs
//
// ========== 架构设计 ==========
//
// - 高内聚：TwoTowerRecall 只负责协调流程，具体逻辑在各服务中
// - 低耦合：通过接口依赖，可替换实现
// - DDD 模式：每个服务都是独立的领域服务
//   - FeatureService：特征领域服务
//   - MLService：模型推理领域服务
//   - ANNService：向量检索领域服务
//
// ========== 扩展性 ==========
//
// 1. 特征服务：可以替换为 Redis、HTTP、Feast 等实现
// 2. 模型服务：可以替换为 TensorFlow Serving、ONNX Runtime 等实现
// 3. 向量服务：可以替换为 Faiss、Elasticsearch 等实现
//
// ========== 性能优化 ==========
//
// 1. 特征服务：支持批量获取（BatchGetUserFeatures）
// 2. 模型服务：支持批量推理（PredictRequest.Instances）
// 3. 向量服务：支持批量检索（SearchRequest.TopK）
//
// ========== 注意事项 ==========
//
// 1. 特征顺序：确保特征顺序与训练时一致（使用 Features 字典格式可避免此问题）
// 2. Embedding 维度：确保 User Embedding 和 Item Embedding 维度一致
// 3. 距离度量：双塔模型通常使用内积（inner_product）或余弦相似度（cosine）
// 4. 模型部署：Item Embedding 需要离线预计算并存入向量数据库

// 注意：不再需要适配器！
// MilvusService 直接实现了 core.VectorService 接口，
// 可以直接传递给 recall.NewTwoTowerRecall
