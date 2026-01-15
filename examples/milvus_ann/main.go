package main

import (
	"context"
	"fmt"
	"time"

	"reckit/core"
	"reckit/pipeline"
	"reckit/recall"
	"reckit/vector"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// ========== 1. 创建 Milvus 服务 ==========
	fmt.Println("=== 创建 Milvus 服务 ===")
	milvusService := vector.NewMilvusService(
		"localhost:19530",
		vector.WithMilvusAuth("root", "Milvus"),
		vector.WithMilvusDatabase("recommend"),
		vector.WithMilvusTimeout(30),
	)
	defer milvusService.Close()

	// ========== 2. 创建集合 ==========
	fmt.Println("\n=== 创建集合 ===")
	collectionName := "items"
	dimension := 128

	// 检查集合是否存在
	exists, err := milvusService.HasCollection(ctx, collectionName)
	if err != nil {
		fmt.Printf("检查集合失败: %v\n", err)
	} else if !exists {
		// 创建集合
		err = milvusService.CreateCollection(ctx, &vector.CreateCollectionRequest{
			Name:      collectionName,
			Dimension: dimension,
			Metric:    "cosine",
		})
		if err != nil {
			fmt.Printf("创建集合失败: %v\n", err)
			return
		}
		fmt.Printf("集合 %s 创建成功\n", collectionName)
	} else {
		fmt.Printf("集合 %s 已存在\n", collectionName)
	}

	// ========== 3. 插入向量（示例） ==========
	fmt.Println("\n=== 插入向量 ===")
	// 生成示例向量
	vectors := [][]float64{
		generateRandomVector(dimension),
		generateRandomVector(dimension),
		generateRandomVector(dimension),
	}
	itemIDs := []int64{1, 2, 3}

	err = milvusService.Insert(ctx, &vector.InsertRequest{
		Collection: collectionName,
		Vectors:    vectors,
		IDs:        itemIDs,
	})
	if err != nil {
		fmt.Printf("插入向量失败: %v\n", err)
	} else {
		fmt.Printf("成功插入 %d 个向量\n", len(vectors))
	}

	// ========== 4. 向量搜索 ==========
	fmt.Println("\n=== 向量搜索 ===")
	userVector := generateRandomVector(dimension)

	searchResult, err := milvusService.Search(ctx, &vector.SearchRequest{
		Collection: collectionName,
		Vector:     userVector,
		TopK:       10,
		Metric:     "cosine",
	})
	if err != nil {
		fmt.Printf("向量搜索失败: %v\n", err)
	} else {
		fmt.Printf("搜索到 %d 个结果:\n", len(searchResult.IDs))
		for i, id := range searchResult.IDs {
			fmt.Printf("  %d. 物品 %d (相似度: %.4f)\n", i+1, id, searchResult.Scores[i])
		}
	}

	// ========== 5. 与 recall.ANN 集成 ==========
	fmt.Println("\n=== 与 recall.ANN 集成 ===")
	adapter := vector.NewVectorStoreAdapter(milvusService, collectionName)

	// 注意：由于 VectorStoreAdapter 的 GetVector 和 ListVectors 返回错误
	// 需要修改 recall.ANN 以支持直接使用 ANNService.Search
	// 或者创建一个增强版的 ANN 实现

	// 创建推荐上下文
	rctx := &core.RecommendContext{
		UserID: 1,
		Scene:  "feed",
		UserProfile: map[string]any{
			"user_vector": userVector,
		},
	}

	// 使用 ANN 召回（注意：当前实现会失败，因为 ListVectors 不支持）
	ann := &recall.ANN{
		Store:      adapter,
		TopK:       10,
		Metric:     "cosine",
		UserVector: userVector,
	}

	items, err := ann.Recall(ctx, rctx)
	if err != nil {
		fmt.Printf("ANN 召回失败: %v\n", err)
		fmt.Println("提示：当前 VectorStoreAdapter 的 ListVectors 不支持，")
		fmt.Println("      建议直接使用 ANNService.Search 或修改 recall.ANN 实现")
	} else {
		fmt.Printf("ANN 召回成功，返回 %d 个物品\n", len(items))
		for i, item := range items {
			fmt.Printf("  %d. 物品 %d (分数: %.4f)\n", i+1, item.ID, item.Score)
		}
	}

	// ========== 6. Pipeline 集成示例 ==========
	fmt.Println("\n=== Pipeline 集成示例 ===")
	// 注意：由于适配器限制，这里只是示例代码结构
	// 实际使用时需要修改 recall.ANN 以支持 ANNService

	p := &pipeline.Pipeline{
		Nodes: []pipeline.Node{
			&recall.Fanout{
				Sources: []recall.Source{
					// 使用 Milvus 的 ANN 召回
					ann,
				},
				Dedup: true,
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

// generateRandomVector 生成随机向量（用于示例）
func generateRandomVector(dimension int) []float64 {
	vec := make([]float64, dimension)
	for i := range vec {
		// 简单的伪随机数生成（实际应该使用真正的随机数）
		vec[i] = float64(i%10) / 10.0
	}
	return vec
}
