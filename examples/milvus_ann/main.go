package main

import (
	"context"
	"fmt"
	"time"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pipeline"
	"github.com/rushteam/reckit/recall"

	milvus "github.com/rushteam/reckit/ext/vector/milvus"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// ========== 1. 创建 Milvus 服务 ==========
	fmt.Println("=== 创建 Milvus 服务 ===")
	milvusService, err := milvus.NewMilvusService(
		"localhost:19530",
		milvus.WithMilvusAuth("root", "Milvus"),
		milvus.WithMilvusDatabase("recommend"),
		milvus.WithMilvusTimeout(30),
	)
	if err != nil {
		fmt.Printf("创建 Milvus 服务失败: %v\n", err)
		return
	}
	defer milvusService.Close(ctx)

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
		err = milvusService.CreateCollection(ctx, &core.VectorCreateCollectionRequest{
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
	itemIDs := []string{"1", "2", "3"}

	err = milvusService.Insert(ctx, &core.VectorInsertRequest{
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

	searchResult, err := milvusService.Search(ctx, &core.VectorSearchRequest{
		Collection: collectionName,
		Vector:     userVector,
		TopK:       10,
		Metric:     "cosine",
	})
	if err != nil {
		fmt.Printf("向量搜索失败: %v\n", err)
	} else {
		fmt.Printf("搜索到 %d 个结果:\n", len(searchResult.Items))
		for i, item := range searchResult.Items {
			fmt.Printf("  %d. 物品 %s (相似度: %.4f)\n", i+1, item.ID, item.Score)
		}
	}

	// ========== 5. 与 recall.ANN 集成 ==========
	fmt.Println("\n=== 与 recall.ANN 集成 ===")

	// 创建推荐上下文
	rctx := &core.RecommendContext{
		UserID: "1",
		Scene:  "feed",
		UserProfile: map[string]any{
			"user_embedding": userVector,
		},
	}

	// 使用 ANN 召回（直接使用 core.VectorService）
	ann := &recall.ANN{
		VectorService: milvusService, // 直接使用 Milvus 服务（实现了 core.VectorService）
		Collection:    collectionName,
		TopK:           10,
		Metric:         "cosine",
		UserEmbedding:  userVector,
	}

	items, err := ann.Recall(ctx, rctx)
	if err != nil {
		fmt.Printf("ANN 召回失败: %v\n", err)
	} else {
		fmt.Printf("ANN 召回成功，返回 %d 个物品\n", len(items))
		for i, item := range items {
			fmt.Printf("  %d. 物品 %s (分数: %.4f)\n", i+1, item.ID, item.Score)
		}
	}

	// ========== 6. Pipeline 集成示例 ==========
	fmt.Println("\n=== Pipeline 集成示例 ===")

	p := &pipeline.Pipeline{
		Nodes: []pipeline.Node{
			&recall.Fanout{
				Sources: []recall.Source{
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

func generateRandomVector(dimension int) []float64 {
	vec := make([]float64, dimension)
	for i := range vec {
		vec[i] = float64(i%10) / 10.0
	}
	return vec
}
