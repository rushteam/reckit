package main

import (
	"context"
	"fmt"
	"time"

	"reckit/core"
	"reckit/pipeline"
	"reckit/recall"
	"reckit/store"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// 1. 创建内存存储和协同过滤适配器
	memStore := store.NewMemoryStore()
	defer memStore.Close()
	cfStore := recall.NewStoreCFAdapter(memStore, "cf")

	// 2. 模拟用户-物品交互数据
	setupTestData(ctx, cfStore)

	// 3. 创建用户协同过滤召回源
	userCF := &recall.UserBasedCF{
		Store:            cfStore,
		TopKSimilarUsers: 10,
		TopKItems:        5,
		SimilarityMetric: "cosine",
		MinCommonItems:   2,
	}

	// 4. 创建物品协同过滤召回源（i2i）
	i2i := &recall.I2IRecall{
		Store:            cfStore,
		TopKSimilarItems: 10,
		TopKItems:        5,
		SimilarityMetric: "cosine",
		MinCommonUsers:   2,
	}

	// 5. 测试用户协同过滤
	fmt.Println("=== 用户协同过滤 (User-Based CF) ===")
	testUserCF(ctx, userCF)

	// 6. 测试物品协同过滤（i2i）
	fmt.Println("\n=== 物品协同过滤 (Item-Based CF / i2i) ===")
	testItemCF(ctx, i2i)

	// 7. 集成到 Pipeline
	fmt.Println("\n=== 集成到 Pipeline ===")
	testPipeline(ctx, userCF, i2i)
}

func setupTestData(ctx context.Context, cfStore *recall.StoreCFAdapter) {
	interactions := []struct {
		UserID string
		ItemID string
		Score  float64
	}{
		{"1", "1", 5.0}, {"1", "2", 4.0}, {"1", "3", 5.0},
		{"2", "2", 5.0}, {"2", "3", 4.0}, {"2", "4", 5.0}, {"2", "1", 4.0},
		{"3", "4", 4.0}, {"3", "5", 5.0}, {"3", "6", 4.0}, {"3", "3", 3.0},
		{"4", "1", 4.0}, {"4", "3", 5.0}, {"4", "5", 4.0}, {"4", "4", 3.0},
		{"5", "1", 5.0}, {"5", "2", 5.0}, {"5", "3", 4.0}, {"5", "7", 5.0}, {"5", "4", 4.0},
	}

	if err := recall.SetupCFTestData(ctx, cfStore, interactions); err != nil {
		panic(fmt.Sprintf("设置测试数据失败: %v", err))
	}
}

func testUserCF(ctx context.Context, userCF *recall.UserBasedCF) {
	rctx := &core.RecommendContext{
		UserID: "1",
		Scene:  "feed",
	}

	items, err := userCF.Recall(ctx, rctx)
	if err != nil {
		fmt.Printf("错误: %v\n", err)
		return
	}

	fmt.Printf("为用户 %s 推荐:\n", rctx.UserID)
	for i, item := range items {
		fmt.Printf("  %d. 物品 %s (分数: %.4f, 来源: %s)\n",
			i+1, item.ID, item.Score, getLabelValue(item, "recall_source"))
	}
}

func testItemCF(ctx context.Context, i2i *recall.I2IRecall) {
	rctx := &core.RecommendContext{
		UserID: "1",
		Scene:  "feed",
	}

	items, err := i2i.Recall(ctx, rctx)
	if err != nil {
		fmt.Printf("错误: %v\n", err)
		return
	}

	fmt.Printf("为用户 %s 推荐 (基于物品相似度):\n", rctx.UserID)
	for i, item := range items {
		fmt.Printf("  %d. 物品 %s (分数: %.4f, 来源: %s)\n",
			i+1, item.ID, item.Score, getLabelValue(item, "recall_source"))
	}
}

func testPipeline(ctx context.Context, userCF *recall.UserBasedCF, i2i *recall.I2IRecall) {
	p := &pipeline.Pipeline{
		Nodes: []pipeline.Node{
			&recall.Fanout{
				Sources: []recall.Source{
					userCF,
					i2i,
				},
				Dedup:             true,
				MergeStrategyName: "priority",
			},
		},
	}

	rctx := &core.RecommendContext{
		UserID: "1",
		Scene:  "feed",
	}

	items, err := p.Run(ctx, rctx, nil)
	if err != nil {
		fmt.Printf("Pipeline 执行错误: %v\n", err)
		return
	}

	fmt.Printf("Pipeline 推荐结果 (用户 %s):\n", rctx.UserID)
	for i, item := range items {
		fmt.Printf("  %d. 物品 %s (分数: %.4f, 来源: %s)\n",
			i+1, item.ID, item.Score, getLabelValue(item, "recall_source"))
	}
}

func getLabelValue(item *core.Item, key string) string {
	if item.Labels != nil {
		if label, ok := item.Labels[key]; ok {
			return label.Value
		}
	}
	return "unknown"
}
