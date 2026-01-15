package main

import (
	"context"
	"fmt"
	"time"

	"reckit/core"
	"reckit/pipeline"
	"reckit/recall"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// 1. 创建内存协同过滤存储
	cfStore := recall.NewMemoryCFStore()

	// 2. 模拟用户-物品交互数据
	// 用户1: 喜欢物品 1, 2, 3
	// 用户2: 喜欢物品 2, 3, 4 (与用户1有共同兴趣)
	// 用户3: 喜欢物品 4, 5, 6 (与用户1、2兴趣不同)
	// 用户4: 喜欢物品 1, 3, 5 (与用户1有部分共同兴趣)
	setupTestData(cfStore)

	// 3. 创建用户协同过滤召回源
	userCF := &recall.UserBasedCF{
		Store:            cfStore,
		TopKSimilarUsers: 10, // 考虑 Top 10 相似用户
		TopKItems:        5,  // 返回 Top 5 物品
		SimilarityMetric: "cosine",
		MinCommonItems:   2, // 至少 2 个共同物品
	}

	// 4. 创建物品协同过滤召回源（i2i）
	// 使用 I2IRecall 别名，更符合工业习惯
	i2i := &recall.I2IRecall{
		Store:            cfStore,
		TopKSimilarItems: 10, // 考虑 Top 10 相似物品
		TopKItems:        5,  // 返回 Top 5 物品
		SimilarityMetric: "cosine",
		MinCommonUsers:   2, // 至少 2 个共同用户
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

func setupTestData(cfStore *recall.MemoryCFStore) {
	// 用户1的交互
	cfStore.AddInteraction(1, 1, 5.0) // 用户1 对物品1 评分5
	cfStore.AddInteraction(1, 2, 4.0) // 用户1 对物品2 评分4
	cfStore.AddInteraction(1, 3, 5.0) // 用户1 对物品3 评分5

	// 用户2的交互（与用户1有共同兴趣）
	cfStore.AddInteraction(2, 2, 5.0) // 用户2 对物品2 评分5
	cfStore.AddInteraction(2, 3, 4.0) // 用户2 对物品3 评分4
	cfStore.AddInteraction(2, 4, 5.0) // 用户2 对物品4 评分5
	cfStore.AddInteraction(2, 1, 4.0) // 用户2 对物品1 评分4（增加共同用户）

	// 用户3的交互（兴趣不同）
	cfStore.AddInteraction(3, 4, 4.0) // 用户3 对物品4 评分4
	cfStore.AddInteraction(3, 5, 5.0) // 用户3 对物品5 评分5
	cfStore.AddInteraction(3, 6, 4.0) // 用户3 对物品6 评分4
	cfStore.AddInteraction(3, 3, 3.0) // 用户3 对物品3 评分3（增加共同用户）

	// 用户4的交互（与用户1有部分共同兴趣）
	cfStore.AddInteraction(4, 1, 4.0) // 用户4 对物品1 评分4
	cfStore.AddInteraction(4, 3, 5.0) // 用户4 对物品3 评分5
	cfStore.AddInteraction(4, 5, 4.0) // 用户4 对物品5 评分4
	cfStore.AddInteraction(4, 4, 3.0) // 用户4 对物品4 评分3（增加共同用户）

	// 用户5的交互（与用户1、2高度相似）
	cfStore.AddInteraction(5, 1, 5.0) // 用户5 对物品1 评分5
	cfStore.AddInteraction(5, 2, 5.0) // 用户5 对物品2 评分5
	cfStore.AddInteraction(5, 3, 4.0) // 用户5 对物品3 评分4
	cfStore.AddInteraction(5, 7, 5.0) // 用户5 对物品7 评分5（新物品）
	cfStore.AddInteraction(5, 4, 4.0) // 用户5 对物品4 评分4（增加共同用户）
}

func testUserCF(ctx context.Context, userCF *recall.UserBasedCF) {
	// 为用户1推荐（用户1已经交互过物品1,2,3）
	rctx := &core.RecommendContext{
		UserID: 1,
		Scene:  "feed",
	}

	items, err := userCF.Recall(ctx, rctx)
	if err != nil {
		fmt.Printf("错误: %v\n", err)
		return
	}

	fmt.Printf("为用户 %d 推荐:\n", rctx.UserID)
	for i, item := range items {
		fmt.Printf("  %d. 物品 %d (分数: %.4f, 来源: %s)\n",
			i+1, item.ID, item.Score, getLabelValue(item, "recall_source"))
	}
}

func testItemCF(ctx context.Context, i2i *recall.I2IRecall) {
	// 为用户1推荐（基于用户1的历史物品1,2,3）
	rctx := &core.RecommendContext{
		UserID: 1,
		Scene:  "feed",
	}

	items, err := i2i.Recall(ctx, rctx)
	if err != nil {
		fmt.Printf("错误: %v\n", err)
		return
	}

	fmt.Printf("为用户 %d 推荐 (基于物品相似度):\n", rctx.UserID)
	for i, item := range items {
		fmt.Printf("  %d. 物品 %d (分数: %.4f, 来源: %s)\n",
			i+1, item.ID, item.Score, getLabelValue(item, "recall_source"))
	}
}

func testPipeline(ctx context.Context, userCF *recall.UserBasedCF, i2i *recall.I2IRecall) {
	// 创建多路召回 Pipeline
	p := &pipeline.Pipeline{
		Nodes: []pipeline.Node{
			&recall.Fanout{
				Sources: []recall.Source{
					userCF, // 用户协同过滤 (u2i)
					i2i,    // 物品协同过滤 (i2i)
				},
				Dedup:         true,
				MergeStrategy: "priority", // 优先使用第一个召回源的结果
			},
		},
	}

	rctx := &core.RecommendContext{
		UserID: 1,
		Scene:  "feed",
	}

	items, err := p.Run(ctx, rctx, nil)
	if err != nil {
		fmt.Printf("Pipeline 执行错误: %v\n", err)
		return
	}

	fmt.Printf("Pipeline 推荐结果 (用户 %d):\n", rctx.UserID)
	for i, item := range items {
		fmt.Printf("  %d. 物品 %d (分数: %.4f, 来源: %s)\n",
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
