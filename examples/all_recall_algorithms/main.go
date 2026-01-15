package main

import (
	"context"
	"encoding/json"
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

	// 1. 初始化存储
	memStore := store.NewMemoryStore()
	defer memStore.Close()

	// 2. 准备测试数据
	setupTestData(ctx, memStore)

	// 3. 创建各种召回源
	recallSources := createRecallSources(ctx, memStore)

	// 4. 测试各个召回算法
	fmt.Println("=== 召回算法测试 ===")

	testRecall(ctx, recallSources["u2i"], "1")
	testRecall(ctx, recallSources["i2i"], "1")
	testRecall(ctx, recallSources["mf"], "1")
	testRecall(ctx, recallSources["emb"], "1")
	testRecall(ctx, recallSources["content"], "1")

	// 5. 集成到 Pipeline
	fmt.Println("\n=== 集成到 Pipeline ===")
	testPipeline(ctx, recallSources)
}

func setupTestData(ctx context.Context, memStore store.Store) {
	// 设置协同过滤数据
	cfStore := recall.NewStoreCFAdapter(memStore, "cf")
	cfInteractions := []struct {
		UserID string
		ItemID string
		Score  float64
	}{
		{"1", "1", 5.0}, {"1", "2", 4.0}, {"1", "3", 5.0},
		{"2", "2", 5.0}, {"2", "3", 4.0}, {"2", "4", 5.0},
		{"3", "4", 4.0}, {"3", "5", 5.0}, {"3", "6", 4.0},
	}
	recall.SetupCFTestData(ctx, cfStore, cfInteractions)

	// 设置矩阵分解数据
	user1Vec, _ := json.Marshal([]float64{0.5, 0.3, 0.2})
	memStore.Set(ctx, "mf:user:1", user1Vec)

	itemVectors := map[string][]float64{
		"1": {0.4, 0.3, 0.3},
		"2": {0.5, 0.2, 0.3},
		"3": {0.3, 0.4, 0.3},
		"4": {0.2, 0.3, 0.5},
		"5": {0.4, 0.4, 0.2},
	}
	itemIDs := make([]string, 0, len(itemVectors))
	for itemID, vec := range itemVectors {
		itemIDs = append(itemIDs, itemID)
		vecData, _ := json.Marshal(vec)
		memStore.Set(ctx, fmt.Sprintf("mf:item:%s", itemID), vecData)
	}
	itemsData, _ := json.Marshal(itemIDs)
	memStore.Set(ctx, "mf:items", itemsData)

	// 设置内容推荐数据
	user1Prefs, _ := json.Marshal(map[string]float64{
		"tech":  0.8,
		"game":  0.6,
		"music": 0.2,
	})
	memStore.Set(ctx, "content:user:1", user1Prefs)

	itemFeatures := map[string]map[string]float64{
		"1": {"tech": 0.9, "game": 0.1, "music": 0.0},
		"2": {"tech": 0.7, "game": 0.3, "music": 0.0},
		"3": {"tech": 0.1, "game": 0.9, "music": 0.0},
		"4": {"tech": 0.0, "game": 0.2, "music": 0.8},
		"5": {"tech": 0.5, "game": 0.5, "music": 0.0},
	}
	contentItemIDs := make([]string, 0, len(itemFeatures))
	for itemID, features := range itemFeatures {
		contentItemIDs = append(contentItemIDs, itemID)
		featData, _ := json.Marshal(features)
		memStore.Set(ctx, fmt.Sprintf("content:item:%s", itemID), featData)
	}
	contentItemsData, _ := json.Marshal(contentItemIDs)
	memStore.Set(ctx, "content:items", contentItemsData)
}

func createRecallSources(ctx context.Context, memStore store.Store) map[string]recall.Source {
	_ = ctx
	cfStore := recall.NewStoreCFAdapter(memStore, "cf")
	mfStore := recall.NewStoreMFAdapter(memStore, "mf")
	contentStore := recall.NewStoreContentAdapter(memStore, "content")

	return map[string]recall.Source{
		"u2i": &recall.U2IRecall{
			Store:            cfStore,
			TopKSimilarUsers: 5,
			TopKItems:        3,
			SimilarityMetric: "cosine",
		},
		"i2i": &recall.I2IRecall{
			Store:            cfStore,
			TopKSimilarItems: 5,
			TopKItems:        3,
			SimilarityMetric: "cosine",
		},
		"mf": &recall.MFRecall{
			Store: mfStore,
			TopK:  3,
		},
		"emb": &recall.EmbRecall{
			UserVector: []float64{0.5, 0.3, 0.2},
			TopK:       3,
			Metric:     "cosine",
		},
		"content": &recall.ContentRecall{
			Store:            contentStore,
			TopK:             3,
			SimilarityMetric: "cosine",
		},
	}
}

func testRecall(ctx context.Context, source recall.Source, userID string) {
	rctx := &core.RecommendContext{
		UserID: userID,
		Scene:  "feed",
	}

	items, err := source.Recall(ctx, rctx)
	if err != nil {
		fmt.Printf("  错误: %v\n", err)
		return
	}

	fmt.Printf("  算法: %s\n", source.Name())
	for i, item := range items {
		sourceLabel := getLabelValue(item, "recall_source")
		fmt.Printf("  %d. 物品 %s (分数: %.4f, 来源: %s)\n",
			i+1, item.ID, item.Score, sourceLabel)
	}
}

func testPipeline(ctx context.Context, sources map[string]recall.Source) {
	p := &pipeline.Pipeline{
		Nodes: []pipeline.Node{
			&recall.Fanout{
				Sources: []recall.Source{
					sources["u2i"],
					sources["i2i"],
					sources["mf"],
					sources["content"],
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
		sourceLabel := getLabelValue(item, "recall_source")
		fmt.Printf("  %d. 物品 %s (分数: %.4f, 来源: %s)\n",
			i+1, item.ID, item.Score, sourceLabel)
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
