package main

import (
	"context"
	"fmt"
	"time"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/model"
	"github.com/rushteam/reckit/recall"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// 1. 创建 Word2Vec 模型（示例：从预训练的词向量）
	// 实际使用时，应该从文件、HTTP API 或 S3 加载预训练的模型
	wordVectors := map[string][]float64{
		"electronics": []float64{0.1, 0.2, 0.3, 0.4},
		"smartphone":  []float64{0.2, 0.3, 0.4, 0.5},
		"tech":        []float64{0.15, 0.25, 0.35, 0.45},
		"mobile":      []float64{0.18, 0.28, 0.38, 0.48},
		"device":      []float64{0.12, 0.22, 0.32, 0.42},
		"laptop":      []float64{0.3, 0.4, 0.5, 0.6},
		"computer":    []float64{0.25, 0.35, 0.45, 0.55},
		"item_1":      []float64{0.1, 0.2, 0.3, 0.4},
		"item_2":      []float64{0.2, 0.3, 0.4, 0.5},
		"item_3":      []float64{0.3, 0.4, 0.5, 0.6},
		"item_4":      []float64{0.4, 0.5, 0.6, 0.7},
	}
	w2vModel := model.NewWord2VecModel(wordVectors, 4)

	// 2. 创建存储（示例：内存存储）
	store := &MemoryWord2VecStore{
		itemTexts: map[string]string{
			"item_1": "electronics smartphone tech",
			"item_2": "smartphone mobile device",
			"item_3": "laptop computer tech",
			"item_4": "electronics device mobile",
		},
		itemTags: map[string][]string{
			"item_1": {"electronics", "smartphone", "tech"},
			"item_2": {"smartphone", "mobile", "device"},
			"item_3": {"laptop", "computer", "tech"},
			"item_4": {"electronics", "device", "mobile"},
		},
		allItems: []string{"item_1", "item_2", "item_3", "item_4"},
	}

	// 3. 创建用户画像（包含最近点击）
	userProfile := core.NewUserProfile("user_1")
	userProfile.AddRecentClick("item_1", 10)

	// 4. 创建召回源（基于文本）
	word2vecRecall := &recall.Word2VecRecall{
		Model:     w2vModel,
		Store:     store,
		TopK:      10,
		Mode:      "text",
		TextField: "title",
	}

	// 5. 执行召回
	rctx := &core.RecommendContext{
		UserID: "user_1",
		Scene:  "feed",
		User:   userProfile,
	}

	items, err := word2vecRecall.Recall(ctx, rctx)
	if err != nil {
		panic(err)
	}

	// 6. 输出结果
	fmt.Println("=== Word2Vec 召回结果（基于文本） ===")
	for i, item := range items {
		fmt.Printf("%d. 物品 %s (相似度: %.4f)\n", i+1, item.ID, item.Score)
	}

	// 7. 测试序列模式
	fmt.Println("\n=== Word2Vec 召回结果（基于序列） ===")
	sequenceRecall := &recall.Word2VecRecall{
		Model: w2vModel,
		Store: store,
		TopK:  10,
		Mode:  "sequence",
	}
	items2, err := sequenceRecall.Recall(ctx, rctx)
	if err != nil {
		panic(err)
	}
	for i, item := range items2 {
		fmt.Printf("%d. 物品 %s (相似度: %.4f)\n", i+1, item.ID, item.Score)
	}
}

// MemoryWord2VecStore 是内存实现的 Word2VecStore（示例）
type MemoryWord2VecStore struct {
	itemTexts map[string]string
	itemTags  map[string][]string
	allItems  []string
}

func (s *MemoryWord2VecStore) GetItemText(ctx context.Context, itemID string) (string, error) {
	return s.itemTexts[itemID], nil
}

func (s *MemoryWord2VecStore) GetItemTags(ctx context.Context, itemID string) ([]string, error) {
	return s.itemTags[itemID], nil
}

func (s *MemoryWord2VecStore) GetUserSequence(ctx context.Context, userID string, maxLen int) ([]string, error) {
	// 示例：返回固定的序列
	return []string{"item_1", "item_2"}, nil
}

func (s *MemoryWord2VecStore) GetAllItems(ctx context.Context) ([]string, error) {
	return s.allItems, nil
}
