package main

import (
	"context"
	"fmt"
	"time"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/model"
	"github.com/rushteam/reckit/recall"
	"github.com/rushteam/reckit/service"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// 1. 创建 BERT 服务客户端（使用 TorchServe 或 TensorFlow Serving）
	// 方式1：使用 TorchServe
	torchServeClient := service.NewTorchServeClient(
		"http://localhost:8080", // TorchServe 端点
		"bert-base",             // 模型名称
		service.WithTorchServeTimeout(5*time.Second),
	)

	// 方式2：使用 TensorFlow Serving
	// tfServingClient := service.NewTFServingClient(
	// 	"http://localhost:8501", // TF Serving REST API 端点
	// 	"bert-base",             // 模型名称
	// 	service.WithTFServingTimeout(5*time.Second),
	// )

	// 2. 创建 BERT 模型
	bertModel := model.NewBERTModel(torchServeClient, 768).
		WithModelName("bert-base").
		WithMaxLength(512).
		WithPoolingStrategy("cls")

	// 3. 创建存储（示例：内存存储）
	store := &MemoryBERTStore{
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

	// 4. 创建用户画像（包含最近点击）
	userProfile := core.NewUserProfile("user_1")
	userProfile.AddRecentClick("item_1", 10)

	// 5. 创建 BERT 召回源（基于文本）
	bertRecall := &recall.BERTRecall{
		Model:     bertModel,
		Store:     store,
		TopK:      10,
		Mode:      "text",
		TextField: "title",
		BatchSize: 32,
	}

	// 6. 执行召回
	rctx := &core.RecommendContext{
		UserID: "user_1",
		Scene:  "feed",
		User:   userProfile,
	}

	// 注意：这里需要实际的 BERT 服务运行才能工作
	// 如果没有运行服务，会返回错误
	items, err := bertRecall.Recall(ctx, rctx)
	if err != nil {
		fmt.Printf("BERT 召回错误: %v\n", err)
		fmt.Println("提示：需要启动 BERT 服务（TorchServe 或 TensorFlow Serving）")
		return
	}

	// 7. 输出结果
	fmt.Println("=== BERT 召回结果（基于文本） ===")
	for i, item := range items {
		fmt.Printf("%d. 物品 %s (相似度: %.4f)\n", i+1, item.ID, item.Score)
	}

	// 8. 测试查询模式
	fmt.Println("\n=== BERT 召回结果（基于查询） ===")
	queryStore := &MemoryBERTStore{
		itemTexts: store.itemTexts,
		itemTags:  store.itemTags,
		allItems:  store.allItems,
		userQuery: "smartphone tech",
	}
	queryRecall := &recall.BERTRecall{
		Model:     bertModel,
		Store:     queryStore,
		TopK:      10,
		Mode:      "query",
		TextField: "title",
		BatchSize: 32,
	}
	items2, err := queryRecall.Recall(ctx, rctx)
	if err != nil {
		fmt.Printf("BERT 召回错误: %v\n", err)
		return
	}
	for i, item := range items2 {
		fmt.Printf("%d. 物品 %s (相似度: %.4f)\n", i+1, item.ID, item.Score)
	}
}

// MemoryBERTStore 是内存实现的 BERTStore（示例）
type MemoryBERTStore struct {
	itemTexts map[string]string
	itemTags  map[string][]string
	allItems  []string
	userQuery string
}

func (s *MemoryBERTStore) GetItemText(ctx context.Context, itemID string) (string, error) {
	return s.itemTexts[itemID], nil
}

func (s *MemoryBERTStore) GetItemTags(ctx context.Context, itemID string) ([]string, error) {
	return s.itemTags[itemID], nil
}

func (s *MemoryBERTStore) GetUserQuery(ctx context.Context, userID string) (string, error) {
	return s.userQuery, nil
}

func (s *MemoryBERTStore) GetAllItems(ctx context.Context) ([]string, error) {
	return s.allItems, nil
}
