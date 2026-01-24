// Package main 演示 Word2Vec（文本）与 Item2Vec（序列）召回。
//
// 1. Word2Vec 文本模式：词为文本词，用物品标题/标签等编码，相似度召回。
// 2. Item2Vec 序列模式：词为物品 ID，用用户行为序列编码，相似度召回。
//
// 模型可来自：
//   - 内联 map（示例）
//   - JSON 文件：train_item2vec.py 导出，或本目录 item2vec_vectors.json
//   - 自建 HTTP/S3 等，解析后 LoadWord2VecFromMap
//
// Python 训练见：python/train/train_item2vec.py，文档见 docs/WORD2VEC_ITEM2VEC.md。
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/model"
	"github.com/rushteam/reckit/recall"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// 1. 加载模型：优先 JSON（与 train_item2vec.py 输出格式一致），否则用内联数据
	w2vModel := loadWord2VecModel()
	if w2vModel == nil {
		panic("加载 Word2Vec 模型失败")
	}

	// 2. 存储与用户画像
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
	userProfile := core.NewUserProfile("user_1")
	userProfile.AddRecentClick("item_1", 10)

	rctx := &core.RecommendContext{
		UserID: "user_1",
		Scene:  "feed",
		User:   userProfile,
	}

	// ---------- Word2Vec 文本模式 ----------
	// 用户向量：最近点击物品的文本（标题等）编码；候选：物品文本编码，相似度召回
	fmt.Println("=== Word2Vec 召回（文本模式） ===")
	textRecall := &recall.Word2VecRecall{
		Model:     w2vModel,
		Store:     store,
		TopK:      10,
		Mode:      "text",
		TextField: "title",
	}
	itemsText, err := textRecall.Recall(ctx, rctx)
	if err != nil {
		panic(err)
	}
	for i, item := range itemsText {
		fmt.Printf("  %d. %s (相似度: %.4f)\n", i+1, item.ID, item.Score)
	}

	// ---------- Item2Vec 序列模式 ----------
	// 用户向量：用户行为序列（物品 ID 列表）编码；候选：物品 ID 向量，相似度召回
	// 序列来自 GetUserSequence 或 User.RecentClicks；词表为物品 ID
	fmt.Println("\n=== Item2Vec 召回（序列模式） ===")
	seqRecall := &recall.Word2VecRecall{
		Model: w2vModel,
		Store: store,
		TopK:  10,
		Mode:  "sequence",
	}
	itemsSeq, err := seqRecall.Recall(ctx, rctx)
	if err != nil {
		panic(err)
	}
	for i, item := range itemsSeq {
		fmt.Printf("  %d. %s (相似度: %.4f)\n", i+1, item.ID, item.Score)
	}

	// 序列向量化示例
	seq := []string{"item_1", "item_2"}
	userVec := w2vModel.EncodeSequence(seq)
	itemVec := w2vModel.GetWordVector("item_3")
	sim := w2vModel.Similarity(userVec, itemVec)
	fmt.Printf("\n序列 %v 与 item_3 相似度: %.4f\n", seq, sim)
}

// loadWord2VecModel 从 JSON 或内联数据加载 Word2Vec 模型。
// 查找顺序：examples/word2vec/item2vec_vectors.json -> python/model/item2vec_vectors.json -> 内联 map。
// 通常从项目根 go run ./examples/word2vec 运行，故路径相对项目根。
func loadWord2VecModel() *model.Word2VecModel {
	candidates := []string{
		filepath.Join("examples", "word2vec", "item2vec_vectors.json"),
		filepath.Join("python", "model", "item2vec_vectors.json"),
	}
	for _, p := range candidates {
		m, err := loadFromJSON(p)
		if err == nil && m != nil {
			return m
		}
	}
	return embedWord2VecModel()
}

func loadFromJSON(path string) (*model.Word2VecModel, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}
	return model.LoadWord2VecFromMap(raw)
}

func embedWord2VecModel() *model.Word2VecModel {
	wordVectors := map[string][]float64{
		"electronics": {0.1, 0.2, 0.3, 0.4},
		"smartphone":  {0.2, 0.3, 0.4, 0.5},
		"tech":        {0.15, 0.25, 0.35, 0.45},
		"mobile":      {0.18, 0.28, 0.38, 0.48},
		"device":      {0.12, 0.22, 0.32, 0.42},
		"laptop":      {0.3, 0.4, 0.5, 0.6},
		"computer":    {0.25, 0.35, 0.45, 0.55},
		"item_1":      {0.1, 0.2, 0.3, 0.4},
		"item_2":      {0.2, 0.3, 0.4, 0.5},
		"item_3":      {0.3, 0.4, 0.5, 0.6},
		"item_4":      {0.4, 0.5, 0.6, 0.7},
	}
	return model.NewWord2VecModel(wordVectors, 4)
}

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
	return []string{"item_1", "item_2"}, nil
}

func (s *MemoryWord2VecStore) GetAllItems(ctx context.Context) ([]string, error) {
	return s.allItems, nil
}
