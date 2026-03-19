package recall

import (
	"context"
	"sort"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/model"
	"github.com/rushteam/reckit/pkg/utils"
)

// Word2VecStore 是 Word2Vec 召回所需的存储接口。
type Word2VecStore interface {
	// GetItemText 获取物品的文本特征（标题、描述、标签等）
	GetItemText(ctx context.Context, itemID string) (string, error)

	// GetItemTags 获取物品的标签列表
	GetItemTags(ctx context.Context, itemID string) ([]string, error)

	// GetUserSequence 获取用户行为序列（点击的物品ID列表）
	GetUserSequence(ctx context.Context, userID string, maxLen int) ([]string, error)

	// GetAllItems 获取所有物品 ID 列表
	GetAllItems(ctx context.Context) ([]string, error)
}

// Word2VecRecall 是基于 Word2Vec 的召回源。
//
// 核心思想：
//   - 将用户行为序列或物品文本特征转换为向量
//   - 通过向量相似度找到相似物品
//
// 使用场景：
//   - I2I 召回：基于物品文本相似度
//   - 序列召回：基于用户行为序列
type Word2VecRecall struct {
	// Model Word2Vec 模型
	Model *model.Word2VecModel

	// Store 存储接口
	Store Word2VecStore

	// TopK 返回 TopK 个物品
	TopK int

	// Mode 召回模式：text（基于文本）或 sequence（基于序列）
	Mode string

	// TextField 文本字段：title / description / tags
	TextField string

	// HistoryFunc 可选，返回用户最近点击的物品 ID 列表。
	// 未设置时从 rctx.Attributes["recent_clicks"] 获取。
	HistoryFunc func(rctx *core.RecommendContext) []string
}

func (r *Word2VecRecall) Name() string {
	return "recall.word2vec"
}

func (r *Word2VecRecall) Recall(
	ctx context.Context,
	rctx *core.RecommendContext,
) ([]*core.Item, error) {
	if r.Model == nil || r.Store == nil || rctx == nil {
		return nil, nil
	}

	// 1. 获取用户向量
	var userEmbedding []float64
	var err error

	switch r.Mode {
	case "sequence":
		sequence, err := r.Store.GetUserSequence(ctx, rctx.UserID, 50)
		if err != nil || len(sequence) == 0 {
			sequence = r.getRecentClicks(rctx)
			if len(sequence) > 50 {
				sequence = sequence[:50]
			}
		}
		if len(sequence) == 0 {
			return nil, nil
		}
		userEmbedding = r.Model.EncodeSequence(sequence)

	case "text":
		fallthrough
	default:
		recentClicks := r.getRecentClicks(rctx)
		if len(recentClicks) > 0 {
			texts := make([]string, 0)
			for _, itemID := range recentClicks {
				text, err := r.Store.GetItemText(ctx, itemID)
				if err == nil && text != "" {
					texts = append(texts, text)
				}
			}
			if len(texts) > 0 {
				combinedText := ""
				for _, t := range texts {
					combinedText += t + " "
				}
				userEmbedding = r.Model.EncodeText(combinedText)
			}
		}
	}

	if len(userEmbedding) == 0 {
		return nil, nil
	}

	// 2. 获取所有物品并计算相似度
	allItems, err := r.Store.GetAllItems(ctx)
	if err != nil {
		return nil, err
	}

	type scoredItem struct {
		itemID string
		score  float64
	}
	scores := make([]scoredItem, 0)

	clickedSet := make(map[string]bool)
	for _, clicked := range r.getRecentClicks(rctx) {
		clickedSet[clicked] = true
	}

	for _, itemID := range allItems {
		if clickedSet[itemID] {
			continue
		}

		// 获取物品向量
		var itemVector []float64
		switch r.TextField {
		case "tags":
			tags, err := r.Store.GetItemTags(ctx, itemID)
			if err == nil && len(tags) > 0 {
				itemVector = r.Model.EncodeWords(tags)
			}
		case "description":
			text, err := r.Store.GetItemText(ctx, itemID)
			if err == nil && text != "" {
				itemVector = r.Model.EncodeText(text)
			}
		default:
			text, err := r.Store.GetItemText(ctx, itemID)
			if err == nil && text != "" {
				itemVector = r.Model.EncodeText(text)
			}
		}

		if len(itemVector) == 0 {
			continue
		}

		// 计算相似度
		score := r.Model.Similarity(userEmbedding, itemVector)
		if score > 0 {
			scores = append(scores, scoredItem{
				itemID: itemID,
				score:  score,
			})
		}
	}

	// 3. 排序取 TopK
	topK := r.TopK
	if topK <= 0 {
		topK = 20
	}
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})
	if len(scores) > topK {
		scores = scores[:topK]
	}

	// 4. 封装结果
	out := make([]*core.Item, 0, len(scores))
	for _, s := range scores {
		it := core.NewItem(s.itemID)
		it.Score = s.score
		it.PutLabel("recall_mode", utils.Label{Value: r.Mode, Source: "recall"})
		out = append(out, it)
	}

	return out, nil
}

func (r *Word2VecRecall) getRecentClicks(rctx *core.RecommendContext) []string {
	if r.HistoryFunc != nil {
		return r.HistoryFunc(rctx)
	}
	if rctx != nil && rctx.Attributes != nil {
		if clicks, ok := rctx.Attributes["recent_clicks"].([]string); ok {
			return clicks
		}
	}
	return nil
}
