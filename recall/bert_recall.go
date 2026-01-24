package recall

import (
	"context"
	"sort"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/model"
	"github.com/rushteam/reckit/pkg/utils"
)

// BERTStore 是 BERT 召回所需的存储接口。
type BERTStore interface {
	// GetItemText 获取物品的文本特征（标题、描述、标签等）
	GetItemText(ctx context.Context, itemID string) (string, error)

	// GetItemTags 获取物品的标签列表
	GetItemTags(ctx context.Context, itemID string) ([]string, error)

	// GetUserQuery 获取用户查询文本（可选，用于搜索场景）
	GetUserQuery(ctx context.Context, userID string) (string, error)

	// GetAllItems 获取所有物品 ID 列表
	GetAllItems(ctx context.Context) ([]string, error)
}

// BERTRecall 是基于 BERT 的召回源。
//
// 核心思想：
//   - 使用 BERT 将文本编码为语义向量
//   - 通过向量相似度找到语义相似的物品
//
// 使用场景：
//   - 文本语义召回：基于物品标题、描述的语义相似度
//   - 搜索推荐：用户查询与物品文本的语义匹配
//   - I2I 召回：基于物品文本语义相似度
type BERTRecall struct {
	// Model BERT 模型
	Model *model.BERTModel

	// Store 存储接口
	Store BERTStore

	// TopK 返回 TopK 个物品
	TopK int

	// Mode 召回模式：text（基于文本）或 query（基于查询）
	Mode string

	// TextField 文本字段：title / description / tags
	TextField string

	// BatchSize 批量编码大小（提高效率）
	BatchSize int
}

func (r *BERTRecall) Name() string {
	return "recall.bert"
}

func (r *BERTRecall) Recall(
	ctx context.Context,
	rctx *core.RecommendContext,
) ([]*core.Item, error) {
	if r.Model == nil || r.Store == nil || rctx == nil {
		return nil, nil
	}

	// 1. 获取用户向量
	var userVector []float64
	var err error

	switch r.Mode {
	case "query":
		// 基于用户查询
		query, err := r.Store.GetUserQuery(ctx, rctx.UserID)
		if err != nil || query == "" {
			return nil, nil
		}
		userVector, err = r.Model.EncodeText(ctx, query)
		if err != nil {
			return nil, err
		}

	case "text":
		fallthrough
	default:
		// 基于用户最近点击的物品文本
		if rctx.User != nil && len(rctx.User.RecentClicks) > 0 {
			// 获取最近点击的物品文本
			texts := make([]string, 0)
			for _, itemID := range rctx.User.RecentClicks {
				text, err := r.Store.GetItemText(ctx, itemID)
				if err == nil && text != "" {
					texts = append(texts, text)
				}
				if len(texts) >= 5 { // 限制最多 5 个文本
					break
				}
			}
			if len(texts) > 0 {
				// 批量编码文本
				vectors, err := r.Model.EncodeTexts(ctx, texts)
				if err != nil {
					return nil, err
				}
				// 对多个文本向量求平均（或使用其他聚合方式）
				userVector = r.aggregateVectors(vectors)
			}
		}
	}

	if len(userVector) == 0 {
		return nil, nil
	}

	// 2. 获取所有物品并批量编码
	allItems, err := r.Store.GetAllItems(ctx)
	if err != nil {
		return nil, err
	}

	// 过滤已点击的物品
	candidateItems := make([]string, 0)
	clickedSet := make(map[string]bool)
	if rctx.User != nil {
		for _, clicked := range rctx.User.RecentClicks {
			clickedSet[clicked] = true
		}
	}
	for _, itemID := range allItems {
		if !clickedSet[itemID] {
			candidateItems = append(candidateItems, itemID)
		}
	}

	// 3. 批量获取物品文本并编码
	batchSize := r.BatchSize
	if batchSize <= 0 {
		batchSize = 32 // 默认批量大小
	}

	type scoredItem struct {
		itemID string
		score  float64
	}
	scores := make([]scoredItem, 0)

	// 分批处理
	for i := 0; i < len(candidateItems); i += batchSize {
		end := i + batchSize
		if end > len(candidateItems) {
			end = len(candidateItems)
		}
		batch := candidateItems[i:end]

		// 获取批量文本
		texts := make([]string, 0, len(batch))
		itemIDs := make([]string, 0, len(batch))
		for _, itemID := range batch {
			var text string
			switch r.TextField {
			case "tags":
				tags, err := r.Store.GetItemTags(ctx, itemID)
				if err == nil && len(tags) > 0 {
					// 将标签列表合并为文本
					text = ""
					for j, tag := range tags {
						if j > 0 {
							text += " "
						}
						text += tag
					}
				}
			case "description":
				text, err = r.Store.GetItemText(ctx, itemID)
			default:
				text, err = r.Store.GetItemText(ctx, itemID)
			}

			if err == nil && text != "" {
				texts = append(texts, text)
				itemIDs = append(itemIDs, itemID)
			}
		}

		if len(texts) == 0 {
			continue
		}

		// 批量编码
		vectors, err := r.Model.EncodeTexts(ctx, texts)
		if err != nil {
			// 如果批量编码失败，跳过这一批
			continue
		}

		// 计算相似度
		for j, itemID := range itemIDs {
			if j < len(vectors) {
				score := r.Model.Similarity(userVector, vectors[j])
				if score > 0 {
					scores = append(scores, scoredItem{
						itemID: itemID,
						score:  score,
					})
				}
			}
		}
	}

	// 4. 排序取 TopK
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

	// 5. 封装结果
	out := make([]*core.Item, 0, len(scores))
	for _, s := range scores {
		it := core.NewItem(s.itemID)
		it.Score = s.score
		it.PutLabel("recall_source", utils.Label{Value: "bert", Source: "recall"})
		it.PutLabel("recall_mode", utils.Label{Value: r.Mode, Source: "recall"})
		out = append(out, it)
	}

	return out, nil
}

// aggregateVectors 聚合多个向量（求平均）。
func (r *BERTRecall) aggregateVectors(vectors [][]float64) []float64 {
	if len(vectors) == 0 {
		return nil
	}
	if len(vectors) == 1 {
		return vectors[0]
	}

	dim := len(vectors[0])
	aggregated := make([]float64, dim)
	for _, vec := range vectors {
		if len(vec) != dim {
			continue
		}
		for i := 0; i < dim; i++ {
			aggregated[i] += vec[i]
		}
	}

	// 求平均
	for i := 0; i < dim; i++ {
		aggregated[i] /= float64(len(vectors))
	}

	return aggregated
}
