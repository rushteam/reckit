package rank

import (
	"context"
	"fmt"
	"sort"

	"reckit/core"
	"reckit/model"
	"reckit/pipeline"
	"reckit/pkg/utils"
)

// DINNode 是使用 DIN 模型的排序 Node。
// DIN 模型需要用户行为序列，从 UserProfile.RecentClicks 获取。
type DINNode struct {
	Model model.RankModel

	// MaxBehaviorSeqLen 是最大行为序列长度
	MaxBehaviorSeqLen int
}

func (n *DINNode) Name() string        { return "rank.din" }
func (n *DINNode) Kind() pipeline.Kind { return pipeline.KindRank }

func (n *DINNode) Process(
	ctx context.Context,
	rctx *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	if n.Model == nil || len(items) == 0 {
		return items, nil
	}

	maxLen := n.MaxBehaviorSeqLen
	if maxLen == 0 {
		maxLen = 10 // 默认最多 10 个历史行为
	}

	// 从 UserProfile 获取用户行为序列
	var behaviorSeq []int64
	if rctx != nil && rctx.User != nil {
		behaviorSeq = rctx.User.RecentClicks
		if len(behaviorSeq) > maxLen {
			behaviorSeq = behaviorSeq[len(behaviorSeq)-maxLen:]
		}
	}

	for _, it := range items {
		if it == nil {
			continue
		}

		// 添加候选物品 ID 到特征中
		features := make(map[string]float64)
		for k, v := range it.Features {
			features[k] = v
		}
		features["candidate_item_id"] = float64(it.ID)

		// 添加用户行为序列到特征中
		for i, itemID := range behaviorSeq {
			if i >= maxLen {
				break
			}
			features[fmt.Sprintf("behavior_item_%d", i)] = float64(itemID)
		}

		score, err := n.Model.Predict(features)
		if err != nil {
			return nil, err
		}
		it.Score = score
		it.PutLabel("rank_model", utils.Label{Value: n.Model.Name(), Source: "rank"})
		it.PutLabel("rank_type", utils.Label{Value: "din", Source: "rank"})
	}

	sort.SliceStable(items, func(i, j int) bool {
		if items[i] == nil {
			return false
		}
		if items[j] == nil {
			return true
		}
		return items[i].Score > items[j].Score
	})
	return items, nil
}
