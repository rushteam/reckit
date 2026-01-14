package rank

import (
	"context"
	"sort"

	"reckit/core"
	"reckit/model"
	"reckit/pipeline"
	"reckit/pkg/utils"
)

// LRNode 是一个使用 RankModel 的排序 Node 示例（不限定模型类型，LR 只是默认实现之一）。
// - 写入 labels：rank_model
// - 更新 item.Score 并按分数降序排序
type LRNode struct {
	Model model.RankModel
}

func (n *LRNode) Name() string        { return "rank.model" }
func (n *LRNode) Kind() pipeline.Kind { return pipeline.KindRank }

func (n *LRNode) Process(
	_ context.Context,
	_ *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	if n.Model == nil || len(items) == 0 {
		return items, nil
	}

	for _, it := range items {
		if it == nil {
			continue
		}
		score, err := n.Model.Predict(it.Features)
		if err != nil {
			return nil, err
		}
		it.Score = score
		it.PutLabel("rank_model", utils.Label{Value: n.Model.Name(), Source: "rank"})
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
