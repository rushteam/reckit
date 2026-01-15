package rank

import (
	"context"
	"sort"

	"reckit/core"
	"reckit/model"
	"reckit/pipeline"
	"reckit/pkg/utils"
)

// TwoTowerNode 是使用两塔模型的排序 Node。
// 两塔模型分别学习用户表示和物品表示，通过相似度计算排序分数。
type TwoTowerNode struct {
	Model model.RankModel
}

func (n *TwoTowerNode) Name() string        { return "rank.two_tower" }
func (n *TwoTowerNode) Kind() pipeline.Kind { return pipeline.KindRank }

func (n *TwoTowerNode) Process(
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
		it.PutLabel("rank_type", utils.Label{Value: "two_tower", Source: "rank"})
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
