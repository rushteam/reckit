package rank

import (
	"context"
	"sort"

	"reckit/core"
	"reckit/model"
	"reckit/pipeline"
	"reckit/pkg/utils"
)

// DNNNode 是使用 DNN 模型的排序 Node。
type DNNNode struct {
	Model model.RankModel
}

func (n *DNNNode) Name() string        { return "rank.dnn" }
func (n *DNNNode) Kind() pipeline.Kind { return pipeline.KindRank }

func (n *DNNNode) Process(
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
		it.PutLabel("rank_type", utils.Label{Value: "dnn", Source: "rank"})
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
