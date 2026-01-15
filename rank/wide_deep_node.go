package rank

import (
	"context"
	"sort"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/model"
	"github.com/rushteam/reckit/pipeline"
	"github.com/rushteam/reckit/pkg/utils"
)

// WideDeepNode 是使用 Wide&Deep 模型的排序 Node。
type WideDeepNode struct {
	Model model.RankModel
}

func (n *WideDeepNode) Name() string        { return "rank.wide_deep" }
func (n *WideDeepNode) Kind() pipeline.Kind { return pipeline.KindRank }

func (n *WideDeepNode) Process(
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
		it.PutLabel("rank_type", utils.Label{Value: "wide_deep", Source: "rank"})
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
