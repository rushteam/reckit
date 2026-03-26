package rerank

import (
	"context"
	"sort"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pipeline"
)

// WeightedInterleaveNode 按通道权重交叉混排。
// 与 FairInterleaveNode 的区别：Fair 等权轮询，Weighted 按 Weights 决定每轮取数。
//
// 算法：
//  1. 按 LabelKey 分组，组内按 Score 降序
//  2. 每轮对权重做归一化，转为概率累积分布；
//     轮询时按 residual（权重剩余额度）贪心选取最大残差通道
//  3. 取到 N 条为止
type WeightedInterleaveNode struct {
	N        int
	LabelKey string
	Weights  map[string]float64
}

func (n *WeightedInterleaveNode) Name() string {
	return "rerank.weighted_interleave"
}

func (n *WeightedInterleaveNode) Kind() pipeline.Kind {
	return pipeline.KindReRank
}

func (n *WeightedInterleaveNode) Process(
	_ context.Context,
	_ *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	if len(items) == 0 {
		return items, nil
	}

	labelKey := n.LabelKey
	if labelKey == "" {
		labelKey = DefaultRecallSourceLabel
	}

	limit := n.N
	if limit <= 0 {
		limit = len(items)
	}

	groups, order := groupByChannel(items, labelKey)

	for _, key := range order {
		g := groups[key]
		sort.SliceStable(g, func(i, j int) bool {
			return g[i].Score > g[j].Score
		})
	}

	totalWeight := 0.0
	for _, key := range order {
		totalWeight += n.weight(key)
	}
	if totalWeight == 0 {
		totalWeight = float64(len(order))
	}

	type channelState struct {
		key     string
		weight  float64
		cursor  int
		residue float64
	}

	states := make([]channelState, 0, len(order))
	for _, key := range order {
		w := n.weight(key)
		if w <= 0 {
			w = 1
		}
		states = append(states, channelState{
			key:    key,
			weight: w / totalWeight,
		})
	}

	out := make([]*core.Item, 0, limit)
	for len(out) < limit {
		for i := range states {
			states[i].residue += states[i].weight
		}

		added := false
		sort.SliceStable(states, func(i, j int) bool {
			return states[i].residue > states[j].residue
		})
		for i := range states {
			if len(out) >= limit {
				break
			}
			g := groups[states[i].key]
			c := states[i].cursor
			if c < len(g) {
				out = append(out, g[c])
				states[i].cursor = c + 1
				states[i].residue -= 1.0
				added = true
			}
		}
		if !added {
			break
		}
	}
	return out, nil
}

func (n *WeightedInterleaveNode) weight(key string) float64 {
	if n.Weights == nil {
		return 1
	}
	if w, ok := n.Weights[key]; ok {
		return w
	}
	return 1
}
