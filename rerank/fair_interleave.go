package rerank

import (
	"context"
	"sort"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pipeline"
)

// FairInterleaveNode 精排后按召回通道公平轮询交叉，保证各通道均等曝光。
// 算法：按 LabelKey 分组 → 组内按 Score 降序 → 等权轮询依次取一条，直至 N 条。
type FairInterleaveNode struct {
	// N 输出数量；<= 0 时取 len(items)
	N int
	// LabelKey 分组键，默认 "recall_source"；取合并标签的首段（与 PrimaryRecallChannel 一致）
	LabelKey string
}

func (n *FairInterleaveNode) Name() string {
	return "rerank.fair_interleave"
}

func (n *FairInterleaveNode) Kind() pipeline.Kind {
	return pipeline.KindReRank
}

func (n *FairInterleaveNode) Process(
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

	cursors := make(map[string]int, len(order))
	out := make([]*core.Item, 0, limit)
	for len(out) < limit {
		added := false
		for _, key := range order {
			if len(out) >= limit {
				break
			}
			g := groups[key]
			c := cursors[key]
			if c < len(g) {
				out = append(out, g[c])
				cursors[key] = c + 1
				added = true
			}
		}
		if !added {
			break
		}
	}
	return out, nil
}

// groupByChannel 按 labelKey 首段分组，返回分组 map 和首次出现顺序。
func groupByChannel(items []*core.Item, labelKey string) (map[string][]*core.Item, []string) {
	groups := make(map[string][]*core.Item)
	var order []string
	seen := make(map[string]bool)
	for _, it := range items {
		if it == nil {
			continue
		}
		ch := PrimaryRecallChannel(it, labelKey)
		groups[ch] = append(groups[ch], it)
		if !seen[ch] {
			order = append(order, ch)
			seen[ch] = true
		}
	}
	return groups, order
}

