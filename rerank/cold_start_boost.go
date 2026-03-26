package rerank

import (
	"context"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pipeline"
)

// ColdStartBoostNode 新物品提权：曝光次数低于阈值时对 Score 加成。
//
// 公式：item.Score += BoostValue * (1 - impressions / Threshold)
// 曝光 >= Threshold 时不加成。
//
// 统计数据来源（按优先级）：
//  1. Provider（BanditStatsProvider）批量查询
//  2. item.Meta[ImpressionKey]（int64/float64）
type ColdStartBoostNode struct {
	Provider      BanditStatsProvider
	ImpressionKey string
	Threshold     int64
	BoostValue    float64
}

func (n *ColdStartBoostNode) Name() string        { return "rerank.cold_start_boost" }
func (n *ColdStartBoostNode) Kind() pipeline.Kind { return pipeline.KindReRank }

func (n *ColdStartBoostNode) Process(
	ctx context.Context,
	rctx *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	if n.Threshold <= 0 || n.BoostValue == 0 || len(items) == 0 {
		return items, nil
	}

	var statsMap map[string]BanditStats
	if n.Provider != nil {
		ids := itemIDs(items)
		sm, err := n.Provider.BatchGetStats(ctx, rctx, ids)
		if err == nil {
			statsMap = sm
		}
	}

	impKey := n.ImpressionKey
	if impKey == "" {
		impKey = "impressions"
	}

	for _, it := range items {
		if it == nil {
			continue
		}
		imp := n.getImpressions(it, statsMap, impKey)
		if imp >= n.Threshold {
			continue
		}
		ratio := 1.0 - float64(imp)/float64(n.Threshold)
		it.Score += n.BoostValue * ratio
	}
	return items, nil
}

func (n *ColdStartBoostNode) getImpressions(it *core.Item, statsMap map[string]BanditStats, key string) int64 {
	if statsMap != nil {
		if s, ok := statsMap[it.ID]; ok {
			return s.Impressions
		}
	}
	if it.Meta != nil {
		if raw, ok := it.Meta[key]; ok {
			switch v := raw.(type) {
			case int64:
				return v
			case int:
				return int64(v)
			case float64:
				return int64(v)
			}
		}
	}
	return 0
}
