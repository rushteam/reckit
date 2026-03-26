package filter

import (
	"context"

	"github.com/rushteam/reckit/core"
)

// QualityGateFilter 分数门槛过滤：Score 低于 MinScore 的物品被过滤。
type QualityGateFilter struct {
	MinScore float64
}

func (f *QualityGateFilter) Name() string { return "filter.quality_gate" }

func (f *QualityGateFilter) ShouldFilter(
	_ context.Context,
	_ *core.RecommendContext,
	item *core.Item,
) (bool, error) {
	if item == nil {
		return true, nil
	}
	return item.Score < f.MinScore, nil
}
