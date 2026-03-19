package filter

import (
	"context"
	"fmt"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pipeline"
	"github.com/rushteam/reckit/pkg/utils"
)

// FilterNode 是过滤 Node，可以组合多个过滤器进行过滤。
// 如果任何一个过滤器返回 true，该物品就会被过滤掉。
type FilterNode struct {
	Filters []Filter
}

func (n *FilterNode) Name() string {
	return "filter.node"
}

func (n *FilterNode) Kind() pipeline.Kind {
	return pipeline.KindFilter
}

func (n *FilterNode) Process(
	ctx context.Context,
	rctx *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	if len(n.Filters) == 0 || len(items) == 0 {
		return items, nil
	}

	// 将 Filters 分为 BatchFilter 和普通 Filter 两组，保持原始顺序
	var batchFilters []BatchFilter
	var itemFilters []Filter
	for _, f := range n.Filters {
		if bf, ok := f.(BatchFilter); ok {
			batchFilters = append(batchFilters, bf)
		} else {
			itemFilters = append(itemFilters, f)
		}
	}

	// 阶段 1：依次执行 BatchFilter（整批过滤，避免 N+1）
	remaining := items
	for _, bf := range batchFilters {
		var err error
		remaining, err = bf.FilterBatch(ctx, rctx, remaining)
		if err != nil {
			return nil, fmt.Errorf("batch filter %q failed: %w", bf.Name(), err)
		}
	}

	// 阶段 2：逐条执行普通 Filter
	if len(itemFilters) == 0 {
		return remaining, nil
	}

	out := make([]*core.Item, 0, len(remaining))
	for _, item := range remaining {
		if item == nil {
			continue
		}

		shouldFilter := false
		filterReason := ""

		for _, f := range itemFilters {
			ok, err := f.ShouldFilter(ctx, rctx, item)
			if err != nil {
				return nil, fmt.Errorf("filter %q failed for item %q: %w", f.Name(), item.ID, err)
			}
			if ok {
				shouldFilter = true
				filterReason = f.Name()
				break
			}
		}

		if shouldFilter {
			item.PutLabel("filtered", utils.Label{
				Value:  "true",
				Source: filterReason,
			})
			continue
		}

		out = append(out, item)
	}

	return out, nil
}
