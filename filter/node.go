package filter

import (
	"context"

	"reckit/core"
	"reckit/pipeline"
	"reckit/pkg/utils"
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

	out := make([]*core.Item, 0, len(items))
	filteredCount := 0

	for _, item := range items {
		if item == nil {
			continue
		}

		shouldFilter := false
		filterReason := ""

		// 依次检查每个过滤器
		for _, f := range n.Filters {
			ok, err := f.ShouldFilter(ctx, rctx, item)
			if err != nil {
				// 过滤器错误时记录但不中断流程
				continue
			}
			if ok {
				shouldFilter = true
				filterReason = f.Name()
				break
			}
		}

		if shouldFilter {
			filteredCount++
			// 记录过滤原因（可选，用于调试/观测）
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
