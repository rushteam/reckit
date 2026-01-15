package rerank

import (
	"context"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pipeline"
)

// Diversity 是一个简单的多样性 ReRank 示例：按类别去重（保留首个出现的类别）。
// 类别来源优先级：
// - label["category"].Value
// - meta["category"] (string)
type Diversity struct {
	LabelKey string // 默认 "category"
}

func (n *Diversity) Name() string {
	return "rerank.diversity"
}

func (n *Diversity) Kind() pipeline.Kind {
	return pipeline.KindReRank
}

func (n *Diversity) Process(
	_ context.Context,
	_ *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	if len(items) == 0 {
		return items, nil
	}

	key := n.LabelKey
	if key == "" {
		key = "category"
	}

	seen := make(map[string]bool, 32)
	out := make([]*core.Item, 0, len(items))

	for _, it := range items {
		if it == nil {
			continue
		}

		cate := ""
		if it.Labels != nil {
			if lbl, ok := it.Labels[key]; ok {
				cate = lbl.Value
			}
		}
		if cate == "" && it.Meta != nil {
			if v, ok := it.Meta[key]; ok {
				if s, ok := v.(string); ok {
					cate = s
				}
			}
		}

		if cate == "" {
			out = append(out, it)
			continue
		}
		if seen[cate] {
			continue
		}
		seen[cate] = true
		out = append(out, it)
	}

	return out, nil
}
