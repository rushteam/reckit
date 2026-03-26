package postprocess

import (
	"context"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pipeline"
)

// PaddingFunc 动态补足策略：返回最多 need 条补位物品。
type PaddingFunc func(ctx context.Context, rctx *core.RecommendContext, need int) ([]*core.Item, error)

// PaddingNode 当结果不足 N 条时，用兜底策略补足。
// 优先使用 FallbackItems 静态列表，不足时调用 FallbackFunc；两者均可选。
// 补位物品会被打上 Label "__padding__"。
type PaddingNode struct {
	N             int
	FallbackItems []*core.Item
	FallbackFunc  PaddingFunc
}

func (n *PaddingNode) Name() string        { return "postprocess.padding" }
func (n *PaddingNode) Kind() pipeline.Kind { return pipeline.KindPostProcess }

func (n *PaddingNode) Process(
	ctx context.Context,
	rctx *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	target := n.N
	if target <= 0 || len(items) >= target {
		return items, nil
	}

	need := target - len(items)
	existing := make(map[string]bool, len(items))
	for _, it := range items {
		if it != nil {
			existing[it.ID] = true
		}
	}

	out := make([]*core.Item, len(items), target)
	copy(out, items)

	out = appendPadding(out, n.FallbackItems, existing, need)

	if len(out) < target && n.FallbackFunc != nil {
		remain := target - len(out)
		extra, err := n.FallbackFunc(ctx, rctx, remain)
		if err != nil {
			return nil, err
		}
		out = appendPadding(out, extra, existing, remain)
	}

	return out, nil
}

func appendPadding(out []*core.Item, candidates []*core.Item, existing map[string]bool, need int) []*core.Item {
	added := 0
	for _, it := range candidates {
		if added >= need {
			break
		}
		if it == nil || existing[it.ID] {
			continue
		}
		existing[it.ID] = true
		it.PutLabel("__padding__", labelPadding)
		out = append(out, it)
		added++
	}
	return out
}
