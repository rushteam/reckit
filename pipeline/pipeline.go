package pipeline

import (
	"context"

	"reckit/core"
)

// Pipeline 是 Reckit 的核心抽象：把推荐逻辑拆成可组合的 Node 链。
type Pipeline struct {
	Nodes []Node
}

func (p *Pipeline) Run(
	ctx context.Context,
	rctx *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	cur := items
	for _, node := range p.Nodes {
		next, err := node.Process(ctx, rctx, cur)
		if err != nil {
			return nil, err
		}
		cur = next
	}
	return cur, nil
}
