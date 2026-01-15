package pipeline

import (
	"context"

	"github.com/rushteam/reckit/core"
)

// PipelineHook 是 Pipeline 执行过程中的 Hook 接口，用于实现中间件功能。
// 用户可以实现此接口来添加日志、监控、缓存、性能分析等功能。
type PipelineHook interface {
	// BeforeNode 在 Node 执行前调用
	// 可以修改 items 或 context，返回修改后的 items
	// 如果返回 error，Pipeline 将中断执行
	BeforeNode(ctx context.Context, rctx *core.RecommendContext, node Node, items []*core.Item) ([]*core.Item, error)

	// AfterNode 在 Node 执行后调用
	// 可以修改 items 或记录执行结果，返回修改后的 items
	// 如果返回 error，Pipeline 将中断执行
	AfterNode(ctx context.Context, rctx *core.RecommendContext, node Node, items []*core.Item, err error) ([]*core.Item, error)
}

// Pipeline 是 Reckit 的核心抽象：把推荐逻辑拆成可组合的 Node 链。
// 支持 Hook 机制，允许用户插入中间件功能。
type Pipeline struct {
	Nodes []Node
	Hooks []PipelineHook // Hook 列表，按顺序执行
}

// Run 执行 Pipeline，依次处理每个 Node。
// 如果设置了 Hooks，会在每个 Node 执行前后调用相应的 Hook。
func (p *Pipeline) Run(
	ctx context.Context,
	rctx *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	cur := items
	for _, node := range p.Nodes {
		// 执行 BeforeNode Hooks
		var err error
		for _, hook := range p.Hooks {
			cur, err = hook.BeforeNode(ctx, rctx, node, cur)
			if err != nil {
				return nil, err
			}
		}

		// 执行 Node
		next, err := node.Process(ctx, rctx, cur)

		// 执行 AfterNode Hooks
		for _, hook := range p.Hooks {
			next, err = hook.AfterNode(ctx, rctx, node, next, err)
			if err != nil {
				return nil, err
			}
		}

		if err != nil {
			return nil, err
		}
		cur = next
	}
	return cur, nil
}
