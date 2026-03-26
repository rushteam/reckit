package filter

import (
	"context"
	"fmt"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pipeline"
)

// Condition 判断 Pipeline 级条件是否成立；返回 true 时 Node 生效。
type Condition interface {
	Evaluate(ctx context.Context, rctx *core.RecommendContext) (bool, error)
}

// ConditionFunc 便捷函数适配器。
type ConditionFunc func(ctx context.Context, rctx *core.RecommendContext) (bool, error)

func (f ConditionFunc) Evaluate(ctx context.Context, rctx *core.RecommendContext) (bool, error) {
	return f(ctx, rctx)
}

// ConditionalNode 条件节点：仅当 Condition 为 true 时执行内部 Node，否则透传。
// 适用于 AB 实验、场景开关、时间段策略等。
type ConditionalNode struct {
	Cond Condition
	Node pipeline.Node
}

func (n *ConditionalNode) Name() string {
	if n.Node != nil {
		return "filter.cond(" + n.Node.Name() + ")"
	}
	return "filter.cond"
}

func (n *ConditionalNode) Kind() pipeline.Kind {
	if n.Node != nil {
		return n.Node.Kind()
	}
	return pipeline.KindFilter
}

func (n *ConditionalNode) Process(
	ctx context.Context,
	rctx *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	if n.Cond == nil || n.Node == nil {
		return items, nil
	}
	active, err := n.Cond.Evaluate(ctx, rctx)
	if err != nil {
		return nil, fmt.Errorf("conditional %q: %w", n.Name(), err)
	}
	if !active {
		return items, nil
	}
	return n.Node.Process(ctx, rctx, items)
}
