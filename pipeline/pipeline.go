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

	// ErrorHooks 全局错误钩子列表。
	// 当 Node（或 PipelineHook）执行出错时，Pipeline 依次调用所有 ErrorHook：
	//   - 若任一 ErrorHook 返回 recovered=true，跳过该 Node 继续执行（使用上一步 items）
	//   - 若全部返回 false，Pipeline 终止并返回错误
	// 未配置 ErrorHooks 时行为与之前完全一致（fail-fast）。
	ErrorHooks []ErrorHook
}

// Run 执行 Pipeline，依次处理每个 Node。
// 如果设置了 Hooks，会在每个 Node 执行前后调用相应的 Hook。
// 如果设置了 ErrorHooks，Node 出错时会尝试降级（跳过该 Node）。
func (p *Pipeline) Run(
	ctx context.Context,
	rctx *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	cur := items
	for _, node := range p.Nodes {
		// BeforeNode Hooks
		var err error
		for _, hook := range p.Hooks {
			cur, err = hook.BeforeNode(ctx, rctx, node, cur)
			if err != nil {
				break
			}
		}
		if err != nil {
			if p.tryRecover(ctx, rctx, node, err) {
				continue
			}
			return nil, err
		}

		// 执行 Node
		next, nodeErr := node.Process(ctx, rctx, cur)

		// AfterNode Hooks
		for _, hook := range p.Hooks {
			next, nodeErr = hook.AfterNode(ctx, rctx, node, next, nodeErr)
			if nodeErr != nil {
				break
			}
		}

		if nodeErr != nil {
			if p.tryRecover(ctx, rctx, node, nodeErr) {
				continue // 降级：跳过该 Node，保留上一步的 items
			}
			return nil, nodeErr
		}
		cur = next
	}
	return cur, nil
}

// tryRecover 依次调用 ErrorHooks，返回是否有任一 hook 声明 recovered。
func (p *Pipeline) tryRecover(ctx context.Context, rctx *core.RecommendContext, node Node, err error) bool {
	recovered := false
	for _, eh := range p.ErrorHooks {
		if eh.OnNodeError(ctx, rctx, node, err) {
			recovered = true
		}
	}
	return recovered
}
