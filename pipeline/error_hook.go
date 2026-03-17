package pipeline

import (
	"context"
	"fmt"
	"io"
	"os"

	"github.com/rushteam/reckit/core"
)

// ErrorHook 是 Pipeline 全局错误钩子，用于错误上报和降级控制。
//
// 当 Node 执行出错时，Pipeline 依次调用所有 ErrorHook（不论前一个返回什么）：
//   - 若任一 ErrorHook 返回 recovered=true，Pipeline 跳过该 Node 继续执行（使用上一步 items）
//   - 若所有 ErrorHook 返回 recovered=false，Pipeline 终止并返回错误
//
// 所有 ErrorHook 都会被调用，便于 metrics/alerting 采集完整数据。
type ErrorHook interface {
	OnNodeError(ctx context.Context, rctx *core.RecommendContext, node Node, err error) (recovered bool)
}

// ---------------------------------------------------------------------------
// 内置实现
// ---------------------------------------------------------------------------

// WarnAndSkipHook 记录日志并跳过失败的 Node（始终降级）。
// 适用于非关键 Node，如 feature.enrich、rerank.diversity 等。
type WarnAndSkipHook struct {
	// Writer 日志输出目标，nil 时使用 os.Stderr。
	Writer io.Writer
}

func (h *WarnAndSkipHook) OnNodeError(_ context.Context, _ *core.RecommendContext, node Node, err error) bool {
	w := h.Writer
	if w == nil {
		w = os.Stderr
	}
	fmt.Fprintf(w, "[WARN] node %q (%s) failed, skipped: %v\n", node.Name(), node.Kind(), err)
	return true
}

// KindRecoveryHook 按 Node Kind 决定是否降级。
// 仅当失败 Node 的 Kind 在 RecoverKinds 中时跳过，其余错误仍终止 Pipeline。
type KindRecoveryHook struct {
	RecoverKinds map[Kind]bool
	// OnError 可选回调，在判断前触发（用于 metrics/alerting）。
	OnError func(node Node, err error)
}

func (h *KindRecoveryHook) OnNodeError(_ context.Context, _ *core.RecommendContext, node Node, err error) bool {
	if h.OnError != nil {
		h.OnError(node, err)
	}
	return h.RecoverKinds[node.Kind()]
}

// ErrorCallbackHook 调用回调函数上报错误，自身不做降级（recovered 始终为 false）。
// 适合只接入 metrics/alerting、不改变执行流的场景。
type ErrorCallbackHook struct {
	Callback func(ctx context.Context, node Node, err error)
}

func (h *ErrorCallbackHook) OnNodeError(ctx context.Context, _ *core.RecommendContext, node Node, err error) bool {
	if h.Callback != nil {
		h.Callback(ctx, node, err)
	}
	return false
}

// CompositeErrorHook 组合多个 ErrorHook，依次调用。
// 任一子 hook 返回 recovered=true 即视为整体 recovered。
type CompositeErrorHook struct {
	Hooks []ErrorHook
}

func (h *CompositeErrorHook) OnNodeError(ctx context.Context, rctx *core.RecommendContext, node Node, err error) bool {
	recovered := false
	for _, hook := range h.Hooks {
		if hook.OnNodeError(ctx, rctx, node, err) {
			recovered = true
		}
	}
	return recovered
}
