package feedback

import (
	"context"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pipeline"
)

// FeedbackHook Pipeline Hook，用于自动记录反馈
type FeedbackHook struct {
	collector Collector
}

// NewFeedbackHook 创建反馈 Hook
func NewFeedbackHook(collector Collector) *FeedbackHook {
	return &FeedbackHook{collector: collector}
}

// BeforeNode 节点执行前（这里不需要做什么）
func (h *FeedbackHook) BeforeNode(ctx context.Context, rctx *core.RecommendContext,
	node pipeline.Node, items []*core.Item) ([]*core.Item, error) {
	return items, nil
}

// AfterNode 节点执行后，如果是最后一个节点则记录曝光
func (h *FeedbackHook) AfterNode(ctx context.Context, rctx *core.RecommendContext,
	node pipeline.Node, items []*core.Item, err error) ([]*core.Item, error) {

	// 如果是最后一个节点（返回最终结果），记录曝光
	// 这里简化处理，实际可以根据 Node Kind 判断
	if err == nil && len(items) > 0 {
		// 异步记录曝光，不阻塞
		_ = h.collector.RecordImpression(ctx, rctx, items)
	}

	return items, err
}
