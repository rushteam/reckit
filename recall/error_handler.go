package recall

import (
	"context"
	"time"

	"github.com/rushteam/reckit/core"
)

// ErrorHandler 是错误处理策略接口，用于自定义召回源失败时的处理逻辑。
type ErrorHandler interface {
	// HandleError 处理召回源错误。
	// 返回 error != nil 时将中断整个 Fanout。
	HandleError(source Source, err error, rctx *core.RecommendContext) ([]*core.Item, error)
}

// IgnoreErrorHandler 忽略错误，返回空结果，不中断其他召回源。
// 可选 OnError 回调，用于 metrics/alerting 上报。
type IgnoreErrorHandler struct {
	OnError func(source Source, err error)
}

func (h *IgnoreErrorHandler) HandleError(source Source, err error, _ *core.RecommendContext) ([]*core.Item, error) {
	if h.OnError != nil {
		h.OnError(source, err)
	}
	return nil, nil
}

// RetryErrorHandler 是重试策略：失败后重试若干次，每次间隔 RetryDelay。
// 全部重试仍失败时返回空结果（降级），不中断 Pipeline。
type RetryErrorHandler struct {
	MaxRetries int
	RetryDelay time.Duration
	// OnRetry 可选回调，每次重试前触发（用于 metrics/日志）。
	OnRetry func(source Source, attempt int, err error)
	// OnGiveUp 可选回调，全部重试失败后触发。
	OnGiveUp func(source Source, err error)
}

func (h *RetryErrorHandler) HandleError(source Source, err error, rctx *core.RecommendContext) ([]*core.Item, error) {
	maxRetries := h.MaxRetries
	if maxRetries <= 0 {
		maxRetries = 1
	}
	var lastErr error
	for i := 0; i < maxRetries; i++ {
		if h.OnRetry != nil {
			h.OnRetry(source, i+1, err)
		}
		if h.RetryDelay > 0 {
			time.Sleep(h.RetryDelay)
		}
		items, retryErr := source.Recall(context.Background(), rctx)
		if retryErr == nil {
			return items, nil
		}
		lastErr = retryErr
	}
	if h.OnGiveUp != nil {
		h.OnGiveUp(source, lastErr)
	}
	return nil, nil
}

// FallbackErrorHandler 是降级策略：使用备用召回源。
type FallbackErrorHandler struct {
	FallbackSource Source
	// OnFallback 可选回调，降级时触发。
	OnFallback func(source Source, err error)
}

func (h *FallbackErrorHandler) HandleError(source Source, err error, rctx *core.RecommendContext) ([]*core.Item, error) {
	if h.OnFallback != nil {
		h.OnFallback(source, err)
	}
	if h.FallbackSource != nil {
		return h.FallbackSource.Recall(context.Background(), rctx)
	}
	return nil, nil
}
