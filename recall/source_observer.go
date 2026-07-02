package recall

import (
	"context"
	"time"

	"github.com/rushteam/reckit/core"
)

// SourceObserver 召回源可观测接口。
// 由业务方实现，接入 Prometheus/OpenTelemetry 等监控体系。
type SourceObserver interface {
	OnRecall(ctx context.Context, source Source, items []*core.Item, err error, duration time.Duration)
}

// ObservableSource 为任意 Source 添加可观测性的装饰器。
// 每次 Recall 调用结束后触发 Observer 回调，上报延迟、条数、错误等指标。
type ObservableSource struct {
	Inner    Source
	Observer SourceObserver
}

func (s *ObservableSource) Name() string { return s.Inner.Name() }

func (s *ObservableSource) Recall(ctx context.Context, rctx *core.RecommendContext) ([]*core.Item, error) {
	start := time.Now()
	items, err := s.Inner.Recall(ctx, rctx)
	if s.Observer != nil {
		s.Observer.OnRecall(ctx, s.Inner, items, err, time.Since(start))
	}
	return items, err
}

// ObserveSources 批量包装 Source 列表，为每个 Source 添加同一 Observer。
func ObserveSources(sources []Source, observer SourceObserver) []Source {
	wrapped := make([]Source, len(sources))
	for i, src := range sources {
		wrapped[i] = &ObservableSource{Inner: src, Observer: observer}
	}
	return wrapped
}
