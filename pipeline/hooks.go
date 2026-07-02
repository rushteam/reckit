package pipeline

import (
	"context"
	"log/slog"
	"time"

	"github.com/rushteam/reckit/core"
)

// NodeMetrics 单个 Node 的执行指标。
type NodeMetrics struct {
	NodeName   string
	Kind       Kind
	InputCount int
	OutputCount int
	Duration   time.Duration
	Err        error
}

// MetricsCollector 指标采集器接口。
// 由业务方实现，接入 Prometheus/OpenTelemetry 等监控体系。
type MetricsCollector interface {
	Collect(ctx context.Context, metrics NodeMetrics)
}

// MetricsHook 采集每个 Node 的执行时间、输入输出条数、错误等指标。
type MetricsHook struct {
	Collector MetricsCollector
	startTime map[Node]time.Time
	inputCnt  map[Node]int
}

func NewMetricsHook(collector MetricsCollector) *MetricsHook {
	return &MetricsHook{
		Collector: collector,
		startTime: make(map[Node]time.Time),
		inputCnt:  make(map[Node]int),
	}
}

func (h *MetricsHook) BeforeNode(_ context.Context, _ *core.RecommendContext, node Node, items []*core.Item) ([]*core.Item, error) {
	h.startTime[node] = time.Now()
	h.inputCnt[node] = len(items)
	return items, nil
}

func (h *MetricsHook) AfterNode(ctx context.Context, _ *core.RecommendContext, node Node, items []*core.Item, err error) ([]*core.Item, error) {
	if h.Collector == nil {
		return items, err
	}
	start := h.startTime[node]
	h.Collector.Collect(ctx, NodeMetrics{
		NodeName:    node.Name(),
		Kind:        node.Kind(),
		InputCount:  h.inputCnt[node],
		OutputCount: len(items),
		Duration:    time.Since(start),
		Err:         err,
	})
	delete(h.startTime, node)
	delete(h.inputCnt, node)
	return items, err
}

// LoggingHook 使用 slog 输出每个 Node 的执行摘要。
type LoggingHook struct {
	// Level 日志级别，默认 slog.LevelDebug。
	Level     slog.Level
	startTime map[Node]time.Time
	inputCnt  map[Node]int
}

func NewLoggingHook(level slog.Level) *LoggingHook {
	return &LoggingHook{
		Level:     level,
		startTime: make(map[Node]time.Time),
		inputCnt:  make(map[Node]int),
	}
}

func (h *LoggingHook) BeforeNode(_ context.Context, _ *core.RecommendContext, node Node, items []*core.Item) ([]*core.Item, error) {
	h.startTime[node] = time.Now()
	h.inputCnt[node] = len(items)
	return items, nil
}

func (h *LoggingHook) AfterNode(ctx context.Context, _ *core.RecommendContext, node Node, items []*core.Item, err error) ([]*core.Item, error) {
	start := h.startTime[node]
	elapsed := time.Since(start)
	delete(h.startTime, node)

	attrs := []any{
		slog.String("node", node.Name()),
		slog.String("kind", string(node.Kind())),
		slog.Int("input", h.inputCnt[node]),
		slog.Int("output", len(items)),
		slog.Duration("elapsed", elapsed),
	}
	delete(h.inputCnt, node)

	if err != nil {
		attrs = append(attrs, slog.String("error", err.Error()))
	}
	slog.Log(ctx, h.Level, "pipeline node executed", attrs...)
	return items, err
}
