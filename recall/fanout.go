package recall

import (
	"context"
	"strconv"
	"sync"
	"time"

	"golang.org/x/sync/errgroup"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pipeline"
	"github.com/rushteam/reckit/pkg/utils"
)

// Fanout 是一个 Recall Node：并发执行多个召回源，并合并结果。
// 同时实现 Source 接口，支持嵌套在另一个 Fanout 中作为子召回源。
type Fanout struct {
	// NodeName 自定义名称（可选），用于嵌套时区分不同 Fanout 实例。
	// 为空时默认 "recall.fanout"。
	NodeName string

	Sources       []Source
	Dedup         bool
	Timeout       time.Duration // 每个召回源的超时时间
	MaxConcurrent int           // 最大并发数（0 表示无限制）
	
	// MergeStrategy 合并策略（必需）
	// 使用内置策略：FirstMergeStrategy、UnionMergeStrategy、PriorityMergeStrategy
	// 或实现自定义策略
	MergeStrategy MergeStrategy
	
	// ErrorHandler 错误处理策略（可选）
	// 如果为 nil，则使用默认策略（IgnoreErrorHandler）
	ErrorHandler ErrorHandler
	
	// SourcePriorities 自定义优先级权重（可选）
	// key: Source 名称，value: 优先级（值越小优先级越高）
	// 如果未设置，则使用 Source 在数组中的索引作为优先级
	SourcePriorities map[string]int
}

func (n *Fanout) Name() string {
	if n.NodeName != "" {
		return n.NodeName
	}
	return "recall.fanout"
}
func (n *Fanout) Kind() pipeline.Kind { return pipeline.KindRecall }

// Recall 使 Fanout 同时实现 Source 接口，支持嵌套在另一个 Fanout 中作为子召回源。
func (n *Fanout) Recall(ctx context.Context, rctx *core.RecommendContext) ([]*core.Item, error) {
	return n.Process(ctx, rctx, nil)
}

func (n *Fanout) Process(
	ctx context.Context,
	rctx *core.RecommendContext,
	_ []*core.Item,
) ([]*core.Item, error) {
	if len(n.Sources) == 0 {
		return nil, nil
	}

	var (
		mu      sync.Mutex
		all     []*core.Item
		eg, egCtx = errgroup.WithContext(ctx)
	)

	// 限流：使用 semaphore 控制并发数
	sem := make(chan struct{}, n.MaxConcurrent)
	if n.MaxConcurrent <= 0 {
		close(sem) // 无限制时直接关闭，避免阻塞
	}

	for i, src := range n.Sources {
		s := src
		// 计算优先级：优先使用自定义权重，否则使用索引
		priority := i
		if n.SourcePriorities != nil {
			if customPriority, ok := n.SourcePriorities[s.Name()]; ok {
				priority = customPriority
			}
		}

		eg.Go(func() error {
			// 限流
			if n.MaxConcurrent > 0 {
				sem <- struct{}{}
				defer func() { <-sem }()
			}

			// 超时控制：基于 errgroup 的 derived context，
			// 任一 source 返回 error 后其它 goroutine 能感知取消。
			recallCtx := egCtx
			if n.Timeout > 0 {
				var cancel context.CancelFunc
				recallCtx, cancel = context.WithTimeout(egCtx, n.Timeout)
				defer cancel()
			}

			items, err := s.Recall(recallCtx, rctx)
			if err != nil {
				// 使用错误处理策略
				handler := n.ErrorHandler
				if handler == nil {
					handler = &IgnoreErrorHandler{} // 默认策略
				}
				
				handledItems, handleErr := handler.HandleError(s, err, rctx)
				if handleErr != nil {
					return handleErr
				}
				items = handledItems
			}

			// 记录召回来源 label，方便 explain / 观测
			for _, it := range items {
				it.PutLabel("recall_source", utils.Label{Value: s.Name(), Source: "recall"})
				it.PutLabel("recall_priority", utils.Label{Value: strconv.Itoa(priority), Source: "recall"})
			}

			mu.Lock()
			all = append(all, items...)
			mu.Unlock()
			return nil
		})
	}

	if err := eg.Wait(); err != nil {
		return nil, err
	}

	// 合并策略（必需）
	// 使用局部变量，避免在并发请求中写共享字段导致 data race。
	strategy := n.MergeStrategy
	if strategy == nil {
		strategy = &FirstMergeStrategy{}
	}

	return strategy.Merge(all, n.Dedup), nil
}

