package recall

import (
	"context"
	"strconv"
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

	// slots[i] 对应 Sources[i] 的结果，保证合并顺序与声明顺序一致，
	// 消除并发 goroutine 完成顺序带来的非确定性（影响 FirstMergeStrategy 等去重行为）。
	slots := make([][]*core.Item, len(n.Sources))
	eg, egCtx := errgroup.WithContext(ctx)

	sem := make(chan struct{}, n.MaxConcurrent)
	if n.MaxConcurrent <= 0 {
		close(sem)
	}

	for i, src := range n.Sources {
		idx := i
		s := src
		priority := i
		if n.SourcePriorities != nil {
			if customPriority, ok := n.SourcePriorities[s.Name()]; ok {
				priority = customPriority
			}
		}

		eg.Go(func() error {
			if n.MaxConcurrent > 0 {
				sem <- struct{}{}
				defer func() { <-sem }()
			}

			recallCtx := egCtx
			if n.Timeout > 0 {
				var cancel context.CancelFunc
				recallCtx, cancel = context.WithTimeout(egCtx, n.Timeout)
				defer cancel()
			}

			items, err := s.Recall(recallCtx, rctx)
			if err != nil {
				handler := n.ErrorHandler
				if handler == nil {
					handler = &IgnoreErrorHandler{}
				}
				handledItems, handleErr := handler.HandleError(recallCtx, s, err, rctx)
				if handleErr != nil {
					return handleErr
				}
				items = handledItems
			}

			// recall_source：直接覆盖（不合并），标识当前 Fanout 分配的来源名。
			// recall_priority：仅在 Source 未自行设置时写入 Fanout 分配的索引。
			priorityStr := strconv.Itoa(priority)
			sourceLbl := utils.Label{Value: s.Name(), Source: "recall"}
			priorityLbl := utils.Label{Value: priorityStr, Source: "recall"}
			for _, it := range items {
				if it == nil {
					continue
				}
				if it.Labels == nil {
					it.Labels = make(map[string]utils.Label)
				}
				it.Labels["recall_source"] = sourceLbl
				if _, exists := it.Labels["recall_priority"]; !exists {
					it.Labels["recall_priority"] = priorityLbl
				}
			}

			slots[idx] = items
			return nil
		})
	}

	if err := eg.Wait(); err != nil {
		return nil, err
	}

	// 按声明顺序拼合
	var all []*core.Item
	for _, s := range slots {
		all = append(all, s...)
	}

	// 合并策略（必需）
	// 使用局部变量，避免在并发请求中写共享字段导致 data race。
	strategy := n.MergeStrategy
	if strategy == nil {
		strategy = &FirstMergeStrategy{}
	}

	return strategy.Merge(all, n.Dedup), nil
}

