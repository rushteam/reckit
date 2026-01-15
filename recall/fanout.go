package recall

import (
	"context"
	"sync"
	"time"

	"golang.org/x/sync/errgroup"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pipeline"
	"github.com/rushteam/reckit/pkg/utils"
)

// MergeStrategy 是合并策略接口，用于自定义多路召回结果的合并逻辑。
type MergeStrategy interface {
	// Merge 合并多个召回源的结果
	// items: 所有召回源的结果（可能包含重复物品）
	// dedup: 是否启用去重
	// 返回: 合并后的物品列表
	Merge(items []*core.Item, dedup bool) []*core.Item
}

// FirstMergeStrategy 是默认的合并策略：按 ID 去重，保留第一个出现的。
type FirstMergeStrategy struct{}

func (s *FirstMergeStrategy) Merge(items []*core.Item, dedup bool) []*core.Item {
	if !dedup {
		return items
	}
	seen := make(map[string]*core.Item, len(items))
	out := make([]*core.Item, 0, len(items))
	for _, it := range items {
		if it == nil {
			continue
		}
		if old, ok := seen[it.ID]; ok {
			// 合并 labels
			for k, v := range it.Labels {
				old.PutLabel(k, v)
			}
			continue
		}
		seen[it.ID] = it
		out = append(out, it)
	}
	return out
}

// UnionMergeStrategy 是并集策略：不去重，保留所有结果。
type UnionMergeStrategy struct{}

func (s *UnionMergeStrategy) Merge(items []*core.Item, dedup bool) []*core.Item {
	return items
}

// PriorityMergeStrategy 是按优先级合并的策略。
// 优先级由 Source 的索引决定（索引越小优先级越高），或通过 PriorityWeights 自定义。
type PriorityMergeStrategy struct {
	// PriorityWeights 自定义优先级权重（可选）
	// key: Source 名称，value: 优先级（值越小优先级越高）
	// 如果未设置，则使用 Source 在数组中的索引作为优先级
	PriorityWeights map[string]int
}

func (s *PriorityMergeStrategy) Merge(items []*core.Item, dedup bool) []*core.Item {
	if !dedup {
		return items
	}
	seen := make(map[string]*core.Item, len(items))
	for _, it := range items {
		if it == nil {
			continue
		}
		old, exists := seen[it.ID]
		if !exists {
			seen[it.ID] = it
			continue
		}
		// 比较优先级
		oldPriority := s.getPriority(old)
		newPriority := s.getPriority(it)
		// 保留优先级更高的（值更小）
		if newPriority < oldPriority {
			seen[it.ID] = it
		} else {
			// 合并 labels
			for k, v := range it.Labels {
				old.PutLabel(k, v)
			}
		}
	}
	out := make([]*core.Item, 0, len(seen))
	for _, it := range seen {
		out = append(out, it)
	}
	return out
}

func (s *PriorityMergeStrategy) getPriority(item *core.Item) int {
	// 优先使用自定义权重
	if s.PriorityWeights != nil {
		if sourceLbl, ok := item.Labels["recall_source"]; ok {
			if weight, ok := s.PriorityWeights[sourceLbl.Value]; ok {
				return weight
			}
		}
	}
	// 回退到从 label 读取（由 Fanout 设置）
	if lbl, ok := item.Labels["recall_priority"]; ok {
		if len(lbl.Value) > 0 {
			return int(lbl.Value[0] - '0')
		}
	}
	return 999 // 默认最低优先级
}

// ErrorHandler 是错误处理策略接口，用于自定义召回源失败时的处理逻辑。
type ErrorHandler interface {
	// HandleError 处理召回源错误
	// source: 失败的召回源
	// err: 错误信息
	// rctx: 推荐上下文
	// 返回: 处理后的物品列表和错误（如果返回错误，将中断 Pipeline）
	HandleError(source Source, err error, rctx *core.RecommendContext) ([]*core.Item, error)
}

// IgnoreErrorHandler 是忽略错误的策略：返回空结果，不中断其他召回源。
type IgnoreErrorHandler struct{}

func (h *IgnoreErrorHandler) HandleError(source Source, err error, rctx *core.RecommendContext) ([]*core.Item, error) {
	return nil, nil // 返回空结果，不中断
}

// RetryErrorHandler 是重试策略（示例实现，实际使用时需要更完整的实现）。
type RetryErrorHandler struct {
	MaxRetries int
	RetryDelay time.Duration
}

func (h *RetryErrorHandler) HandleError(source Source, err error, rctx *core.RecommendContext) ([]*core.Item, error) {
	// 简化实现：实际使用时需要实现真正的重试逻辑
	return nil, nil
}

// FallbackErrorHandler 是降级策略：使用备用召回源。
type FallbackErrorHandler struct {
	FallbackSource Source
}

func (h *FallbackErrorHandler) HandleError(source Source, err error, rctx *core.RecommendContext) ([]*core.Item, error) {
	if h.FallbackSource != nil {
		// 使用备用召回源
		return h.FallbackSource.Recall(context.Background(), rctx)
	}
	return nil, nil
}

// Fanout 是一个 Recall Node：并发执行多个召回源，并合并结果。
// 支持超时、限流、优先级合并策略。
type Fanout struct {
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

func (n *Fanout) Name() string        { return "recall.fanout" }
func (n *Fanout) Kind() pipeline.Kind { return pipeline.KindRecall }

func (n *Fanout) Process(
	ctx context.Context,
	rctx *core.RecommendContext,
	_ []*core.Item,
) ([]*core.Item, error) {
	if len(n.Sources) == 0 {
		return nil, nil
	}

	var (
		mu    sync.Mutex
		all   []*core.Item
		eg, _ = errgroup.WithContext(ctx)
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

			// 超时控制
			recallCtx := ctx
			if n.Timeout > 0 {
				var cancel context.CancelFunc
				recallCtx, cancel = context.WithTimeout(ctx, n.Timeout)
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
				it.PutLabel("recall_priority", utils.Label{Value: string(rune('0' + priority)), Source: "recall"})
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
	if n.MergeStrategy == nil {
		// 如果没有设置，使用默认策略
		n.MergeStrategy = &FirstMergeStrategy{}
	}
	
	return n.MergeStrategy.Merge(all, n.Dedup), nil
}

